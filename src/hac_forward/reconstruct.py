"""Reconstruct released HAC models from real intensity tables in one command.

This command reads lightcurves and released acquisition/cylinder metadata only.
Every model is reconstructed independently, in float64, with complete settings
and source identities recorded beside its serialized STL.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .area_measure_transform import pushforward_through_submission_frame
from .atomic_refinement_solver import triangle_area_measure
from .baseline import ForwardModelBaseline
from .bhs_phase1 import (
    BHSOptimizerConfig,
    fit_bhs_surface_measure,
    harmonic_coefficient_metadata,
    quadrature_masses,
)
from .constants import CYLINDER_RADII
from .gaussian_bhs_inversion import (
    GaussianBHSObjectiveConfig,
    GaussianBHSShapeConfig,
    predict_selected_gaussian_measure_table_jax,
    surface_reference_masses,
)
from .inversion import InversionTarget, prepare_inversion_target, weighted_curve_misfit_jax
from .io_utils import file_provenance, write_json
from .paths import lightcurve_path
from .semidiscrete_ot_quantization import SemidiscreteOTQuantizerConfig, semidiscrete_ot_quantize
from .stl import load_stl_triangles, write_ascii_stl
from .submission_validation import normalize_submission_frame, validate_submission_stl
from .surface_measure_artifact import SurfaceMeasureArtifact, write_surface_measure_artifact
from .variational_minkowski import VariationalMinkowskiConfig, solve_variational_minkowski


def strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): strict_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, np.generic):
        return strict_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path: Path, value: Any) -> None:
    write_json(path, strict_json(value))


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(strict_json(value), sort_keys=True).encode()).hexdigest()


def source_identity() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*.py"))
    }


def runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "scipy", "ml_dtypes", "opt_einsum")
        },
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
    }


def _complete_dataclass(cls: type, values: dict[str, Any]) -> Any:
    expected = {field.name for field in fields(cls)}
    if set(values) != expected:
        raise ValueError(
            f"{cls.__name__}: missing {sorted(expected - set(values))}, unexpected {sorted(set(values) - expected)}"
        )
    return cls(**values)


def effective_config(config: dict[str, Any], model_id: int) -> tuple[Any, ...]:
    if config["schema_version"] != 1 or config["precision"] != "float64":
        raise ValueError("unsupported configuration version or precision")
    if model_id not in CYLINDER_RADII:
        raise ValueError(f"unknown released model {model_id}")
    if config["cylinder_radii"] != {str(k): v for k, v in CYLINDER_RADII.items()}:
        raise ValueError("configuration cylinder radii differ from released metadata")
    acquisition = config["acquisition"]
    if acquisition["source"] != "real" or acquisition["curve_type"] != "intensity":
        raise ValueError("production profile supports real intensity observations")
    baseline = dict(acquisition["baseline"])
    baseline["angle_phase_shifts_degrees"] = {
        int(k): v for k, v in baseline["angle_phase_shifts_degrees"].items()
    }
    calibration = _complete_dataclass(ForwardModelBaseline, baseline)
    shape = dict(config["phase1"]["shape"])
    if shape["reference_spheroid_radius"] != "released_cylinder_radius":
        raise ValueError("reference radius must use released metadata")
    shape["reference_spheroid_radius"] = CYLINDER_RADII[model_id]
    optimizer = dict(config["phase1"]["optimizer"])
    if config["phase1"]["optimizer_seed_rule"] != "base_seed_plus_model_id":
        raise ValueError("unknown optimizer seed rule")
    optimizer["seed"] += model_id
    if not config["frame"]["match_radius"] or (
        config["frame"]["z_min"],
        config["frame"]["z_max"],
    ) != (-1.0, 1.0):
        raise ValueError("production normalization requires released cylinder and z contacts")
    return (
        calibration,
        _complete_dataclass(GaussianBHSShapeConfig, shape),
        _complete_dataclass(GaussianBHSObjectiveConfig, config["phase1"]["objective"]),
        _complete_dataclass(BHSOptimizerConfig, optimizer),
        _complete_dataclass(SemidiscreteOTQuantizerConfig, config["quantization"]["config"]),
        _complete_dataclass(VariationalMinkowskiConfig, config["phase2"]),
    )


def predict_measure(
    masses: np.ndarray, normals: np.ndarray, shape: GaussianBHSShapeConfig, target: InversionTarget
) -> np.ndarray:
    values = np.asarray(
        predict_selected_gaussian_measure_table_jax(
            jnp.asarray(masses, dtype=jnp.float64),
            jnp.asarray(normals, dtype=jnp.float64),
            shape,
            target,
        ),
        dtype=np.float64,
    )
    if values.shape != target.observed.shape or not np.all(np.isfinite(values)):
        raise RuntimeError("surface-measure rerender has invalid values or shape")
    return values


def weighted_rmse(predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> float:
    return float(
        np.sqrt(
            2.0
            * float(
                weighted_curve_misfit_jax(
                    jnp.asarray(predicted - observed, dtype=jnp.float64),
                    np.asarray(weights, dtype=np.float64),
                )
            )
        )
    )


def run_model(
    data_dir: Path,
    model_id: int,
    output_dir: Path,
    config: dict[str, Any],
    *,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Run a complete independent fit; an existing output is never overwritten."""
    with jax.enable_x64():
        return _run_model_x64(
            data_dir.resolve(), model_id, output_dir.resolve(), config, command=command
        )


def _run_model_x64(
    data_dir: Path,
    model_id: int,
    output_dir: Path,
    config: dict[str, Any],
    *,
    command: list[str] | None,
) -> dict[str, Any]:
    baseline, shape, objective, optimizer, quantizer, solver = effective_config(config, model_id)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    input_path = lightcurve_path(data_dir, model_id, "intensity", "real")
    record = {
        "schema_version": 1,
        "model_id": model_id,
        "status": "running",
        "public_target_read": False,
        "config": config,
        "config_sha256": identity(config),
        "input": file_provenance(input_path),
        "source_sha256": source_identity(),
        "runtime": runtime_identity(),
        "command": command or sys.argv,
        "effective": {
            "shape": asdict(shape),
            "objective": asdict(objective),
            "optimizer": asdict(optimizer),
            "quantizer": asdict(quantizer),
            "solver": asdict(solver),
        },
    }
    save_json(output_dir / "run.json", record)
    try:
        acquisition = config["acquisition"]
        target = prepare_inversion_target(
            data_dir,
            model_id=model_id,
            curve_type="intensity",
            source="real",
            n_samples=acquisition["target_samples"],
            angles_degrees=acquisition["angles_degrees"],
            view_labels_selected=acquisition["view_labels"],
            baseline=baseline,
            calibration_mode=acquisition["calibration_mode"],
        )
        print(f"Model {model_id:02d}: Phase 1 fit", flush=True)
        fit = fit_bhs_surface_measure(target, shape, objective, optimizer)
        degrees, orders, components = harmonic_coefficient_metadata(shape)
        phase1 = SurfaceMeasureArtifact(
            model_id=model_id,
            representation="dense_harmonic",
            masses=fit.masses,
            normals=fit.normals,
            quadrature_masses=quadrature_masses(shape),
            reference_masses=surface_reference_masses(shape),
            harmonic_coefficients=fit.harmonic_coefficients,
            harmonic_degrees=degrees,
            harmonic_orders=orders,
            harmonic_components=components,
            seed=optimizer.seed,
            config={
                "measure_method": "harmonic",
                "method_config": {
                    "shape": asdict(shape),
                    "objective": asdict(objective),
                    "optimizer": asdict(optimizer),
                },
                "acquisition": {**acquisition, "calibration": acquisition["calibration_mode"]},
            },
            input_data={"lightcurve": record["input"]},
            diagnostics=strict_json(
                {
                    **fit.diagnostics,
                    "objective_terms": fit.objective_terms,
                    "optimizer": fit.optimizer_diagnostics,
                    "curve_rmses": fit.curve_rmses,
                    "curve_corrs": fit.curve_corrs,
                }
            ),
            provenance={"source_sha256": record["source_sha256"], "command": record["command"]},
        )
        write_surface_measure_artifact(output_dir / "phase1", phase1)
        np.savez_compressed(
            output_dir / "phase1/fit_curves.npz",
            observed=target.observed,
            predicted=fit.predicted,
            curve_weights=target.effective_curve_weights,
            phases=target.phases,
            view_directions=target.view_directions,
            angles_degrees=np.asarray(target.angles_degrees),
            view_labels=np.asarray(target.view_labels),
            curve_rmses=fit.curve_rmses,
            curve_corrs=fit.curve_corrs,
        )
        save_json(
            output_dir / "phase1/optimizer.json",
            {"diagnostics": fit.optimizer_diagnostics, "adam_history": fit.adam_history},
        )
        print(f"Model {model_id:02d}: semidiscrete OT quantization", flush=True)
        count = config["quantization"]["requested_facets"]
        quant = semidiscrete_ot_quantize(fit.masses, fit.normals, count, config=quantizer)
        quant_artifact = SurfaceMeasureArtifact(
            model_id=model_id,
            representation="quantized_atomic",
            masses=quant.masses,
            normals=quant.normals,
            assignments=quant.assignments,
            cluster_source_masses=quant.cluster_source_masses,
            seed=quantizer.seed,
            config={
                "method": "semidiscrete_ot",
                "requested_facets": count,
                "config": asdict(quantizer),
            },
            input_data={"phase1": file_provenance(output_dir / "phase1/surface_measure.npz")},
            diagnostics={"quantization": asdict(quant.diagnostics)},
            provenance={"source_sha256": record["source_sha256"]},
        )
        write_surface_measure_artifact(output_dir / "quantized", quant_artifact)
        print(
            f"Model {model_id:02d}: strict Minkowski realization ({len(quant.masses)} facets)",
            flush=True,
        )
        reconstruction = solve_variational_minkowski(quant.masses, quant.normals, config=solver)
        intrinsic = reconstruction.mesh.vertices
        triangles, frame = normalize_submission_frame(
            intrinsic, model_id=model_id, match_radius=True
        )
        pushed = pushforward_through_submission_frame(quant.masses, quant.normals, frame)
        write_ascii_stl(
            output_dir / "intrinsic_shape.stl",
            intrinsic,
            solid_name=f"bhs_intrinsic_model{model_id:02d}",
        )
        write_ascii_stl(
            output_dir / "submission_shape.stl",
            triangles,
            solid_name=f"bhs_submission_model{model_id:02d}",
        )
        np.savez_compressed(
            output_dir / "framed_surface_measure.npz",
            masses=pushed.masses,
            normals=pushed.normals,
            pushed_areas=pushed.pushed_areas,
            linear_map=pushed.linear_map,
            cofactor_map=pushed.cofactor_map,
            source_masses=quant.masses,
            source_normals=quant.normals,
        )
        validation = validate_submission_stl(
            output_dir / "submission_shape.stl", cylinder_radius=CYLINDER_RADII[model_id]
        )
        validation.require_valid()
        save_json(
            output_dir / "reconstruction.json",
            {
                "schema_version": 1,
                "model_id": model_id,
                "solver_config": asdict(solver),
                "solver_diagnostics": asdict(reconstruction.diagnostics),
                "submission_frame": frame,
                "framed_surface_measure": asdict(pushed.diagnostics),
                "validation": validation.to_dict(),
            },
        )
        triangle_masses, triangle_normals = triangle_area_measure(
            load_stl_triangles(output_dir / "submission_shape.stl")
        )
        parent_prediction = predict_measure(fit.masses, fit.normals, shape, target)
        intrinsic_prediction = predict_measure(quant.masses, quant.normals, shape, target)
        framed_prediction = predict_measure(pushed.masses, pushed.normals, shape, target)
        triangle_prediction = predict_measure(triangle_masses, triangle_normals, shape, target)
        metrics = {
            "requested_k": count,
            "effective_k": len(quant.masses),
            "phase1_weighted_rmse": weighted_rmse(
                fit.predicted, target.observed, target.effective_curve_weights
            ),
            "intrinsic_weighted_rmse": weighted_rmse(
                intrinsic_prediction, target.observed, target.effective_curve_weights
            ),
            "framed_weighted_rmse": weighted_rmse(
                framed_prediction, target.observed, target.effective_curve_weights
            ),
            "triangle_weighted_rmse": weighted_rmse(
                triangle_prediction, target.observed, target.effective_curve_weights
            ),
            "parent_prediction_max_abs": float(np.max(np.abs(parent_prediction - fit.predicted))),
            "triangle_prediction_max_abs": float(
                np.max(np.abs(triangle_prediction - framed_prediction))
            ),
            "framed_closure_norm": float(np.linalg.norm(pushed.masses @ pushed.normals)),
            "area_log_rmse": reconstruction.diagnostics.area_log_rmse,
        }
        verification = config["verification"]
        for key, tolerance in [
            ("parent_prediction_max_abs", "parent_prediction_max_abs_tolerance"),
            ("triangle_prediction_max_abs", "triangle_prediction_max_abs_tolerance"),
            ("framed_closure_norm", "framed_closure_tolerance"),
        ]:
            if metrics[key] > verification[tolerance]:
                raise RuntimeError(f"{key}={metrics[key]} exceeds {verification[tolerance]}")
        np.savez_compressed(
            output_dir / "lightcurves.npz",
            observed=target.observed,
            phase1_predicted=fit.predicted,
            intrinsic_predicted=intrinsic_prediction,
            framed_predicted=framed_prediction,
            emitted_triangle_predicted=triangle_prediction,
            curve_weights=target.effective_curve_weights,
        )
        record.update(
            status="complete", metrics=metrics, elapsed_seconds=time.monotonic() - started
        )
        record["artifacts"] = {
            str(path.relative_to(output_dir)): file_provenance(path, relative_to=output_dir)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path.name != "run.json"
        }
        save_json(output_dir / "run.json", record)
        print(
            f"Model {model_id:02d}: complete, triangle RMSE {metrics['triangle_weighted_rmse']:.8f}",
            flush=True,
        )
        return record
    except Exception as exc:
        record.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.monotonic() - started,
        )
        save_json(output_dir / "run.json", record)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models", default="4,5,6,7,8,9,10")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/submission.json"))
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    models = [int(value) for value in args.models.split(",")]
    if not models or len(set(models)) != len(models):
        parser.error("--models must contain distinct released model IDs")
    for model in models:
        effective_config(config, model)
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    records = []
    for model in models:
        record = run_model(args.data_dir, model, args.output_dir / f"model{model:02d}", config)
        records.append(
            {
                "model_id": model,
                "status": record["status"],
                "metrics": record["metrics"],
                "run": f"model{model:02d}/run.json",
            }
        )
        save_json(
            args.output_dir / "summary.json",
            {"schema_version": 1, "models": records, "config_sha256": identity(config)},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
