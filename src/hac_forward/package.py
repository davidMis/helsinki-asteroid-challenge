"""Certify and atomically package exactly the seven HAC challenge models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .certification import certify_stl

MODELS = tuple(range(4, 11))
REQUIRED_ARTIFACTS = frozenset(
    {
        "framed_surface_measure.npz",
        "intrinsic_shape.stl",
        "lightcurves.npz",
        "phase1/fit_curves.npz",
        "phase1/optimizer.json",
        "phase1/surface_measure.json",
        "phase1/surface_measure.npz",
        "quantized/surface_measure.json",
        "quantized/surface_measure.npz",
        "reconstruction.json",
        "submission_shape.stl",
    }
)
NUMERICAL_PACKAGES = {
    "jax": "0.10.1",
    "jaxlib": "0.10.1",
    "numpy": "2.3.5",
    "scipy": "1.17.1",
    "ml_dtypes": "0.5.4",
    "opt_einsum": "3.4.0",
}


def validate_runtime(runtime: dict) -> None:
    """Require the pinned numerical environment and actual double precision."""
    try:
        python_version = tuple(int(part) for part in runtime["python"].split(".")[:2])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid recorded Python version") from exc
    if not (3, 11) <= python_version < (3, 14):
        raise ValueError("recorded Python version is outside >=3.11,<3.14")
    if runtime.get("jax_enable_x64") is not True or runtime.get("jax_backend") != "cpu":
        raise ValueError("production runtime must use float64 JAX on CPU")
    if runtime.get("packages") != NUMERICAL_PACKAGES:
        raise ValueError("production numerical package versions differ from the frozen pins")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def package(run_dir: Path, output_dir: Path, config_path: Path) -> dict:
    """Require complete, current-source, input-pinned and certified runs."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite package: {output_dir}")
    config = json.loads(config_path.read_text())
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    source_dir = Path(__file__).resolve().parent
    sources = {p.name: sha256(p) for p in sorted(source_dir.glob("*.py"))}
    repository = config_path.resolve().parent.parent
    data_manifest = json.loads(
        (repository / "data_manifest/real_intensity_20260907.json").read_text()
    )
    inputs = {row["model_id"]: row for row in data_manifest["files"]}
    actual_models = {
        int(p.name.removeprefix("model")) for p in run_dir.glob("model[0-9][0-9]") if p.is_dir()
    }
    if actual_models != set(MODELS):
        raise ValueError(f"expected only model directories 4–10; found {sorted(actual_models)}")
    records = {}
    for model in MODELS:
        directory = run_dir / f"model{model:02d}"
        run_path = directory / "run.json"
        run = json.loads(run_path.read_text())
        if run["status"] != "complete" or run["model_id"] != model or run["public_target_read"]:
            raise ValueError(f"model {model}: incomplete, misidentified or truth-dependent run")
        if run["config"] != config or run["config_sha256"] != config_hash:
            raise ValueError(f"model {model}: configuration mismatch")
        if run["source_sha256"] != sources:
            raise ValueError(f"model {model}: source differs from the production run")
        if run["input"]["sha256"] != inputs[model]["sha256"]:
            raise ValueError(f"model {model}: input differs from the certified release")
        validate_runtime(run["runtime"])
        missing = REQUIRED_ARTIFACTS - set(run["artifacts"])
        if missing:
            raise ValueError(f"model {model}: incomplete artifact provenance: {sorted(missing)}")
        for name, artifact in run["artifacts"].items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("artifact path escapes model directory")
            if artifact["path"] != name:
                raise ValueError(f"model {model}: artifact path/record mismatch: {name}")
            if sha256(path) != artifact["sha256"] or path.stat().st_size != artifact["size_bytes"]:
                raise ValueError(f"model {model}: modified artifact {name}")
        metrics = run["metrics"]
        if any(not np.isfinite(value) or value < 0 for value in metrics.values()):
            raise ValueError(f"model {model}: nonfinite or negative diagnostics")
        limits = config["verification"]
        for name, limit in (
            ("parent_prediction_max_abs", limits["parent_prediction_max_abs_tolerance"]),
            ("triangle_prediction_max_abs", limits["triangle_prediction_max_abs_tolerance"]),
            ("framed_closure_norm", limits["framed_closure_tolerance"]),
            ("area_log_rmse", config["phase2"]["area_log_rmse_tolerance"]),
        ):
            if metrics[name] > limit:
                raise ValueError(f"model {model}: failed {name} gate")
        stl = directory / "submission_shape.stl"
        expected_stl_hash = run["artifacts"]["submission_shape.stl"]["sha256"]
        certificate = certify_stl(stl, cylinder_radius=config["cylinder_radii"][str(model)])
        if certificate["stl"]["sha256"] != expected_stl_hash:
            raise RuntimeError(f"model {model}: candidate changed before certification")
        if not certificate["valid"]:
            raise ValueError(f"model {model}: failed convex certificate: {certificate['errors']}")
        records[model] = {
            "model_id": model,
            "stl": f"AsteroidModel{model:02d}.stl",
            "stl_sha256": expected_stl_hash,
            "input": inputs[model],
            "run_manifest_sha256": sha256(run_path),
            "runtime": run["runtime"],
            "metrics": metrics,
            "certificate": certificate,
            "elapsed_seconds": run["elapsed_seconds"],
        }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hac-package-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "payload"
        staging.mkdir()
        for model, record in records.items():
            destination = staging / record["stl"]
            destination.write_bytes(
                (run_dir / f"model{model:02d}/submission_shape.stl").read_bytes()
            )
            if sha256(destination) != record["stl_sha256"]:
                raise RuntimeError("candidate changed during packaging")
            write_json(staging / f"certificates/model{model:02d}.json", record["certificate"])
        manifest = {
            "schema_version": 1,
            "status": "ready_for_first_submission",
            "models": list(MODELS),
            "config_sha256": config_hash,
            "config": config,
            "source_sha256": sources,
            "records": [
                {key: value for key, value in record.items() if key != "certificate"}
                for record in records.values()
            ],
            "validation_scope": "Topology and HAC frame; numerical convex-boundary single-cover certificate; strict facet-area fit; emitted-triangle brightness replay. No hidden-shape accuracy claim.",
        }
        write_json(staging / "manifest.json", manifest)
        checksums = "".join(
            f"{sha256(path)}  {path.relative_to(staging).as_posix()}\n"
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        )
        (staging / "SHA256SUMS").write_text(checksums)
        os.replace(staging, output_dir)
    return {
        "output_dir": str(output_dir),
        "models": list(MODELS),
        "status": "ready_for_first_submission",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/submission.json"))
    args = parser.parse_args()
    print(json.dumps(package(args.run_dir, args.output_dir, args.config), indent=2))


if __name__ == "__main__":
    main()
