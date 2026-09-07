#!/usr/bin/env python3
"""Run the unchanged released HAC evaluators on public reconstruction STLs.

This optional adapter is independent of inference. Obtain the public target
meshes and evaluation source from the official data release. No MATLAB port,
target simplification, or cached projection approximation is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from hac_forward import stl

SOURCE_SHA256 = {
    "README_HAC2026_evaluation_codes.txt": "5d38706471454e90dd00ed7eb94b4b032dc9315785def38e51fcfbf2b1342903",
    "Voxel measure - Python/metrics.py": "5a4663e119832e567971b5791463e6e176c054fa32ca6fa188923d0c9d13ce00",
    "Voxel measure - Python/comparison_example.py": "823d6a1b5b070628b123b8f93eac2ecdc7e5e0633dda8ee001911028bc205e38",
    "Projection measure - MATLAB/twoDmetric.m": "2dd2970e6711176d9803f3afbba9ff646277337574b3698349507af19e44c0ba",
    "Projection measure - MATLAB/ComparisonExample2Dprojection.m": "12413a07679fecdffd669fbb0769cbcd411838f37f67c6307212a122a1d41a6e",
}


def file_identity(path: Path) -> dict:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_sources(source_dir: Path) -> dict:
    records = {}
    for name, expected in SOURCE_SHA256.items():
        record = file_identity(source_dir / name)
        if record["sha256"] != expected:
            raise ValueError(f"official evaluation source checksum mismatch: {name}")
        records[name] = record
    return records


def load_voxel_evaluator(source_dir: Path):
    verify_sources(source_dir)
    spec = importlib.util.spec_from_file_location(
        "hac_official_released_voxel_metrics", source_dir / "Voxel measure - Python/metrics.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot import the released Python evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def voxel_scores(module, target: Path, candidate: Path, pitch: float) -> dict:
    started = time.monotonic()
    first, second = module.relative_volume_difference_voxelized(
        str(target.resolve()), str(candidate.resolve()), pitch=pitch
    )
    first, second = float(first), float(second)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (first, second)):
        raise ValueError("official voxel losses must be finite and within [0,1]")
    return {
        "measure1_loss": first,
        "measure2_loss": second,
        "voxel_iou_score": 1 - first,
        "voxel_score": 1 - second,
        "elapsed_seconds": time.monotonic() - started,
    }


def normalize_target(raw_path: Path, normalized_path: Path, model: int) -> dict:
    """Preserve every triangle; translate z only, then scale uniformly."""
    raw = file_identity(raw_path)
    triangles = stl.validate_triangle_array(stl.load_stl_triangles(raw_path))
    points = triangles.reshape(-1, 3)
    low, high = points.min(axis=0), points.max(axis=0)
    midpoint = (low[2] + high[2]) / 2
    half_extent = (high[2] - low[2]) / 2
    if not np.isfinite(half_extent) or half_extent <= 0:
        raise ValueError("public target has invalid z extent")
    offset = np.array([0.0, 0.0, midpoint])
    normalized = (triangles - offset) / half_extent
    stl.write_ascii_stl(
        normalized_path, normalized, solid_name=f"normalized_public_model{model:02d}"
    )
    if file_identity(raw_path) != raw:
        raise RuntimeError("raw target changed during normalization")
    return {
        "raw_stl": raw,
        "normalized_stl": file_identity(normalized_path),
        "normalization": {
            "offset": offset.tolist(),
            "z_half_extent_divisor": float(half_extent),
            "center_xy": False,
            "all_triangles_preserved": True,
            "triangle_count": int(len(triangles)),
            "serialization": "ASCII_STL_9_significant_digits",
        },
    }


def run_matlab(jobs: list[dict], source_dir: Path, executable: Path, output_dir: Path) -> dict:
    driver = Path(__file__).with_name("hac_evaluate_public.m").resolve()
    request_path = output_dir / "matlab_request.json"
    result_path = output_dir / "matlab_result.json"
    request = {
        "source_dir": str((source_dir / "Projection measure - MATLAB").resolve()),
        "jobs": jobs,
        "result_path": str(result_path.resolve()),
    }
    write_json(request_path, request)

    def quote(path: Path) -> str:
        return "'" + str(path).replace("'", "''") + "'"

    expression = (
        f"addpath({quote(driver.parent)}); hac_evaluate_public({quote(request_path.resolve())});"
    )
    with (output_dir / "matlab.log").open("w") as log:
        process = subprocess.run(
            [str(executable.resolve()), "-batch", expression],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(f"native MATLAB evaluation failed; see {output_dir / 'matlab.log'}")
    result = json.loads(result_path.read_text())
    rows = result["rows"]
    if isinstance(rows, dict):
        rows = [rows]
    if len(rows) != len(jobs):
        raise ValueError("native MATLAB result has an incorrect row count")
    for job, row in zip(jobs, rows, strict=True):
        if (
            row["model_id"] != job["model_id"]
            or row["theta"] != job["theta"]
            or row["status"] != "ok"
        ):
            raise ValueError("native MATLAB result does not match its requested job")
        if not math.isfinite(row["projection_score"]) or not 0 <= row["projection_score"] <= 1:
            raise ValueError("native MATLAB score is invalid")
    result["rows"] = rows
    return result


def evaluate(args: argparse.Namespace) -> dict:
    if not math.isfinite(args.pitch) or args.pitch <= 0 or not math.isfinite(args.theta):
        raise ValueError("pitch must be positive and finite; theta must be finite")
    models = [int(value) for value in args.models.split(",")]
    if (
        not models
        or len(set(models)) != len(models)
        or any(model not in (1, 2, 3) for model in models)
    ):
        raise ValueError("--models must contain distinct public model IDs from 1,2,3")
    sources = verify_sources(args.evaluation_dir)
    versions = {
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "meshio", "trimesh")
    }
    for name, expected in (("meshio", "5.3.5"), ("trimesh", "4.12.2")):
        if versions[name] != expected:
            raise ValueError(f"install the evaluation extra: required {name}=={expected}")
    executable = args.matlab.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"MATLAB executable missing: {executable}")
    candidates = []
    for model in models:
        model_dir = args.run_dir / f"model{model:02d}"
        run_path = model_dir / "run.json"
        run = json.loads(run_path.read_text())
        candidate = file_identity(model_dir / "submission_shape.stl")
        if run["status"] != "complete" or run["model_id"] != model:
            raise ValueError(f"Model {model} reconstruction is not complete")
        recorded = run["artifacts"]["submission_shape.stl"]
        if any(candidate[key] != recorded[key] for key in ("sha256", "size_bytes")):
            raise ValueError(f"Model {model} STL differs from its run manifest")
        raw = args.data_dir / f"AsteroidModel{model:02d}_shape_public" / f"asteroid{model}.stl"
        if not raw.is_file():
            raise FileNotFoundError(raw)
        candidates.append(
            {
                "model_id": model,
                "candidate_stl": candidate,
                "run_manifest": file_identity(run_path),
                "raw_target_path": raw,
            }
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    local_sources = {
        "python_adapter": file_identity(Path(__file__)),
        "matlab_driver": file_identity(Path(__file__).with_name("hac_evaluate_public.m")),
        "stl_loader_writer": file_identity(Path(stl.__file__)),
    }
    report = {
        "schema_version": 1,
        "status": "running",
        "settings": {
            "pitch": args.pitch,
            "theta_degrees": args.theta,
            "voxel_score": "1-measure2",
            "projection_score": "unmodified_native_matlab_twoDmetric",
            "combined_score": "voxel_score+projection_score",
            "calibration_convention": "released_example_settings_not_final_leaderboard_guarantee",
            "candidate_normalization": "none",
        },
        "official_sources": sources,
        "local_sources": local_sources,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": versions,
            "matlab_executable": str(executable),
        },
        "rows": [],
    }
    report_path = output / "evaluation.json"
    write_json(report_path, report)
    try:
        module = load_voxel_evaluator(args.evaluation_dir)
        jobs = []
        for candidate in candidates:
            model = candidate["model_id"]
            target_path = output / "targets" / f"target_model{model:02d}.stl"
            target = normalize_target(candidate["raw_target_path"], target_path, model)
            row = {key: value for key, value in candidate.items() if key != "raw_target_path"}
            row["target"] = target
            row["voxel"] = voxel_scores(
                module, target_path, Path(candidate["candidate_stl"]["path"]), args.pitch
            )
            report["rows"].append(row)
            jobs.append(
                {
                    "model_id": model,
                    "target_stl": str(target_path),
                    "candidate_stl": candidate["candidate_stl"]["path"],
                    "theta": args.theta,
                }
            )
            write_json(report_path, report)
            print(f"Model {model:02d}: voxel score {row['voxel']['voxel_score']:.9f}", flush=True)
        native = run_matlab(jobs, args.evaluation_dir, executable, output)
        report["runtime"]["matlab"] = native["runtime"]
        for row, native_row in zip(report["rows"], native["rows"], strict=True):
            row["projection"] = native_row
            row["total_score"] = row["voxel"]["voxel_score"] + native_row["projection_score"]
        unchanged = list(sources.values()) + list(local_sources.values())
        for row in report["rows"]:
            unchanged.extend(
                [
                    row["candidate_stl"],
                    row["run_manifest"],
                    row["target"]["raw_stl"],
                    row["target"]["normalized_stl"],
                ]
            )
        if any(file_identity(Path(record["path"])) != record for record in unchanged):
            raise RuntimeError("an evaluation input or implementation changed during the run")
        report["mean_total_score"] = sum(row["total_score"] for row in report["rows"]) / len(
            report["rows"]
        )
        report["status"] = "complete"
        write_json(report_path, report)
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(report_path, report)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matlab", type=Path, required=True)
    parser.add_argument("--models", default="1,2,3")
    parser.add_argument("--pitch", type=float, default=0.05)
    parser.add_argument("--theta", type=float, default=0.0)
    result = evaluate(parser.parse_args())
    print(f"Complete: mean combined score {result['mean_total_score']:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
