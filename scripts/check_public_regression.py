#!/usr/bin/env python3
"""Compare a fresh public replay to the pre-extraction calibrated references."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from hac_forward.stl import load_stl_triangles

ROOT = Path(__file__).resolve().parents[1]


def check(run_dir: Path) -> dict:
    fixture = ROOT / "tests/fixtures/public_reference.npz"
    manifest = json.loads((fixture.with_suffix(".json")).read_text())
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != manifest["npz_sha256"]:
        raise ValueError("public reference fixture checksum mismatch")
    tolerance = manifest["absolute_tolerance"]
    records = []
    with np.load(fixture, allow_pickle=False) as reference:
        for row in manifest["records"]:
            model = row["model_id"]
            prefix = f"model{model:02d}"
            directory = run_dir / prefix
            run = json.loads((directory / "run.json").read_text())
            if run["status"] != "complete" or run["public_target_read"]:
                raise ValueError(f"{prefix}: run is incomplete or truth-dependent")
            if run["input"]["sha256"] != row["input_sha256"]:
                raise ValueError(f"{prefix}: input differs from calibrated release")
            errors = {}
            with np.load(directory / "phase1/surface_measure.npz", allow_pickle=False) as actual:
                for name in ("masses", "normals", "harmonic_coefficients"):
                    errors[name] = float(
                        np.max(np.abs(actual[name] - reference[prefix + "_" + name]))
                    )
            with np.load(directory / "phase1/fit_curves.npz", allow_pickle=False) as actual:
                errors["predicted"] = float(
                    np.max(np.abs(actual["predicted"] - reference[prefix + "_predicted"]))
                )
            path = directory / "submission_shape.stl"
            actual_points = load_stl_triangles(path).reshape(-1, 3)
            expected_points = reference[prefix + "_triangles"].reshape(-1, 3)
            errors["vertex_hausdorff"] = max(
                float(np.max(cKDTree(actual_points).query(expected_points)[0])),
                float(np.max(cKDTree(expected_points).query(actual_points)[0])),
            )
            errors["triangle_weighted_rmse"] = abs(
                run["metrics"]["triangle_weighted_rmse"] - row["triangle_weighted_rmse"]
            )
            passed = (
                all(np.isfinite(error) and error <= tolerance for error in errors.values())
                and run["metrics"]["effective_k"] == row["effective_k"]
            )
            records.append(
                {
                    "model_id": model,
                    "passed": passed,
                    "errors": errors,
                    "stl_byte_identical": hashlib.sha256(path.read_bytes()).hexdigest()
                    == row["original_stl_sha256"],
                }
            )
    return {
        "schema_version": 1,
        "all_passed": all(row["passed"] for row in records),
        "tolerance": tolerance,
        "reference_sha256": manifest["npz_sha256"],
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.run_dir)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
