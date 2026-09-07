"""Package boundary tests using synthetic certified cubes and hash records.

These fixtures exercise packaging integrity, not inverse reconstruction accuracy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import ConvexHull

import hac_forward.package as packaging
from hac_forward.stl import write_ascii_stl

ROOT = Path(__file__).resolve().parents[1]


def cube(radius: float) -> np.ndarray:
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    vertices[:, :2] *= radius / np.sqrt(2)
    hull = ConvexHull(vertices)
    triangles = vertices[hull.simplices].copy()
    for index, triangle in enumerate(triangles):
        if (
            np.dot(
                np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]),
                hull.equations[index, :3],
            )
            < 0
        ):
            triangles[index] = triangle[[0, 2, 1]]
    return triangles


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runs = self.root / "runs"
        self.output = self.root / "submission"
        self.config_path = self.root / "configs/submission.json"
        self.config_path.parent.mkdir()
        shutil.copy2(ROOT / "configs/submission.json", self.config_path)
        self.config = json.loads(self.config_path.read_text())
        inputs = [
            {"model_id": m, "sha256": hashlib.sha256(f"input-{m}".encode()).hexdigest()}
            for m in packaging.MODELS
        ]
        packaging.write_json(
            self.root / "data_manifest/real_intensity_20260907.json", {"files": inputs}
        )
        source = Path(packaging.__file__).parent
        sources = {p.name: packaging.sha256(p) for p in sorted(source.glob("*.py"))}
        config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        for row in inputs:
            model = row["model_id"]
            directory = self.runs / f"model{model:02d}"
            directory.mkdir(parents=True)
            for name in packaging.REQUIRED_ARTIFACTS:
                path = directory / name
                path.parent.mkdir(exist_ok=True)
                if name.endswith(".stl"):
                    write_ascii_stl(path, cube(self.config["cylinder_radii"][str(model)]))
                elif name.endswith(".npz"):
                    np.savez_compressed(path, fixture=np.array([model]))
                else:
                    packaging.write_json(path, {"synthetic_test_fixture": True})
            artifacts = {
                name: {
                    "path": name,
                    "sha256": packaging.sha256(directory / name),
                    "size_bytes": (directory / name).stat().st_size,
                }
                for name in packaging.REQUIRED_ARTIFACTS
            }
            run = {
                "model_id": model,
                "status": "complete",
                "public_target_read": False,
                "config": self.config,
                "config_sha256": config_hash,
                "source_sha256": sources,
                "input": row,
                "artifacts": artifacts,
                "elapsed_seconds": 0.1,
                "runtime": {
                    "python": "3.12.14",
                    "jax_enable_x64": True,
                    "jax_backend": "cpu",
                    "packages": packaging.NUMERICAL_PACKAGES,
                },
                "metrics": {
                    "requested_k": 64,
                    "effective_k": 6,
                    "parent_prediction_max_abs": 0.0,
                    "triangle_prediction_max_abs": 0.0,
                    "framed_closure_norm": 0.0,
                    "area_log_rmse": 0.0,
                },
            }
            packaging.write_json(directory / "run.json", run)

    def read(self, model=4):
        return json.loads((self.runs / f"model{model:02d}/run.json").read_text())

    def write(self, run, model=4):
        packaging.write_json(self.runs / f"model{model:02d}/run.json", run)

    def package(self):
        return packaging.package(self.runs, self.output, self.config_path)

    def test_complete_seven_model_package_and_checksums(self):
        result = self.package()
        self.assertEqual(result["status"], "ready_for_first_submission")
        self.assertEqual(len(list(self.output.glob("*.stl"))), 7)
        self.assertEqual(len(list((self.output / "certificates").glob("*.json"))), 7)
        for line in (self.output / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(packaging.sha256(self.output / name), expected)

    def test_missing_model_rejected(self):
        shutil.rmtree(self.runs / "model10")
        with self.assertRaisesRegex(ValueError, "expected only model"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_each_required_artifact_record_is_mandatory(self):
        original = self.read()
        for name in packaging.REQUIRED_ARTIFACTS:
            with self.subTest(name=name):
                run = json.loads(json.dumps(original))
                del run["artifacts"][name]
                self.write(run)
                with self.assertRaisesRegex(ValueError, "incomplete artifact provenance"):
                    self.package()
        self.assertFalse(self.output.exists())

    def test_modified_artifact_rejected(self):
        path = self.runs / "model04/lightcurves.npz"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "modified artifact"):
            self.package()

    def test_false_precision_unpinned_packages_and_wrong_backend_rejected(self):
        original = self.read()
        changes = [
            {"jax_enable_x64": False},
            {"jax_enable_x64": 1},
            {"jax_backend": "gpu"},
            {"python": "3.14.0"},
            {"python": "3.10.1"},
            {"packages": {**packaging.NUMERICAL_PACKAGES, "ml_dtypes": "0.6.0"}},
        ]
        for change in changes:
            with self.subTest(change=change):
                run = json.loads(json.dumps(original))
                run["runtime"].update(change)
                self.write(run)
                with self.assertRaises(ValueError):
                    self.package()

    def test_failed_rerender_gate_rejected(self):
        run = self.read()
        run["metrics"]["triangle_prediction_max_abs"] = 1.0
        self.write(run)
        with self.assertRaisesRegex(ValueError, "failed triangle_prediction"):
            self.package()

    def test_changed_source_and_release_input_rejected(self):
        original = self.read()
        for name in ["source_sha256", "input"]:
            with self.subTest(name=name):
                run = json.loads(json.dumps(original))
                run[name] = {} if name == "source_sha256" else {"sha256": "incorrect"}
                self.write(run)
                with self.assertRaises(ValueError):
                    self.package()

    def test_candidate_changed_before_certification_rejected(self):
        original = packaging.certify_stl

        def mutate_then_certify(path, **kwargs):
            path.write_bytes(path.read_bytes() + b"\n")
            return original(path, **kwargs)

        with patch.object(packaging, "certify_stl", mutate_then_certify):
            with self.assertRaisesRegex(RuntimeError, "changed before certification"):
                self.package()
        self.assertFalse(self.output.exists())

    def test_candidate_changed_after_certification_rejected(self):
        original = packaging.certify_stl

        def certify_then_mutate(path, **kwargs):
            result = original(path, **kwargs)
            path.write_bytes(path.read_bytes() + b"\n")
            return result

        with patch.object(packaging, "certify_stl", certify_then_mutate):
            with self.assertRaisesRegex(RuntimeError, "changed during packaging"):
                self.package()
        self.assertFalse(self.output.exists())

    def test_existing_package_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "sentinel"
        sentinel.write_text("retain")
        with self.assertRaises(FileExistsError):
            self.package()
        self.assertEqual(sentinel.read_text(), "retain")


if __name__ == "__main__":
    unittest.main()
