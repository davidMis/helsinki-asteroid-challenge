"""Small optional integration controls; no challenge target meshes required."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from scipy.spatial import ConvexHull

from hac_forward import stl

SPEC = importlib.util.spec_from_file_location(
    "evaluate_public", Path(__file__).with_name("evaluate_public.py")
)
assert SPEC is not None and SPEC.loader is not None
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


def cube():
    points = np.array(
        [[x + 0.3, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float
    )
    hull = ConvexHull(points)
    triangles = points[hull.simplices].copy()
    for index, triangle in enumerate(triangles):
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if np.dot(normal, hull.equations[index, :3]) < 0:
            triangles[index] = triangle[[0, 2, 1]]
    return triangles


class OfficialAdapterTests(unittest.TestCase):
    def test_normalization_preserves_all_triangles_and_xy_axis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = 10 * cube() + [0, 0, 5]
            stl.write_ascii_stl(root / "raw.stl", raw)
            result = EVALUATE.normalize_target(root / "raw.stl", root / "normalized.stl", 1)
            normalized = stl.load_stl_triangles(root / "normalized.stl")
            np.testing.assert_allclose(normalized, cube(), rtol=0, atol=1e-15)
            self.assertEqual(result["normalization"]["triangle_count"], 12)
            self.assertFalse(result["normalization"]["center_xy"])
            self.assertEqual(result["normalization"]["offset"], [0, 0, 5])

    @unittest.skipUnless(
        os.environ.get("HAC_EVALUATION_DIR"), "set HAC_EVALUATION_DIR to official sources"
    )
    def test_altered_official_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(os.environ["HAC_EVALUATION_DIR"])
            copy = Path(temporary) / "evaluation"
            shutil.copytree(source, copy)
            with (copy / "Voxel measure - Python/metrics.py").open("a") as stream:
                stream.write("\n# tampered source\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                EVALUATE.verify_sources(copy)

    @unittest.skipUnless(
        os.environ.get("HAC_EVALUATION_DIR") and os.environ.get("HAC_MATLAB"),
        "set HAC_EVALUATION_DIR and HAC_MATLAB for native identity control",
    )
    def test_identity_voxel_and_native_matlab(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "data/AsteroidModel01_shape_public/asteroid1.stl"
            candidate = root / "run/model01/submission_shape.stl"
            stl.write_ascii_stl(target, 10 * cube() + [0, 0, 5])
            stl.write_ascii_stl(candidate, cube())
            before = EVALUATE.file_identity(candidate)
            EVALUATE.write_json(
                candidate.with_name("run.json"),
                {
                    "model_id": 1,
                    "status": "complete",
                    "artifacts": {"submission_shape.stl": before},
                },
            )
            args = argparse.Namespace(
                data_dir=root / "data",
                run_dir=root / "run",
                output_dir=root / "output",
                evaluation_dir=Path(os.environ["HAC_EVALUATION_DIR"]),
                matlab=Path(os.environ["HAC_MATLAB"]),
                models="1",
                pitch=0.25,
                theta=45.0,
            )
            try:
                result = EVALUATE.evaluate(args)
            except RuntimeError:
                log = root / "output/matlab.log"
                self.fail(log.read_text() if log.exists() else "MATLAB did not create a log")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["rows"][0]["voxel"]["measure1_loss"], 0)
            self.assertEqual(result["rows"][0]["voxel"]["measure2_loss"], 0)
            self.assertEqual(result["rows"][0]["projection"]["projection_score"], 1)
            self.assertEqual(result["mean_total_score"], 2)
            self.assertEqual(EVALUATE.file_identity(candidate), before)
            self.assertEqual(
                json.loads((root / "output/evaluation.json").read_text())["status"], "complete"
            )
            with self.assertRaises(FileExistsError):
                EVALUATE.evaluate(args)


if __name__ == "__main__":
    unittest.main()
