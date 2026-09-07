"""Small independent numerical contracts for the extracted inference path."""

import json
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from hac_forward.area_measure_transform import pushforward_through_submission_frame
from hac_forward.gaussian_bhs import close_masses_jax, close_masses_np
from hac_forward.damit_model import gaussian_image_grid
from hac_forward.reconstruct import effective_config
from hac_forward.semidiscrete_ot_quantization import (
    SemidiscreteOTQuantizerConfig,
    semidiscrete_ot_quantize,
)
from hac_forward.stl import write_ascii_stl
from hac_forward.submission_validation import normalize_submission_frame, validate_submission_stl
from hac_forward.variational_minkowski import (
    VariationalMinkowskiConfig,
    solve_variational_minkowski,
)

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_full_configuration_has_per_model_radius_and_seed(self):
        config = json.loads((ROOT / "configs/submission.json").read_text())
        first = effective_config(config, 1)
        last = effective_config(config, 10)
        self.assertEqual(first[1].reference_spheroid_radius, 1.12)
        self.assertEqual(last[1].reference_spheroid_radius, 3.95)
        self.assertEqual(last[3].seed - first[3].seed, 9)
        self.assertEqual(first[2].surface_wasserstein_weight, 0.03)
        self.assertEqual(first[4].normalized_output_mass_floor, 1e-8)
        self.assertEqual(first[5].area_log_rmse_tolerance, 1e-6)
        del config["phase1"]["shape"]["epsilon"]
        with self.assertRaisesRegex(ValueError, "missing"):
            effective_config(config, 1)

    def test_exact_closure_and_custom_derivative_survive_extraction(self):
        grid = gaussian_image_grid(2)
        reference = grid.base_areas / np.sum(grid.base_areas)
        rng = np.random.default_rng(912)
        values = rng.normal(scale=0.3, size=len(reference))
        direction = rng.normal(size=len(reference))
        masses = close_masses_np(values, reference, grid.normals)
        self.assertGreater(np.min(masses), 0)
        np.testing.assert_allclose(np.sum(masses), 1.0, atol=2e-14)
        np.testing.assert_allclose(masses @ grid.normals, 0.0, atol=2e-12)
        with jax.enable_x64():

            def func(x):
                return close_masses_jax(x, jnp.asarray(reference), jnp.asarray(grid.normals))

            output, derivative = jax.jvp(func, (jnp.asarray(values),), (jnp.asarray(direction),))
            step = 1e-5
            difference = (
                close_masses_np(values + step * direction, reference, grid.normals)
                - close_masses_np(values - step * direction, reference, grid.normals)
            ) / (2 * step)
        np.testing.assert_allclose(output, masses, rtol=2e-11, atol=1e-12)
        np.testing.assert_allclose(derivative, difference, rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(np.asarray(derivative) @ grid.normals, 0.0, atol=1e-10)

    def test_quantization_realization_and_frame_contract(self):
        grid = gaussian_image_grid(2)
        masses = grid.base_areas / np.sum(grid.base_areas)
        quantized = semidiscrete_ot_quantize(
            masses,
            grid.normals,
            8,
            config=SemidiscreteOTQuantizerConfig(
                seed=21, n_starts=2, normalized_output_mass_floor=1e-8
            ),
        )
        np.testing.assert_allclose(quantized.masses @ quantized.normals, 0.0, atol=1e-12)
        result = solve_variational_minkowski(
            quantized.masses,
            quantized.normals,
            config=VariationalMinkowskiConfig(area_log_rmse_tolerance=1e-6),
        )
        self.assertLess(result.diagnostics.area_log_rmse, 1e-6)
        framed, frame = normalize_submission_frame(
            result.mesh.vertices, model_id=1, match_radius=True
        )
        push = pushforward_through_submission_frame(quantized.masses, quantized.normals, frame)
        self.assertLess(push.diagnostics.pushed_closure_norm, 1e-10)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shape.stl"
            write_ascii_stl(path, framed)
            validate_submission_stl(path, cylinder_radius=1.12).require_valid()
        self.assertAlmostEqual(np.min(framed[:, :, 2]), -1.0, places=12)
        self.assertAlmostEqual(np.max(framed[:, :, 2]), 1.0, places=12)
        self.assertAlmostEqual(np.max(np.linalg.norm(framed[:, :, :2], axis=-1)), 1.12, places=12)


if __name__ == "__main__":
    unittest.main()
