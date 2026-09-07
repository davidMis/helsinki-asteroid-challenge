import unittest

import numpy as np
from scipy.spatial import ConvexHull

from hac_forward.certification import certify_convex_mesh


def cube():
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
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


def cube_with_top_dent(depth):
    triangles = cube()
    triangles = triangles[~np.all(triangles[:, :, 2] == 1, axis=1)]
    corners = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    dented_top = np.array([[corners[i], corners[(i + 1) % 4], [0, 0, 1 - depth]] for i in range(4)])
    return np.concatenate([triangles, dented_top])


class ConvexCertificateTests(unittest.TestCase):
    def test_cube_is_certified(self):
        result = certify_convex_mesh(cube(), cylinder_radius=np.sqrt(2))
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["non_self_intersection"]["certified_numerically"])

    def test_inward_face_rejected(self):
        triangles = cube()
        triangles[0] = triangles[0, [0, 2, 1]]
        self.assertFalse(certify_convex_mesh(triangles, cylinder_radius=np.sqrt(2))["valid"])

    def test_closed_concave_mesh_rejected(self):
        result = certify_convex_mesh(cube_with_top_dent(1), cylinder_radius=np.sqrt(2))
        self.assertTrue(result["structural"]["valid"], result["structural"]["errors"])
        self.assertFalse(result["checks"]["outward_supporting_faces"])
        self.assertFalse(result["valid"])

    def test_self_crossing_closed_sphere_rejected(self):
        # The apex passes through the flat bottom face. Its four incident
        # triangles cross that face along a square strictly inside its edges.
        # Topology and signed volume alone cannot detect this intersection.
        triangles = cube_with_top_dent(2.5)
        triangles[:, :, 2] = (triangles[:, :, 2] + 0.25) / 1.25
        result = certify_convex_mesh(triangles, cylinder_radius=np.sqrt(2))
        self.assertTrue(result["structural"]["valid"], result["structural"]["errors"])
        self.assertTrue(result["checks"]["spherical_topology"])
        self.assertTrue(result["checks"]["manifold_vertex_links"])
        self.assertFalse(result["valid"])
        self.assertFalse(result["non_self_intersection"]["certified_numerically"])

    def test_subtolerance_defect_is_explicitly_numerical(self):
        triangles = cube_with_top_dent(1e-8)
        result = certify_convex_mesh(triangles, cylinder_radius=np.sqrt(2))
        self.assertTrue(result["valid"], result["errors"])
        self.assertGreater(result["metrics"]["maximum_triangle_to_hull_plane_distance"], 0)
        self.assertIn(
            "below the reported tolerances", result["non_self_intersection"]["limitation"]
        )
        strict = certify_convex_mesh(
            triangles,
            cylinder_radius=np.sqrt(2),
            relative_plane_tolerance=1e-10,
            relative_measure_tolerance=1e-10,
        )
        self.assertFalse(strict["valid"])
        self.assertFalse(strict["checks"]["outward_supporting_faces"])

    def test_resolved_dent_rejected_even_when_area_matches(self):
        result = certify_convex_mesh(cube_with_top_dent(1e-5), cylinder_radius=np.sqrt(2))
        self.assertTrue(result["structural"]["valid"], result["structural"]["errors"])
        self.assertTrue(result["checks"]["convex_hull_area_agreement"])
        self.assertFalse(result["checks"]["outward_supporting_faces"])
        self.assertFalse(result["valid"])

    def test_duplicate_cover_rejected(self):
        result = certify_convex_mesh(np.concatenate([cube(), cube()]), cylinder_radius=np.sqrt(2))
        self.assertFalse(result["valid"])

    def test_nonfinite_rejected(self):
        triangles = cube()
        triangles[0, 0, 0] = np.nan
        self.assertFalse(certify_convex_mesh(triangles, cylinder_radius=np.sqrt(2))["valid"])

    def test_missing_face_rejected(self):
        result = certify_convex_mesh(cube()[1:], cylinder_radius=np.sqrt(2))
        self.assertFalse(result["valid"])
        self.assertFalse(result["non_self_intersection"]["certified_numerically"])

    def test_invalid_tolerances_rejected(self):
        for name in ("relative_plane_tolerance", "relative_measure_tolerance"):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    certify_convex_mesh(cube(), cylinder_radius=np.sqrt(2), **{name: value})


if __name__ == "__main__":
    unittest.main()
