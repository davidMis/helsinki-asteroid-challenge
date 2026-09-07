"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
import numpy as np


def triangle_area_measure(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized triangle areas and oriented unit normals."""
    mesh = np.asarray(triangles, dtype=np.float64)
    if mesh.ndim != 3 or mesh.shape[1:] != (3, 3) or mesh.shape[0] < 4:
        raise ValueError("triangles must have shape (n, 3, 3), n >= 4")
    if not np.all(np.isfinite(mesh)):
        raise ValueError("triangles must be finite")
    cross = np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0])
    double_areas = np.linalg.norm(cross, axis=1)
    if np.any(double_areas <= 0.0) or not np.all(np.isfinite(double_areas)):
        raise ValueError("triangles must be nondegenerate")
    masses = double_areas / np.sum(double_areas)
    normals = cross / double_areas[:, None]
    return masses, normals

