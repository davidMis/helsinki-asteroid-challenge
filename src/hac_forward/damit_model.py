"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass(frozen=True)
class GaussianImageGrid:
    """Fixed normal fan used by the DAMIT Gaussian-image parameterization."""

    n_rows: int
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    base_areas: np.ndarray
    theta: np.ndarray
    phi: np.ndarray



def harmonic_degrees(l_max: int, m_max: int) -> np.ndarray:
    """Return the spherical degree associated with each coefficient column."""
    degrees: list[int] = []
    for m in range(m_max + 1):
        for degree in range(m, l_max + 1):
            degrees.append(degree)
            if m != 0:
                degrees.append(degree)
    return np.asarray(degrees, dtype=np.float64)



@lru_cache(maxsize=16)
def gaussian_image_grid(n_rows: int) -> GaussianImageGrid:
    """Return the standard DAMIT-like triangulation of the unit sphere."""
    if n_rows < 1:
        raise ValueError("n_rows must be positive")

    theta_values: list[float] = [0.0]
    phi_values: list[float] = [0.0]
    dtheta = np.pi / (2.0 * n_rows)
    for row in range(1, n_rows + 1):
        dphi = np.pi / (2.0 * row)
        for col in range(4 * row):
            theta_values.append(row * dtheta)
            phi_values.append(col * dphi)
    for row in range(n_rows - 1, 0, -1):
        dphi = np.pi / (2.0 * row)
        for col in range(4 * row):
            theta_values.append(np.pi - row * dtheta)
            phi_values.append(col * dphi)
    theta_values.append(np.pi)
    phi_values.append(0.0)

    theta = np.asarray(theta_values, dtype=np.float64)
    phi = np.asarray(phi_values, dtype=np.float64)
    vertices = np.column_stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )
    faces = _standard_tri_faces(n_rows)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centroids = np.mean(triangles, axis=1)
    inward = np.einsum("ij,ij->i", cross, centroids) < 0.0
    if np.any(inward):
        faces = np.array(faces, copy=True)
        faces[inward, 1], faces[inward, 2] = (
            faces[inward, 2],
            faces[inward, 1].copy(),
        )
        triangles = vertices[faces]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
    double_areas = np.linalg.norm(cross, axis=1)
    normals = cross / double_areas[:, None]
    base_areas = 0.5 * double_areas
    normal_theta = np.arccos(np.clip(normals[:, 2], -1.0, 1.0))
    normal_phi = np.arctan2(normals[:, 1], normals[:, 0])
    return GaussianImageGrid(
        n_rows=n_rows,
        vertices=vertices,
        faces=faces,
        normals=normals.astype(np.float64),
        base_areas=base_areas.astype(np.float64),
        theta=normal_theta.astype(np.float64),
        phi=normal_phi.astype(np.float64),
    )



@lru_cache(maxsize=64)
def harmonic_basis(n_rows: int, l_max: int, m_max: int) -> np.ndarray:
    """Return unnormalized real basis columns in DAMIT coefficient order."""
    grid = gaussian_image_grid(n_rows)
    cos_theta = np.cos(grid.theta)
    sin_theta = np.sin(grid.theta)
    p_lm = _associated_legendre_no_condon(cos_theta, sin_theta, l_max, m_max)
    columns: list[np.ndarray] = []
    for m in range(m_max + 1):
        for degree in range(m, l_max + 1):
            base = p_lm[:, degree, m]
            if m == 0:
                columns.append(base)
            else:
                columns.append(base * np.cos(m * grid.phi))
                columns.append(base * np.sin(m * grid.phi))
    return np.column_stack(columns).astype(np.float64)



def _standard_tri_faces(n_rows: int) -> np.ndarray:
    nod: dict[int, list[int]] = {0: [0]}
    node_index = 0
    for row in range(1, n_rows + 1):
        values: list[int] = []
        for _col in range(4 * row):
            node_index += 1
            values.append(node_index)
        values.append(values[0])
        nod[row] = values
    for row in range(n_rows - 1, 0, -1):
        values = []
        for _col in range(4 * row):
            node_index += 1
            values.append(node_index)
        values.append(values[0])
        nod[2 * n_rows - row] = values
    nod[2 * n_rows] = [node_index + 1]

    faces: list[tuple[int, int, int]] = []
    for j1 in range(1, n_rows + 1):
        for j3 in range(1, 5):
            j0 = (j3 - 1) * j1
            faces.append((nod[j1 - 1][j0 - (j3 - 1)], nod[j1][j0], nod[j1][j0 + 1]))
            for j2 in range(j0 + 1, j0 + j1):
                faces.append(
                    (
                        nod[j1][j2],
                        nod[j1 - 1][j2 - (j3 - 1)],
                        nod[j1 - 1][j2 - 1 - (j3 - 1)],
                    )
                )
                faces.append(
                    (
                        nod[j1 - 1][j2 - (j3 - 1)],
                        nod[j1][j2],
                        nod[j1][j2 + 1],
                    )
                )

    for j1 in range(n_rows + 1, 2 * n_rows + 1):
        for j3 in range(1, 5):
            j0 = (j3 - 1) * (2 * n_rows - j1)
            faces.append(
                (
                    nod[j1][j0],
                    nod[j1 - 1][j0 + 1 + (j3 - 1)],
                    nod[j1 - 1][j0 + (j3 - 1)],
                )
            )
            for j2 in range(j0 + 1, j0 + (2 * n_rows - j1) + 1):
                faces.append((nod[j1][j2], nod[j1 - 1][j2 + (j3 - 1)], nod[j1][j2 - 1]))
                faces.append(
                    (
                        nod[j1][j2],
                        nod[j1 - 1][j2 + 1 + (j3 - 1)],
                        nod[j1 - 1][j2 + (j3 - 1)],
                    )
                )
    return np.asarray(faces, dtype=np.int32)



def _associated_legendre_no_condon(
    cos_theta: np.ndarray,
    sin_theta: np.ndarray,
    l_max: int,
    m_max: int,
) -> np.ndarray:
    p_lm = np.zeros((cos_theta.shape[0], l_max + 1, m_max + 1), dtype=np.float64)
    p_lm[:, 0, 0] = 1.0
    for m in range(1, min(m_max, l_max) + 1):
        p_lm[:, m, m] = (2 * m - 1) * sin_theta * p_lm[:, m - 1, m - 1]
    for m in range(0, min(m_max, l_max) + 1):
        if m + 1 <= l_max:
            p_lm[:, m + 1, m] = (2 * m + 1) * cos_theta * p_lm[:, m, m]
        for degree in range(m + 2, l_max + 1):
            p_lm[:, degree, m] = (
                (2 * degree - 1) * cos_theta * p_lm[:, degree - 1, m]
                - (degree + m - 1) * p_lm[:, degree - 2, m]
            ) / (degree - m)
    return p_lm

