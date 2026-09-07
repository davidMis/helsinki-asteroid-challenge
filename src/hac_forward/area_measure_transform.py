"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class AreaMeasurePushforwardDiagnostics:
    """Scalar checks describing one linear area-measure pushforward."""

    determinant: float
    input_total_area: float
    pushed_total_area: float
    total_area_scale: float
    input_closure_norm: float
    pushed_closure_norm: float
    paired_mass_l1: float
    hellinger_distance: float
    weighted_mean_normal_displacement_degrees: float
    weighted_rms_normal_displacement_degrees: float
    maximum_normal_displacement_degrees: float



@dataclass(frozen=True)
class AreaMeasurePushforward:
    """A normalized pushed measure plus its absolute area-vector state."""

    masses: np.ndarray
    normals: np.ndarray
    pushed_areas: np.ndarray
    linear_map: np.ndarray
    cofactor_map: np.ndarray
    diagnostics: AreaMeasurePushforwardDiagnostics



def submission_frame_linear_map(
    frame: Mapping[str, int | float | bool],
) -> np.ndarray:
    """Return the linear part of :func:`normalize_submission_frame`.

    The z-centering translation has no effect on a surface-area measure.  The
    remaining map first scales every coordinate by ``uniform_scale`` and then
    applies ``xy_scale`` to x and y only.
    """
    try:
        uniform_scale = float(frame["uniform_scale"])
        xy_scale = float(frame["xy_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "submission frame must contain numeric uniform_scale and xy_scale"
        ) from exc
    if (
        not np.isfinite(uniform_scale)
        or not np.isfinite(xy_scale)
        or uniform_scale <= 0.0
        or xy_scale <= 0.0
    ):
        raise ValueError("submission-frame scales must be finite and positive")
    return np.diag(
        [
            uniform_scale * xy_scale,
            uniform_scale * xy_scale,
            uniform_scale,
        ]
    )



def pushforward_area_measure(
    masses: np.ndarray,
    normals: np.ndarray,
    linear_map: np.ndarray,
) -> AreaMeasurePushforward:
    """Push a discrete surface-area measure through a linear map.

    Parameters
    ----------
    masses, normals:
        Strictly positive facet areas and corresponding outward unit normals.
        The masses need not sum to one; returned ``masses`` always do.
    linear_map:
        A finite, nonsingular, orientation-preserving ``(3, 3)`` matrix.

    Notes
    -----
    Atom identities and ordering are preserved.  ``pushed_areas`` retains the
    absolute scale induced by the supplied input areas, whereas ``masses`` is
    their probability normalization for downstream photometric prediction.
    """
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    directions = np.asarray(normals, dtype=np.float64)
    linear = np.asarray(linear_map, dtype=np.float64)
    if values.size < 1 or directions.shape != (values.size, 3):
        raise ValueError("masses and normals have incompatible shapes")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("masses must be finite and strictly positive")
    if not np.all(np.isfinite(directions)):
        raise ValueError("normals must be finite")
    lengths = np.linalg.norm(directions, axis=1)
    if not np.allclose(lengths, 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("normals must have unit length")
    if linear.shape != (3, 3) or not np.all(np.isfinite(linear)):
        raise ValueError("linear_map must be a finite 3 by 3 matrix")
    determinant = float(np.linalg.det(linear))
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("linear_map must be nonsingular and orientation preserving")

    cofactor = determinant * np.linalg.inv(linear).T
    area_vectors = values[:, None] * directions
    pushed_vectors = area_vectors @ cofactor.T
    pushed_areas = np.linalg.norm(pushed_vectors, axis=1)
    if not np.all(np.isfinite(pushed_areas)) or np.any(pushed_areas <= 0.0):
        raise ValueError("linear map produced invalid pushed facet areas")
    pushed_normals = pushed_vectors / pushed_areas[:, None]
    input_total = float(np.sum(values))
    pushed_total = float(np.sum(pushed_areas))
    source_masses = values / input_total
    pushed_masses = pushed_areas / pushed_total

    cosines = np.clip(
        np.einsum("ij,ij->i", directions, pushed_normals),
        -1.0,
        1.0,
    )
    angles = np.arccos(cosines)
    mass_l1 = float(np.sum(np.abs(source_masses - pushed_masses)))
    diagnostics = AreaMeasurePushforwardDiagnostics(
        determinant=determinant,
        input_total_area=input_total,
        pushed_total_area=pushed_total,
        total_area_scale=pushed_total / input_total,
        input_closure_norm=float(np.linalg.norm(values @ directions)),
        pushed_closure_norm=float(np.linalg.norm(pushed_masses @ pushed_normals)),
        paired_mass_l1=mass_l1,
        hellinger_distance=float(
            np.linalg.norm(np.sqrt(source_masses) - np.sqrt(pushed_masses)) / np.sqrt(2.0)
        ),
        weighted_mean_normal_displacement_degrees=float(np.degrees(np.sum(source_masses * angles))),
        weighted_rms_normal_displacement_degrees=float(
            np.degrees(np.sqrt(np.sum(source_masses * angles**2)))
        ),
        maximum_normal_displacement_degrees=float(np.degrees(np.max(angles))),
    )
    return AreaMeasurePushforward(
        masses=pushed_masses,
        normals=pushed_normals,
        pushed_areas=pushed_areas,
        linear_map=linear.copy(),
        cofactor_map=cofactor,
        diagnostics=diagnostics,
    )



def pushforward_through_submission_frame(
    masses: np.ndarray,
    normals: np.ndarray,
    frame: Mapping[str, int | float | bool],
) -> AreaMeasurePushforward:
    """Push a measure through the linear part of a submission-frame map."""
    return pushforward_area_measure(
        masses,
        normals,
        submission_frame_linear_map(frame),
    )

