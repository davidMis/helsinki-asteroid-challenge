"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
import numpy as np
from .constants import ANGLES_DEGREES, TOP_CAMERA_ALPHA_DEGREES, VIEW_LABELS


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        raise ValueError("zero-length vector")
    return vector / norm



def horizontal_direction(
    angle_degrees: float,
    *,
    azimuth_offset_degrees: float = 0.0,
    azimuth_sign: int = 1,
) -> np.ndarray:
    """Return the object-to-camera direction for a horizontal measurement angle."""
    theta = np.deg2rad(azimuth_offset_degrees + azimuth_sign * angle_degrees)
    return np.asarray([-np.cos(theta), -np.sin(theta), 0.0], dtype=np.float64)



def tilted_direction(
    angle_degrees: float,
    alpha_degrees: float,
    *,
    z_sign: float,
    alpha_from: str = "vertical",
    azimuth_offset_degrees: float = 0.0,
    azimuth_sign: int = 1,
) -> np.ndarray:
    """Return a tilted top/bottom camera direction.

    ``alpha_from='vertical'`` treats the published alpha as the tilt away from
    the z-axis. This is an explicit hypothesis to validate against public data.
    """
    horizontal = horizontal_direction(
        angle_degrees,
        azimuth_offset_degrees=azimuth_offset_degrees,
        azimuth_sign=azimuth_sign,
    )
    alpha = np.deg2rad(alpha_degrees)
    if alpha_from == "vertical":
        vector = np.sin(alpha) * horizontal + z_sign * np.cos(alpha) * np.array(
            [0.0, 0.0, 1.0]
        )
    elif alpha_from == "horizontal":
        vector = np.cos(alpha) * horizontal + z_sign * np.sin(alpha) * np.array(
            [0.0, 0.0, 1.0]
        )
    else:
        raise ValueError(f"unsupported alpha_from: {alpha_from!r}")
    return unit_vector(vector)



def view_directions(
    alpha_from: str = "vertical",
    *,
    azimuth_offset_degrees: float = 0.0,
    azimuth_sign: int = 1,
    top_z_sign: float = 1.0,
) -> np.ndarray:
    """Return view directions with shape ``(angle, view, xyz)``."""
    directions = np.zeros((len(ANGLES_DEGREES), len(VIEW_LABELS), 3), dtype=np.float64)
    for angle_index, angle in enumerate(ANGLES_DEGREES):
        horizontal = horizontal_direction(
            angle,
            azimuth_offset_degrees=azimuth_offset_degrees,
            azimuth_sign=azimuth_sign,
        )
        alpha = TOP_CAMERA_ALPHA_DEGREES[angle]
        directions[angle_index, 0] = horizontal
        directions[angle_index, 1] = horizontal
        directions[angle_index, 2] = tilted_direction(
            angle,
            alpha,
            z_sign=top_z_sign,
            alpha_from=alpha_from,
            azimuth_offset_degrees=azimuth_offset_degrees,
            azimuth_sign=azimuth_sign,
        )
        directions[angle_index, 3] = tilted_direction(
            angle,
            alpha,
            z_sign=-top_z_sign,
            alpha_from=alpha_from,
            azimuth_offset_degrees=azimuth_offset_degrees,
            azimuth_sign=azimuth_sign,
        )
    return directions

