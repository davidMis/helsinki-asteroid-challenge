"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations


ANGLES_DEGREES: tuple[int, ...] = (0, 45, 90, 135, 225, 270, 315)



VIEW_LABELS: tuple[str, ...] = (
    "horizontal_1",
    "horizontal_2",
    "top",
    "bottom",
)



TOP_CAMERA_ALPHA_DEGREES: dict[int, float] = {
    0: 21.0,
    45: 26.0,
    90: 26.0,
    135: 26.0,
    225: 24.0,
    270: 24.0,
    315: 24.0,
}



CYLINDER_RADII: dict[int, float] = {
    1: 1.12,
    2: 1.42,
    3: 0.88,
    4: 1.475,
    5: 1.22,
    6: 0.925,
    7: 1.205,
    8: 1.24,
    9: 0.67,
    10: 3.95,
}



LIGHTCURVE_FILE_MODEL_TOKENS: dict[int, str] = {
    **{model_id: f"{model_id:02d}" for model_id in range(1, 10)},
    10: "010",
}



RELEASED_MODEL_IDS: tuple[int, ...] = tuple(LIGHTCURVE_FILE_MODEL_TOKENS)



LIGHT_DIRECTION = (-1.0, 0.0, 0.0)

