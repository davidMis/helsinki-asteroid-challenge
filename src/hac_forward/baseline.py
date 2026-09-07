"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ForwardModelBaseline:
    """A frozen forward-model/acquisition calibration checkpoint."""

    name: str
    target_source: str
    curve_type: str
    phase_sign: int
    azimuth_offset: float
    azimuth_sign: int
    alpha_from: str
    top_z_sign: int
    scattering: str
    lommel_seeliger_weight: float
    visibility_mode: str
    occlusion_grid_size: int
    occlusion_depth_tolerance: float
    occlusion_softness: float
    phase_batch_size: int
    angle_phase_shifts_degrees: dict[int, float]
    contrast_gain: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

