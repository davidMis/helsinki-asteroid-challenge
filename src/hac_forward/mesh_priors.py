"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from .constants import CYLINDER_RADII


def released_cylinder_radius(model_id: int) -> float:
    """Return the released bounding-cylinder radius for ``model_id``."""
    try:
        return float(CYLINDER_RADII[model_id])
    except KeyError as exc:
        raise ValueError(f"no released cylinder radius for model {model_id}") from exc

