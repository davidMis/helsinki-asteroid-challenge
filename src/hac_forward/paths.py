"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from pathlib import Path
from .constants import LIGHTCURVE_FILE_MODEL_TOKENS, RELEASED_MODEL_IDS


def require_released_model_id(model_id: int) -> int:
    """Return a valid released model id or raise a descriptive error."""
    if isinstance(model_id, bool) or model_id not in RELEASED_MODEL_IDS:
        raise ValueError(
            f"model_id must identify a released model in {RELEASED_MODEL_IDS}, "
            f"got {model_id!r}"
        )
    return int(model_id)



def model_root(data_dir: Path, model_id: int) -> Path:
    """Return the directory containing one asteroid model's released files."""
    model_id = require_released_model_id(model_id)
    visibility = "public" if model_id <= 3 else "secret"
    return data_dir / f"AsteroidModel{model_id:02d}_shape_{visibility}"



def lightcurve_path(
    data_dir: Path,
    model_id: int,
    curve_type: str,
    source: str,
) -> Path:
    """Return the expected path for a lightcurve text file."""
    if curve_type not in {"intensity", "binary"}:
        raise ValueError(f"unsupported curve_type: {curve_type!r}")
    if source not in {"real", "blender"}:
        raise ValueError(f"unsupported source: {source!r}")

    model_id = require_released_model_id(model_id)
    suffix = "_blender" if source == "blender" else ""
    file_model_token = LIGHTCURVE_FILE_MODEL_TOKENS[model_id]
    return (
        model_root(data_dir, model_id)
        / f"Asteroid{model_id}_lightcurve_data"
        / f"Asteroid{file_model_token}_lightcurve_{curve_type}{suffix}.txt"
    )

