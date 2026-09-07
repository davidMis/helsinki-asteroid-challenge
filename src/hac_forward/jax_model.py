"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
import jax.numpy as jnp


def rotate_lab_direction_to_body(direction: jnp.ndarray, phases: jnp.ndarray) -> jnp.ndarray:
    """Rotate lab-frame directions by ``-phase`` into the body frame."""
    cos_phase = jnp.cos(phases)
    sin_phase = jnp.sin(phases)
    x, y, z = direction
    return jnp.stack(
        [
            cos_phase * x + sin_phase * y,
            -sin_phase * x + cos_phase * y,
            jnp.full_like(phases, z),
        ],
        axis=1,
    )

