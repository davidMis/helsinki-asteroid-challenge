"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
import numpy as np


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("correlation inputs must be nonempty with matching shapes")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("correlation inputs must be finite")
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denominator = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(a_centered, b_centered) / denominator)

