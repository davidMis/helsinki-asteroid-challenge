"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Collection
import numpy as np
from .constants import ANGLES_DEGREES, VIEW_LABELS
from .paths import lightcurve_path


CURVE_SHAPE = (len(ANGLES_DEGREES), len(VIEW_LABELS))



NORMALIZATION_ATOL = 2.0e-3



PROJECTION_LABORATORY = "laboratory"



PROJECTION_LEGACY_PERSPECTIVE = "legacy_perspective"



PROJECTION_ORTHOGRAPHIC = "orthographic"



PROJECTION_UNVERIFIED = "unverified"



MODEL1_BLENDER_PROJECTION_BY_SHA256 = {
    "a31e205f8e14dc1c74532e7a4a1ef21a62e817fc694cf1e7a20fd2424a1859e5":
        PROJECTION_LEGACY_PERSPECTIVE,
    "49c62f0c3db610b5be5a4422388e63346ac1b9496b76826b2524c039af685968":
        PROJECTION_LEGACY_PERSPECTIVE,
    "32e88e7920395e3a33381d56f8ac42d3acda12a38c6ec154a1bd9d6d04fd31fd":
        PROJECTION_ORTHOGRAPHIC,
    "f3b1b688d85cbcd06d85bb736ccc52f1348d2e8dfa6f69fddc2d6d947b774fc9":
        PROJECTION_ORTHOGRAPHIC,
}



@dataclass(frozen=True)
class LightcurveTable:
    """A 29-column HAC lightcurve file reshaped into angle/view axes."""

    model_id: int
    curve_type: str
    source: str
    path: Path
    time: np.ndarray
    values: np.ndarray
    validity_weights: np.ndarray | None = None
    independence_weights: np.ndarray | None = None
    projection_provenance: np.ndarray | None = None
    normalization_means: np.ndarray | None = None

    def __post_init__(self) -> None:
        time = np.asarray(self.time)
        values = np.asarray(self.values)
        if time.ndim != 1:
            raise ValueError(f"time must be one-dimensional, got {time.shape}")
        if values.ndim != 3 or values.shape[1:] != CURVE_SHAPE:
            raise ValueError(
                "values must have shape "
                f"(n, {CURVE_SHAPE[0]}, {CURVE_SHAPE[1]}), got {values.shape}"
            )
        if values.shape[0] != time.shape[0]:
            raise ValueError(
                f"time has {time.shape[0]} samples but values has {values.shape[0]}"
            )
        if values.shape[0] < 2:
            raise ValueError("a lightcurve table must contain at least two samples")
        for name, weights in (
            ("validity_weights", self.validity_weights),
            ("independence_weights", self.independence_weights),
        ):
            if weights is None:
                continue
            array = np.asarray(weights, dtype=np.float64)
            if array.shape != CURVE_SHAPE:
                raise ValueError(f"{name} must have shape {CURVE_SHAPE}, got {array.shape}")
            if not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError(f"{name} must contain finite non-negative values")
        if self.projection_provenance is not None:
            provenance = np.asarray(self.projection_provenance)
            if provenance.shape != CURVE_SHAPE:
                raise ValueError(
                    "projection_provenance must have shape "
                    f"{CURVE_SHAPE}, got {provenance.shape}"
                )
        if self.normalization_means is not None:
            means = np.asarray(self.normalization_means, dtype=np.float64)
            if means.shape != CURVE_SHAPE:
                raise ValueError(
                    f"normalization_means must have shape {CURVE_SHAPE}, "
                    f"got {means.shape}"
                )
            if not np.all(np.isfinite(means)) or np.any(means <= 0.0):
                raise ValueError("normalization_means must be finite and positive")

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def phases(self) -> np.ndarray:
        return np.linspace(0.0, 2.0 * np.pi, self.n_samples, endpoint=False)

    @property
    def curve_weights(self) -> np.ndarray:
        """Return validity x independence weights with shape ``(angle, view)``."""
        validity = (
            np.ones(CURVE_SHAPE, dtype=np.float64)
            if self.validity_weights is None
            else np.asarray(self.validity_weights, dtype=np.float64)
        )
        independence = (
            np.ones(CURVE_SHAPE, dtype=np.float64)
            if self.independence_weights is None
            else np.asarray(self.independence_weights, dtype=np.float64)
        )
        return validity * independence

    def projection_provenance_weights(
        self,
        allowed_provenance: Collection[str],
    ) -> np.ndarray:
        """Return curve weights masked to selected camera-projection provenance."""
        allowed = frozenset(str(item) for item in allowed_provenance)
        if self.projection_provenance is None:
            return np.zeros(CURVE_SHAPE)
        mask = np.isin(np.asarray(self.projection_provenance), tuple(allowed))
        return self.curve_weights * mask

    def curve(self, angle_degrees: int, view_label: str) -> np.ndarray:
        angle_index = ANGLES_DEGREES.index(angle_degrees)
        view_index = VIEW_LABELS.index(view_label)
        return self.values[:, angle_index, view_index]

    def curve_weight(self, angle_degrees: int, view_label: str) -> float:
        """Return the released-data effective weight for one curve."""
        angle_index = ANGLES_DEGREES.index(angle_degrees)
        view_index = VIEW_LABELS.index(view_label)
        return float(self.curve_weights[angle_index, view_index])

    def curve_projection_provenance(
        self,
        angle_degrees: int,
        view_label: str,
    ) -> str | None:
        """Return the known camera-projection provenance for one curve."""
        if self.projection_provenance is None:
            return None
        angle_index = ANGLES_DEGREES.index(angle_degrees)
        view_index = VIEW_LABELS.index(view_label)
        return str(self.projection_provenance[angle_index, view_index])



def load_lightcurve(
    data_dir: Path,
    model_id: int,
    curve_type: str = "intensity",
    source: str = "blender",
) -> LightcurveTable:
    """Load a HAC lightcurve file and reshape columns to ``time, angle, view``."""
    path = lightcurve_path(data_dir, model_id, curve_type, source)
    if not path.exists():
        raise FileNotFoundError(path)

    matrix = np.loadtxt(path, delimiter=",")
    _validate_lightcurve_matrix(matrix, path)

    values = matrix[:, 1:].reshape(
        matrix.shape[0],
        len(ANGLES_DEGREES),
        len(VIEW_LABELS),
    )
    _validate_released_curve_relationships(
        values,
        path=path,
        model_id=model_id,
        curve_type=curve_type,
        source=source,
    )
    normalization_means = np.mean(values, axis=0)
    values = values / normalization_means[None, :, :]
    validity_weights, independence_weights, projection_provenance = (
        released_curve_metadata(
            model_id, curve_type, source,
            file_sha256=(hashlib.sha256(path.read_bytes()).hexdigest()
                         if model_id == 1 and source == "blender" else None),
        )
    )
    return LightcurveTable(
        model_id=model_id,
        curve_type=curve_type,
        source=source,
        path=path,
        time=matrix[:, 0],
        values=values,
        validity_weights=validity_weights,
        independence_weights=independence_weights,
        projection_provenance=projection_provenance,
        normalization_means=normalization_means,
    )



def released_curve_metadata(
    model_id: int,
    curve_type: str,
    source: str,
    *,
    file_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return validity, independence, and projection metadata for a release."""
    # Route through the path helper so model/source/type validation stays in
    # one place without depending on a path existing on disk.
    lightcurve_path(Path("."), model_id, curve_type, source)

    validity = np.ones(CURVE_SHAPE, dtype=np.float64)
    independence = np.ones(CURVE_SHAPE, dtype=np.float64)

    if source == "real":
        provenance = np.full(CURVE_SHAPE, PROJECTION_LABORATORY, dtype="<U24")
        if model_id == 3 and curve_type == "intensity":
            validity[
                ANGLES_DEGREES.index(0),
                VIEW_LABELS.index("horizontal_1"),
            ] = 0.0
        return validity, independence, provenance

    # Every current Blender table repeats horizontal_1 as horizontal_2.  Half
    # weights preserve one independent horizontal observation per angle while
    # retaining a symmetric table for JAX code.
    independence[:, VIEW_LABELS.index("horizontal_1")] = 0.5
    independence[:, VIEW_LABELS.index("horizontal_2")] = 0.5

    if model_id == 1:
        provenance = np.full(
            CURVE_SHAPE,
            MODEL1_BLENDER_PROJECTION_BY_SHA256.get(file_sha256 or "", PROJECTION_UNVERIFIED),
            dtype="<U24",
        )
    elif model_id == 2:
        provenance = np.full(
            CURVE_SHAPE,
            PROJECTION_LEGACY_PERSPECTIVE,
            dtype="<U24",
        )
        provenance[:, VIEW_LABELS.index("horizontal_1")] = PROJECTION_ORTHOGRAPHIC
        provenance[:, VIEW_LABELS.index("horizontal_2")] = PROJECTION_ORTHOGRAPHIC
    elif model_id == 3:
        provenance = np.full(CURVE_SHAPE, PROJECTION_ORTHOGRAPHIC, dtype="<U24")
    else:
        provenance = np.full(CURVE_SHAPE, PROJECTION_UNVERIFIED, dtype="<U24")
    return validity, independence, provenance



def resample_periodic_table(values: np.ndarray, n_samples: int) -> np.ndarray:
    """Periodically resample and mean-normalize a full HAC curve table."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError("n_samples must be an integer of at least two")
    target_n = int(n_samples)
    if target_n < 2:
        raise ValueError("n_samples must be an integer of at least two")
    array = _validate_curve_table_values(values)
    source_n = array.shape[0]

    if source_n == target_n:
        result = np.array(array, copy=True)
    else:
        source_x = np.arange(source_n + 1, dtype=np.float64) / float(source_n)
        target_x = np.arange(target_n, dtype=np.float64) / float(target_n)
        flat = array.reshape(source_n, -1)
        periodic_flat = np.vstack([flat, flat[:1]])
        result_flat = np.empty((target_n, flat.shape[1]), dtype=np.float64)
        for curve_index in range(flat.shape[1]):
            result_flat[:, curve_index] = np.interp(
                target_x,
                source_x,
                periodic_flat[:, curve_index],
            )
        result = result_flat.reshape((target_n, *CURVE_SHAPE))

    means = np.mean(result, axis=0)
    if not np.all(np.isfinite(means)) or np.any(means <= 0.0):
        raise ValueError("resampled curves must have finite positive means")
    return result / means[None, :, :]



def _validate_lightcurve_matrix(matrix: np.ndarray, path: Path) -> None:
    """Validate the numeric contract shared by all released HAC tables."""
    if matrix.ndim != 2 or matrix.shape[1] != 29:
        raise ValueError(f"{path} should have shape (n, 29), got {matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError(f"{path} must contain at least two samples")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{path} contains non-finite timestamps or lightcurve values")
    time = matrix[:, 0]
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{path} timestamps must be strictly increasing")
    values = matrix[:, 1:]
    if np.any(values < 0.0):
        raise ValueError(f"{path} contains negative lightcurve values")
    means = np.mean(values, axis=0)
    if np.any(means <= 0.0):
        raise ValueError(f"{path} contains a zero-mean lightcurve")
    max_error = float(np.max(np.abs(means - 1.0)))
    if max_error > NORMALIZATION_ATOL:
        raise ValueError(
            f"{path} curves are not mean-normalized: max |mean - 1| "
            f"is {max_error:.6g}, tolerance is {NORMALIZATION_ATOL:.6g}"
        )



def _validate_curve_table_values(values: np.ndarray) -> np.ndarray:
    """Return a finite float64 HAC table after validating its array contract."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != CURVE_SHAPE:
        raise ValueError(
            f"values must have shape (n, {CURVE_SHAPE[0]}, {CURVE_SHAPE[1]}), "
            f"got {array.shape}"
        )
    if array.shape[0] < 2:
        raise ValueError("values must contain at least two source samples")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite samples")
    return array



def _validate_released_curve_relationships(
    values: np.ndarray,
    *,
    path: Path,
    model_id: int,
    curve_type: str,
    source: str,
) -> None:
    """Fail loudly if release-specific duplicate assumptions stop holding."""
    if source == "blender" and not np.array_equal(values[:, :, 0], values[:, :, 1]):
        raise ValueError(
            f"{path} no longer has identical Blender horizontal_1/horizontal_2 "
            "curves; update released-data independence metadata"
        )
    if model_id == 3 and curve_type == "intensity" and source == "real":
        angle_index = ANGLES_DEGREES.index(0)
        horizontal_index = VIEW_LABELS.index("horizontal_1")
        top_index = VIEW_LABELS.index("top")
        if not np.array_equal(
            values[:, angle_index, horizontal_index],
            values[:, angle_index, top_index],
        ):
            raise ValueError(
                f"{path} no longer has the documented model-3 0-degree "
                "horizontal_1/top duplicate; update released-data validity metadata"
            )

