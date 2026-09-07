"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np
from .baseline import ForwardModelBaseline
from .constants import ANGLES_DEGREES, VIEW_LABELS
from .geometry import view_directions
from .lightcurves import load_lightcurve, resample_periodic_table


@dataclass(frozen=True)
class PredictionCalibration:
    """Fixed prediction-side calibration for an inversion target."""

    angle_phase_shifts_degrees: dict[int, float]
    contrast_gain: float = 1.0



@dataclass(frozen=True)
class InversionTarget:
    """Observed lightcurves and geometry used by the radial inversion code."""

    model_id: int
    curve_type: str
    source: str
    observed: np.ndarray
    phases: np.ndarray
    view_directions: np.ndarray
    angles_degrees: tuple[int, ...]
    view_labels: tuple[str, ...]
    calibration: PredictionCalibration
    lightcurve_path: Path
    curve_weights: np.ndarray | None = None
    projection_provenance: np.ndarray | None = None

    def __post_init__(self) -> None:
        observed = np.asarray(self.observed)
        expected_curve_shape = (len(self.angles_degrees), len(self.view_labels))
        if observed.ndim != 3 or observed.shape[1:] != expected_curve_shape:
            raise ValueError(
                "observed must have shape (time, angle, view), got "
                f"{observed.shape} for curve shape {expected_curve_shape}"
            )
        if observed.shape[0] < 2 or not np.all(np.isfinite(observed)):
            raise ValueError("observed must contain at least two finite samples")
        phases = np.asarray(self.phases)
        if phases.shape != (observed.shape[0],) or not np.all(np.isfinite(phases)):
            raise ValueError("phases must contain one finite value per observed sample")
        directions = np.asarray(self.view_directions)
        if directions.shape != (expected_curve_shape[0] * expected_curve_shape[1], 3):
            raise ValueError(
                "view_directions must have shape "
                f"({expected_curve_shape[0] * expected_curve_shape[1]}, 3), "
                f"got {directions.shape}"
            )
        if self.curve_weights is not None:
            weights = np.asarray(self.curve_weights, dtype=np.float64)
            if weights.shape != expected_curve_shape:
                raise ValueError(
                    f"curve_weights must have shape {expected_curve_shape}, "
                    f"got {weights.shape}"
                )
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("curve_weights must contain finite non-negative values")
            if float(np.sum(weights)) <= 0.0:
                raise ValueError("at least one selected curve must have positive weight")
        if self.projection_provenance is not None:
            provenance = np.asarray(self.projection_provenance)
            if provenance.shape != expected_curve_shape:
                raise ValueError(
                    "projection_provenance must have shape "
                    f"{expected_curve_shape}, got {provenance.shape}"
                )

    @property
    def effective_curve_weights(self) -> np.ndarray:
        """Return explicit curve weights, defaulting to legacy all-ones behavior."""
        shape = (len(self.angles_degrees), len(self.view_labels))
        if self.curve_weights is None:
            return np.ones(shape, dtype=np.float64)
        return np.asarray(self.curve_weights, dtype=np.float64)



@dataclass(frozen=True)
class AdamResult:
    """Output from a deterministic Adam inversion run."""

    best_parameters: np.ndarray
    final_parameters: np.ndarray
    best_loss: float
    final_loss: float
    history: list[dict[str, float | int]]



def prepare_inversion_target(
    data_dir: Path,
    *,
    model_id: int,
    curve_type: str,
    source: str,
    n_samples: int,
    angles_degrees: list[int],
    view_labels_selected: list[str],
    baseline: ForwardModelBaseline,
    calibration_mode: str = "none",
) -> InversionTarget:
    """Load, resample, select, and package lightcurves for inversion."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    if not angles_degrees or len(set(angles_degrees)) != len(angles_degrees):
        raise ValueError("angles_degrees must contain unique released HAC angles")
    if not view_labels_selected or len(set(view_labels_selected)) != len(
        view_labels_selected
    ):
        raise ValueError("view_labels_selected must contain unique HAC view labels")
    if calibration_mode == "baseline" and (
        source != baseline.target_source or curve_type != baseline.curve_type
    ):
        raise ValueError(
            f"baseline {baseline.name!r} calibrates "
            f"{baseline.target_source}/{baseline.curve_type}, not {source}/{curve_type}; "
            "use calibration_mode='none' or select a compatible baseline"
        )
    table = load_lightcurve(data_dir, model_id, curve_type, source)
    values = resample_periodic_table(table.values, n_samples)
    angle_indices = [ANGLES_DEGREES.index(angle) for angle in angles_degrees]
    view_indices = [VIEW_LABELS.index(view) for view in view_labels_selected]
    selected = values[:, angle_indices][:, :, view_indices]
    selected_weights = table.curve_weights[angle_indices][:, view_indices]
    selected_provenance = (
        None
        if table.projection_provenance is None
        else np.asarray(table.projection_provenance)[angle_indices][:, view_indices]
    )

    all_directions = view_directions(
        alpha_from=baseline.alpha_from,
        azimuth_offset_degrees=baseline.azimuth_offset,
        azimuth_sign=baseline.azimuth_sign,
        top_z_sign=baseline.top_z_sign,
    )
    selected_directions = all_directions[angle_indices][:, view_indices].reshape(-1, 3)
    phases = baseline.phase_sign * np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)

    if calibration_mode == "none":
        calibration = PredictionCalibration(angle_phase_shifts_degrees={}, contrast_gain=1.0)
    elif calibration_mode == "baseline":
        calibration = PredictionCalibration(
            angle_phase_shifts_degrees=dict(baseline.angle_phase_shifts_degrees),
            contrast_gain=baseline.contrast_gain,
        )
    else:
        raise ValueError(f"unsupported calibration_mode: {calibration_mode!r}")

    return InversionTarget(
        model_id=model_id,
        curve_type=curve_type,
        source=source,
        observed=selected.astype(np.float64),
        phases=phases.astype(np.float64),
        view_directions=selected_directions.astype(np.float64),
        angles_degrees=tuple(angles_degrees),
        view_labels=tuple(view_labels_selected),
        calibration=calibration,
        lightcurve_path=table.path,
        curve_weights=selected_weights.astype(np.float64),
        projection_provenance=selected_provenance,
    )



def weighted_curve_misfit_jax(
    residual: jnp.ndarray,
    curve_weights: jnp.ndarray | np.ndarray | None = None,
) -> jnp.ndarray:
    """Return half the curve-weighted mean squared residual.

    Weights apply to the trailing ``(angle, view)`` axes and are normalized by
    their sum.  All-one weights therefore exactly recover ``0.5 * mean(r**2)``;
    zero-weight invalid curves do not change the scale of priors or other loss
    terms.
    """
    values = jnp.asarray(residual)
    if values.ndim != 3:
        raise ValueError("residual must have shape (time, angle, view)")
    if curve_weights is None:
        return 0.5 * jnp.mean(values**2)
    weights = jnp.asarray(curve_weights, dtype=values.dtype)
    if tuple(weights.shape) != tuple(values.shape[1:]):
        raise ValueError(
            f"curve_weights shape {weights.shape} does not match {values.shape[1:]}"
        )
    denominator = values.shape[0] * jnp.sum(weights)
    return 0.5 * jnp.sum(values**2 * weights[None, :, :]) / denominator



def apply_prediction_calibration_jax(
    predicted: jnp.ndarray,
    angles_degrees: tuple[int, ...],
    calibration: PredictionCalibration,
) -> jnp.ndarray:
    """Apply fixed per-angle circular shifts and contrast gain to predictions."""
    shifted = predicted
    n_samples = int(predicted.shape[0])
    for angle_index, angle in enumerate(angles_degrees):
        shift_degrees = calibration.angle_phase_shifts_degrees.get(int(angle), 0.0)
        if shift_degrees == 0.0:
            continue
        sample_shift = int(round(float(shift_degrees) * n_samples / 360.0))
        shifted = shifted.at[:, angle_index, :].set(
            jnp.roll(shifted[:, angle_index, :], sample_shift, axis=0)
        )
    if calibration.contrast_gain != 1.0:
        shifted = 1.0 + float(calibration.contrast_gain) * (shifted - 1.0)
    return shifted



def run_adam(
    objective: Callable[[jnp.ndarray], jnp.ndarray],
    initial_parameters: np.ndarray,
    *,
    n_steps: int,
    learning_rate: float,
    log_every: int = 25,
) -> AdamResult:
    """Minimize ``objective`` from one initialization using Adam."""
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if isinstance(log_every, bool) or int(log_every) != log_every or log_every < 1:
        raise ValueError("log_every must be a positive integer")
    initial = np.asarray(initial_parameters)
    if initial.ndim != 1 or initial.size == 0:
        raise ValueError("initial_parameters must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(initial)):
        raise ValueError("initial_parameters must be finite")
    value_and_grad = jax.jit(jax.value_and_grad(objective))
    params = jnp.asarray(initial)
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8

    best_params = np.asarray(params)
    best_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for step in range(1, n_steps + 1):
        loss, grad = value_and_grad(params)
        loss_value = float(loss)
        grad_norm = float(jnp.linalg.norm(grad))
        if not np.isfinite(loss_value) or not np.isfinite(grad_norm):
            raise FloatingPointError(
                f"non-finite Adam objective/gradient at step {step}: "
                f"loss={loss_value}, grad_norm={grad_norm}"
            )
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = np.asarray(params)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        params = params - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

        if step == 1 or step == n_steps or step % log_every == 0:
            history.append(
                {
                    "step": step,
                    "loss": loss_value,
                    "grad_norm": grad_norm,
                }
            )

    final_loss = float(objective(params))
    if not np.isfinite(final_loss) or not np.all(np.isfinite(np.asarray(params))):
        raise FloatingPointError("Adam produced non-finite final parameters or loss")
    if final_loss < best_loss:
        best_loss = final_loss
        best_params = np.asarray(params)
    return AdamResult(
        best_parameters=best_params,
        final_parameters=np.asarray(params),
        best_loss=best_loss,
        final_loss=final_loss,
        history=history,
    )



def run_multistart_adam(
    objective: Callable[[jnp.ndarray], jnp.ndarray],
    initial_parameters: np.ndarray,
    *,
    n_steps: int,
    learning_rate: float,
    log_every: int = 25,
) -> AdamResult:
    """Run Adam from several initializations and keep the best result."""
    starts = np.asarray(initial_parameters)
    if starts.ndim != 2 or starts.shape[0] == 0 or starts.shape[1] == 0:
        raise ValueError(
            "initial_parameters must have shape (n_starts, n_parameters) with both positive"
        )
    if not np.all(np.isfinite(starts)):
        raise ValueError("initial_parameters must be finite")
    results = [
        run_adam(
            objective,
            start,
            n_steps=n_steps,
            learning_rate=learning_rate,
            log_every=log_every,
        )
        for start in starts
    ]
    best_index = int(np.argmin([result.best_loss for result in results]))
    best = results[best_index]
    history: list[dict[str, float | int]] = []
    for start_index, result in enumerate(results):
        for row in result.history:
            history.append({"start": start_index, **row})
    return AdamResult(
        best_parameters=best.best_parameters,
        final_parameters=best.final_parameters,
        best_loss=best.best_loss,
        final_loss=best.final_loss,
        history=history,
    )

