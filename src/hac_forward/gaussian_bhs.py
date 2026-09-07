"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ClosureConfig:
    """Numerical controls for the three-dimensional closure solve."""

    tolerance: float = 1.0e-12
    max_iterations: int = 64
    max_backtracks: int = 20
    armijo_fraction: float = 1.0e-4
    hessian_relative_floor: float = 1.0e-12
    max_step_norm: float = 20.0
    condition_warning: float = 1.0e10
    raise_on_failure: bool = True

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_backtracks <= 0:
            raise ValueError("max_backtracks must be positive")
        if not 0.0 < self.armijo_fraction < 0.5:
            raise ValueError("armijo_fraction must lie in (0, 0.5)")
        if self.hessian_relative_floor <= 0.0:
            raise ValueError("hessian_relative_floor must be positive")
        if self.max_step_norm <= 0.0:
            raise ValueError("max_step_norm must be positive")
        if self.condition_warning <= 1.0:
            raise ValueError("condition_warning must exceed one")



@dataclass(frozen=True)
class ClosureDiagnostics:
    """Convergence and conditioning information for a NumPy closure solve."""

    converged: bool
    iterations: int
    backtracks: int
    closure_norm: float
    mass_error: float
    condition_number: float
    min_eigenvalue: float
    max_eigenvalue: float
    tilt_norm: float
    min_mass: float
    objective: float
    ill_conditioned: bool



@dataclass(frozen=True)
class ClosureResult:
    """Closed masses, exponential tilt, and NumPy diagnostics."""

    masses: np.ndarray
    tilt: np.ndarray
    diagnostics: ClosureDiagnostics



class JaxClosureDiagnostics(NamedTuple):
    """JAX-array diagnostics returned by :func:`solve_closure_jax`."""

    converged: jax.Array
    iterations: jax.Array
    backtracks: jax.Array
    closure_norm: jax.Array
    mass_error: jax.Array
    condition_number: jax.Array
    min_eigenvalue: jax.Array
    max_eigenvalue: jax.Array
    tilt_norm: jax.Array
    min_mass: jax.Array
    objective: jax.Array
    ill_conditioned: jax.Array



DEFAULT_CLOSURE_CONFIG = ClosureConfig()



_FLOAT64_EPS = float(np.finfo(np.float64).eps)



_FLOAT64_TINY = float(np.finfo(np.float64).tiny)



def close_masses_np(
    logits: np.ndarray,
    base_masses: np.ndarray,
    normals: np.ndarray,
    *,
    config: ClosureConfig = DEFAULT_CLOSURE_CONFIG,
) -> np.ndarray:
    """Return positive, mass-one, Minkowski-closed NumPy weights.

    ``base_masses`` may contain quadrature weights, reference facet areas, or
    any other strictly positive base measure.  Multiplying all base masses by
    a common constant, or adding a common constant to all logits, leaves the
    result unchanged.
    """
    return solve_closure_np(logits, base_masses, normals, config=config).masses



def solve_closure_np(
    logits: np.ndarray,
    base_masses: np.ndarray,
    normals: np.ndarray,
    *,
    config: ClosureConfig = DEFAULT_CLOSURE_CONFIG,
) -> ClosureResult:
    """Solve the exponential-tilt closure problem with damped Newton steps.

    The origin must lie in the interior of the convex hull of the supplied
    normals.  This is automatic for a full antipodally symmetric normal grid.
    If the Newton solve does not converge, the default behavior is to raise a
    ``RuntimeError``.  Set ``config.raise_on_failure=False`` to inspect the
    returned diagnostics instead.
    """
    logits_np, base_np, normals_np = _validate_numpy_inputs(
        logits,
        base_masses,
        normals,
    )
    normal_scale = float(np.max(np.linalg.norm(normals_np, axis=1)))
    scaled_normals = normals_np / normal_scale
    base_log = logits_np + np.log(base_np)
    scaled_tilt = np.zeros(3, dtype=np.float64)
    tolerance = max(config.tolerance, 8.0 * _FLOAT64_EPS)
    total_backtracks = 0
    iterations = 0

    for _iteration in range(config.max_iterations):
        masses, mean, covariance, objective = _tilt_statistics_np(
            base_log,
            scaled_normals,
            scaled_tilt,
        )
        closure_norm = float(np.linalg.norm(mean))
        if not np.isfinite(closure_norm) or closure_norm <= tolerance:
            break

        direction = _newton_direction_np(mean, covariance, config)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > config.max_step_norm:
            direction = direction * (config.max_step_norm / direction_norm)
        descent = float(mean @ direction)
        if not np.isfinite(descent) or descent <= 0.0:
            direction = mean.copy()
            descent = float(mean @ mean)

        candidate, accepted, used_backtracks = _line_search_np(
            base_log,
            scaled_normals,
            scaled_tilt,
            direction,
            objective,
            descent,
            closure_norm,
            config,
        )
        total_backtracks += used_backtracks
        iterations += 1
        if not accepted:
            break
        scaled_tilt = candidate

    masses, scaled_mean, scaled_covariance, objective = _tilt_statistics_np(
        base_log,
        scaled_normals,
        scaled_tilt,
    )
    tilt = scaled_tilt / normal_scale
    mean = scaled_mean * normal_scale
    diagnostics = _diagnostics_np(
        masses,
        mean,
        scaled_covariance,
        objective,
        tilt,
        iterations,
        total_backtracks,
        tolerance * normal_scale,
        config,
    )
    result = ClosureResult(masses=masses, tilt=tilt, diagnostics=diagnostics)
    if config.raise_on_failure and not diagnostics.converged:
        raise RuntimeError(
            "exponential-tilt closure did not converge: "
            f"closure_norm={diagnostics.closure_norm:.3e}, "
            f"iterations={diagnostics.iterations}, "
            f"condition_number={diagnostics.condition_number:.3e}"
        )
    return result



def close_masses_jax(
    logits: jax.Array,
    base_masses: jax.Array,
    normals: jax.Array,
    *,
    config: ClosureConfig = DEFAULT_CLOSURE_CONFIG,
) -> jax.Array:
    """Return differentiable positive, mass-one, closed JAX weights.

    The custom derivative is obtained from the closure constraint by implicit
    differentiation.  It supports derivatives with respect to logits, base
    masses, and normals.  Use :func:`solve_closure_jax` when the tilt and
    convergence diagnostics are also needed.  If that solve is nonconverged,
    ill-conditioned, or loses strict positivity at the working precision, this
    convenience function returns NaNs rather than silently exposing an invalid
    implicit derivative.  Float64 is recommended for geometric inversion.
    """
    logits_jax, base_jax, normals_jax = _prepare_jax_inputs(
        logits,
        base_masses,
        normals,
    )
    return _close_masses_jax_implicit(logits_jax, base_jax, normals_jax, config)



def _validate_numpy_inputs(
    logits: np.ndarray,
    base_masses: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits_np = np.asarray(logits, dtype=np.float64)
    base_np = np.asarray(base_masses, dtype=np.float64)
    normals_np = np.asarray(normals, dtype=np.float64)
    if logits_np.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if base_np.shape != logits_np.shape:
        raise ValueError("base_masses must have the same shape as logits")
    if normals_np.shape != (logits_np.size, 3):
        raise ValueError("normals must have shape (n_masses, 3)")
    if logits_np.size < 4:
        raise ValueError("at least four normals are required")
    if not np.all(np.isfinite(logits_np)):
        raise ValueError("logits must be finite")
    if not np.all(np.isfinite(base_np)) or np.any(base_np <= 0.0):
        raise ValueError("base_masses must be finite and strictly positive")
    if not np.all(np.isfinite(normals_np)):
        raise ValueError("normals must be finite")
    if np.linalg.matrix_rank(normals_np) < 3:
        raise ValueError("normals must span three dimensions")
    return logits_np, base_np, normals_np



def _prepare_jax_inputs(
    logits: jax.Array,
    base_masses: jax.Array,
    normals: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    logits_array = jnp.asarray(logits)
    base_array = jnp.asarray(base_masses)
    normals_array = jnp.asarray(normals)
    if logits_array.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if base_array.shape != logits_array.shape:
        raise ValueError("base_masses must have the same shape as logits")
    if normals_array.shape != (logits_array.shape[0], 3):
        raise ValueError("normals must have shape (n_masses, 3)")
    if logits_array.shape[0] < 4:
        raise ValueError("at least four normals are required")
    dtype = jnp.result_type(logits_array, base_array, normals_array, jnp.float32)
    return (
        jnp.asarray(logits_array, dtype=dtype),
        jnp.asarray(base_array, dtype=dtype),
        jnp.asarray(normals_array, dtype=dtype),
    )



def _tilt_statistics_np(
    base_log: np.ndarray,
    normals: np.ndarray,
    tilt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scores = base_log + normals @ tilt
    shift = float(np.max(scores))
    floor = np.log(_FLOAT64_TINY)
    unnormalized = np.exp(np.maximum(scores - shift, floor))
    total = float(np.sum(unnormalized))
    masses = unnormalized / total
    objective = shift + float(np.log(total))
    mean = masses @ normals
    centered = normals - mean
    covariance = (centered.T * masses) @ centered
    covariance = 0.5 * (covariance + covariance.T)
    return masses, mean, covariance, objective



def _newton_direction_np(
    mean: np.ndarray,
    covariance: np.ndarray,
    config: ClosureConfig,
) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(covariance)
    maximum = max(float(eigenvalues[-1]), _FLOAT64_TINY)
    relative_floor = max(
        config.hessian_relative_floor,
        8.0 * _FLOAT64_EPS,
    )
    damping = max(relative_floor * maximum - float(eigenvalues[0]), 0.0)
    regularized = covariance + damping * np.eye(3, dtype=np.float64)
    try:
        return np.linalg.solve(regularized, mean)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(regularized, rcond=relative_floor) @ mean



def _line_search_np(
    base_log: np.ndarray,
    normals: np.ndarray,
    tilt: np.ndarray,
    direction: np.ndarray,
    objective: float,
    descent: float,
    closure_norm: float,
    config: ClosureConfig,
) -> tuple[np.ndarray, bool, int]:
    alpha = 1.0
    objective_slack = 64.0 * _FLOAT64_EPS * max(1.0, abs(objective))
    for attempt in range(config.max_backtracks):
        candidate = tilt - alpha * direction
        _, candidate_mean, _, candidate_objective = _tilt_statistics_np(
            base_log,
            normals,
            candidate,
        )
        candidate_closure_norm = float(np.linalg.norm(candidate_mean))
        closure_merit = (
            np.isfinite(candidate_objective)
            and candidate_objective <= objective + objective_slack
            and np.isfinite(candidate_closure_norm)
            and candidate_closure_norm < closure_norm
        )
        armijo_bound = objective - config.armijo_fraction * alpha * descent
        satisfies_armijo = np.isfinite(candidate_objective) and candidate_objective <= armijo_bound
        if satisfies_armijo or closure_merit:
            return candidate, True, attempt
        alpha *= 0.5
    return tilt, False, config.max_backtracks



def _diagnostics_np(
    masses: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    objective: float,
    tilt: np.ndarray,
    iterations: int,
    backtracks: int,
    tolerance: float,
    config: ClosureConfig,
) -> ClosureDiagnostics:
    eigenvalues = np.linalg.eigvalsh(covariance)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    denominator = max(minimum, _FLOAT64_TINY)
    condition_number = maximum / denominator
    closure_norm = float(np.linalg.norm(mean))
    mass_error = abs(float(np.sum(masses)) - 1.0)
    effective_relative_floor = max(config.hessian_relative_floor, 8.0 * _FLOAT64_EPS)
    ill_conditioned = bool(
        not np.isfinite(condition_number)
        or condition_number >= config.condition_warning
        or minimum <= effective_relative_floor * max(maximum, _FLOAT64_TINY)
    )
    return ClosureDiagnostics(
        converged=bool(
            np.isfinite(closure_norm)
            and closure_norm <= tolerance
            and mass_error <= 8.0 * _FLOAT64_EPS
            and np.all(np.isfinite(masses))
            and np.all(masses > 0.0)
        ),
        iterations=iterations,
        backtracks=backtracks,
        closure_norm=closure_norm,
        mass_error=mass_error,
        condition_number=condition_number,
        min_eigenvalue=minimum,
        max_eigenvalue=maximum,
        tilt_norm=float(np.linalg.norm(tilt)),
        min_mass=float(np.min(masses)),
        objective=float(objective),
        ill_conditioned=ill_conditioned,
    )



def _tilt_statistics_jax(
    base_log: jax.Array,
    normals: jax.Array,
    tilt: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    scores = base_log + normals @ tilt
    shift = jnp.max(scores)
    floor = jnp.log(jnp.finfo(scores.dtype).tiny)
    unnormalized = jnp.exp(jnp.maximum(scores - shift, floor))
    total = jnp.sum(unnormalized)
    masses = unnormalized / total
    objective = shift + jnp.log(total)
    mean = masses @ normals
    centered = normals - mean
    covariance = (centered.T * masses) @ centered
    covariance = 0.5 * (covariance + covariance.T)
    return masses, mean, covariance, objective



def _newton_direction_jax(
    mean: jax.Array,
    covariance: jax.Array,
    config: ClosureConfig,
) -> jax.Array:
    eigenvalues = jnp.linalg.eigvalsh(covariance)
    tiny = jnp.finfo(covariance.dtype).tiny
    eps = jnp.finfo(covariance.dtype).eps
    maximum = jnp.maximum(eigenvalues[-1], tiny)
    relative_floor = jnp.maximum(
        jnp.asarray(config.hessian_relative_floor, dtype=covariance.dtype),
        8.0 * eps,
    )
    damping = jnp.maximum(relative_floor * maximum - eigenvalues[0], 0.0)
    regularized = covariance + damping * jnp.eye(3, dtype=covariance.dtype)
    direction = jnp.linalg.solve(regularized, mean)
    direction_norm = jnp.linalg.norm(direction)
    scale = jnp.minimum(
        1.0,
        jnp.asarray(config.max_step_norm, dtype=direction.dtype)
        / jnp.maximum(direction_norm, tiny),
    )
    direction = direction * scale
    descent = mean @ direction
    use_gradient = jnp.logical_or(~jnp.isfinite(descent), descent <= 0.0)
    return jnp.where(use_gradient, mean, direction)



def _line_search_jax(
    base_log: jax.Array,
    normals: jax.Array,
    tilt: jax.Array,
    direction: jax.Array,
    objective: jax.Array,
    config: ClosureConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    _, mean, _, _ = _tilt_statistics_jax(base_log, normals, tilt)
    descent = jnp.maximum(mean @ direction, 0.0)
    closure_norm = jnp.linalg.norm(mean)
    objective_slack = (
        64.0
        * jnp.finfo(objective.dtype).eps
        * jnp.maximum(jnp.asarray(1.0, dtype=objective.dtype), jnp.abs(objective))
    )
    initial = (
        jnp.asarray(1.0, dtype=tilt.dtype),
        jnp.asarray(False),
        tilt,
        jnp.asarray(0, dtype=jnp.int32),
    )

    def body(
        _index: int,
        state: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        alpha, accepted, chosen, attempts = state
        candidate = tilt - alpha * direction
        _, candidate_mean, _, candidate_objective = _tilt_statistics_jax(
            base_log,
            normals,
            candidate,
        )
        candidate_closure_norm = jnp.linalg.norm(candidate_mean)
        active = ~accepted
        finite = jnp.isfinite(candidate_objective)
        closure_finite = jnp.isfinite(candidate_closure_norm)
        closure_merit = jnp.logical_and(
            jnp.logical_and(finite, candidate_objective <= objective + objective_slack),
            jnp.logical_and(closure_finite, candidate_closure_norm < closure_norm),
        )
        bound = objective - config.armijo_fraction * alpha * descent
        satisfies_armijo = jnp.logical_and(finite, candidate_objective <= bound)
        take = jnp.logical_and(active, jnp.logical_or(satisfies_armijo, closure_merit))
        chosen = jnp.where(take, candidate, chosen)
        accepted = jnp.logical_or(accepted, take)
        attempts = attempts + active.astype(jnp.int32)
        alpha = jnp.where(accepted, alpha, 0.5 * alpha)
        return alpha, accepted, chosen, attempts

    _, accepted, chosen, attempts = jax.lax.fori_loop(
        0,
        config.max_backtracks,
        body,
        initial,
    )
    backtracks = jnp.maximum(attempts - 1, 0)
    return chosen, accepted, backtracks



def _solve_closure_jax_raw(
    logits: jax.Array,
    base_masses: jax.Array,
    normals: jax.Array,
    config: ClosureConfig,
) -> tuple[jax.Array, jax.Array, JaxClosureDiagnostics]:
    base_log = logits + jnp.log(base_masses)
    eps = jnp.finfo(logits.dtype).eps
    tiny = jnp.finfo(logits.dtype).tiny
    normal_scale = jnp.max(jnp.linalg.norm(normals, axis=1))
    valid_normal_scale = jnp.logical_and(jnp.isfinite(normal_scale), normal_scale > tiny)
    safe_normal_scale = jnp.where(valid_normal_scale, normal_scale, 1.0)
    scaled_normals = normals / safe_normal_scale
    tolerance = jnp.maximum(
        jnp.asarray(config.tolerance, dtype=logits.dtype),
        8.0 * eps,
    )
    initial = (
        jnp.zeros((3,), dtype=logits.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(True),
    )

    def body(
        _index: int,
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        tilt, iterations, backtracks, running = state
        _, mean, covariance, objective = _tilt_statistics_jax(
            base_log,
            scaled_normals,
            tilt,
        )
        closure_norm = jnp.linalg.norm(mean)
        active = jnp.logical_and(
            running,
            jnp.logical_and(jnp.isfinite(closure_norm), closure_norm > tolerance),
        )
        direction = _newton_direction_jax(mean, covariance, config)
        candidate, accepted, used_backtracks = _line_search_jax(
            base_log,
            scaled_normals,
            tilt,
            direction,
            objective,
            config,
        )
        update = jnp.logical_and(active, accepted)
        tilt = jnp.where(update, candidate, tilt)
        iterations = iterations + active.astype(jnp.int32)
        backtracks = backtracks + jnp.where(active, used_backtracks, 0)
        running = jnp.logical_and(active, accepted)
        return tilt, iterations, backtracks, running

    scaled_tilt, iterations, backtracks, _ = jax.lax.fori_loop(
        0,
        config.max_iterations,
        body,
        initial,
    )
    masses, scaled_mean, scaled_covariance, objective = _tilt_statistics_jax(
        base_log,
        scaled_normals,
        scaled_tilt,
    )
    tilt = scaled_tilt / safe_normal_scale
    mean = scaled_mean * safe_normal_scale
    eigenvalues = jnp.linalg.eigvalsh(scaled_covariance)
    minimum = eigenvalues[0]
    maximum = eigenvalues[-1]
    condition_number = maximum / jnp.maximum(minimum, tiny)
    closure_norm = jnp.linalg.norm(mean)
    mass_error = jnp.abs(jnp.sum(masses) - 1.0)
    effective_relative_floor = jnp.maximum(
        jnp.asarray(config.hessian_relative_floor, dtype=logits.dtype),
        8.0 * eps,
    )
    ill_conditioned = jnp.logical_or(
        jnp.logical_or(
            ~jnp.isfinite(condition_number), condition_number >= config.condition_warning
        ),
        minimum <= effective_relative_floor * jnp.maximum(maximum, tiny),
    )
    inputs_valid = jnp.logical_and(
        valid_normal_scale,
        jnp.logical_and(
            jnp.all(jnp.isfinite(logits)),
            jnp.logical_and(
                jnp.all(jnp.logical_and(jnp.isfinite(base_masses), base_masses > 0.0)),
                jnp.all(jnp.isfinite(normals)),
            ),
        ),
    )
    strictly_positive = jnp.logical_and(
        jnp.all(jnp.isfinite(masses)),
        jnp.all(masses > 0.0),
    )
    diagnostics = JaxClosureDiagnostics(
        converged=jnp.logical_and(
            jnp.logical_and(inputs_valid, strictly_positive),
            jnp.logical_and(
                jnp.logical_and(
                    jnp.isfinite(closure_norm),
                    closure_norm <= tolerance * safe_normal_scale,
                ),
                mass_error <= 8.0 * eps,
            ),
        ),
        iterations=iterations,
        backtracks=backtracks,
        closure_norm=closure_norm,
        mass_error=mass_error,
        condition_number=condition_number,
        min_eigenvalue=minimum,
        max_eigenvalue=maximum,
        tilt_norm=jnp.linalg.norm(tilt),
        min_mass=jnp.min(masses),
        objective=objective,
        ill_conditioned=ill_conditioned,
    )
    return masses, tilt, diagnostics



@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _close_masses_jax_implicit(
    logits: jax.Array,
    base_masses: jax.Array,
    normals: jax.Array,
    config: ClosureConfig,
) -> jax.Array:
    masses, _, diagnostics = _solve_closure_jax_raw(logits, base_masses, normals, config)
    valid = jnp.logical_and(diagnostics.converged, ~diagnostics.ill_conditioned)
    return jnp.where(valid, masses, jnp.full_like(masses, jnp.nan))



@_close_masses_jax_implicit.defjvp
def _close_masses_jax_implicit_jvp(
    config: ClosureConfig,
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    logits, base_masses, normals = primals
    logits_dot, base_masses_dot, normals_dot = tangents
    masses, tilt, diagnostics = _solve_closure_jax_raw(
        logits,
        base_masses,
        normals,
        config,
    )
    mean = masses @ normals
    centered_normals = normals - mean
    normal_scale = jnp.max(jnp.linalg.norm(normals, axis=1))
    safe_normal_scale = jnp.where(
        jnp.logical_and(jnp.isfinite(normal_scale), normal_scale > 0.0),
        normal_scale,
        1.0,
    )
    scaled_centered_normals = centered_normals / safe_normal_scale
    scaled_covariance = (scaled_centered_normals.T * masses) @ scaled_centered_normals
    scaled_covariance = 0.5 * (scaled_covariance + scaled_covariance.T)

    base_score_dot = (
        logits_dot + base_masses_dot / base_masses + jnp.sum(normals_dot * tilt, axis=1)
    )
    right_hand_side = jnp.sum(
        masses[:, None] * scaled_centered_normals * base_score_dot[:, None],
        axis=0,
    ) + jnp.sum(masses[:, None] * normals_dot / safe_normal_scale, axis=0)
    scaled_tilt_dot = -jnp.linalg.solve(scaled_covariance, right_hand_side)
    tilt_dot = scaled_tilt_dot / safe_normal_scale
    score_dot = base_score_dot + normals @ tilt_dot
    centered_score_dot = score_dot - jnp.sum(masses * score_dot)
    masses_dot = masses * centered_score_dot
    valid = jnp.logical_and(diagnostics.converged, ~diagnostics.ill_conditioned)
    invalid_masses = jnp.full_like(masses, jnp.nan)
    return (
        jnp.where(valid, masses, invalid_masses),
        jnp.where(valid, masses_dot, invalid_masses),
    )

