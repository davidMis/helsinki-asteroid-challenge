"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np


CHORDAL_GROUND_COST = "chordal"



GEODESIC_GROUND_COST = "geodesic"



WASSERSTEIN_GROUND_COSTS = (CHORDAL_GROUND_COST, GEODESIC_GROUND_COST)



def squared_geodesic_cost_matrix(normals: np.ndarray) -> np.ndarray:
    """Return pairwise squared great-circle distances in radians squared."""

    directions = np.asarray(normals, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3 or directions.shape[0] < 1:
        raise ValueError("normals must have shape (n, 3) with n positive")
    if not np.all(np.isfinite(directions)):
        raise ValueError("normals must be finite")
    lengths = np.linalg.norm(directions, axis=1)
    if not np.allclose(lengths, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("normals must be unit length")
    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    cost = np.arccos(cosine) ** 2
    # Exact zeros on the diagonal make identity and symmetry tests independent
    # of tiny roundoff in the supplied unit normals.
    np.fill_diagonal(cost, 0.0)
    return (0.5 * (cost + cost.T)).astype(np.float64, copy=False)



def squared_chordal_cost_matrix(normals: np.ndarray) -> np.ndarray:
    """Return pairwise squared Euclidean distances between unit normals."""

    directions = _validated_unit_normals(normals)
    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    cost = np.maximum(2.0 - 2.0 * cosine, 0.0)
    np.fill_diagonal(cost, 0.0)
    return (0.5 * (cost + cost.T)).astype(np.float64, copy=False)



def squared_ground_cost_matrix(normals: np.ndarray, ground_cost: str) -> np.ndarray:
    """Return the declared chordal or intrinsic spherical squared cost."""

    if ground_cost == CHORDAL_GROUND_COST:
        return squared_chordal_cost_matrix(normals)
    if ground_cost == GEODESIC_GROUND_COST:
        return squared_geodesic_cost_matrix(normals)
    raise ValueError(f"ground_cost must be one of {WASSERSTEIN_GROUND_COSTS}")



def sinkhorn_divergence_jax(
    masses: jnp.ndarray,
    reference: jnp.ndarray,
    cost: jnp.ndarray,
    *,
    epsilon: float,
    iterations: int,
) -> jnp.ndarray:
    """Return the debiased entropic approximation to discrete ``W_2^2``.

    The supports must be the same and ``cost[i, j]`` is their squared ground
    distance.  The fixed matrix-scaling iterations are differentiated in full,
    so the reported gradient is the exact derivative of the implemented
    finite-iteration divergence and approaches the envelope gradient as the
    marginal residuals converge.
    """

    values, target, ground_cost = _validated_jax_inputs(masses, reference, cost)
    regularization = float(epsilon)
    n_iterations = _validate_sinkhorn_controls(regularization, iterations)
    cross = _entropic_transport_cost_jax(
        values,
        target,
        ground_cost,
        epsilon=regularization,
        iterations=n_iterations,
    )
    self_values = _entropic_transport_cost_jax(
        values,
        values,
        ground_cost,
        epsilon=regularization,
        iterations=n_iterations,
    )
    self_target = _entropic_transport_cost_jax(
        target,
        target,
        ground_cost,
        epsilon=regularization,
        iterations=n_iterations,
    )
    return 0.5 * ((cross - self_values) + (cross - self_target))



def sinkhorn_marginal_error_jax(
    masses: jnp.ndarray,
    reference: jnp.ndarray,
    cost: jnp.ndarray,
    *,
    epsilon: float,
    iterations: int,
) -> jnp.ndarray:
    """Return the maximum absolute marginal residual of the cross solve."""

    values, target, ground_cost = _validated_jax_inputs(masses, reference, cost)
    regularization = float(epsilon)
    n_iterations = _validate_sinkhorn_controls(regularization, iterations)
    first_scaling, second_scaling, kernel = _sinkhorn_scalings_jax(
        values,
        target,
        ground_cost,
        epsilon=regularization,
        iterations=n_iterations,
    )
    plan = (values * first_scaling)[:, None] * kernel * (target * second_scaling)[None, :]
    row_error = jnp.max(jnp.abs(jnp.sum(plan, axis=1) - values))
    column_error = jnp.max(jnp.abs(jnp.sum(plan, axis=0) - target))
    return jnp.maximum(row_error, column_error)



def sinkhorn_divergence_marginal_error_jax(
    masses: jnp.ndarray,
    reference: jnp.ndarray,
    cost: jnp.ndarray,
    *,
    epsilon: float,
    iterations: int,
) -> jnp.ndarray:
    """Return the worst marginal residual over all three divergence solves."""

    cross = sinkhorn_marginal_error_jax(
        masses,
        reference,
        cost,
        epsilon=epsilon,
        iterations=iterations,
    )
    self_masses = sinkhorn_marginal_error_jax(
        masses,
        masses,
        cost,
        epsilon=epsilon,
        iterations=iterations,
    )
    self_reference = sinkhorn_marginal_error_jax(
        reference,
        reference,
        cost,
        epsilon=epsilon,
        iterations=iterations,
    )
    return jnp.maximum(cross, jnp.maximum(self_masses, self_reference))



def _entropic_transport_cost_jax(
    first_marginal: jnp.ndarray,
    second_marginal: jnp.ndarray,
    cost: jnp.ndarray,
    *,
    epsilon: float,
    iterations: int,
) -> jnp.ndarray:
    first_scaling, second_scaling, kernel = _sinkhorn_scalings_jax(
        first_marginal,
        second_marginal,
        cost,
        epsilon=epsilon,
        iterations=iterations,
    )
    plan = (
        (first_marginal * first_scaling)[:, None]
        * kernel
        * (second_marginal * second_scaling)[None, :]
    )
    log_density_ratio = (
        jnp.log(first_scaling)[:, None] + jnp.log(second_scaling)[None, :] + jnp.log(kernel)
    )
    return jnp.sum(plan * (cost + epsilon * log_density_ratio))



def _sinkhorn_scalings_jax(
    first_marginal: jnp.ndarray,
    second_marginal: jnp.ndarray,
    cost: jnp.ndarray,
    *,
    epsilon: float,
    iterations: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dtype = first_marginal.dtype
    tiny = jnp.finfo(dtype).tiny
    log_tiny = jnp.log(tiny)
    kernel = jnp.exp(jnp.maximum(-cost / epsilon, log_tiny))
    first = jnp.ones_like(first_marginal)
    second = jnp.ones_like(second_marginal)

    def update(
        _iteration: int,
        state: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        _current_first, current_second = state
        next_first = 1.0 / jnp.maximum(
            kernel @ (second_marginal * current_second),
            tiny,
        )
        next_second = 1.0 / jnp.maximum(
            kernel.T @ (first_marginal * next_first),
            tiny,
        )
        # Fix the multiplicative gauge to keep both factors away from the
        # floating-point extremes during long solves.
        scale = jnp.exp(jnp.vdot(first_marginal, jnp.log(next_first)))
        return next_first / scale, next_second * scale

    first, second = jax.lax.fori_loop(0, iterations, update, (first, second))
    return first, second, kernel



def _validated_jax_inputs(
    masses: jnp.ndarray,
    reference: jnp.ndarray,
    cost: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    values = jnp.ravel(jnp.asarray(masses))
    target = jnp.ravel(jnp.asarray(reference, dtype=values.dtype))
    ground_cost = jnp.asarray(cost, dtype=values.dtype)
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError("masses must be a nonempty vector")
    if target.shape != values.shape:
        raise ValueError("masses and reference must have the same shape")
    if ground_cost.shape != (values.shape[0], values.shape[0]):
        raise ValueError("cost must have shape (n, n)")
    tiny = jnp.finfo(values.dtype).tiny
    values = jnp.maximum(values, tiny)
    target = jnp.maximum(target, tiny)
    return values / jnp.sum(values), target / jnp.sum(target), ground_cost



def _validate_sinkhorn_controls(epsilon: float, iterations: int) -> int:
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    return int(iterations)



def _validated_unit_normals(normals: np.ndarray) -> np.ndarray:
    directions = np.asarray(normals, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3 or directions.shape[0] < 1:
        raise ValueError("normals must have shape (n, 3) with n positive")
    if not np.all(np.isfinite(directions)):
        raise ValueError("normals must be finite")
    lengths = np.linalg.norm(directions, axis=1)
    if not np.allclose(lengths, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("normals must be unit length")
    return directions

