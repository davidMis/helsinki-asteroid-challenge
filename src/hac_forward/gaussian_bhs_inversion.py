"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
from .constants import LIGHT_DIRECTION
from .damit_model import gaussian_image_grid, harmonic_basis, harmonic_degrees
from .gaussian_bhs import close_masses_jax, close_masses_np
from .inversion import InversionTarget, apply_prediction_calibration_jax, weighted_curve_misfit_jax
from .jax_model import rotate_lab_direction_to_body
from .metrics import correlation
from .spherical_wasserstein import CHORDAL_GROUND_COST, WASSERSTEIN_GROUND_COSTS, sinkhorn_divergence_jax, sinkhorn_divergence_marginal_error_jax, squared_ground_cost_matrix


@dataclass(frozen=True)
class GaussianBHSShapeConfig:
    """Finite Bayes--Hilbert chart and convex scattering configuration."""

    n_rows: int = 8
    l_max: int = 6
    m_max: int = 6
    min_degree: int = 2
    lambert_coefficient: float = 100.0
    lommel_seeliger_coefficient: float = 1.0
    phase_amplitude: float = 0.0
    phase_width: float = 0.1
    phase_slope: float = 0.0
    epsilon: float = 1.0e-9
    reference_spheroid_radius: float = 1.0
    reference_spheroid_gamma: float = 0.0

    def __post_init__(self) -> None:
        if self.n_rows < 1:
            raise ValueError("n_rows must be positive")
        if self.l_max < 0 or self.m_max < 0 or self.m_max > self.l_max:
            raise ValueError("require 0 <= m_max <= l_max")
        if self.min_degree < 0 or self.min_degree > self.l_max:
            raise ValueError("min_degree must lie between zero and l_max")
        if self.lambert_coefficient < 0.0:
            raise ValueError("lambert_coefficient must be non-negative")
        if self.lommel_seeliger_coefficient < 0.0:
            raise ValueError("lommel_seeliger_coefficient must be non-negative")
        if self.phase_width <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("phase_width and epsilon must be positive")
        if not np.isfinite(self.reference_spheroid_radius) or self.reference_spheroid_radius <= 0.0:
            raise ValueError("reference_spheroid_radius must be finite and positive")
        if (
            not np.isfinite(self.reference_spheroid_gamma)
            or not 0.0 <= self.reference_spheroid_gamma <= 1.0
        ):
            raise ValueError("reference_spheroid_gamma must lie in [0, 1]")
        if self.n_parameters < 1:
            raise ValueError("the requested harmonic chart has no active columns")

    @property
    def n_facets(self) -> int:
        return int(8 * self.n_rows * self.n_rows)

    @property
    def n_parameters(self) -> int:
        return int(np.count_nonzero(self.active_degree_mask))

    @property
    def active_degree_mask(self) -> np.ndarray:
        return harmonic_degrees(self.l_max, self.m_max) >= self.min_degree

    @property
    def active_degrees(self) -> np.ndarray:
        return harmonic_degrees(self.l_max, self.m_max)[self.active_degree_mask]



@dataclass(frozen=True)
class GaussianBHSObjectiveConfig:
    """Weights for data fidelity and surface-measure regularization."""

    raw_weight: float = 1.0
    lightcurve_bhs_weight: float = 0.05
    surface_bhs_weight: float = 1.0e-4
    surface_wasserstein_weight: float = 0.0
    surface_moment_weight: float = 0.0
    sobolev_weight: float = 1.0e-4
    sobolev_order: float = 2.0
    lightcurve_epsilon: float = 1.0e-3
    surface_wasserstein_ground_cost: str = CHORDAL_GROUND_COST
    surface_wasserstein_epsilon: float = 0.05
    surface_wasserstein_iterations: int = 200

    def __post_init__(self) -> None:
        for name in (
            "raw_weight",
            "lightcurve_bhs_weight",
            "surface_bhs_weight",
            "surface_wasserstein_weight",
            "surface_moment_weight",
            "sobolev_weight",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.sobolev_order) or self.sobolev_order < 0.0:
            raise ValueError("sobolev_order must be finite and non-negative")
        if not np.isfinite(self.lightcurve_epsilon) or self.lightcurve_epsilon <= 0.0:
            raise ValueError("lightcurve_epsilon must be finite and positive")
        if self.surface_wasserstein_ground_cost not in WASSERSTEIN_GROUND_COSTS:
            raise ValueError(
                f"surface_wasserstein_ground_cost must be one of {WASSERSTEIN_GROUND_COSTS}"
            )
        if (
            not np.isfinite(self.surface_wasserstein_epsilon)
            or self.surface_wasserstein_epsilon <= 0.0
        ):
            raise ValueError("surface_wasserstein_epsilon must be finite and positive")
        if (
            isinstance(self.surface_wasserstein_iterations, bool)
            or int(self.surface_wasserstein_iterations) != self.surface_wasserstein_iterations
            or self.surface_wasserstein_iterations < 1
        ):
            raise ValueError("surface_wasserstein_iterations must be a positive integer")



@lru_cache(maxsize=64)
def active_harmonic_basis(
    n_rows: int,
    l_max: int,
    m_max: int,
    min_degree: int,
) -> np.ndarray:
    """Return centered, unit-BHS-norm real-harmonic columns.

    The legacy DAMIT basis is intentionally unnormalized and its high-degree
    columns differ in scale by many orders of magnitude.  Normalizing with the
    spherical quadrature reference makes coefficient priors and optimizer step
    sizes meaningful without changing the represented span.
    """
    degrees = harmonic_degrees(l_max, m_max)
    basis = harmonic_basis(n_rows, l_max, m_max)[:, degrees >= min_degree]
    base = gaussian_image_grid(n_rows).base_areas
    reference = base / np.sum(base)
    basis = basis - reference @ basis
    norms = np.sqrt(np.sum(reference[:, None] * basis**2, axis=0))
    if np.any(norms <= 1.0e-12):
        raise ValueError("active harmonic basis contains a degenerate column")
    return (basis / norms).astype(np.float64, copy=False)



@lru_cache(maxsize=64)
def spheroid_reference_weights(
    n_rows: int,
    equatorial_radius: float,
    gamma: float,
) -> np.ndarray:
    """Return quadrature weights for a clr-interpolated spheroid measure.

    The full reference is the surface-area measure of the axis-aligned
    ellipsoid with semi-axes ``(r, r, 1)``.  Its density with respect to
    spherical area is proportional to

    ``r**4 / (r**2 * (nx**2 + ny**2) + nz**2)**2``.

    Raising that density to ``gamma`` gives the Bayes--Hilbert (clr-linear)
    interpolation from the uniform sphere at zero to the full spheroid at
    one.  The returned weights need not sum to one because the closure chart
    is invariant to their common scale.  ``gamma=0`` deliberately returns the
    legacy spherical quadrature weights without another numerical transform.
    """
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    radius = float(equatorial_radius)
    strength = float(gamma)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("equatorial_radius must be finite and positive")
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")

    grid = gaussian_image_grid(n_rows)
    if strength == 0.0 or radius == 1.0:
        return grid.base_areas

    normals = grid.normals
    xy_squared = normals[:, 0] ** 2 + normals[:, 1] ** 2
    z_squared = normals[:, 2] ** 2
    log_radius = np.log(radius)
    with np.errstate(divide="ignore"):
        log_denominator = np.logaddexp(
            2.0 * log_radius + np.log(xy_squared),
            np.log(z_squared),
        )
    log_density = strength * (4.0 * log_radius - 2.0 * log_denominator)
    log_density -= float(np.max(log_density))
    log_density = np.maximum(log_density, np.log(np.finfo(np.float64).tiny))
    return grid.base_areas * np.exp(log_density)



def surface_reference_masses(config: GaussianBHSShapeConfig) -> np.ndarray:
    """Return the unit-mass reference used by the chart and surface prior."""
    weights = spheroid_reference_weights(
        config.n_rows,
        config.reference_spheroid_radius,
        config.reference_spheroid_gamma,
    )
    return weights / np.sum(weights)



def log_density_jax(
    parameters: jnp.ndarray,
    config: GaussianBHSShapeConfig,
) -> jnp.ndarray:
    """Evaluate the unconstrained log-density on the normal grid."""
    params = jnp.ravel(jnp.asarray(parameters))
    if params.shape[0] != config.n_parameters:
        raise ValueError(f"expected {config.n_parameters} parameters, got {params.shape[0]}")
    basis = jnp.asarray(
        active_harmonic_basis(
            config.n_rows,
            config.l_max,
            config.m_max,
            config.min_degree,
        ),
        dtype=params.dtype,
    )
    return basis @ params



def closed_gaussian_masses_jax(
    parameters: jnp.ndarray,
    config: GaussianBHSShapeConfig,
) -> jnp.ndarray:
    """Return positive, unit-mass facet weights with exact vector closure."""
    logits = log_density_jax(parameters, config)
    grid = gaussian_image_grid(config.n_rows)
    reference = spheroid_reference_weights(
        config.n_rows,
        config.reference_spheroid_radius,
        config.reference_spheroid_gamma,
    )
    return close_masses_jax(
        logits,
        jnp.asarray(reference, dtype=logits.dtype),
        jnp.asarray(grid.normals, dtype=logits.dtype),
    )



def closed_gaussian_masses_np(
    parameters: np.ndarray,
    config: GaussianBHSShapeConfig,
) -> np.ndarray:
    """NumPy diagnostic/export version of :func:`closed_gaussian_masses_jax`."""
    params = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if params.shape[0] != config.n_parameters:
        raise ValueError(f"expected {config.n_parameters} parameters, got {params.shape[0]}")
    logits = (
        active_harmonic_basis(
            config.n_rows,
            config.l_max,
            config.m_max,
            config.min_degree,
        )
        @ params
    )
    grid = gaussian_image_grid(config.n_rows)
    reference = spheroid_reference_weights(
        config.n_rows,
        config.reference_spheroid_radius,
        config.reference_spheroid_gamma,
    )
    return np.asarray(close_masses_np(logits, reference, grid.normals))



def area_normal_table(
    parameters: np.ndarray,
    config: GaussianBHSShapeConfig,
) -> np.ndarray:
    """Return the exactly closed ``(area, nx, ny, nz)`` export table."""
    masses = closed_gaussian_masses_np(parameters, config)
    normals = gaussian_image_grid(config.n_rows).normals
    return np.column_stack([masses, normals]).astype(np.float64, copy=False)



def photometric_kernel_for_normals_jax(
    config: GaussianBHSShapeConfig,
    normals: jnp.ndarray,
    phases: jnp.ndarray,
    light_direction: jnp.ndarray,
    view_directions: jnp.ndarray,
) -> jnp.ndarray:
    """Return the linear kernel for an arbitrary array of unit normals.

    This is the observation operator needed after closure-preserving atomic
    coarsening moves Gaussian-image atoms off the original quadrature grid.
    It deliberately shares every scattering and phase convention with
    :func:`photometric_kernel_jax`.
    """
    phases = jnp.asarray(phases)
    dtype = phases.dtype
    normals = jnp.asarray(normals, dtype=dtype)
    if normals.ndim != 2 or normals.shape[1] != 3 or normals.shape[0] < 1:
        raise ValueError("normals must have shape (n, 3) with n positive")
    light = _normalize(jnp.asarray(light_direction, dtype=dtype))
    views = _normalize(jnp.asarray(view_directions, dtype=dtype))
    light_body = rotate_lab_direction_to_body(light, phases)

    def one_view(view: jnp.ndarray) -> jnp.ndarray:
        view_body = rotate_lab_direction_to_body(view, phases)
        raw_mu0 = light_body @ normals.T
        raw_mu = view_body @ normals.T
        visible = (raw_mu0 > 0.0) & (raw_mu > 0.0)
        mu0 = jnp.where(visible, raw_mu0, 0.0)
        mu = jnp.where(visible, raw_mu, 0.0)
        kernel = (
            mu0
            * mu
            * (
                float(config.lambert_coefficient)
                + float(config.lommel_seeliger_coefficient) / (mu0 + mu + float(config.epsilon))
            )
        )
        cos_alpha = jnp.clip(jnp.sum(view_body * light_body, axis=1), -1.0, 1.0)
        alpha = jnp.arccos(cos_alpha)
        phase_scale = (
            1.0
            + float(config.phase_amplitude) * jnp.exp(-alpha / float(config.phase_width))
            + float(config.phase_slope) * alpha
        )
        return phase_scale[:, None] * kernel

    return jax.vmap(one_view)(views)



def predict_gaussian_measure_lightcurves_jax(
    masses: jnp.ndarray,
    normals: jnp.ndarray,
    config: GaussianBHSShapeConfig,
    phases: jnp.ndarray,
    light_direction: jnp.ndarray,
    view_directions: jnp.ndarray,
) -> jnp.ndarray:
    """Predict normalized curves from any discrete Gaussian image."""
    values = jnp.ravel(jnp.asarray(masses))
    directions = jnp.asarray(normals, dtype=values.dtype)
    if directions.shape != (values.shape[0], 3):
        raise ValueError("masses and normals have incompatible shapes")
    kernel = photometric_kernel_for_normals_jax(
        config,
        directions,
        phases,
        light_direction,
        view_directions,
    )
    brightness = jnp.einsum("vti,i->vt", kernel, values)
    curve_means = jnp.mean(brightness, axis=1, keepdims=True)
    # Unit-mass BHS coordinates remove global area scale.  Avoid an additive
    # normalization epsilon, which would quietly reintroduce scale dependence.
    # An exactly dark acquisition is represented by the zero curve.
    safe_means = jnp.where(curve_means > 0.0, curve_means, 1.0)
    return jnp.where(curve_means > 0.0, brightness / safe_means, 0.0)



def predict_selected_gaussian_bhs_table_jax(
    parameters: jnp.ndarray,
    config: GaussianBHSShapeConfig,
    target: InversionTarget,
) -> jnp.ndarray:
    """Predict a selected HAC table with shape ``(time, angle, view)``."""
    masses = closed_gaussian_masses_jax(parameters, config)
    normals = jnp.asarray(
        gaussian_image_grid(config.n_rows).normals,
        dtype=jnp.asarray(parameters).dtype,
    )
    return predict_selected_gaussian_measure_table_jax(
        masses,
        normals,
        config,
        target,
    )



def predict_selected_gaussian_measure_table_jax(
    masses: jnp.ndarray,
    normals: jnp.ndarray,
    config: GaussianBHSShapeConfig,
    target: InversionTarget,
) -> jnp.ndarray:
    """Predict a selected HAC table from any discrete Gaussian image."""
    values = jnp.ravel(jnp.asarray(masses))
    curves = predict_gaussian_measure_lightcurves_jax(
        values,
        normals,
        config,
        jnp.asarray(target.phases),
        jnp.asarray(LIGHT_DIRECTION, dtype=values.dtype),
        jnp.asarray(target.view_directions),
    )
    predicted = jnp.moveaxis(
        curves.reshape(
            len(target.angles_degrees),
            len(target.view_labels),
            target.observed.shape[0],
        ),
        2,
        0,
    )
    return apply_prediction_calibration_jax(
        predicted,
        target.angles_degrees,
        target.calibration,
    )



def surface_bhs_coordinates_jax(
    masses: jnp.ndarray,
    config: GaussianBHSShapeConfig,
) -> jnp.ndarray:
    """Return clr coordinates relative to the configured surface reference."""
    values = jnp.asarray(masses)
    base = jnp.asarray(gaussian_image_grid(config.n_rows).base_areas, dtype=values.dtype)
    quadrature = base / jnp.sum(base)
    reference = jnp.asarray(surface_reference_masses(config), dtype=values.dtype)
    log_ratio = jnp.log(values) - jnp.log(reference)
    return log_ratio - jnp.sum(quadrature * log_ratio)



def surface_bhs_energy_jax(
    masses: jnp.ndarray,
    config: GaussianBHSShapeConfig,
) -> jnp.ndarray:
    """Squared BHS distance to the configured reference surface measure."""
    values = jnp.asarray(masses)
    base = jnp.asarray(gaussian_image_grid(config.n_rows).base_areas, dtype=values.dtype)
    quadrature = base / jnp.sum(base)
    clr = surface_bhs_coordinates_jax(values, config)
    return 0.5 * jnp.sum(quadrature * clr**2)



@lru_cache(maxsize=32)
def surface_wasserstein_cost_matrix(
    n_rows: int,
    ground_cost: str = CHORDAL_GROUND_COST,
) -> np.ndarray:
    """Return the fixed-grid squared chordal or intrinsic transport cost."""

    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    return squared_ground_cost_matrix(gaussian_image_grid(n_rows).normals, ground_cost)



def surface_wasserstein2_energy_jax(
    masses: jnp.ndarray,
    config: GaussianBHSShapeConfig,
    *,
    ground_cost: str = CHORDAL_GROUND_COST,
    epsilon: float,
    iterations: int,
) -> jnp.ndarray:
    """Return ``0.5`` times the spherical Sinkhorn approximation to ``W_2^2``.

    The balanced unit measures share the fixed DAMIT normal fan.  The default
    squared chordal cost gives an extrinsic spherical ``W_2`` approximation;
    intrinsic squared great-circle cost is an explicit sensitivity option.
    Debiasing makes the energy vanish at the configured spheroid reference.
    ``epsilon`` remains a declared approximation scale rather than a second
    tuned prior weight.
    """

    values = jnp.asarray(masses)
    reference = jnp.asarray(surface_reference_masses(config), dtype=values.dtype)
    cost = jnp.asarray(
        surface_wasserstein_cost_matrix(config.n_rows, ground_cost),
        dtype=values.dtype,
    )
    return 0.5 * sinkhorn_divergence_jax(
        values,
        reference,
        cost,
        epsilon=epsilon,
        iterations=iterations,
    )



def surface_reference_moment_energy_jax(
    masses: jnp.ndarray,
    config: GaussianBHSShapeConfig,
) -> jnp.ndarray:
    """Penalize the observable degree-two moment away from the reference.

    The normalized tensor ``sum_i p_i n_i n_i.T`` captures the broad
    polar-versus-equatorial allocation of Gaussian-image mass.  Matching this
    tensor targets the aspect weak mode without damping the higher spherical
    harmonics that encode boxiness and other resolved silhouette structure.
    """
    values = jnp.asarray(masses)
    normals = jnp.asarray(
        gaussian_image_grid(config.n_rows).normals,
        dtype=values.dtype,
    )
    reference = jnp.asarray(surface_reference_masses(config), dtype=values.dtype)
    moment = jnp.einsum("i,ij,ik->jk", values / jnp.sum(values), normals, normals)
    reference_moment = jnp.einsum("i,ij,ik->jk", reference, normals, normals)
    return 0.5 * jnp.sum((moment - reference_moment) ** 2)



def spectral_sobolev_energy_jax(
    parameters: jnp.ndarray,
    config: GaussianBHSShapeConfig,
    *,
    order: float,
) -> jnp.ndarray:
    """Degree-weighted coefficient energy used as a smoothness penalty."""
    params = jnp.ravel(jnp.asarray(parameters))
    degrees = jnp.asarray(config.active_degrees, dtype=params.dtype)
    weights = (degrees * (degrees + 1.0)) ** float(order)
    weights = weights / jnp.maximum(jnp.mean(weights), 1.0)
    return 0.5 * jnp.mean(weights * params**2)



def lightcurve_bhs_loss_jax(
    predicted: jnp.ndarray,
    observed: jnp.ndarray,
    curve_weights: jnp.ndarray,
    *,
    epsilon: float,
) -> jnp.ndarray:
    """Dense clr loss using a scale-relative positive floor per curve."""
    predicted_clr = _relative_clr(predicted, epsilon=epsilon)
    observed_clr = _relative_clr(observed, epsilon=epsilon)
    residual = predicted_clr - observed_clr
    weights = jnp.asarray(curve_weights, dtype=residual.dtype)
    return 0.5 * jnp.sum(residual**2 * weights[None, :, :]) / (residual.shape[0] * jnp.sum(weights))



def gaussian_bhs_objective_terms_jax(
    parameters: jnp.ndarray,
    target: InversionTarget,
    shape_config: GaussianBHSShapeConfig,
    objective_config: GaussianBHSObjectiveConfig,
) -> dict[str, jnp.ndarray]:
    """Return named MAP objective components for diagnostics and optimization."""
    predicted = predict_selected_gaussian_bhs_table_jax(
        parameters,
        shape_config,
        target,
    )
    observed = jnp.asarray(target.observed, dtype=predicted.dtype)
    curve_weights = jnp.asarray(target.effective_curve_weights, dtype=predicted.dtype)
    raw = weighted_curve_misfit_jax(predicted - observed, curve_weights)
    curve_bhs = lightcurve_bhs_loss_jax(
        predicted,
        observed,
        curve_weights,
        epsilon=objective_config.lightcurve_epsilon,
    )
    masses = closed_gaussian_masses_jax(parameters, shape_config)
    surface_bhs = surface_bhs_energy_jax(masses, shape_config)
    if objective_config.surface_wasserstein_weight > 0.0:
        surface_wasserstein2 = surface_wasserstein2_energy_jax(
            masses,
            shape_config,
            ground_cost=objective_config.surface_wasserstein_ground_cost,
            epsilon=objective_config.surface_wasserstein_epsilon,
            iterations=objective_config.surface_wasserstein_iterations,
        )
    else:
        # The O(N^2) Sinkhorn solve must remain absent from legacy objectives.
        surface_wasserstein2 = jnp.asarray(0.0, dtype=masses.dtype)
    surface_moment = surface_reference_moment_energy_jax(masses, shape_config)
    sobolev = spectral_sobolev_energy_jax(
        parameters,
        shape_config,
        order=objective_config.sobolev_order,
    )
    total = (
        objective_config.raw_weight * raw
        + objective_config.lightcurve_bhs_weight * curve_bhs
        + objective_config.surface_bhs_weight * surface_bhs
        + objective_config.surface_wasserstein_weight * surface_wasserstein2
        + objective_config.surface_moment_weight * surface_moment
        + objective_config.sobolev_weight * sobolev
    )
    return {
        "total": total,
        "raw": raw,
        "lightcurve_bhs": curve_bhs,
        "surface_bhs": surface_bhs,
        "surface_wasserstein2": surface_wasserstein2,
        "surface_moment": surface_moment,
        "sobolev": sobolev,
    }



def make_gaussian_bhs_objective(
    target: InversionTarget,
    shape_config: GaussianBHSShapeConfig,
    objective_config: GaussianBHSObjectiveConfig,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create the scalar closure-charted Bayes--Hilbert MAP objective."""

    def objective(parameters: jnp.ndarray) -> jnp.ndarray:
        return gaussian_bhs_objective_terms_jax(
            parameters,
            target,
            shape_config,
            objective_config,
        )["total"]

    return objective



def evaluate_gaussian_bhs_parameters(
    parameters: np.ndarray,
    target: InversionTarget,
    shape_config: GaussianBHSShapeConfig,
    objective_config: GaussianBHSObjectiveConfig,
) -> dict[str, Any]:
    """Evaluate fit, objective components, closure, and density diagnostics."""
    params = jnp.asarray(parameters)
    predicted = np.asarray(predict_selected_gaussian_bhs_table_jax(params, shape_config, target))
    residual = predicted - target.observed
    curve_rmses = np.sqrt(np.mean(residual**2, axis=0))
    curve_corrs = np.empty(curve_rmses.shape, dtype=np.float64)
    for angle_index in range(curve_rmses.shape[0]):
        for view_index in range(curve_rmses.shape[1]):
            curve_corrs[angle_index, view_index] = correlation(
                target.observed[:, angle_index, view_index],
                predicted[:, angle_index, view_index],
            )
    masses = closed_gaussian_masses_np(np.asarray(parameters), shape_config)
    normals = gaussian_image_grid(shape_config.n_rows).normals
    terms = gaussian_bhs_objective_terms_jax(
        params,
        target,
        shape_config,
        objective_config,
    )
    wasserstein_marginal_error = 0.0
    if objective_config.surface_wasserstein_weight > 0.0:
        reference = jnp.asarray(surface_reference_masses(shape_config), dtype=params.dtype)
        cost = jnp.asarray(
            surface_wasserstein_cost_matrix(
                shape_config.n_rows,
                objective_config.surface_wasserstein_ground_cost,
            ),
            dtype=params.dtype,
        )
        wasserstein_marginal_error = float(
            sinkhorn_divergence_marginal_error_jax(
                jnp.asarray(masses, dtype=params.dtype),
                reference,
                cost,
                epsilon=objective_config.surface_wasserstein_epsilon,
                iterations=objective_config.surface_wasserstein_iterations,
            )
        )
    return {
        "predicted": predicted,
        "mean_rmse": float(np.mean(curve_rmses)),
        "max_rmse": float(np.max(curve_rmses)),
        "median_corr": float(np.nanmedian(curve_corrs)),
        "min_corr": float(np.nanmin(curve_corrs)),
        "curve_rmses": curve_rmses,
        "curve_corrs": curve_corrs,
        "mass_sum": float(np.sum(masses)),
        "closure_vector": np.asarray(masses @ normals),
        "closure_norm": float(np.linalg.norm(masses @ normals)),
        "min_mass": float(np.min(masses)),
        "max_mass": float(np.max(masses)),
        "surface_wasserstein_sinkhorn_marginal_error": wasserstein_marginal_error,
        "objective_terms": {name: float(value) for name, value in terms.items()},
    }



def _relative_clr(curves: jnp.ndarray, *, epsilon: float) -> jnp.ndarray:
    values = jnp.maximum(jnp.asarray(curves), 0.0)
    mean = jnp.mean(values, axis=0, keepdims=True)
    logged = jnp.log(values + float(epsilon) * jnp.maximum(mean, 1.0e-12))
    return logged - jnp.mean(logged, axis=0, keepdims=True)



def _normalize(vectors: jnp.ndarray) -> jnp.ndarray:
    return vectors / jnp.linalg.norm(vectors, axis=-1, keepdims=True)

