"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from .damit_model import gaussian_image_grid
from .gaussian_bhs_inversion import GaussianBHSObjectiveConfig, GaussianBHSShapeConfig, area_normal_table, evaluate_gaussian_bhs_parameters, make_gaussian_bhs_objective
from .inversion import InversionTarget, run_multistart_adam


@dataclass(frozen=True)
class BHSOptimizerConfig:
    """Deterministic multistart Adam and optional L-BFGS configuration."""

    starts: int = 3
    steps: int = 300
    learning_rate: float = 0.03
    initial_scale: float = 0.15
    lbfgs_iterations: int = 100
    seed: int = 20260820

    def __post_init__(self) -> None:
        for name in ("starts", "steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.lbfgs_iterations, bool)
            or int(self.lbfgs_iterations) != self.lbfgs_iterations
            or self.lbfgs_iterations < 0
        ):
            raise ValueError("lbfgs_iterations must be a non-negative integer")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.initial_scale) or self.initial_scale < 0.0:
            raise ValueError("initial_scale must be finite and non-negative")



@dataclass(frozen=True)
class BHSPhase1Result:
    """Numerical result and diagnostics from one Phase 1 fit."""

    harmonic_coefficients: np.ndarray
    masses: np.ndarray
    normals: np.ndarray
    predicted: np.ndarray
    curve_rmses: np.ndarray
    curve_corrs: np.ndarray
    objective_terms: dict[str, float]
    diagnostics: dict[str, Any]
    optimizer_diagnostics: dict[str, Any]
    adam_history: tuple[dict[str, float | int], ...]



def harmonic_coefficient_metadata(
    config: GaussianBHSShapeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return degree, order, and real-component labels in active basis order."""

    degrees: list[int] = []
    orders: list[int] = []
    components: list[str] = []
    for order in range(config.m_max + 1):
        for degree in range(order, config.l_max + 1):
            if degree < config.min_degree:
                continue
            if order == 0:
                degrees.append(degree)
                orders.append(order)
                components.append("zonal")
            else:
                for component in ("cos", "sin"):
                    degrees.append(degree)
                    orders.append(order)
                    components.append(component)
    if len(degrees) != config.n_parameters:
        raise RuntimeError("harmonic metadata does not match the active basis")
    return (
        np.asarray(degrees, dtype=np.int64),
        np.asarray(orders, dtype=np.int64),
        np.asarray(components, dtype="<U5"),
    )



def fit_bhs_surface_measure(
    target: InversionTarget,
    shape_config: GaussianBHSShapeConfig,
    objective_config: GaussianBHSObjectiveConfig,
    optimizer_config: BHSOptimizerConfig,
) -> BHSPhase1Result:
    """Fit one positive closed surface measure to an inversion target."""

    objective = make_gaussian_bhs_objective(target, shape_config, objective_config)
    rng = np.random.default_rng(optimizer_config.seed)
    starts = rng.normal(
        scale=optimizer_config.initial_scale / np.sqrt(shape_config.n_parameters),
        size=(optimizer_config.starts, shape_config.n_parameters),
    )
    # Always include the configured reference measure as a deterministic start.
    starts[0] = 0.0
    adam = run_multistart_adam(
        objective,
        starts,
        n_steps=optimizer_config.steps,
        learning_rate=optimizer_config.learning_rate,
        log_every=max(1, min(25, optimizer_config.steps)),
    )
    coefficients = np.asarray(adam.best_parameters, dtype=np.float64)
    polish: dict[str, Any] = {
        "used": False,
        "accepted": False,
        "initial_loss": float(adam.best_loss),
    }
    if optimizer_config.lbfgs_iterations > 0:
        coefficients, polish = _polish_lbfgs(
            objective,
            coefficients,
            max_iterations=optimizer_config.lbfgs_iterations,
        )

    evaluation = evaluate_gaussian_bhs_parameters(
        coefficients,
        target,
        shape_config,
        objective_config,
    )
    table = area_normal_table(coefficients, shape_config)
    arrays_to_check = (
        coefficients,
        table,
        evaluation["predicted"],
        evaluation["curve_rmses"],
    )
    if not all(np.all(np.isfinite(values)) for values in arrays_to_check):
        raise FloatingPointError("Phase 1 produced non-finite output")
    # Correlation is intentionally allowed to be NaN for constant curves.
    diagnostics = {
        name: value
        for name, value in evaluation.items()
        if name not in {"predicted", "curve_rmses", "curve_corrs", "objective_terms"}
    }
    optimizer_diagnostics = {
        "adam_best_loss": float(adam.best_loss),
        "adam_final_loss": float(adam.final_loss),
        "polish": polish,
        "selected_objective": float(evaluation["objective_terms"]["total"]),
    }
    return BHSPhase1Result(
        harmonic_coefficients=coefficients,
        masses=table[:, 0],
        normals=table[:, 1:4],
        predicted=np.asarray(evaluation["predicted"], dtype=np.float64),
        curve_rmses=np.asarray(evaluation["curve_rmses"], dtype=np.float64),
        curve_corrs=np.asarray(evaluation["curve_corrs"], dtype=np.float64),
        objective_terms={
            name: float(value) for name, value in evaluation["objective_terms"].items()
        },
        diagnostics=diagnostics,
        optimizer_diagnostics=optimizer_diagnostics,
        adam_history=tuple(dict(row) for row in adam.history),
    )



def quadrature_masses(config: GaussianBHSShapeConfig) -> np.ndarray:
    """Return normalized spherical quadrature masses for artifact metadata."""

    weights = np.asarray(gaussian_image_grid(config.n_rows).base_areas, dtype=np.float64)
    return weights / np.sum(weights)



def _polish_lbfgs(
    objective: Any,
    parameters: np.ndarray,
    *,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def fun(values: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(jnp.asarray(values))
        return float(value), np.asarray(gradient, dtype=np.float64)

    initial_loss = float(objective(jnp.asarray(parameters)))
    result = minimize(
        fun,
        parameters,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iterations, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    polished = np.asarray(result.x, dtype=np.float64)
    accepted = bool(
        np.isfinite(result.fun)
        and np.all(np.isfinite(polished))
        and float(result.fun) <= initial_loss
    )
    payload = {
        "used": True,
        "accepted": accepted,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "initial_loss": initial_loss,
        "final_loss": float(result.fun),
    }
    return (polished if accepted else np.asarray(parameters, dtype=np.float64)), payload

