"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from .atomic_gaussian_image import closure_preserving_atomic_coarsen


@dataclass(frozen=True)
class SemidiscreteOTQuantizerConfig:
    """Numerical controls for deterministic weighted spherical Lloyd descent."""

    seed: int = 0
    n_starts: int = 8
    max_iterations: int = 200
    objective_tolerance: float = 1.0e-13
    center_tolerance_radians: float = 1.0e-12
    closure_tolerance: float = 1.0e-10
    unit_normal_tolerance: float = 1.0e-10
    minimum_resultant: float = float(np.finfo(np.float64).tiny)
    transport_feasibility_tolerance: float = 1.0e-10
    transport_mass_scale: float = 1.0e8
    transport_marginal_residual_tolerance: float = 5.0e-10
    restart_wasserstein2_tie_tolerance: float = 1.0e-12
    normalized_output_mass_floor: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("n_starts", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "objective_tolerance",
            "center_tolerance_radians",
            "closure_tolerance",
            "unit_normal_tolerance",
            "minimum_resultant",
            "transport_feasibility_tolerance",
            "transport_mass_scale",
            "transport_marginal_residual_tolerance",
            "restart_wasserstein2_tie_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.transport_feasibility_tolerance < 1.0e-10:
            raise ValueError("transport_feasibility_tolerance must be at least 1e-10")
        floor = self.normalized_output_mass_floor
        if floor is not None:
            if isinstance(floor, bool):
                raise ValueError("normalized_output_mass_floor must be finite and positive")
            try:
                floor_value = float(floor)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "normalized_output_mass_floor must be finite and positive"
                ) from exc
            if not np.isfinite(floor_value) or floor_value <= 0.0:
                raise ValueError("normalized_output_mass_floor must be finite and positive")
            if floor_value > 0.25:
                raise ValueError(
                    "normalized_output_mass_floor cannot exceed 0.25 for at least four atoms"
                )
            object.__setattr__(self, "normalized_output_mass_floor", floor_value)



@dataclass(frozen=True)
class SemidiscreteOTIterationDiagnostics:
    """One Lloyd assignment-and-centroid update."""

    iteration: int
    half_squared_chordal_energy: float
    objective_improvement: float
    maximum_center_shift_degrees: float
    assignment_changes: int | None
    empty_cluster_repairs: int



@dataclass(frozen=True)
class SemidiscreteOTStartDiagnostics:
    """Complete convergence record for one deterministic initialization."""

    start_index: int
    initialization_kind: str
    initial_seed_canonical_indices: tuple[int, ...]
    initial_half_squared_chordal_energy: float
    final_half_squared_chordal_energy: float | None
    converged: bool
    iteration_count: int
    total_empty_cluster_repairs: int
    valid: bool
    rejection_reason: str | None
    closed_output_wasserstein2_squared: float | None
    closed_output_transport_maximum_marginal_residual: float | None
    closed_output_transport_accepted: bool | None
    cell_mass_to_output_mass_l1: float | None
    history: tuple[SemidiscreteOTIterationDiagnostics, ...]



@dataclass(frozen=True)
class MassFloorMergeDiagnostics:
    """One deterministic low-resultant merge in the optional mass-safe reduction."""

    step: int
    atom_count_before: int
    atom_count_after: int
    low_atom_canonical_index_before: int
    partner_canonical_index_before: int
    low_atom_normal_before: tuple[float, float, float]
    partner_normal_before: tuple[float, float, float]
    low_atom_normalized_mass_before: float
    partner_normalized_mass_before: float
    merged_atom_normalized_mass_after: float
    minimum_normalized_mass_after: float
    resultant_total_before: float
    resultant_total_after: float
    half_squared_chordal_energy_increase: float
    squared_chordal_partition_cost_increase: float



@dataclass(frozen=True)
class SemidiscreteOTQuantizationDiagnostics:
    """Auditable OT, convergence, closure, and distortion diagnostics."""

    method: str
    ground_cost: str
    initialization: str
    seed: int
    input_atom_count: int
    requested_output_atom_count: int
    output_atom_count: int
    effective_output_atom_count: int
    configured_start_count: int
    effective_start_count: int
    selected_start_index: int
    selected_start_converged: bool
    selected_iteration_count: int
    restart_selection_criterion: str
    restart_wasserstein2_tie_tolerance: float
    closed_output_wasserstein2_best_numeric: float
    closed_output_wasserstein2_selected_excess_over_best: float
    closed_output_wasserstein2_runner_up_gap: float | None
    closed_output_transport_feasibility_tolerance: float
    closed_output_transport_mass_scale: float
    closed_output_transport_marginal_residual_tolerance: float
    mass_floor_reduction_enabled: bool
    normalized_output_mass_floor: float | None
    mass_floor_validation_tolerance: float | None
    mass_floor_merge_count: int
    mass_floor_merge_trace: tuple[MassFloorMergeDiagnostics, ...]
    input_closure_norm: float
    output_closure_norm: float
    lloyd_cell_mass_closure_norm: float
    lloyd_cell_masses: tuple[float, ...]
    closure_output_masses: tuple[float, ...]
    lloyd_half_squared_chordal_energy: float
    semidiscrete_partition_wasserstein2_squared: float
    pre_reduction_half_squared_chordal_energy: float
    post_reduction_half_squared_chordal_energy: float
    mass_floor_half_squared_chordal_energy_delta: float
    pre_reduction_squared_chordal_partition_cost: float
    post_reduction_squared_chordal_partition_cost: float
    mass_floor_squared_chordal_partition_cost_delta: float
    closed_output_wasserstein2_squared: float
    closed_output_wasserstein2_energy: float
    closed_output_transport_solver_status: int
    closed_output_transport_solver_message: str
    closed_output_transport_maximum_marginal_residual: float
    pre_reduction_closed_output_wasserstein2_squared: float
    post_reduction_closed_output_wasserstein2_squared: float
    mass_floor_closed_output_wasserstein2_squared_delta: float
    total_resultant_before_normalization: float
    resultant_mass_contraction: float
    pre_reduction_resultant_mass_contraction: float
    post_reduction_resultant_mass_contraction: float
    mass_floor_resultant_mass_contraction_delta: float
    ward_partition_half_squared_chordal_energy: float
    ward_start_final_half_squared_chordal_energy: float
    ward_start_improvement_from_ward_partition: float
    selected_improvement_from_ward_partition: float
    cell_mass_to_output_mass_l1: float
    paired_cluster_mass_l1: float
    weighted_mean_angular_displacement_degrees: float
    weighted_rms_angular_displacement_degrees: float
    mass_99_angular_displacement_degrees: float
    maximum_angular_displacement_degrees: float
    minimum_output_normal_separation_degrees: float
    inverse_simpson_effective_atom_count: float
    minimum_output_mass: float
    maximum_output_mass: float
    starts: tuple[SemidiscreteOTStartDiagnostics, ...]



@dataclass(frozen=True)
class SemidiscreteOTQuantization:
    """One closure-compatible atomic measure selected by semidiscrete OT."""

    masses: np.ndarray
    normals: np.ndarray
    assignments: np.ndarray
    cluster_source_masses: np.ndarray
    diagnostics: SemidiscreteOTQuantizationDiagnostics



@dataclass(frozen=True)
class RectangularWassersteinResult:
    """Exact finite-support transport result for different source/target supports."""

    squared_distance: float
    energy: float
    solver_status: int
    solver_message: str
    maximum_marginal_residual: float



@dataclass(frozen=True)
class _LloydTrial:
    centers: np.ndarray | None
    assignments: np.ndarray | None
    energy: float | None
    diagnostics: SemidiscreteOTStartDiagnostics
    closed_transport: RectangularWassersteinResult | None = None
    cell_mass_to_output_mass_l1: float | None = None
    canonical_output_signature: tuple[float, ...] = ()



@dataclass(frozen=True)
class _ReductionCluster:
    vector: np.ndarray
    source_mass: float
    members: tuple[int, ...]



@dataclass(frozen=True)
class _MassFloorReduction:
    assignments: np.ndarray
    pre_reduction_half_squared_chordal_energy: float
    post_reduction_half_squared_chordal_energy: float
    validation_tolerance: float
    trace: tuple[MassFloorMergeDiagnostics, ...]



def semidiscrete_ot_quantize(
    masses: np.ndarray,
    normals: np.ndarray,
    atom_count: int,
    *,
    config: SemidiscreteOTQuantizerConfig | None = None,
) -> SemidiscreteOTQuantization:
    """Quantize one closed spherical measure to exactly ``atom_count`` atoms.

    The procedure is deterministic and invariant to input-row permutation in
    its emitted masses and normals, apart from unavoidable floating-point ties
    between exactly duplicate source rows.  Every source atom is assigned to
    one nonempty Lloyd cell.  The output has positive masses, unit normals,
    unit total mass, and Minkowski closure to the configured tolerance.
    """

    settings = config or SemidiscreteOTQuantizerConfig()
    values, directions, canonical_to_original = _validate_measure(
        masses,
        normals,
        settings,
    )
    if isinstance(atom_count, bool) or int(atom_count) != atom_count:
        raise ValueError("atom_count must be an integer")
    count = int(atom_count)
    if count < 4 or count > values.size:
        raise ValueError("atom_count must lie between 4 and the input atom count")

    unique_indices, unique_masses = _unique_direction_data(values, directions)
    if count > unique_indices.size:
        raise ValueError("atom_count cannot exceed the number of distinct input normal directions")
    seeded_start_count = min(settings.n_starts - 1, int(unique_indices.size))
    effective_starts = 1 + seeded_start_count
    ward = closure_preserving_atomic_coarsen(
        values,
        directions,
        count,
        closure_tolerance=settings.closure_tolerance,
        unit_normal_tolerance=settings.unit_normal_tolerance,
        minimum_resultant=settings.minimum_resultant,
    )
    ward_partition_energy = _assigned_energy(
        values,
        directions,
        ward.assignments,
        ward.normals,
    )
    trials: list[_LloydTrial] = [
        _run_lloyd(
            values,
            directions,
            ward.normals,
            np.empty(0, dtype=np.int64),
            start_index=0,
            initialization_kind="closure_preserving_resultant_ward",
            config=settings,
        )
    ]
    if seeded_start_count:
        initial_anchor = _seeded_weighted_anchor(unique_masses, settings.seed)
        anchors = _weighted_farthest_sequence(
            directions[unique_indices],
            unique_masses,
            seeded_start_count,
            first_index=initial_anchor,
        )
    else:
        anchors = np.empty(0, dtype=np.int64)
    for offset, unique_anchor in enumerate(anchors, start=1):
        seed_unique_indices = _weighted_farthest_seeds(
            directions[unique_indices],
            unique_masses,
            count,
            first_index=int(unique_anchor),
        )
        seed_indices = unique_indices[seed_unique_indices]
        trials.append(
            _run_lloyd(
                values,
                directions,
                directions[seed_indices],
                seed_indices,
                start_index=offset,
                initialization_kind="seeded_weighted_farthest_first",
                config=settings,
            )
        )

    usable = [trial for trial in trials if trial.diagnostics.valid]
    if not usable:
        reasons = "; ".join(
            f"start {trial.diagnostics.start_index}: {trial.diagnostics.rejection_reason}"
            for trial in trials
        )
        raise RuntimeError(f"all semidiscrete OT starts were invalid: {reasons}")
    converged = [trial for trial in usable if trial.diagnostics.converged]
    if not converged:
        raise RuntimeError(
            "no semidiscrete OT start reached a self-consistent Lloyd partition "
            f"within {settings.max_iterations} iterations"
        )
    converged_indices = {trial.diagnostics.start_index for trial in converged}
    trials = [
        (
            _score_closed_trial(values, directions, trial, settings)
            if trial.diagnostics.start_index in converged_indices
            else trial
        )
        for trial in trials
    ]
    eligible = [
        trial
        for trial in trials
        if trial.diagnostics.converged
        and trial.diagnostics.closed_output_transport_accepted is True
    ]
    if not eligible:
        raise RuntimeError(
            "no converged semidiscrete OT start passed the exact-transport marginal gate"
        )
    numeric_order = sorted(
        eligible,
        key=lambda trial: (
            _closed_wasserstein2(trial),
            trial.diagnostics.start_index,
        ),
    )
    best_numeric = _closed_wasserstein2(numeric_order[0])
    tied = [
        trial
        for trial in eligible
        if _closed_wasserstein2(trial) <= best_numeric + settings.restart_wasserstein2_tie_tolerance
    ]
    # Exact balanced W2 to the emitted closed measure is the primary key.  A
    # fixed absolute band absorbs LP noise; only candidates inside that band
    # are tied by Lloyd Q, closure-mass adjustment, canonical support, and id.
    selected = min(tied, key=_trial_secondary_selection_key)
    selected_wasserstein2 = _closed_wasserstein2(selected)
    runner_up_gap = (
        _closed_wasserstein2(numeric_order[1]) - best_numeric if len(numeric_order) > 1 else None
    )
    if selected.assignments is None or selected.centers is None or selected.energy is None:
        raise RuntimeError("selected semidiscrete OT start has no solution")  # pragma: no cover

    mass_floor_reduction = (
        _reduce_selected_partition_to_mass_floor(values, directions, selected, settings)
        if settings.normalized_output_mass_floor is not None
        else None
    )

    return _build_output(
        values,
        directions,
        canonical_to_original,
        selected,
        tuple(trial.diagnostics for trial in trials),
        configured_starts=settings.n_starts,
        effective_starts=effective_starts,
        ward_partition_energy=ward_partition_energy,
        best_numeric_wasserstein2=best_numeric,
        selected_wasserstein2_excess=selected_wasserstein2 - best_numeric,
        runner_up_wasserstein2_gap=runner_up_gap,
        requested_output_count=count,
        mass_floor_reduction=mass_floor_reduction,
        config=settings,
    )



def _run_lloyd(
    masses: np.ndarray,
    normals: np.ndarray,
    initial_centers: np.ndarray,
    seed_indices: np.ndarray,
    *,
    start_index: int,
    initialization_kind: str,
    config: SemidiscreteOTQuantizerConfig,
) -> _LloydTrial:
    centers = np.array(initial_centers, dtype=np.float64, copy=True)
    initial_energy = _nearest_site_energy(masses, normals, centers)
    previous_energy = initial_energy
    previous_assignments: np.ndarray | None = None
    history: list[SemidiscreteOTIterationDiagnostics] = []
    total_repairs = 0
    converged = False
    assignments: np.ndarray | None = None
    rejection_reason: str | None = None

    for iteration in range(1, config.max_iterations + 1):
        assignments, repairs = _assign_nonempty(masses, normals, centers)
        total_repairs += repairs
        resultants = _cluster_resultants(masses, normals, assignments, centers.shape[0])
        resultant_norms = _row_norms(resultants)
        if np.any(resultant_norms <= config.minimum_resultant):
            rejection_reason = (
                "a Lloyd cell has resultant at or below minimum_resultant "
                f"({float(np.min(resultant_norms)):.3e})"
            )
            assignments = None
            break
        updated_centers = resultants / resultant_norms[:, None]
        energy = _assigned_energy(masses, normals, assignments, updated_centers)
        improvement = previous_energy - energy
        shifts = np.arccos(np.clip(np.einsum("ij,ij->i", centers, updated_centers), -1.0, 1.0))
        assignment_changes = (
            None
            if previous_assignments is None
            else int(np.count_nonzero(assignments != previous_assignments))
        )
        history.append(
            SemidiscreteOTIterationDiagnostics(
                iteration=iteration,
                half_squared_chordal_energy=float(energy),
                objective_improvement=float(improvement),
                maximum_center_shift_degrees=float(np.max(shifts) * 180.0 / np.pi),
                assignment_changes=assignment_changes,
                empty_cluster_repairs=repairs,
            )
        )

        centers = updated_centers
        stable_assignments = assignment_changes == 0
        small_update = (
            abs(improvement) <= config.objective_tolerance * max(1.0, abs(previous_energy))
            and float(np.max(shifts)) <= config.center_tolerance_radians
        )
        previous_energy = energy
        previous_assignments = assignments.copy()
        if stable_assignments or small_update:
            checked_assignments, checked_repairs = _assign_nonempty(
                masses,
                normals,
                centers,
            )
            if np.array_equal(checked_assignments, assignments):
                total_repairs += checked_repairs
                converged = True
                break

    valid = assignments is not None and bool(np.isfinite(previous_energy))
    diagnostics = SemidiscreteOTStartDiagnostics(
        start_index=start_index,
        initialization_kind=initialization_kind,
        initial_seed_canonical_indices=tuple(int(index) for index in seed_indices),
        initial_half_squared_chordal_energy=float(initial_energy),
        final_half_squared_chordal_energy=float(previous_energy) if valid else None,
        converged=converged,
        iteration_count=len(history),
        total_empty_cluster_repairs=total_repairs,
        valid=valid,
        rejection_reason=rejection_reason,
        closed_output_wasserstein2_squared=None,
        closed_output_transport_maximum_marginal_residual=None,
        closed_output_transport_accepted=None,
        cell_mass_to_output_mass_l1=None,
        history=tuple(history),
    )
    return _LloydTrial(
        centers=centers if valid else None,
        assignments=assignments if valid else None,
        energy=float(previous_energy) if valid else None,
        diagnostics=diagnostics,
    )



def _reduce_selected_partition_to_mass_floor(
    source_masses: np.ndarray,
    source_normals: np.ndarray,
    selected: _LloydTrial,
    config: SemidiscreteOTQuantizerConfig,
) -> _MassFloorReduction:
    """Greedily merge low-resultant atoms until the normalized floor holds."""

    floor = config.normalized_output_mass_floor
    if floor is None:  # pragma: no cover - caller guards the opt-in path
        raise RuntimeError("mass-floor reduction requires an enabled floor")
    if selected.assignments is None or selected.energy is None:
        raise RuntimeError("mass-floor reduction requires a valid selected partition")
    requested_count = int(np.max(selected.assignments)) + 1
    initial_resultants = _cluster_resultants(
        source_masses,
        source_normals,
        selected.assignments,
        requested_count,
    )
    clusters: list[_ReductionCluster] = []
    for index in range(requested_count):
        members = tuple(int(value) for value in np.flatnonzero(selected.assignments == index))
        clusters.append(
            _ReductionCluster(
                vector=initial_resultants[index],
                source_mass=float(np.sum(source_masses[np.asarray(members, dtype=np.int64)])),
                members=members,
            )
        )
    clusters = _canonical_reduction_clusters(clusters)
    validation_tolerance = _mass_floor_validation_tolerance(floor)
    trace: list[MassFloorMergeDiagnostics] = []

    while True:
        raw_masses, normals, output_masses, resultant_total = _reduction_cluster_state(
            clusters,
            config.minimum_resultant,
        )
        low_indices = np.flatnonzero(output_masses < floor)
        if low_indices.size == 0:
            break
        if len(clusters) <= 4:
            raise ValueError(
                "normalized_output_mass_floor is infeasible without fewer than four atoms; "
                f"floor={floor:.6g}, minimum_at_four={float(np.min(output_masses)):.6g}"
            )

        candidates: list[tuple[float, int, int, np.ndarray, float]] = []
        for low_index_raw in low_indices:
            low_index = int(low_index_raw)
            for partner_index in range(len(clusters)):
                if partner_index == low_index:
                    continue
                merged_vector = clusters[low_index].vector + clusters[partner_index].vector
                merged_norm = _stable_vector_norm(merged_vector)
                if merged_norm <= config.minimum_resultant:
                    continue
                increase = _resultant_merge_loss(
                    clusters[low_index].vector,
                    clusters[partner_index].vector,
                    merged_norm=merged_norm,
                )
                candidates.append(
                    (
                        increase,
                        low_index,
                        partner_index,
                        merged_vector,
                        merged_norm,
                    )
                )
        if not candidates:
            raise ValueError(
                "normalized_output_mass_floor is infeasible because every admissible "
                "low-resultant merge cancels numerically"
            )
        increase, low_index, partner_index, merged_vector, _merged_norm = min(
            candidates,
            key=lambda candidate: (candidate[0], candidate[1], candidate[2]),
        )
        low_cluster = clusters[low_index]
        partner_cluster = clusters[partner_index]
        merged_members = tuple(sorted(low_cluster.members + partner_cluster.members))
        merged_cluster = _ReductionCluster(
            vector=merged_vector,
            source_mass=low_cluster.source_mass + partner_cluster.source_mass,
            members=merged_members,
        )
        remaining = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {low_index, partner_index}
        ]
        clusters = _canonical_reduction_clusters([*remaining, merged_cluster])
        (
            _next_raw_masses,
            _next_normals,
            next_output_masses,
            next_resultant_total,
        ) = _reduction_cluster_state(clusters, config.minimum_resultant)
        merged_index = next(
            index for index, cluster in enumerate(clusters) if cluster.members == merged_members
        )
        actual_increase = max(0.0, resultant_total - next_resultant_total)
        trace.append(
            MassFloorMergeDiagnostics(
                step=len(trace) + 1,
                atom_count_before=len(clusters) + 1,
                atom_count_after=len(clusters),
                low_atom_canonical_index_before=low_index,
                partner_canonical_index_before=partner_index,
                low_atom_normal_before=(
                    float(normals[low_index, 0]),
                    float(normals[low_index, 1]),
                    float(normals[low_index, 2]),
                ),
                partner_normal_before=(
                    float(normals[partner_index, 0]),
                    float(normals[partner_index, 1]),
                    float(normals[partner_index, 2]),
                ),
                low_atom_normalized_mass_before=float(output_masses[low_index]),
                partner_normalized_mass_before=float(output_masses[partner_index]),
                merged_atom_normalized_mass_after=float(next_output_masses[merged_index]),
                minimum_normalized_mass_after=float(np.min(next_output_masses)),
                resultant_total_before=resultant_total,
                resultant_total_after=next_resultant_total,
                half_squared_chordal_energy_increase=increase,
                squared_chordal_partition_cost_increase=2.0 * increase,
            )
        )
        if abs(actual_increase - increase) > 128.0 * np.finfo(np.float64).eps:
            raise RuntimeError("mass-floor merge cost identity failed")

    final_raw_masses, final_normals, final_output_masses, final_resultant_total = (
        _reduction_cluster_state(clusters, config.minimum_resultant)
    )
    if float(np.min(final_output_masses)) + validation_tolerance < floor:
        raise RuntimeError("mass-floor reduction stopped below its configured floor")
    final_assignments = np.empty(source_masses.size, dtype=np.int64)
    for output_index, cluster in enumerate(clusters):
        final_assignments[np.asarray(cluster.members, dtype=np.int64)] = output_index
    post_energy = _assigned_energy(
        source_masses,
        source_normals,
        final_assignments,
        final_normals,
    )
    contraction = float(1.0 - final_resultant_total)
    if abs(post_energy - contraction) > 128.0 * np.finfo(np.float64).eps:
        raise RuntimeError("mass-floor partition energy/resultant identity failed")
    trace_increase = math.fsum(entry.half_squared_chordal_energy_increase for entry in trace)
    if abs((post_energy - float(selected.energy)) - trace_increase) > (
        256.0 * np.finfo(np.float64).eps
    ):
        raise RuntimeError("mass-floor merge trace does not telescope to the final energy")
    closure = final_output_masses @ final_normals
    if float(np.linalg.norm(closure)) > config.closure_tolerance:
        raise RuntimeError("mass-floor resultant reduction failed Minkowski closure")
    if np.any(final_raw_masses <= config.minimum_resultant):  # pragma: no cover - state checks
        raise RuntimeError("mass-floor reduction emitted a degenerate resultant")
    return _MassFloorReduction(
        assignments=final_assignments,
        pre_reduction_half_squared_chordal_energy=float(selected.energy),
        post_reduction_half_squared_chordal_energy=post_energy,
        validation_tolerance=validation_tolerance,
        trace=tuple(trace),
    )



def _canonical_reduction_clusters(
    clusters: list[_ReductionCluster],
) -> list[_ReductionCluster]:
    """Return a canonical order used by every deterministic merge tie."""

    def key(cluster: _ReductionCluster) -> tuple[float, float, float, float, tuple[int, ...]]:
        norm = _stable_vector_norm(cluster.vector)
        normal = cluster.vector / norm
        return (
            float(normal[0]),
            float(normal[1]),
            float(normal[2]),
            norm,
            cluster.members,
        )

    return sorted(clusters, key=key)



def _reduction_cluster_state(
    clusters: list[_ReductionCluster],
    minimum_resultant: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    vectors = np.stack([cluster.vector for cluster in clusters])
    raw_masses = _row_norms(vectors)
    if np.any(raw_masses <= minimum_resultant):
        raise ValueError("mass-floor reduction encountered a numerically zero resultant")
    normals = vectors / raw_masses[:, None]
    total = float(np.sum(raw_masses))
    return raw_masses, normals, raw_masses / total, total



def _stable_vector_norm(vector: np.ndarray) -> float:
    return float(math.hypot(*(float(value) for value in np.asarray(vector).reshape(-1))))



def _resultant_merge_loss(
    first: np.ndarray,
    second: np.ndarray,
    *,
    merged_norm: float,
) -> float:
    """Return ``||r||+||s||-||r+s||`` without subtractive cancellation."""

    first_norm = _stable_vector_norm(first)
    second_norm = _stable_vector_norm(second)
    cosine = float(
        np.clip(
            np.dot(first / first_norm, second / second_norm),
            -1.0,
            1.0,
        )
    )
    denominator = first_norm + second_norm + merged_norm
    numerator = 2.0 * first_norm * second_norm * max(0.0, 1.0 - cosine)
    return float(numerator / denominator)



def _mass_floor_validation_tolerance(floor: float) -> float:
    return max(float(128.0 * np.finfo(np.float64).eps), abs(floor) * 1.0e-12)



def _build_output(
    input_masses: np.ndarray,
    input_normals: np.ndarray,
    canonical_to_original: np.ndarray,
    selected: _LloydTrial,
    starts: tuple[SemidiscreteOTStartDiagnostics, ...],
    *,
    configured_starts: int,
    effective_starts: int,
    ward_partition_energy: float,
    best_numeric_wasserstein2: float,
    selected_wasserstein2_excess: float,
    runner_up_wasserstein2_gap: float | None,
    requested_output_count: int,
    mass_floor_reduction: _MassFloorReduction | None,
    config: SemidiscreteOTQuantizerConfig,
) -> SemidiscreteOTQuantization:
    if selected.assignments is None:
        raise RuntimeError("selected trial is missing assignments")  # pragma: no cover
    partition_assignments = (
        mass_floor_reduction.assignments
        if mass_floor_reduction is not None
        else selected.assignments
    )
    count = int(np.max(partition_assignments)) + 1
    resultants = _cluster_resultants(
        input_masses,
        input_normals,
        partition_assignments,
        count,
    )
    raw_masses = _row_norms(resultants)
    if np.any(raw_masses <= config.minimum_resultant):
        raise RuntimeError("selected semidiscrete OT solution contains a degenerate cell")
    output_normals = resultants / raw_masses[:, None]
    source_masses = np.bincount(
        partition_assignments,
        weights=input_masses,
        minlength=count,
    ).astype(np.float64, copy=False)
    resultant_total = float(np.sum(raw_masses))
    output_masses = raw_masses / resultant_total

    order = np.lexsort(
        (output_masses, output_normals[:, 2], output_normals[:, 1], output_normals[:, 0])
    )
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(count)
    output_normals = output_normals[order]
    output_masses = output_masses[order]
    source_masses = source_masses[order]
    canonical_assignments = inverse_order[partition_assignments]
    assignments = np.empty_like(canonical_assignments)
    assignments[canonical_to_original] = canonical_assignments

    assigned_normals = output_normals[canonical_assignments]
    cosines = np.clip(np.einsum("ij,ij->i", input_normals, assigned_normals), -1.0, 1.0)
    angles = np.arccos(cosines)
    mean_angle = float(np.sum(input_masses * angles))
    rms_angle = float(np.sqrt(np.sum(input_masses * angles**2)))
    angle_order = np.argsort(angles, kind="stable")
    cumulative = np.cumsum(input_masses[angle_order])
    quantile_index = min(
        int(np.searchsorted(cumulative, 0.99, side="left")),
        angles.size - 1,
    )
    if count > 1:
        cosine_matrix = np.clip(output_normals @ output_normals.T, -1.0, 1.0)
        np.fill_diagonal(cosine_matrix, -1.0)
        minimum_separation = float(np.arccos(np.max(cosine_matrix)))
    else:  # pragma: no cover - public API requires at least four atoms
        minimum_separation = np.pi

    input_closure = float(np.linalg.norm(input_masses @ input_normals))
    output_closure = float(np.linalg.norm(output_masses @ output_normals))
    if output_closure > config.closure_tolerance:
        raise RuntimeError(
            "closure-preserving resultant normalization exceeded tolerance; "
            f"norm={output_closure:.3e}"
        )
    lloyd_energy = _assigned_energy(
        input_masses,
        input_normals,
        canonical_assignments,
        output_normals,
    )
    contraction = float(1.0 - resultant_total)
    roundoff_scale = 64.0 * np.finfo(np.float64).eps
    if abs(lloyd_energy - contraction) > roundoff_scale:
        raise RuntimeError(
            "Lloyd energy/resultant-contraction identity failed; "
            f"difference={lloyd_energy - contraction:.3e}"
        )
    if selected.closed_transport is None:
        raise RuntimeError("selected trial is missing its exact transport score")
    exact_output_transport = (
        exact_rectangular_wasserstein2(
            input_masses,
            input_normals,
            output_masses,
            output_normals,
            feasibility_tolerance=config.transport_feasibility_tolerance,
            mass_scale=config.transport_mass_scale,
        )
        if mass_floor_reduction is not None
        else selected.closed_transport
    )
    if (
        exact_output_transport.maximum_marginal_residual
        > config.transport_marginal_residual_tolerance
    ):
        raise RuntimeError(
            "mass-floor output failed the exact-transport marginal gate; "
            f"residual={exact_output_transport.maximum_marginal_residual:.3e}"
        )
    pre_reduction_energy = (
        mass_floor_reduction.pre_reduction_half_squared_chordal_energy
        if mass_floor_reduction is not None
        else lloyd_energy
    )
    post_reduction_energy = lloyd_energy
    if mass_floor_reduction is not None and abs(
        post_reduction_energy - mass_floor_reduction.post_reduction_half_squared_chordal_energy
    ) > (256.0 * np.finfo(np.float64).eps):
        raise RuntimeError("mass-floor emitted partition does not replay its recorded energy")
    pre_reduction_wasserstein2 = selected.closed_transport.squared_distance
    mass_floor_enabled = mass_floor_reduction is not None
    mass_floor_tolerance = (
        mass_floor_reduction.validation_tolerance if mass_floor_reduction is not None else None
    )
    mass_floor_trace = mass_floor_reduction.trace if mass_floor_reduction is not None else ()
    floor = config.normalized_output_mass_floor
    if mass_floor_enabled and floor is not None and mass_floor_tolerance is not None:
        if float(np.min(output_masses)) + mass_floor_tolerance < floor:
            raise RuntimeError(
                "mass-floor reduction emitted a mass below the configured floor; "
                f"minimum={float(np.min(output_masses)):.6g}, floor={floor:.6g}"
            )
    cell_mass_closure = float(np.linalg.norm(source_masses @ output_normals))
    degrees = 180.0 / np.pi
    diagnostics = SemidiscreteOTQuantizationDiagnostics(
        method="deterministic_multistart_weighted_spherical_lloyd_with_resultant_closure",
        ground_cost="half_squared_chordal_on_unit_sphere",
        initialization=("resultant_Ward_start_0_plus_seeded_weighted_farthest_first_restarts"),
        seed=config.seed,
        input_atom_count=int(input_masses.size),
        requested_output_atom_count=requested_output_count,
        output_atom_count=count,
        effective_output_atom_count=count,
        configured_start_count=configured_starts,
        effective_start_count=effective_starts,
        selected_start_index=selected.diagnostics.start_index,
        selected_start_converged=selected.diagnostics.converged,
        selected_iteration_count=selected.diagnostics.iteration_count,
        restart_selection_criterion=(
            "minimum exact chordal W2^2 to closure-emitted measure; values within "
            "absolute tie tolerance use Lloyd Q, cell/output mass L1, canonical support, "
            "then restart id"
        ),
        restart_wasserstein2_tie_tolerance=config.restart_wasserstein2_tie_tolerance,
        closed_output_wasserstein2_best_numeric=best_numeric_wasserstein2,
        closed_output_wasserstein2_selected_excess_over_best=(selected_wasserstein2_excess),
        closed_output_wasserstein2_runner_up_gap=runner_up_wasserstein2_gap,
        closed_output_transport_feasibility_tolerance=(config.transport_feasibility_tolerance),
        closed_output_transport_mass_scale=config.transport_mass_scale,
        closed_output_transport_marginal_residual_tolerance=(
            config.transport_marginal_residual_tolerance
        ),
        mass_floor_reduction_enabled=mass_floor_enabled,
        normalized_output_mass_floor=floor,
        mass_floor_validation_tolerance=mass_floor_tolerance,
        mass_floor_merge_count=len(mass_floor_trace),
        mass_floor_merge_trace=mass_floor_trace,
        input_closure_norm=input_closure,
        output_closure_norm=output_closure,
        lloyd_cell_mass_closure_norm=cell_mass_closure,
        lloyd_cell_masses=tuple(float(value) for value in source_masses),
        closure_output_masses=tuple(float(value) for value in output_masses),
        lloyd_half_squared_chordal_energy=float(lloyd_energy),
        semidiscrete_partition_wasserstein2_squared=float(2.0 * lloyd_energy),
        pre_reduction_half_squared_chordal_energy=float(pre_reduction_energy),
        post_reduction_half_squared_chordal_energy=float(post_reduction_energy),
        mass_floor_half_squared_chordal_energy_delta=float(
            post_reduction_energy - pre_reduction_energy
        ),
        pre_reduction_squared_chordal_partition_cost=float(2.0 * pre_reduction_energy),
        post_reduction_squared_chordal_partition_cost=float(2.0 * post_reduction_energy),
        mass_floor_squared_chordal_partition_cost_delta=float(
            2.0 * (post_reduction_energy - pre_reduction_energy)
        ),
        closed_output_wasserstein2_squared=exact_output_transport.squared_distance,
        closed_output_wasserstein2_energy=exact_output_transport.energy,
        closed_output_transport_solver_status=exact_output_transport.solver_status,
        closed_output_transport_solver_message=exact_output_transport.solver_message,
        closed_output_transport_maximum_marginal_residual=(
            exact_output_transport.maximum_marginal_residual
        ),
        pre_reduction_closed_output_wasserstein2_squared=pre_reduction_wasserstein2,
        post_reduction_closed_output_wasserstein2_squared=(exact_output_transport.squared_distance),
        mass_floor_closed_output_wasserstein2_squared_delta=float(
            exact_output_transport.squared_distance - pre_reduction_wasserstein2
        ),
        total_resultant_before_normalization=resultant_total,
        resultant_mass_contraction=contraction,
        pre_reduction_resultant_mass_contraction=float(pre_reduction_energy),
        post_reduction_resultant_mass_contraction=contraction,
        mass_floor_resultant_mass_contraction_delta=float(contraction - pre_reduction_energy),
        ward_partition_half_squared_chordal_energy=float(ward_partition_energy),
        ward_start_final_half_squared_chordal_energy=float(
            starts[0].final_half_squared_chordal_energy
            if starts[0].final_half_squared_chordal_energy is not None
            else np.nan
        ),
        ward_start_improvement_from_ward_partition=float(
            ward_partition_energy
            - (
                starts[0].final_half_squared_chordal_energy
                if starts[0].final_half_squared_chordal_energy is not None
                else np.nan
            )
        ),
        selected_improvement_from_ward_partition=float(ward_partition_energy - lloyd_energy),
        cell_mass_to_output_mass_l1=float(np.sum(np.abs(source_masses - output_masses))),
        paired_cluster_mass_l1=float(np.sum(np.abs(source_masses - output_masses))),
        weighted_mean_angular_displacement_degrees=mean_angle * degrees,
        weighted_rms_angular_displacement_degrees=rms_angle * degrees,
        mass_99_angular_displacement_degrees=float(angles[angle_order[quantile_index]]) * degrees,
        maximum_angular_displacement_degrees=float(np.max(angles)) * degrees,
        minimum_output_normal_separation_degrees=minimum_separation * degrees,
        inverse_simpson_effective_atom_count=float(1.0 / np.sum(output_masses**2)),
        minimum_output_mass=float(np.min(output_masses)),
        maximum_output_mass=float(np.max(output_masses)),
        starts=starts,
    )
    return SemidiscreteOTQuantization(
        masses=output_masses,
        normals=output_normals,
        assignments=assignments,
        cluster_source_masses=source_masses,
        diagnostics=diagnostics,
    )



def _score_closed_trial(
    source_masses: np.ndarray,
    source_normals: np.ndarray,
    trial: _LloydTrial,
    config: SemidiscreteOTQuantizerConfig,
) -> _LloydTrial:
    if trial.assignments is None or trial.centers is None or trial.energy is None:
        return trial  # pragma: no cover - called only for valid trials
    count = trial.centers.shape[0]
    resultants = _cluster_resultants(
        source_masses,
        source_normals,
        trial.assignments,
        count,
    )
    raw_masses = _row_norms(resultants)
    output_normals = resultants / raw_masses[:, None]
    output_masses = raw_masses / np.sum(raw_masses)
    cell_masses = np.bincount(
        trial.assignments,
        weights=source_masses,
        minlength=count,
    ).astype(np.float64, copy=False)
    order = np.lexsort(
        (output_masses, output_normals[:, 2], output_normals[:, 1], output_normals[:, 0])
    )
    canonical_normals = output_normals[order]
    canonical_masses = output_masses[order]
    signature = tuple(
        float(value) for value in np.column_stack([canonical_normals, canonical_masses]).reshape(-1)
    )
    transport = exact_rectangular_wasserstein2(
        source_masses,
        source_normals,
        canonical_masses,
        canonical_normals,
        feasibility_tolerance=config.transport_feasibility_tolerance,
        mass_scale=config.transport_mass_scale,
    )
    accepted = transport.maximum_marginal_residual <= config.transport_marginal_residual_tolerance
    mass_l1 = float(np.sum(np.abs(cell_masses[order] - canonical_masses)))
    diagnostics = replace(
        trial.diagnostics,
        closed_output_wasserstein2_squared=transport.squared_distance,
        closed_output_transport_maximum_marginal_residual=(transport.maximum_marginal_residual),
        closed_output_transport_accepted=accepted,
        cell_mass_to_output_mass_l1=mass_l1,
    )
    return replace(
        trial,
        diagnostics=diagnostics,
        closed_transport=transport,
        cell_mass_to_output_mass_l1=mass_l1,
        canonical_output_signature=signature,
    )



def _closed_wasserstein2(trial: _LloydTrial) -> float:
    if trial.closed_transport is None:
        return math.inf
    return trial.closed_transport.squared_distance



def _trial_secondary_selection_key(
    trial: _LloydTrial,
) -> tuple[float, float, tuple[float, ...], int]:
    return (
        float(trial.energy) if trial.energy is not None else math.inf,
        (
            float(trial.cell_mass_to_output_mass_l1)
            if trial.cell_mass_to_output_mass_l1 is not None
            else math.inf
        ),
        trial.canonical_output_signature,
        trial.diagnostics.start_index,
    )



def _assign_nonempty(
    masses: np.ndarray,
    normals: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, int]:
    cosine = np.clip(normals @ centers.T, -1.0, 1.0)
    assignments = np.argmax(cosine, axis=1).astype(np.int64, copy=False)
    count = centers.shape[0]
    cluster_counts = np.bincount(assignments, minlength=count)
    repairs = 0
    for empty in np.flatnonzero(cluster_counts == 0):
        donors = cluster_counts[assignments] > 1
        if not np.any(donors):  # pragma: no cover - atom_count <= input count
            raise RuntimeError("cannot repair an empty Lloyd cell")
        current_cosine = cosine[np.arange(normals.shape[0]), assignments]
        scores = masses * np.maximum(1.0 - current_cosine, 0.0)
        scores[~donors] = -np.inf
        moved = int(np.argmax(scores))
        old = int(assignments[moved])
        assignments[moved] = int(empty)
        cluster_counts[old] -= 1
        cluster_counts[empty] += 1
        repairs += 1
    return assignments, repairs



def _cluster_resultants(
    masses: np.ndarray,
    normals: np.ndarray,
    assignments: np.ndarray,
    count: int,
) -> np.ndarray:
    resultants = np.zeros((count, 3), dtype=np.float64)
    np.add.at(resultants, assignments, masses[:, None] * normals)
    return resultants



def _assigned_energy(
    masses: np.ndarray,
    normals: np.ndarray,
    assignments: np.ndarray,
    centers: np.ndarray,
) -> float:
    cosines = np.clip(np.einsum("ij,ij->i", normals, centers[assignments]), -1.0, 1.0)
    return float(np.sum(masses * np.maximum(1.0 - cosines, 0.0)))



def _nearest_site_energy(
    masses: np.ndarray,
    normals: np.ndarray,
    centers: np.ndarray,
) -> float:
    nearest_cosines = np.max(np.clip(normals @ centers.T, -1.0, 1.0), axis=1)
    return float(np.sum(masses * np.maximum(1.0 - nearest_cosines, 0.0)))



def _weighted_farthest_sequence(
    directions: np.ndarray,
    masses: np.ndarray,
    count: int,
    *,
    first_index: int,
) -> np.ndarray:
    return _weighted_farthest_seeds(
        directions,
        masses,
        count,
        first_index=first_index,
    )



def _seeded_weighted_anchor(masses: np.ndarray, seed: int) -> int:
    """Draw one reproducible mass-weighted anchor on canonicalized rows."""

    generator = np.random.default_rng(seed)
    threshold = float(generator.random()) * float(np.sum(masses))
    cumulative = np.cumsum(masses)
    return min(int(np.searchsorted(cumulative, threshold, side="right")), masses.size - 1)



def _weighted_farthest_seeds(
    directions: np.ndarray,
    masses: np.ndarray,
    count: int,
    *,
    first_index: int,
) -> np.ndarray:
    selected = [int(first_index)]
    excluded = np.zeros(directions.shape[0], dtype=bool)
    excluded[first_index] = True
    nearest_cost = np.maximum(1.0 - directions @ directions[first_index], 0.0)
    while len(selected) < count:
        scores = masses * nearest_cost
        scores[excluded] = -np.inf
        next_index = int(np.argmax(scores))
        if not np.isfinite(scores[next_index]):  # pragma: no cover - validated unique count
            raise RuntimeError("deterministic farthest-first initialization was exhausted")
        selected.append(next_index)
        excluded[next_index] = True
        next_cost = np.maximum(1.0 - directions @ directions[next_index], 0.0)
        nearest_cost = np.minimum(nearest_cost, next_cost)
    return np.asarray(selected, dtype=np.int64)



def _unique_direction_data(
    masses: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distinct = np.ones(normals.shape[0], dtype=bool)
    if normals.shape[0] > 1:
        distinct[1:] = np.any(normals[1:] != normals[:-1], axis=1)
    starts = np.flatnonzero(distinct)
    group_ids = np.cumsum(distinct) - 1
    group_masses = np.bincount(group_ids, weights=masses).astype(np.float64, copy=False)
    return starts, group_masses



def _validate_measure(
    masses: np.ndarray,
    normals: np.ndarray,
    config: SemidiscreteOTQuantizerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    directions = np.asarray(normals, dtype=np.float64)
    if values.size < 4 or directions.shape != (values.size, 3):
        raise ValueError("masses/normals must describe at least four spherical atoms")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("masses must be finite and strictly positive")
    if not np.all(np.isfinite(directions)):
        raise ValueError("normals must be finite")
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("masses must have finite positive total mass")
    values = values / total
    lengths = _row_norms(directions)
    if np.max(np.abs(lengths - 1.0)) > config.unit_normal_tolerance:
        raise ValueError("normals must be unit length")
    closure = values @ directions
    if float(np.linalg.norm(closure)) > config.closure_tolerance:
        raise ValueError("input measure does not satisfy Minkowski closure")
    order = np.lexsort((values, directions[:, 2], directions[:, 1], directions[:, 0]))
    return values[order], directions[order], order



def _row_norms(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [math.hypot(*(float(value) for value in row)) for row in values],
        dtype=np.float64,
    )



def exact_rectangular_wasserstein2(
    source_masses: np.ndarray,
    source_normals: np.ndarray,
    target_masses: np.ndarray,
    target_normals: np.ndarray,
    *,
    feasibility_tolerance: float = 1.0e-10,
    mass_scale: float = 1.0e8,
) -> RectangularWassersteinResult:
    """Return exact chordal ``W_2^2`` between two finite spherical measures.

    Unlike the Lloyd partition objective, this balanced transport solve uses
    the closure-adjusted output masses as its target marginal.  It does not
    alter Lloyd descent; the quantizer uses it only to rank the converged,
    closure-emitted restarts.
    """

    first, first_directions = _validated_transport_measure("source", source_masses, source_normals)
    second, second_directions = _validated_transport_measure(
        "target", target_masses, target_normals
    )
    if not np.isfinite(feasibility_tolerance) or feasibility_tolerance < 1.0e-10:
        raise ValueError("feasibility_tolerance must be finite and at least 1e-10")
    if not np.isfinite(mass_scale) or mass_scale < 1.0:
        raise ValueError("mass_scale must be finite and at least one")
    source_count = first.size
    target_count = second.size
    cost = np.maximum(2.0 - 2.0 * (first_directions @ second_directions.T), 0.0)
    row_constraints = sparse.kron(
        sparse.eye(source_count, format="csr"),
        np.ones((1, target_count), dtype=np.float64),
        format="csr",
    )
    column_constraints = sparse.kron(
        np.ones((1, source_count), dtype=np.float64),
        sparse.eye(target_count, format="csr"),
        format="csr",
    )
    # The last target constraint is implied by the other marginals.
    equality = sparse.vstack(
        [row_constraints, column_constraints[: target_count - 1]],
        format="csr",
    )
    # HiGHS treats probability-scale right-hand sides below roughly 1e-9 as
    # numerical zero even when its feasibility tolerances are tighter.  Solve
    # in equivalent mass units so marginals relevant at the 1e-12 restart tie
    # scale remain explicit, then divide the plan and objective back down.
    right_hand_side = np.concatenate([first, second[: target_count - 1]]) * mass_scale
    result = linprog(
        cost.reshape(-1),
        A_eq=equality,
        b_eq=right_hand_side,
        bounds=(0.0, None),
        method="highs",
        options={
            "primal_feasibility_tolerance": float(feasibility_tolerance),
            "dual_feasibility_tolerance": float(feasibility_tolerance),
        },
    )
    if not result.success or result.fun is None or result.x is None:
        raise RuntimeError(f"exact rectangular spherical W2 solve failed: {result.message}")
    plan = np.asarray(result.x, dtype=np.float64).reshape(source_count, target_count) / mass_scale
    marginal_residual = max(
        float(np.max(np.abs(np.sum(plan, axis=1) - first))),
        float(np.max(np.abs(np.sum(plan, axis=0) - second))),
    )
    squared_distance = float(max(float(result.fun) / mass_scale, 0.0))
    return RectangularWassersteinResult(
        squared_distance=squared_distance,
        energy=0.5 * squared_distance,
        solver_status=int(result.status),
        solver_message=str(result.message),
        maximum_marginal_residual=marginal_residual,
    )



def _validated_transport_measure(
    label: str,
    masses: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    directions = np.asarray(normals, dtype=np.float64)
    if values.size < 1 or directions.shape != (values.size, 3):
        raise ValueError(f"{label} masses/normals have incompatible shapes")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{label} masses must be finite and non-negative")
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{label} masses must have finite positive total mass")
    if not np.all(np.isfinite(directions)):
        raise ValueError(f"{label} normals must be finite")
    lengths = _row_norms(directions)
    if not np.allclose(lengths, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{label} normals must be unit length")
    return values / total, directions

