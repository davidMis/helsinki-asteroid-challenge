"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import dataclass
import heapq
from itertools import combinations
import math
import numpy as np


@dataclass(frozen=True)
class AtomicCoarseningDiagnostics:
    """Auditable distortion and admissibility diagnostics for one cut."""

    input_atom_count: int
    output_atom_count: int
    input_closure_norm: float
    output_closure_norm: float
    total_resultant_before_normalization: float
    resultant_mass_contraction: float
    paired_cluster_mass_l1: float
    weighted_mean_angular_displacement_degrees: float
    weighted_rms_angular_displacement_degrees: float
    mass_99_angular_displacement_degrees: float
    maximum_angular_displacement_degrees: float
    minimum_output_normal_separation_degrees: float
    inverse_simpson_effective_atom_count: float
    minimum_output_mass: float
    maximum_output_mass: float



@dataclass(frozen=True)
class AtomicGaussianImage:
    """One closure-preserving atomic approximation of an input measure."""

    masses: np.ndarray
    normals: np.ndarray
    assignments: np.ndarray
    cluster_source_masses: np.ndarray
    diagnostics: AtomicCoarseningDiagnostics



@dataclass(frozen=True)
class _Cluster:
    vector: np.ndarray
    source_mass: float
    members: tuple[int, ...]

    @property
    def resultant_mass(self) -> float:
        return _stable_norm(self.vector)



def _stable_norm(vector: np.ndarray) -> float:
    """Return a Euclidean norm without squaring tiny resultants to zero."""

    values = np.asarray(vector, dtype=np.float64)
    return float(math.hypot(*(float(value) for value in values.flat)))



def closure_preserving_atomic_hierarchy(
    masses: np.ndarray,
    normals: np.ndarray,
    atom_counts: list[int] | tuple[int, ...] | np.ndarray,
    *,
    closure_tolerance: float = 1.0e-10,
    unit_normal_tolerance: float = 1.0e-10,
    minimum_resultant: float = 1.0e-14,
) -> dict[int, AtomicGaussianImage]:
    """Return deterministic closure-preserving cuts of one merge hierarchy.

    All input atoms participate in every output; no mass is discarded.  Input
    order is canonicalized lexicographically, making the hierarchy invariant
    to a permutation of otherwise identical input rows (apart from exact
    floating-point ties between duplicate rows).
    """
    values, directions, canonical_to_original = _validate_measure(
        masses,
        normals,
        closure_tolerance=closure_tolerance,
        unit_normal_tolerance=unit_normal_tolerance,
    )
    requested = sorted({int(count) for count in atom_counts}, reverse=True)
    if not requested:
        raise ValueError("atom_counts must contain at least one count")
    if requested[0] > values.size or requested[-1] < 4:
        raise ValueError("atom counts must lie between 4 and the input atom count")
    if minimum_resultant <= 0.0 or not np.isfinite(minimum_resultant):
        raise ValueError("minimum_resultant must be finite and positive")

    active: dict[int, _Cluster] = {
        index: _Cluster(
            vector=values[index] * directions[index],
            source_mass=float(values[index]),
            members=(index,),
        )
        for index in range(values.size)
    }
    next_id = values.size
    heap: list[tuple[float, float, int, int]] = []
    for first, second in combinations(active, 2):
        heapq.heappush(heap, _merge_key(first, active[first], second, active[second]))

    outputs: dict[int, AtomicGaussianImage] = {}
    if values.size in requested:
        outputs[values.size] = _snapshot(
            active,
            values,
            directions,
            canonical_to_original,
        )

    minimum_requested = requested[-1]
    requested_set = set(requested)
    while len(active) > minimum_requested:
        while heap:
            _, _, first, second = heapq.heappop(heap)
            if first in active and second in active:
                break
        else:  # pragma: no cover - defensive; a valid active set always has pairs
            raise RuntimeError("atomic merge heap was exhausted unexpectedly")

        first_cluster = active.pop(first)
        second_cluster = active.pop(second)
        merged_vector = first_cluster.vector + second_cluster.vector
        resultant = _stable_norm(merged_vector)
        if resultant <= minimum_resultant:
            # A nearly cancelling pair is unsuitable as an atom.  Keep it out
            # of consideration and choose the next pair.  Reinsert the nodes
            # and mark only this pair as forbidden.
            active[first] = first_cluster
            active[second] = second_cluster
            continue

        merged = _Cluster(
            vector=merged_vector,
            source_mass=first_cluster.source_mass + second_cluster.source_mass,
            members=tuple(sorted(first_cluster.members + second_cluster.members)),
        )
        merged_id = next_id
        next_id += 1
        for other_id, other in active.items():
            heapq.heappush(heap, _merge_key(merged_id, merged, other_id, other))
        active[merged_id] = merged

        if len(active) in requested_set:
            outputs[len(active)] = _snapshot(
                active,
                values,
                directions,
                canonical_to_original,
            )

    missing = requested_set.difference(outputs)
    if missing:  # pragma: no cover - guarded by the loop and count validation
        raise RuntimeError(f"failed to construct hierarchy cuts {sorted(missing)}")
    return {count: outputs[count] for count in sorted(outputs)}



def closure_preserving_atomic_coarsen(
    masses: np.ndarray,
    normals: np.ndarray,
    atom_count: int,
    **kwargs: float,
) -> AtomicGaussianImage:
    """Convenience wrapper returning one atomic hierarchy cut."""
    return closure_preserving_atomic_hierarchy(
        masses,
        normals,
        [atom_count],
        **kwargs,
    )[int(atom_count)]



def _validate_measure(
    masses: np.ndarray,
    normals: np.ndarray,
    *,
    closure_tolerance: float,
    unit_normal_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    directions = np.asarray(normals, dtype=np.float64)
    if values.size < 4 or directions.shape != (values.size, 3):
        raise ValueError("masses/normals must describe at least four spherical atoms")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("masses must be finite and strictly positive")
    if not np.all(np.isfinite(directions)):
        raise ValueError("normals must be finite")
    if closure_tolerance <= 0.0 or unit_normal_tolerance <= 0.0:
        raise ValueError("validation tolerances must be positive")
    lengths = np.linalg.norm(directions, axis=1)
    if np.max(np.abs(lengths - 1.0)) > unit_normal_tolerance:
        raise ValueError("normals must be unit length")
    values = values / np.sum(values)
    closure = np.sum(values[:, None] * directions, axis=0)
    if np.linalg.norm(closure) > closure_tolerance:
        raise ValueError("input measure does not satisfy Minkowski closure")

    # np.lexsort uses the last key as primary.  Include mass only as a final
    # tie-break for duplicate normals.
    order = np.lexsort((values, directions[:, 2], directions[:, 1], directions[:, 0]))
    return values[order], directions[order], order



def _merge_key(
    first_id: int,
    first: _Cluster,
    second_id: int,
    second: _Cluster,
) -> tuple[float, float, int, int]:
    if second_id < first_id:
        first_id, second_id = second_id, first_id
        first, second = second, first
    first_mass = first.resultant_mass
    second_mass = second.resultant_mass
    merged_mass = _stable_norm(first.vector + second.vector)
    loss = max(0.0, first_mass + second_mass - merged_mass)
    first_unit = first.vector / first_mass
    second_unit = second.vector / second_mass
    cosine = math.fsum(
        float(left) * float(right) for left, right in zip(first_unit, second_unit, strict=True)
    )
    angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return loss, angle, first_id, second_id



def _snapshot(
    active: dict[int, _Cluster],
    input_masses: np.ndarray,
    input_normals: np.ndarray,
    canonical_to_original: np.ndarray,
) -> AtomicGaussianImage:
    clusters = list(active.values())
    vectors = np.stack([cluster.vector for cluster in clusters])
    raw_masses = np.asarray([_stable_norm(vector) for vector in vectors])
    output_normals = vectors / raw_masses[:, None]
    order = np.lexsort(
        (raw_masses, output_normals[:, 2], output_normals[:, 1], output_normals[:, 0])
    )
    output_normals = output_normals[order]
    raw_masses = raw_masses[order]
    clusters = [clusters[index] for index in order]
    resultant_total = float(np.sum(raw_masses))
    output_masses = raw_masses / resultant_total
    source_masses = np.asarray([cluster.source_mass for cluster in clusters])

    canonical_assignments = np.empty(input_masses.size, dtype=np.int64)
    for output_index, cluster in enumerate(clusters):
        canonical_assignments[np.asarray(cluster.members, dtype=np.int64)] = output_index
    assignments = np.empty_like(canonical_assignments)
    assignments[canonical_to_original] = canonical_assignments

    assigned_normals = output_normals[canonical_assignments]
    cosines = np.einsum("ij,ij->i", input_normals, assigned_normals)
    angles = np.arccos(np.clip(cosines, -1.0, 1.0))
    mean_angle = float(np.sum(input_masses * angles))
    rms_angle = float(np.sqrt(np.sum(input_masses * angles**2)))
    angle_order = np.argsort(angles, kind="stable")
    cumulative = np.cumsum(input_masses[angle_order])
    quantile_index = min(int(np.searchsorted(cumulative, 0.99, side="left")), angles.size - 1)
    mass_99_angle = float(angles[angle_order[quantile_index]])

    if output_masses.size > 1:
        cos_matrix = np.clip(output_normals @ output_normals.T, -1.0, 1.0)
        np.fill_diagonal(cos_matrix, -1.0)
        minimum_separation = float(np.arccos(np.max(cos_matrix)))
    else:  # pragma: no cover - public API requires at least four atoms
        minimum_separation = np.pi

    input_closure = float(np.linalg.norm(np.sum(input_masses[:, None] * input_normals, axis=0)))
    output_closure = float(np.linalg.norm(np.sum(output_masses[:, None] * output_normals, axis=0)))
    effective_count = float(1.0 / np.sum(output_masses**2))
    degrees = 180.0 / np.pi
    diagnostics = AtomicCoarseningDiagnostics(
        input_atom_count=int(input_masses.size),
        output_atom_count=int(output_masses.size),
        input_closure_norm=input_closure,
        output_closure_norm=output_closure,
        total_resultant_before_normalization=resultant_total,
        resultant_mass_contraction=float(1.0 - resultant_total),
        paired_cluster_mass_l1=float(np.sum(np.abs(source_masses - output_masses))),
        weighted_mean_angular_displacement_degrees=mean_angle * degrees,
        weighted_rms_angular_displacement_degrees=rms_angle * degrees,
        mass_99_angular_displacement_degrees=mass_99_angle * degrees,
        maximum_angular_displacement_degrees=float(np.max(angles)) * degrees,
        minimum_output_normal_separation_degrees=minimum_separation * degrees,
        inverse_simpson_effective_atom_count=effective_count,
        minimum_output_mass=float(np.min(output_masses)),
        maximum_output_mass=float(np.max(output_masses)),
    )
    return AtomicGaussianImage(
        masses=output_masses,
        normals=output_normals,
        assignments=assignments,
        cluster_source_masses=source_masses,
        diagnostics=diagnostics,
    )

