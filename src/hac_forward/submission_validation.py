"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from .mesh_priors import released_cylinder_radius
from .stl import load_stl_triangles, validate_triangle_array


@dataclass(frozen=True)
class SubmissionMeshValidation:
    """Machine-readable validation result for one proposed submission mesh."""

    valid: bool
    errors: tuple[str, ...]
    checks: dict[str, bool | None]
    metrics: dict[str, int | float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
            "self_intersection": {
                "checked": False,
                "note": "self-intersection is not checked by this validator",
            },
        }

    def require_valid(self) -> None:
        """Raise ``ValueError`` when validation failed."""
        if not self.valid:
            raise ValueError("invalid submission mesh: " + "; ".join(self.errors))



def normalize_submission_frame(
    triangles: np.ndarray,
    model_id: int,
    match_radius: bool = True,
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """Normalize a mesh to the fixed HAC frame and return transform diagnostics.

    The transform translates z only, uniformly scales all coordinates until the
    z extrema are exactly ``[-1, 1]``, and optionally scales x/y about the fixed
    origin until the maximum cylindrical radius matches the released prior.
    The x/y origin is never inferred from the candidate bounding box.
    """
    mesh = validate_triangle_array(triangles)
    flat = mesh.reshape(-1, 3)
    z_min_before = float(np.min(flat[:, 2]))
    z_max_before = float(np.max(flat[:, 2]))
    z_center = 0.5 * (z_min_before + z_max_before)
    z_half_extent = 0.5 * (z_max_before - z_min_before)
    if not np.isfinite(z_half_extent) or z_half_extent <= 0.0:
        raise ValueError("submission mesh has non-positive z extent")

    normalized = np.array(mesh, dtype=np.float64, copy=True)
    normalized[..., 2] -= z_center
    uniform_scale = 1.0 / z_half_extent
    normalized *= uniform_scale
    # Guarantee the two required contacts even when the midpoint arithmetic
    # incurred a last-bit rounding difference.
    normalized[..., 2][mesh[..., 2] == z_min_before] = -1.0
    normalized[..., 2][mesh[..., 2] == z_max_before] = 1.0

    radius_before = float(np.max(np.linalg.norm(flat[:, :2], axis=1)))
    radius_after_z_scale = float(
        np.max(np.linalg.norm(normalized.reshape(-1, 3)[:, :2], axis=1))
    )
    target_radius = released_cylinder_radius(model_id)
    xy_scale = 1.0
    if match_radius:
        if radius_after_z_scale <= 0.0:
            raise ValueError("cannot radius-match a mesh with zero xy radius")
        xy_scale = target_radius / radius_after_z_scale
        normalized[..., :2] *= xy_scale

    normalized = validate_triangle_array(normalized)
    normalized_flat = normalized.reshape(-1, 3)
    diagnostics: dict[str, int | float | bool] = {
        "model_id": int(model_id),
        "match_radius": bool(match_radius),
        "xy_translation": 0.0,
        "z_min_before": z_min_before,
        "z_max_before": z_max_before,
        "z_center_translation": -z_center,
        "z_half_extent_before": z_half_extent,
        "uniform_scale": uniform_scale,
        "xy_radius_before": radius_before,
        "xy_radius_after_z_scale": radius_after_z_scale,
        "released_cylinder_radius": target_radius,
        "xy_scale": xy_scale,
        "z_min_after": float(np.min(normalized_flat[:, 2])),
        "z_max_after": float(np.max(normalized_flat[:, 2])),
        "xy_radius_after": float(
            np.max(np.linalg.norm(normalized_flat[:, :2], axis=1))
        ),
    }
    return normalized, diagnostics



def validate_submission_stl(
    path: Path,
    *,
    cylinder_radius: float,
    minimum_triangle_area: float = 0.0,
    z_tolerance: float = 1.0e-6,
    cylinder_containment_tolerance: float = 1.0e-6,
    cylinder_contact_tolerance: float = 1.0e-6,
    positive_volume_tolerance: float = 0.0,
) -> SubmissionMeshValidation:
    """Load and validate one ASCII or binary STL submission candidate."""
    return validate_submission_mesh(
        load_stl_triangles(Path(path)),
        cylinder_radius=cylinder_radius,
        minimum_triangle_area=minimum_triangle_area,
        z_tolerance=z_tolerance,
        cylinder_containment_tolerance=cylinder_containment_tolerance,
        cylinder_contact_tolerance=cylinder_contact_tolerance,
        positive_volume_tolerance=positive_volume_tolerance,
    )



def validate_submission_mesh(
    triangles: np.ndarray,
    *,
    cylinder_radius: float,
    minimum_triangle_area: float = 0.0,
    z_tolerance: float = 1.0e-6,
    cylinder_containment_tolerance: float = 1.0e-6,
    cylinder_contact_tolerance: float = 1.0e-6,
    positive_volume_tolerance: float = 0.0,
) -> SubmissionMeshValidation:
    """Validate topology, orientation, volume, and HAC cylinder constraints.

    Vertex identity is exact, matching STL's facet representation: adjacent
    facets must repeat exactly the same endpoint coordinates. Self-intersection
    is intentionally outside this dependency-free validator's scope and is
    reported as unchecked rather than inferred from watertightness.
    """
    cylinder_radius = _positive_finite(cylinder_radius, "cylinder_radius")
    minimum_triangle_area = _nonnegative_finite(
        minimum_triangle_area, "minimum_triangle_area"
    )
    z_tolerance = _nonnegative_finite(z_tolerance, "z_tolerance")
    cylinder_containment_tolerance = _nonnegative_finite(
        cylinder_containment_tolerance, "cylinder_containment_tolerance"
    )
    cylinder_contact_tolerance = _nonnegative_finite(
        cylinder_contact_tolerance, "cylinder_contact_tolerance"
    )
    positive_volume_tolerance = _nonnegative_finite(
        positive_volume_tolerance, "positive_volume_tolerance"
    )

    try:
        mesh = validate_triangle_array(
            triangles,
            minimum_area=minimum_triangle_area,
        )
    except (TypeError, ValueError) as exc:
        return SubmissionMeshValidation(
            valid=False,
            errors=(str(exc),),
            checks={
                "finite": False,
                "nondegenerate": False,
                "edge_manifold": False,
                "watertight": False,
                "opposite_edge_orientation": False,
                "connected": False,
                "positive_signed_volume": False,
                "z_extrema": False,
                "cylinder_containment": False,
                "cylinder_contact": False,
                "self_intersection_free": None,
            },
            metrics={},
        )

    flat = mesh.reshape(-1, 3)
    unique_vertices, inverse = np.unique(flat, axis=0, return_inverse=True)
    face_vertices = inverse.reshape(-1, 3)
    edge_owners: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_index, (a, b, c) in enumerate(face_vertices):
        for start, end in ((a, b), (b, c), (c, a)):
            start_id = int(start)
            end_id = int(end)
            key = (
                (start_id, end_id)
                if start_id < end_id
                else (end_id, start_id)
            )
            edge_owners.setdefault(key, []).append(
                (face_index, start_id, end_id)
            )

    edge_counts = np.fromiter(
        (len(owners) for owners in edge_owners.values()),
        dtype=np.int64,
        count=len(edge_owners),
    )
    boundary_edges = int(np.count_nonzero(edge_counts == 1))
    nonmanifold_edges = int(np.count_nonzero(edge_counts > 2))
    edge_manifold = nonmanifold_edges == 0
    watertight = bool(edge_counts.size > 0 and np.all(edge_counts == 2))
    orientation_mismatches = sum(
        1
        for owners in edge_owners.values()
        if len(owners) != 2
        or not (
            owners[0][1] == owners[1][2]
            and owners[0][2] == owners[1][1]
        )
    )
    opposite_orientation = orientation_mismatches == 0
    component_count = _face_component_count(mesh.shape[0], edge_owners)
    connected = component_count == 1

    signed_terms = np.einsum(
        "ij,ij->i",
        mesh[:, 0],
        np.cross(mesh[:, 1], mesh[:, 2]),
    )
    signed_volume = float(np.sum(signed_terms, dtype=np.float64) / 6.0)
    positive_volume = signed_volume > positive_volume_tolerance

    z_min = float(np.min(flat[:, 2]))
    z_max = float(np.max(flat[:, 2]))
    z_min_error = abs(z_min + 1.0)
    z_max_error = abs(z_max - 1.0)
    z_extrema = z_min_error <= z_tolerance and z_max_error <= z_tolerance

    radii = np.linalg.norm(flat[:, :2], axis=1)
    max_radius = float(np.max(radii))
    radial_error = abs(max_radius - cylinder_radius)
    cylinder_containment = (
        max_radius <= cylinder_radius + cylinder_containment_tolerance
    )
    cylinder_contact = radial_error <= cylinder_contact_tolerance

    checks: dict[str, bool | None] = {
        "finite": True,
        "nondegenerate": True,
        "edge_manifold": edge_manifold,
        "watertight": watertight,
        "opposite_edge_orientation": opposite_orientation,
        "connected": connected,
        "positive_signed_volume": positive_volume,
        "z_extrema": z_extrema,
        "cylinder_containment": cylinder_containment,
        "cylinder_contact": cylinder_contact,
        "self_intersection_free": None,
    }
    errors: list[str] = []
    if not edge_manifold:
        errors.append(f"mesh has {nonmanifold_edges} non-manifold edge(s)")
    if not watertight:
        errors.append(
            "mesh is not watertight "
            f"({boundary_edges} boundary edge(s), {nonmanifold_edges} edge(s) "
            "with more than two incident faces)"
        )
    if not opposite_orientation:
        errors.append(
            f"mesh has {orientation_mismatches} edge(s) without opposite facet orientation"
        )
    if not connected:
        errors.append(f"mesh has {component_count} disconnected face components")
    if not positive_volume:
        errors.append(
            f"signed volume {signed_volume:.12g} is not greater than "
            f"{positive_volume_tolerance:.12g}"
        )
    if not z_extrema:
        errors.append(
            f"z extrema [{z_min:.12g}, {z_max:.12g}] do not match [-1, 1] "
            f"within {z_tolerance:.12g}"
        )
    if not cylinder_containment:
        errors.append(
            f"maximum xy radius {max_radius:.12g} exceeds cylinder radius "
            f"{cylinder_radius:.12g} plus tolerance "
            f"{cylinder_containment_tolerance:.12g}"
        )
    if not cylinder_contact:
        errors.append(
            f"maximum xy radius {max_radius:.12g} does not contact cylinder radius "
            f"{cylinder_radius:.12g} within {cylinder_contact_tolerance:.12g}"
        )

    metrics: dict[str, int | float] = {
        "triangle_count": int(mesh.shape[0]),
        "unique_vertex_count": int(unique_vertices.shape[0]),
        "edge_count": int(len(edge_owners)),
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "orientation_mismatch_edge_count": int(orientation_mismatches),
        "face_component_count": component_count,
        "signed_volume": signed_volume,
        "z_min": z_min,
        "z_max": z_max,
        "z_min_error": z_min_error,
        "z_max_error": z_max_error,
        "xy_radius_max": max_radius,
        "cylinder_radius": cylinder_radius,
        "cylinder_contact_error": radial_error,
    }
    return SubmissionMeshValidation(
        valid=not errors,
        errors=tuple(errors),
        checks=checks,
        metrics=metrics,
    )



def _face_component_count(
    n_faces: int,
    edge_owners: dict[tuple[int, int], list[tuple[int, int, int]]],
) -> int:
    adjacency: list[set[int]] = [set() for _ in range(n_faces)]
    for owners in edge_owners.values():
        faces = [owner[0] for owner in owners]
        if len(faces) < 2:
            continue
        first = faces[0]
        for other in faces[1:]:
            adjacency[first].add(other)
            adjacency[other].add(first)

    remaining = set(range(n_faces))
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            face = stack.pop()
            neighbors = adjacency[face] & remaining
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
    return components



def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result



def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result

