"""Independent numerical convex-boundary certificate for serialized STLs.

This is a sufficient check specialized to convex submissions, not a generic
triangle-intersection algorithm. In exact arithmetic an outward, closed
triangle 2-cycle supported on the boundary of its full-dimensional convex hull
has constant positive covering multiplicity. Agreement with the hull volume
forces that multiplicity to be one. Hence its triangles cover the hull once,
without folds or interior overlaps. We also check area and vertex links.

All geometric comparisons here use reported floating-point tolerances. This
is a numerical certificate, not a proof using exact predicates. The input is
the STL as read back from disk, including its serialization error. Features
or intersections below the reported tolerances remain unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .stl import load_stl_triangles
from .submission_validation import validate_submission_mesh


def _vertex_links_are_cycles(faces: np.ndarray, vertex_count: int) -> bool:
    """Each vertex of a closed manifold must have one circular triangle fan."""
    for vertex in range(vertex_count):
        incident = faces[np.any(faces == vertex, axis=1)]
        adjacency: dict[int, list[int]] = {}
        for face in incident:
            a, b = (int(v) for v in face if v != vertex)
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
            return False
        remaining = set(adjacency)
        stack = [remaining.pop()]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        if remaining:
            return False
    return True


def certify_convex_mesh(
    triangles: np.ndarray,
    *,
    cylinder_radius: float,
    relative_plane_tolerance: float = 1e-7,
    relative_measure_tolerance: float = 1e-7,
) -> dict:
    """Return a fail-closed certificate for topology, HAC frame and convexity."""
    for value in (relative_plane_tolerance, relative_measure_tolerance):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("certificate tolerances must be positive and finite")
    structural = validate_submission_mesh(triangles, cylinder_radius=cylinder_radius)
    payload = {
        "schema_version": 1,
        "valid": False,
        "method": "outward_convex_boundary_single_cover",
        "arithmetic": "float64 numerical certificate; no exact-predicate claim",
        "structural": structural.to_dict(),
        "checks": {},
        "metrics": {},
        "tolerances": {
            "relative_support_plane": relative_plane_tolerance,
            "relative_area_and_volume": relative_measure_tolerance,
        },
        "errors": list(structural.errors),
        "non_self_intersection": {
            "certified_numerically": False,
            "scope": "convex-boundary single-cover check at stated tolerances",
            "generic_triangle_intersection_test": False,
            "limitation": "Nonconvex features or intersections below the reported tolerances remain unresolved.",
        },
    }
    if not structural.valid:
        return payload
    mesh = np.asarray(triangles, dtype=np.float64)
    vertices, inverse = np.unique(mesh.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    # Centering improves conditioning without changing the STL or frame checks.
    origin = vertices.mean(axis=0)
    points = vertices - origin
    centered = mesh - origin
    length_scale = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    plane_tolerance = relative_plane_tolerance * length_scale
    crosses = np.cross(centered[:, 1] - centered[:, 0], centered[:, 2] - centered[:, 0])
    twice_areas = np.linalg.norm(crosses, axis=1)
    normals = crosses / twice_areas[:, None]
    supports = np.einsum("ij,ij->i", normals, centered[:, 0])
    max_support_violation = float(np.max(normals @ points.T - supports[:, None]))
    area = float(twice_areas.sum() / 2)
    volume = float(
        np.einsum("ij,ij->i", centered[:, 0], np.cross(centered[:, 1], centered[:, 2])).sum() / 6
    )
    try:
        hull = ConvexHull(points)
    except QhullError as exc:
        payload["errors"].append(f"independent convex hull failed: {exc}")
        return payload
    # Test triangle vertices against independent hull planes. A normal computed
    # from a very thin STL triangle amplifies decimal-rounding error; distances
    # to the hull planes measure the actual geometric defect instead.
    distances = (
        np.einsum("fvc,hc->fhv", centered, hull.equations[:, :3]) + hull.equations[None, :, 3, None]
    )
    plane_residuals = np.max(np.abs(distances), axis=2)
    nearest_planes = np.argmin(plane_residuals, axis=1)
    triangle_boundary_distances = plane_residuals[np.arange(len(mesh)), nearest_planes]
    orientation = np.einsum("ij,ij->i", normals, hull.equations[nearest_planes, :3])
    maximum_boundary_distance = float(np.max(triangle_boundary_distances))
    volume_error = abs(volume - hull.volume) / hull.volume
    area_error = abs(area - hull.area) / hull.area
    euler = len(vertices) - structural.metrics["edge_count"] + len(faces)
    checks = {
        "outward_supporting_faces": maximum_boundary_distance <= plane_tolerance
        and bool(np.all(orientation > 0)),
        "convex_hull_volume_agreement": volume_error <= relative_measure_tolerance,
        "convex_hull_area_agreement": area_error <= relative_measure_tolerance,
        "spherical_topology": euler == 2,
        "manifold_vertex_links": _vertex_links_are_cycles(faces, len(vertices)),
        "no_duplicate_triangles": len(np.unique(np.sort(faces, axis=1), axis=0)) == len(faces),
    }
    payload["checks"] = {name: bool(passed) for name, passed in checks.items()}
    payload["metrics"] = {
        "length_scale": length_scale,
        "support_plane_absolute_tolerance": plane_tolerance,
        "maximum_support_plane_violation": max_support_violation,
        "maximum_triangle_to_hull_plane_distance": maximum_boundary_distance,
        "minimum_alignment_with_hull_normal": float(np.min(orientation)),
        "mesh_volume": volume,
        "convex_hull_volume": float(hull.volume),
        "relative_volume_error": float(volume_error),
        "mesh_area": area,
        "convex_hull_area": float(hull.area),
        "relative_area_error": float(area_error),
        "euler_characteristic": int(euler),
    }
    payload["errors"].extend(name for name, passed in checks.items() if not passed)
    payload["valid"] = all(checks.values())
    payload["non_self_intersection"]["certified_numerically"] = bool(payload["valid"])
    return payload


def certify_stl(path: Path, *, cylinder_radius: float) -> dict:
    """Bind the numerical certificate to the exact serialized file bytes."""
    path = Path(path)
    before = path.read_bytes()
    certificate = certify_convex_mesh(load_stl_triangles(path), cylinder_radius=cylinder_radius)
    if path.read_bytes() != before:
        raise RuntimeError("STL changed during certification")
    certificate["stl"] = {
        "name": path.name,
        "sha256": hashlib.sha256(before).hexdigest(),
        "size_bytes": len(before),
    }
    return certificate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl", type=Path)
    parser.add_argument("--radius", required=True, type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    certificate = certify_stl(args.stl, cylinder_radius=args.radius)
    encoded = json.dumps(certificate, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if certificate["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
