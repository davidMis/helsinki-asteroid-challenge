"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations, product
import numpy as np
from .stl import MeshArrays, mesh_from_triangles


@dataclass(frozen=True)
class VariationalMinkowskiConfig:
    """Numerical policy for :func:`solve_variational_minkowski`."""

    area_log_rmse_tolerance: float = 1.0e-5
    max_newton_iterations: int = 30
    max_continuation_stages: int = 160
    initial_continuation_step: float = 0.08
    minimum_continuation_step: float = 2.5e-5
    continuation_growth: float = 1.5
    max_line_search_steps: int = 24
    armijo_fraction: float = 1.0e-4
    minimum_support_fraction: float = 1.0e-9
    closure_tolerance: float = 1.0e-10
    normal_rank_tolerance: float = 1.0e-10
    vertex_merge_tolerance: float = 2.0e-9
    qhull_options: str | None = "Qx"

    def __post_init__(self) -> None:
        if self.area_log_rmse_tolerance <= 0.0:
            raise ValueError("area_log_rmse_tolerance must be positive")
        if self.max_newton_iterations < 1:
            raise ValueError("max_newton_iterations must be positive")
        if self.max_continuation_stages < 1:
            raise ValueError("max_continuation_stages must be positive")
        if not 0.0 < self.initial_continuation_step <= 1.0:
            raise ValueError("initial_continuation_step must lie in (0, 1]")
        if not 0.0 < self.minimum_continuation_step <= self.initial_continuation_step:
            raise ValueError(
                "minimum_continuation_step must be positive and no larger than initial"
            )
        if self.continuation_growth <= 1.0:
            raise ValueError("continuation_growth must exceed one")
        if self.max_line_search_steps < 1:
            raise ValueError("max_line_search_steps must be positive")
        if not 0.0 < self.armijo_fraction < 1.0:
            raise ValueError("armijo_fraction must lie in (0, 1)")
        if self.minimum_support_fraction <= 0.0:
            raise ValueError("minimum_support_fraction must be positive")
        if self.closure_tolerance <= 0.0:
            raise ValueError("closure_tolerance must be positive")
        if self.normal_rank_tolerance <= 0.0:
            raise ValueError("normal_rank_tolerance must be positive")
        if self.vertex_merge_tolerance <= 0.0:
            raise ValueError("vertex_merge_tolerance must be positive")



@dataclass(frozen=True)
class VariationalMinkowskiDiagnostics:
    """Convergence and geometry checks for a variational reconstruction."""

    converged: bool
    continuation_parameter: float
    continuation_stages: int
    rejected_stages: int
    newton_iterations: int
    line_search_rejections: int
    area_log_rmse: float
    target_weighted_log_rmse: float
    area_relative_rmse: float
    area_mass_l1: float
    area_total_variation: float
    area_hellinger: float
    area_vector_angle_sine: float
    minimum_area_ratio: float
    target_area_dynamic_range: float
    reconstructed_area_dynamic_range: float
    minimum_normal_separation_degrees: float
    target_closure_norm: float
    reconstructed_closure_norm: float
    translation_gauge_norm: float
    volume_centroid_before_centering: tuple[float, float, float]
    volume_centroid_after_centering_norm: float
    target_surface_area: float
    reconstructed_surface_area: float
    volume: float
    mixed_width: float
    euler_identity_relative_error: float
    minimum_support: float
    n_facets: int
    n_vertices: int
    n_triangles: int



@dataclass(frozen=True)
class VariationalMinkowskiResult:
    """A convex mesh and the complete support-number reconstruction state."""

    mesh: MeshArrays
    supports: np.ndarray
    target_areas: np.ndarray
    target_normals: np.ndarray
    reconstructed_areas: np.ndarray
    vertices: np.ndarray
    faces: tuple[np.ndarray, ...]
    diagnostics: VariationalMinkowskiDiagnostics



@dataclass(frozen=True)
class _SupportGeometry:
    vertices: np.ndarray
    faces: tuple[np.ndarray, ...]
    triangles: np.ndarray
    facet_areas: np.ndarray
    edge_lengths: np.ndarray
    volume: float



class VariationalMinkowskiError(RuntimeError):
    """Raised when the strict all-facet reconstruction cannot be certified."""



def solve_variational_minkowski(
    areas: np.ndarray,
    normals: np.ndarray,
    *,
    config: VariationalMinkowskiConfig | None = None,
) -> VariationalMinkowskiResult:
    """Recover the unique (up to translation) polytope with the given measure.

    Parameters
    ----------
    areas, normals:
        Strictly positive facet areas and corresponding outward unit normals.
        The measure must satisfy ``sum_i areas[i] * normals[i] == 0`` within
        ``config.closure_tolerance``.  Inputs are never silently modified.
    config:
        Continuation and certification policy.

    Raises
    ------
    ValueError
        If the prescribed measure violates the discrete Minkowski hypotheses.
    VariationalMinkowskiError
        If continuation cannot reach the requested measure while preserving all
        facets or if the final reconstruction misses the requested tolerance.
    """
    cfg = config or VariationalMinkowskiConfig()
    target, unit_normals = _validate_measure(areas, normals, cfg)
    n_facets = target.size
    target_surface_area = float(np.sum(target))

    # An equal-support body is strictly feasible for a spherical normal grid.
    # Scale it so that its actual (not continuum-approximated) area equals the
    # prescribed area.  Its facet measure supplies a closed continuation start.
    supports = np.full(
        n_facets,
        np.sqrt(target_surface_area / (4.0 * np.pi)),
        dtype=np.float64,
    )
    geometry = _geometry_from_positive_supports(unit_normals, supports, cfg)
    supports *= np.sqrt(target_surface_area / float(np.sum(geometry.facet_areas)))
    geometry = _geometry_from_positive_supports(unit_normals, supports, cfg)
    initial_areas = geometry.facet_areas.copy()
    translation_gauge = unit_normals.T @ supports

    tau = 0.0
    step = cfg.initial_continuation_step
    stages = 0
    rejected_stages = 0
    total_iterations = 0
    line_search_rejections = 0

    while tau < 1.0 - 8.0 * np.finfo(np.float64).eps:
        if stages + rejected_stages >= cfg.max_continuation_stages:
            raise VariationalMinkowskiError(
                "continuation stage limit reached before the target measure"
            )
        proposed_tau = min(1.0, tau + step)
        stage_target = (1.0 - proposed_tau) * initial_areas + proposed_tau * target
        solved = _newton_stage(
            supports,
            unit_normals,
            stage_target,
            cfg,
        )
        total_iterations += solved.iterations
        line_search_rejections += solved.line_search_rejections
        if solved.converged:
            supports = solved.supports
            geometry = solved.geometry
            tau = proposed_tau
            stages += 1
            step = min(1.0 - tau, step * cfg.continuation_growth)
            continue

        rejected_stages += 1
        step *= 0.5
        if step < cfg.minimum_continuation_step:
            raise VariationalMinkowskiError(
                "continuation stalled at "
                f"tau={tau:.8f}; proposed tau={proposed_tau:.8f}, "
                f"stage log-area RMSE={solved.area_log_rmse:.6g}"
            )

    # Re-evaluate without relying on a cached line-search state, then certify.
    geometry = _geometry_from_positive_supports(unit_normals, supports, cfg)
    area_log_rmse = _area_log_rmse(geometry.facet_areas, target)
    if area_log_rmse > cfg.area_log_rmse_tolerance:
        raise VariationalMinkowskiError(
            "final area fidelity failed: "
            f"log-area RMSE {area_log_rmse:.6g} exceeds "
            f"{cfg.area_log_rmse_tolerance:.6g}"
        )

    volume_centroid = _volume_centroid(geometry.triangles)
    triangles = geometry.triangles - volume_centroid
    vertices = geometry.vertices - volume_centroid
    centered_supports = supports - unit_normals @ volume_centroid
    if np.any(centered_supports <= 0.0):
        raise VariationalMinkowskiError("volume-centroid gauge produced non-positive supports")
    # Challenge framing is deliberately outside this solver.  Applying an
    # anisotropic radius transform here would make supports, areas, and volume
    # refer to a different body than the returned mesh.
    mesh = mesh_from_triangles(triangles)
    relative = (geometry.facet_areas - target) / target
    target_probability = target / np.sum(target)
    reconstructed_probability = geometry.facet_areas / np.sum(geometry.facet_areas)
    log_ratio = np.log(geometry.facet_areas) - np.log(target)
    target_closure = target @ unit_normals
    reconstructed_closure = geometry.facet_areas @ unit_normals
    support_gauge = unit_normals.T @ supports
    area_cosine = float(
        (geometry.facet_areas @ target)
        / (np.linalg.norm(geometry.facet_areas) * np.linalg.norm(target))
    )
    off_diagonal_gram = unit_normals @ unit_normals.T
    np.fill_diagonal(off_diagonal_gram, -1.0)
    diagnostics = VariationalMinkowskiDiagnostics(
        converged=True,
        continuation_parameter=float(tau),
        continuation_stages=int(stages),
        rejected_stages=int(rejected_stages),
        newton_iterations=int(total_iterations),
        line_search_rejections=int(line_search_rejections),
        area_log_rmse=float(area_log_rmse),
        target_weighted_log_rmse=float(np.sqrt(np.sum(target_probability * log_ratio**2))),
        area_relative_rmse=float(np.sqrt(np.mean(relative**2))),
        area_mass_l1=float(np.sum(np.abs(geometry.facet_areas - target)) / target_surface_area),
        area_total_variation=float(
            0.5 * np.sum(np.abs(reconstructed_probability - target_probability))
        ),
        area_hellinger=float(
            np.linalg.norm(np.sqrt(reconstructed_probability) - np.sqrt(target_probability))
            / np.sqrt(2.0)
        ),
        area_vector_angle_sine=float(np.sqrt(max(0.0, 1.0 - np.clip(area_cosine, -1.0, 1.0) ** 2))),
        minimum_area_ratio=float(np.min(geometry.facet_areas / target)),
        target_area_dynamic_range=float(np.max(target) / np.min(target)),
        reconstructed_area_dynamic_range=float(
            np.max(geometry.facet_areas) / np.min(geometry.facet_areas)
        ),
        minimum_normal_separation_degrees=float(
            np.degrees(np.arccos(np.clip(np.max(off_diagonal_gram), -1.0, 1.0)))
        ),
        target_closure_norm=float(np.linalg.norm(target_closure)),
        reconstructed_closure_norm=float(np.linalg.norm(reconstructed_closure)),
        translation_gauge_norm=float(np.linalg.norm(support_gauge - translation_gauge)),
        volume_centroid_before_centering=(
            float(volume_centroid[0]),
            float(volume_centroid[1]),
            float(volume_centroid[2]),
        ),
        volume_centroid_after_centering_norm=float(
            np.linalg.norm(_volume_centroid(geometry.triangles - volume_centroid))
        ),
        target_surface_area=target_surface_area,
        reconstructed_surface_area=float(np.sum(geometry.facet_areas)),
        volume=float(geometry.volume),
        mixed_width=float(target @ centered_supports),
        euler_identity_relative_error=float(
            abs(float(geometry.facet_areas @ centered_supports) - 3.0 * geometry.volume)
            / max(
                abs(float(geometry.facet_areas @ centered_supports)),
                3.0 * geometry.volume,
                np.finfo(float).tiny,
            )
        ),
        minimum_support=float(np.min(centered_supports)),
        n_facets=n_facets,
        n_vertices=int(vertices.shape[0]),
        n_triangles=int(mesh.n_triangles),
    )
    return VariationalMinkowskiResult(
        mesh=mesh,
        supports=centered_supports.copy(),
        target_areas=target.copy(),
        target_normals=unit_normals.copy(),
        reconstructed_areas=geometry.facet_areas.copy(),
        vertices=vertices.copy(),
        faces=tuple(face.copy() for face in geometry.faces),
        diagnostics=diagnostics,
    )



@dataclass(frozen=True)
class _StageResult:
    supports: np.ndarray
    geometry: _SupportGeometry
    converged: bool
    iterations: int
    line_search_rejections: int
    area_log_rmse: float



def _newton_stage(
    initial_supports: np.ndarray,
    normals: np.ndarray,
    target_areas: np.ndarray,
    config: VariationalMinkowskiConfig,
) -> _StageResult:
    supports = initial_supports.copy()
    geometry = _geometry_from_positive_supports(normals, supports, config)
    rejections = 0

    for iteration in range(config.max_newton_iterations + 1):
        merit = _area_log_rmse(geometry.facet_areas, target_areas)
        if merit <= config.area_log_rmse_tolerance:
            return _StageResult(supports, geometry, True, iteration, rejections, merit)
        if iteration == config.max_newton_iterations:
            break

        try:
            hessian = _area_jacobian(normals, geometry.edge_lengths)
            delta = _gauge_fixed_newton_step(
                hessian,
                target_areas - geometry.facet_areas,
                normals,
            )
        except VariationalMinkowskiError:
            break
        support_floor = config.minimum_support_fraction * float(np.mean(supports))
        accepted = False
        alpha = 1.0
        for _ in range(config.max_line_search_steps):
            candidate_supports = supports + alpha * delta
            if float(np.min(candidate_supports)) <= support_floor:
                alpha *= 0.5
                rejections += 1
                continue
            try:
                candidate_geometry = _geometry_from_positive_supports(
                    normals,
                    candidate_supports,
                    config,
                )
            except VariationalMinkowskiError:
                alpha *= 0.5
                rejections += 1
                continue
            candidate_merit = _area_log_rmse(candidate_geometry.facet_areas, target_areas)
            required = merit * (1.0 - config.armijo_fraction * alpha)
            if np.isfinite(candidate_merit) and candidate_merit < required:
                supports = candidate_supports
                geometry = candidate_geometry
                accepted = True
                break
            alpha *= 0.5
            rejections += 1
        if not accepted:
            break

    return _StageResult(
        supports,
        geometry,
        False,
        config.max_newton_iterations,
        rejections,
        _area_log_rmse(geometry.facet_areas, target_areas),
    )



def _validate_measure(
    areas: np.ndarray,
    normals: np.ndarray,
    config: VariationalMinkowskiConfig,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(areas, dtype=np.float64).reshape(-1)
    unit_normals = np.asarray(normals, dtype=np.float64)
    if unit_normals.ndim != 2 or unit_normals.shape[1] != 3:
        raise ValueError("normals must have shape (n, 3)")
    if target.size != unit_normals.shape[0]:
        raise ValueError("areas and normals must have matching lengths")
    if target.size < 4:
        raise ValueError("at least four facets are required")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(unit_normals)):
        raise ValueError("areas and normals must be finite")
    if np.any(target <= 0.0):
        raise ValueError("all target facet areas must be strictly positive")
    lengths = np.linalg.norm(unit_normals, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("normal vectors must be nonzero")
    if not np.allclose(
        lengths,
        1.0,
        rtol=0.0,
        atol=10.0 * config.normal_rank_tolerance,
    ):
        raise ValueError("facet normals must be unit length")
    unit_normals = unit_normals / lengths[:, None]
    if np.linalg.matrix_rank(unit_normals, tol=config.normal_rank_tolerance) < 3:
        raise ValueError("facet normals must span three dimensions")
    gram = unit_normals @ unit_normals.T
    np.fill_diagonal(gram, -np.inf)
    if float(np.max(gram)) > 1.0 - config.normal_rank_tolerance:
        raise ValueError("duplicate or numerically indistinguishable normals are not supported")
    surface_area = float(np.sum(target))
    closure_norm = float(np.linalg.norm(target @ unit_normals))
    if closure_norm > config.closure_tolerance * surface_area:
        raise ValueError(
            "area-normal measure is not Minkowski closed: "
            f"relative closure norm {closure_norm / surface_area:.6g} exceeds "
            f"{config.closure_tolerance:.6g}"
        )
    return target, unit_normals



def _geometry_from_positive_supports(
    normals: np.ndarray,
    supports: np.ndarray,
    config: VariationalMinkowskiConfig,
) -> _SupportGeometry:
    """Construct geometry through the polar hull, retaining facet identities."""
    from scipy.spatial import ConvexHull, QhullError

    supports = np.asarray(supports, dtype=np.float64).reshape(-1)
    if supports.size != normals.shape[0] or not np.all(np.isfinite(supports)):
        raise VariationalMinkowskiError("invalid support vector")
    if np.any(supports <= 0.0):
        raise VariationalMinkowskiError("support vector no longer contains the origin strictly")

    dual_points = normals / supports[:, None]
    try:
        hull = ConvexHull(dual_points, qhull_options=config.qhull_options)
    except QhullError as exc:
        raise VariationalMinkowskiError("polar convex hull failed") from exc
    active = np.sort(np.asarray(hull.vertices, dtype=np.int64))
    if active.size != supports.size or not np.array_equal(active, np.arange(supports.size)):
        inactive = np.setdiff1d(np.arange(supports.size), active)
        raise VariationalMinkowskiError(
            f"{inactive.size} prescribed normals became inactive support planes"
        )

    raw_vertices: list[np.ndarray] = []
    simplex_vertices: list[int] = []
    merge_scale = float(np.max(supports))
    merge_tolerance = config.vertex_merge_tolerance * merge_scale
    vertex_bins: dict[tuple[int, int, int], list[int]] = {}
    simplices = np.asarray(hull.simplices, dtype=np.int64)
    for simplex in simplices:
        matrix = normals[simplex]
        try:
            vertex = np.linalg.solve(matrix, supports[simplex])
        except np.linalg.LinAlgError as exc:
            raise VariationalMinkowskiError("singular polar-hull simplex") from exc
        if not np.all(np.isfinite(vertex)):
            raise VariationalMinkowskiError("non-finite primal vertex")
        key = tuple(np.rint(vertex / merge_tolerance).astype(np.int64))
        matched: int | None = None
        for offset in product((-1, 0, 1), repeat=3):
            neighboring_key = tuple(key[axis] + offset[axis] for axis in range(3))
            for candidate in vertex_bins.get(neighboring_key, []):
                if np.linalg.norm(raw_vertices[candidate] - vertex) <= merge_tolerance:
                    matched = candidate
                    break
            if matched is not None:
                break
        if matched is None:
            matched = len(raw_vertices)
            raw_vertices.append(vertex)
            vertex_bins.setdefault(key, []).append(matched)
        simplex_vertices.append(matched)
    vertices = np.asarray(raw_vertices, dtype=np.float64)

    face_vertex_sets: list[set[int]] = [set() for _ in range(supports.size)]
    edge_vertex_sets: dict[tuple[int, int], set[int]] = {}
    for simplex, vertex_index in zip(simplices, simplex_vertices, strict=True):
        for normal_index in simplex:
            face_vertex_sets[int(normal_index)].add(vertex_index)
        for left, right in combinations(sorted(int(i) for i in simplex), 2):
            edge_vertex_sets.setdefault((left, right), set()).add(vertex_index)

    faces: list[np.ndarray] = []
    facet_areas = np.zeros(supports.size, dtype=np.float64)
    triangles: list[np.ndarray] = []
    for index, normal in enumerate(normals):
        vertex_indices = np.asarray(sorted(face_vertex_sets[index]), dtype=np.int64)
        if vertex_indices.size < 3:
            raise VariationalMinkowskiError(f"facet {index} has fewer than three vertices")
        ordered = _order_face(vertex_indices, vertices, normal)
        polygon = vertices[ordered]
        area = _polygon_area(polygon, normal)
        if not np.isfinite(area) or area <= 0.0:
            raise VariationalMinkowskiError(f"facet {index} has non-positive area")
        faces.append(ordered)
        facet_areas[index] = area
        triangles.extend(_triangulate_polygon(polygon, normal))

    edge_lengths = np.zeros((supports.size, supports.size), dtype=np.float64)
    edge_floor = config.vertex_merge_tolerance * merge_scale
    for (left, right), edge_vertex_indices in edge_vertex_sets.items():
        indices = np.asarray(sorted(edge_vertex_indices), dtype=np.int64)
        if indices.size < 2:
            continue
        points = vertices[indices]
        differences = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        length = float(np.max(distances))
        if length <= edge_floor:
            continue
        edge_lengths[left, right] = length
        edge_lengths[right, left] = length

    triangles_array = np.asarray(triangles, dtype=np.float64)
    volume = float(facet_areas @ supports / 3.0)
    if not np.isfinite(volume) or volume <= 0.0:
        raise VariationalMinkowskiError("reconstructed polytope has non-positive volume")
    closure_error = float(np.linalg.norm(facet_areas @ normals))
    if closure_error > 2.0e-7 * float(np.sum(facet_areas)):
        raise VariationalMinkowskiError(
            f"reconstructed facet geometry failed closure ({closure_error:.6g})"
        )
    return _SupportGeometry(
        vertices=vertices,
        faces=tuple(faces),
        triangles=triangles_array,
        facet_areas=facet_areas,
        edge_lengths=edge_lengths,
        volume=volume,
    )



def _area_jacobian(normals: np.ndarray, edge_lengths: np.ndarray) -> np.ndarray:
    """Return ``d area_i / d support_j`` from primal edge lengths."""
    dots = np.clip(normals @ normals.T, -1.0, 1.0)
    sine = np.sqrt(np.maximum(1.0 - dots**2, 0.0))
    jacobian = np.zeros_like(edge_lengths)
    adjacent = edge_lengths > 0.0
    np.fill_diagonal(adjacent, False)
    if np.any(sine[adjacent] <= 1.0e-12):
        raise VariationalMinkowskiError("adjacent facet normals are numerically parallel")
    jacobian[adjacent] = edge_lengths[adjacent] / sine[adjacent]
    jacobian = 0.5 * (jacobian + jacobian.T)
    jacobian[np.diag_indices_from(jacobian)] = -np.sum(jacobian * dots, axis=1)
    return jacobian



def _gauge_fixed_newton_step(
    hessian: np.ndarray,
    area_residual: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """Solve the support Newton equation modulo the translation nullspace."""
    from scipy.linalg import LinAlgWarning, solve
    import warnings

    n_facets = area_residual.size
    kkt = np.zeros((n_facets + 3, n_facets + 3), dtype=np.float64)
    kkt[:n_facets, :n_facets] = hessian
    kkt[:n_facets, n_facets:] = normals
    kkt[n_facets:, :n_facets] = normals.T
    rhs = np.concatenate([area_residual, np.zeros(3, dtype=np.float64)])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            solution = solve(kkt, rhs, assume_a="sym")
    except (np.linalg.LinAlgError, LinAlgWarning, ValueError):
        solution, _, rank, _ = np.linalg.lstsq(kkt, rhs, rcond=1.0e-11)
        if rank < n_facets + 3:
            raise VariationalMinkowskiError(
                "support-area Newton system is rank deficient"
            ) from None
        residual = kkt @ solution - rhs
        relative_residual = np.linalg.norm(residual) / max(
            np.linalg.norm(rhs),
            np.finfo(np.float64).tiny,
        )
        if relative_residual > 1.0e-8:
            raise VariationalMinkowskiError(
                "support-area Newton system has excessive residual"
            ) from None
    delta = np.asarray(solution[:n_facets], dtype=np.float64)
    if not np.all(np.isfinite(delta)):
        raise VariationalMinkowskiError("support-area Newton step is non-finite")
    return delta



def _area_log_rmse(current: np.ndarray, target: np.ndarray) -> float:
    if np.any(current <= 0.0) or np.any(target <= 0.0):
        return float("inf")
    residual = np.log(current) - np.log(target)
    return float(np.sqrt(np.mean(residual**2)))



def _order_face(indices: np.ndarray, vertices: np.ndarray, normal: np.ndarray) -> np.ndarray:
    points = vertices[indices]
    center = np.mean(points, axis=0)
    basis_u, basis_v = _plane_basis(normal)
    centered = points - center
    angles = np.arctan2(centered @ basis_v, centered @ basis_u)
    ordered = indices[np.argsort(angles)]
    polygon = vertices[ordered]
    area_vector = np.sum(np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0)
    if float(area_vector @ normal) < 0.0:
        ordered = ordered[::-1]
    return ordered



def _polygon_area(polygon: np.ndarray, normal: np.ndarray) -> float:
    area_vector = 0.5 * np.sum(np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0)
    return abs(float(area_vector @ normal))



def _triangulate_polygon(polygon: np.ndarray, normal: np.ndarray) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for index in range(1, polygon.shape[0] - 1):
        triangle = np.asarray([polygon[0], polygon[index], polygon[index + 1]])
        cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if np.linalg.norm(cross) <= 1.0e-14:
            continue
        if float(cross @ normal) < 0.0:
            triangle = triangle[[0, 2, 1]]
        result.append(triangle)
    if not result:
        raise VariationalMinkowskiError("facet triangulation produced no triangles")
    return result



def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(reference @ normal)) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    basis_u = np.cross(reference, normal)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    return basis_u, basis_v



def _volume_centroid(triangles: np.ndarray) -> np.ndarray:
    """Return the exact centroid of an oriented closed triangle mesh."""
    triangles = np.asarray(triangles, dtype=np.float64)
    signed_six_volumes = np.einsum(
        "ij,ij->i",
        triangles[:, 0],
        np.cross(triangles[:, 1], triangles[:, 2]),
    )
    signed_volume = float(np.sum(signed_six_volumes) / 6.0)
    if not np.isfinite(signed_volume) or signed_volume <= 0.0:
        raise VariationalMinkowskiError("cannot center a mesh with non-positive signed volume")
    weighted = signed_six_volumes[:, None] * np.sum(triangles, axis=1)
    centroid = np.sum(weighted, axis=0) / (24.0 * signed_volume)
    if not np.all(np.isfinite(centroid)):
        raise VariationalMinkowskiError("volume centroid is non-finite")
    return np.asarray(centroid, dtype=np.float64)

