"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import struct
import tempfile
import numpy as np


@dataclass(frozen=True)
class MeshArrays:
    """Triangle mesh arrays prepared for forward modeling."""

    vertices: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    centroids: np.ndarray
    normalization_offset: np.ndarray
    normalization_scale: float

    @property
    def n_triangles(self) -> int:
        return int(self.normals.shape[0])

    @property
    def total_area(self) -> float:
        return float(np.sum(self.areas))



def load_stl_triangles(path: Path) -> np.ndarray:
    """Load triangles from an ASCII or binary STL file."""
    if not path.exists():
        raise FileNotFoundError(path)
    if _looks_like_binary_stl(path):
        return _load_binary_stl(path)
    return _load_ascii_stl(path)



def mesh_from_triangles(
    triangles: np.ndarray,
    normalization_offset: np.ndarray | None = None,
    normalization_scale: float = 1.0,
) -> MeshArrays:
    """Compute normals, areas, and centroids from triangle vertices."""
    triangles = validate_triangle_array(triangles)
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge_1, edge_2)
    double_area = np.linalg.norm(cross, axis=1)

    normals = cross / double_area[:, None]
    areas = 0.5 * double_area
    centroids = np.mean(triangles, axis=1)
    offset = (
        np.zeros(3, dtype=np.float64)
        if normalization_offset is None
        else np.asarray(normalization_offset, dtype=np.float64)
    )
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("normalization_offset must contain three finite values")
    scale = float(normalization_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization_scale must be finite and positive")
    return MeshArrays(
        vertices=triangles,
        normals=normals,
        areas=areas,
        centroids=centroids,
        normalization_offset=offset,
        normalization_scale=scale,
    )



def validate_triangle_array(
    triangles: np.ndarray,
    *,
    minimum_area: float = 0.0,
) -> np.ndarray:
    """Return a finite, non-empty, nondegenerate ``(n, 3, 3)`` float array.

    Mesh construction and export are deliberately strict. A malformed facet is
    evidence that an upstream geometry calculation failed; silently dropping it
    can turn that failure into a plausible-looking but open STL.
    """
    minimum_area = float(minimum_area)
    if not np.isfinite(minimum_area) or minimum_area < 0.0:
        raise ValueError("minimum_area must be finite and non-negative")

    array = np.asarray(triangles, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(
            f"triangles must have shape (n, 3, 3), received {array.shape}"
        )
    if array.shape[0] == 0:
        raise ValueError("triangle mesh must contain at least one triangle")
    if not np.all(np.isfinite(array)):
        raise ValueError("triangle mesh contains non-finite coordinates")

    with np.errstate(over="ignore", invalid="ignore"):
        cross = np.cross(array[:, 1] - array[:, 0], array[:, 2] - array[:, 0])
        areas = 0.5 * np.linalg.norm(cross, axis=1)
    if not np.all(np.isfinite(cross)) or not np.all(np.isfinite(areas)):
        raise ValueError("triangle geometry overflows finite normal/area calculations")
    degenerate = areas <= minimum_area
    if np.any(degenerate):
        indices = np.flatnonzero(degenerate)
        preview = ", ".join(str(int(index)) for index in indices[:5])
        suffix = "..." if indices.size > 5 else ""
        raise ValueError(
            f"triangle mesh contains {indices.size} degenerate triangle(s) "
            f"at indices {preview}{suffix}"
        )
    return array



def write_ascii_stl(
    path: Path,
    triangles: np.ndarray,
    *,
    solid_name: str = "asteroid",
) -> None:
    """Atomically write triangles to a simple ASCII STL file."""
    mesh = mesh_from_triangles(triangles)
    if not solid_name or "\n" in solid_name or "\r" in solid_name:
        raise ValueError("solid_name must be non-empty and contain no newlines")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"solid {solid_name}\n")
            for normal, triangle in zip(mesh.normals, mesh.vertices, strict=True):
                handle.write(
                    "  facet normal "
                    f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n"
                )
                handle.write("    outer loop\n")
                for vertex in triangle:
                    handle.write(
                        f"      vertex {vertex[0]:.9g} {vertex[1]:.9g} "
                        f"{vertex[2]:.9g}\n"
                    )
                handle.write("    endloop\n")
                handle.write("  endfacet\n")
            handle.write(f"endsolid {solid_name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
        temporary_path.chmod(mode)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)



def _looks_like_binary_stl(path: Path) -> bool:
    size = path.stat().st_size
    if size < 84:
        return False
    with path.open("rb") as handle:
        handle.seek(80)
        n_triangles = struct.unpack("<I", handle.read(4))[0]
    return size == 84 + 50 * n_triangles



def _load_binary_stl(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        handle.seek(80)
        n_triangles = struct.unpack("<I", handle.read(4))[0]
        data = np.frombuffer(handle.read(50 * n_triangles), dtype=np.uint8)
    records = data.reshape(n_triangles, 50)
    floats = records[:, :48].copy().view("<f4").reshape(n_triangles, 12)
    return floats[:, 3:].reshape(n_triangles, 3, 3).astype(np.float64)



def _load_ascii_stl(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("vertex "):
                continue
            parts = stripped.split()
            if len(parts) != 4:
                raise ValueError(f"invalid STL vertex line in {path}: {line!r}")
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(vertices) % 3 != 0:
        raise ValueError(f"STL vertex count is not divisible by 3: {path}")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)

