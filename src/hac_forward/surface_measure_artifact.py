"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping
import numpy as np
from .io_utils import file_provenance, read_json, write_json


SCHEMA_NAME = "hac.surface_measure"



SCHEMA_VERSION = 1



NPZ_FILENAME = "surface_measure.npz"



METADATA_FILENAME = "surface_measure.json"



MASS_ATOL = 1.0e-10



NORMAL_ATOL = 1.0e-10



CLOSURE_ATOL = 1.0e-10



_REPRESENTATION_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")



_HARMONIC_COMPONENTS = frozenset({"zonal", "cos", "sin"})



_ARRAY_DTYPES = {
    "masses": "<f8",
    "normals": "<f8",
    "harmonic_coefficients": "<f8",
    "harmonic_degrees": "<i8",
    "harmonic_orders": "<i8",
    "harmonic_components": "<U5",
    "quadrature_masses": "<f8",
    "reference_masses": "<f8",
    "assignments": "<i8",
    "cluster_source_masses": "<f8",
}



_HARMONIC_BASIS_CONVENTION = {
    "family": "real associated Legendre basis without the Condon--Shortley phase",
    "ordering": "order m outer, degree l inner; zonal for m=0, then cos and sin",
    "centering": "quadrature-mass weighted mean removed from every active column",
    "normalization": "each centered column has unit quadrature-mass weighted L2 norm",
    "free_degrees": "stored degree/order arrays identify the active modes",
    "closure": "masses include an exponential degree-1 closure tilt not stored as a coefficient",
}



@dataclass(frozen=True)
class SurfaceMeasureArtifact:
    """In-memory representation of one surface-measure artifact.

    ``masses`` and ``normals`` are the phase boundary: downstream code must not
    need harmonic coordinates to solve the Minkowski problem.  The remaining
    fields make the result reproducible and, when present, allow the dense
    harmonic representation to be reconstructed exactly.
    """

    model_id: int
    representation: str
    masses: np.ndarray
    normals: np.ndarray
    config: Mapping[str, Any]
    input_data: Mapping[str, Any]
    seed: int
    diagnostics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    harmonic_coefficients: np.ndarray | None = None
    harmonic_degrees: np.ndarray | None = None
    harmonic_orders: np.ndarray | None = None
    harmonic_components: np.ndarray | None = None
    quadrature_masses: np.ndarray | None = None
    reference_masses: np.ndarray | None = None
    assignments: np.ndarray | None = None
    cluster_source_masses: np.ndarray | None = None



def validate_surface_measure_artifact(artifact: SurfaceMeasureArtifact) -> None:
    """Validate the complete in-memory artifact contract.

    The closure and normalization conditions are exact mathematical constraints
    represented with conservative float64 tolerances.  Invalid artifacts fail
    before any files are published.
    """

    if isinstance(artifact.model_id, bool) or int(artifact.model_id) != artifact.model_id:
        raise ValueError("model_id must be a positive integer")
    if int(artifact.model_id) < 1:
        raise ValueError("model_id must be a positive integer")
    if not isinstance(artifact.representation, str) or not _REPRESENTATION_PATTERN.fullmatch(
        artifact.representation
    ):
        raise ValueError("representation must match ^[a-z][a-z0-9_]*$")
    if isinstance(artifact.seed, bool) or int(artifact.seed) != artifact.seed:
        raise ValueError("seed must be a non-negative integer")
    if int(artifact.seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    for name in ("config", "input_data", "diagnostics", "provenance"):
        value = getattr(artifact, name)
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"{name} must be a nonempty mapping")

    masses = _float_vector("masses", artifact.masses)
    normals = np.asarray(artifact.normals, dtype=np.float64)
    if normals.shape != (masses.size, 3):
        raise ValueError(f"normals must have shape ({masses.size}, 3), got {normals.shape}")
    if masses.size < 4:
        raise ValueError("a surface measure must contain at least four atoms")
    if not np.all(np.isfinite(normals)):
        raise ValueError("normals must contain only finite values")
    if np.any(masses <= 0.0):
        raise ValueError("masses must be strictly positive")
    if not np.isclose(float(np.sum(masses)), 1.0, atol=MASS_ATOL, rtol=0.0):
        raise ValueError("masses must sum to one")
    lengths = np.linalg.norm(normals, axis=1)
    if not np.allclose(lengths, 1.0, atol=NORMAL_ATOL, rtol=0.0):
        raise ValueError("normals must be unit vectors")
    closure = masses @ normals
    if float(np.linalg.norm(closure)) > CLOSURE_ATOL:
        raise ValueError(
            "surface measure must satisfy Minkowski closure; "
            f"norm={float(np.linalg.norm(closure)):.3e}"
        )

    harmonic_values = (
        artifact.harmonic_coefficients,
        artifact.harmonic_degrees,
        artifact.harmonic_orders,
        artifact.harmonic_components,
    )
    if any(value is not None for value in harmonic_values) and not all(
        value is not None for value in harmonic_values
    ):
        raise ValueError(
            "harmonic coefficients, degrees, orders, and components must be provided together"
        )
    if artifact.harmonic_coefficients is not None:
        if artifact.quadrature_masses is None or artifact.reference_masses is None:
            raise ValueError("harmonic artifacts require quadrature_masses and reference_masses")
        coefficients = _float_vector("harmonic_coefficients", artifact.harmonic_coefficients)
        degrees = _integer_vector("harmonic_degrees", artifact.harmonic_degrees)
        orders = _integer_vector("harmonic_orders", artifact.harmonic_orders)
        components = np.asarray(artifact.harmonic_components)
        if components.ndim != 1 or components.shape != coefficients.shape:
            raise ValueError("harmonic_components must match harmonic_coefficients")
        if components.dtype.kind not in {"U", "S"}:
            raise ValueError("harmonic_components must contain strings")
        component_strings = components.astype(str)
        if degrees.shape != coefficients.shape or orders.shape != coefficients.shape:
            raise ValueError("harmonic degree/order metadata must match coefficients")
        if np.any(degrees < 0) or np.any(orders < 0) or np.any(orders > degrees):
            raise ValueError("invalid harmonic degree/order metadata")
        if not set(component_strings.tolist()).issubset(_HARMONIC_COMPONENTS):
            raise ValueError("unknown harmonic component label")
        expected_zonal = orders == 0
        if not np.array_equal(component_strings == "zonal", expected_zonal):
            raise ValueError("zonal harmonic components must have order zero, and conversely")
        if np.any((orders > 0) & (component_strings == "zonal")):
            raise ValueError("positive harmonic orders require cos or sin components")

    for name in ("quadrature_masses", "reference_masses"):
        value = getattr(artifact, name)
        if value is None:
            continue
        weights = _float_vector(name, value)
        if weights.shape != masses.shape:
            raise ValueError(f"{name} must match masses")
        if np.any(weights <= 0.0):
            raise ValueError(f"{name} must be strictly positive")
        if not np.isclose(float(np.sum(weights)), 1.0, atol=MASS_ATOL, rtol=0.0):
            raise ValueError(f"{name} must sum to one")

    atomic_lineage = (artifact.assignments, artifact.cluster_source_masses)
    if any(value is not None for value in atomic_lineage) and not all(
        value is not None for value in atomic_lineage
    ):
        raise ValueError("assignments and cluster_source_masses must be provided together")
    if artifact.assignments is not None:
        assignments = _integer_vector("assignments", artifact.assignments)
        if assignments.size < masses.size:
            raise ValueError("assignments must contain at least one source atom per output atom")
        if np.any(assignments < 0) or np.any(assignments >= masses.size):
            raise ValueError("assignments must index output atoms")
        if not np.array_equal(np.unique(assignments), np.arange(masses.size)):
            raise ValueError("assignments must use every output atom")
        source_masses = _float_vector("cluster_source_masses", artifact.cluster_source_masses)
        if source_masses.shape != masses.shape:
            raise ValueError("cluster_source_masses must match output masses")
        if np.any(source_masses <= 0.0):
            raise ValueError("cluster_source_masses must be strictly positive")
        if not np.isclose(float(np.sum(source_masses)), 1.0, atol=MASS_ATOL, rtol=0.0):
            raise ValueError("cluster_source_masses must sum to one")



def write_surface_measure_artifact(
    directory: Path,
    artifact: SurfaceMeasureArtifact,
) -> dict[str, Any]:
    """Atomically create ``directory`` and write a validated artifact there.

    Existing directories are never overwritten.  The returned mapping is the
    exact JSON payload written to ``surface_measure.json``.
    """

    validate_surface_measure_artifact(artifact)
    destination = Path(directory).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    published = False
    try:
        arrays = _artifact_arrays(artifact)
        npz_path = staging / NPZ_FILENAME
        with npz_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
            handle.flush()
            os.fsync(handle.fileno())
        npz_record = file_provenance(npz_path, relative_to=staging)
        metadata = _metadata_payload(artifact, arrays, npz_record)
        write_json(
            staging / METADATA_FILENAME,
            metadata,
            default=_json_default,
        )
        # Exercise the same strict loader used by downstream phases before the
        # directory becomes visible at its final path.
        load_surface_measure_artifact(staging)
        os.replace(staging, destination)
        published = True
        return metadata
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)



def load_surface_measure_artifact(directory: Path) -> SurfaceMeasureArtifact:
    """Load and fully validate a surface-measure artifact directory."""

    root = Path(directory).resolve()
    metadata_path = root / METADATA_FILENAME
    npz_path = root / NPZ_FILENAME
    if not metadata_path.is_file():
        raise ValueError(f"missing artifact metadata: {metadata_path}")
    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError("surface-measure metadata must be a JSON object")
    schema = metadata.get("schema")
    if schema != {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}:
        raise ValueError(
            f"unsupported surface-measure schema: {schema!r}; "
            f"expected {SCHEMA_NAME!r} version {SCHEMA_VERSION}"
        )
    npz_metadata = metadata.get("npz")
    if not isinstance(npz_metadata, dict) or npz_metadata.get("path") != NPZ_FILENAME:
        raise ValueError("artifact metadata must reference surface_measure.npz")
    actual_record = file_provenance(npz_path, relative_to=root)
    for key in ("path", "size_bytes", "sha256"):
        if actual_record[key] != npz_metadata.get(key):
            raise ValueError(f"surface_measure.npz {key} does not match metadata")

    array_metadata = metadata.get("arrays")
    if not isinstance(array_metadata, dict) or not array_metadata:
        raise ValueError("artifact array metadata must be a nonempty object")
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != set(array_metadata):
            raise ValueError("NPZ array set does not match metadata")
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    if not {"masses", "normals"}.issubset(arrays):
        raise ValueError("NPZ must contain masses and normals")
    unknown_arrays = sorted(set(arrays) - set(_ARRAY_DTYPES))
    if unknown_arrays:
        raise ValueError(f"NPZ contains unsupported arrays: {unknown_arrays}")
    for name, values in arrays.items():
        record = array_metadata.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"missing array metadata for {name}")
        if record.get("shape") != list(values.shape):
            raise ValueError(f"array shape metadata mismatch for {name}")
        if record.get("dtype") != values.dtype.str:
            raise ValueError(f"array dtype metadata mismatch for {name}")
        if values.dtype.str != _ARRAY_DTYPES[name]:
            raise ValueError(f"array dtype is not canonical for {name}")

    required_top_level = (
        "model_id",
        "representation",
        "config",
        "input",
        "seed",
        "diagnostics",
        "provenance",
    )
    missing = [name for name in required_top_level if name not in metadata]
    if missing:
        raise ValueError(f"artifact metadata is missing fields: {', '.join(missing)}")
    artifact = SurfaceMeasureArtifact(
        model_id=metadata["model_id"],
        representation=metadata["representation"],
        masses=arrays["masses"],
        normals=arrays["normals"],
        config=metadata["config"],
        input_data=metadata["input"],
        seed=metadata["seed"],
        diagnostics=metadata["diagnostics"],
        provenance=metadata["provenance"],
        harmonic_coefficients=arrays.get("harmonic_coefficients"),
        harmonic_degrees=arrays.get("harmonic_degrees"),
        harmonic_orders=arrays.get("harmonic_orders"),
        harmonic_components=arrays.get("harmonic_components"),
        quadrature_masses=arrays.get("quadrature_masses"),
        reference_masses=arrays.get("reference_masses"),
        assignments=arrays.get("assignments"),
        cluster_source_masses=arrays.get("cluster_source_masses"),
    )
    validate_surface_measure_artifact(artifact)
    if metadata.get("measure") != _measure_metadata(artifact.masses, artifact.normals):
        raise ValueError("surface-measure summary metadata does not match arrays")
    if metadata.get("harmonic_basis") != _harmonic_basis_metadata(artifact):
        raise ValueError("harmonic-basis metadata does not match arrays")
    return artifact



def _artifact_arrays(artifact: SurfaceMeasureArtifact) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "masses": np.asarray(artifact.masses, dtype=_ARRAY_DTYPES["masses"]),
        "normals": np.asarray(artifact.normals, dtype=_ARRAY_DTYPES["normals"]),
    }
    optional = {
        "harmonic_coefficients": artifact.harmonic_coefficients,
        "harmonic_degrees": artifact.harmonic_degrees,
        "harmonic_orders": artifact.harmonic_orders,
        "harmonic_components": artifact.harmonic_components,
        "quadrature_masses": artifact.quadrature_masses,
        "reference_masses": artifact.reference_masses,
        "assignments": artifact.assignments,
        "cluster_source_masses": artifact.cluster_source_masses,
    }
    for name, value in optional.items():
        if value is None:
            continue
        arrays[name] = np.asarray(value, dtype=_ARRAY_DTYPES[name])
    return arrays



def _metadata_payload(
    artifact: SurfaceMeasureArtifact,
    arrays: Mapping[str, np.ndarray],
    npz_record: Mapping[str, Any],
) -> dict[str, Any]:
    masses = np.asarray(artifact.masses, dtype=np.float64)
    normals = np.asarray(artifact.normals, dtype=np.float64)
    payload: dict[str, Any] = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "model_id": int(artifact.model_id),
        "representation": artifact.representation,
        "measure": _measure_metadata(masses, normals),
        "arrays": {
            name: {"shape": list(values.shape), "dtype": values.dtype.str}
            for name, values in arrays.items()
        },
        "npz": dict(npz_record),
        "config": artifact.config,
        "input": artifact.input_data,
        "seed": int(artifact.seed),
        "diagnostics": artifact.diagnostics,
        "provenance": artifact.provenance,
    }
    payload["harmonic_basis"] = _harmonic_basis_metadata(artifact)
    return payload



def _measure_metadata(masses: np.ndarray, normals: np.ndarray) -> dict[str, Any]:
    values = np.asarray(masses, dtype=np.float64)
    directions = np.asarray(normals, dtype=np.float64)
    closure = values @ directions
    return {
        "atom_count": int(values.size),
        "mass_normalization": "unit",
        "mass_sum": float(np.sum(values)),
        "closure_vector": closure.tolist(),
        "closure_norm": float(np.linalg.norm(closure)),
        "minimum_mass": float(np.min(values)),
    }



def _harmonic_basis_metadata(
    artifact: SurfaceMeasureArtifact,
) -> dict[str, Any] | None:
    if artifact.harmonic_coefficients is None:
        return None
    degrees = np.asarray(artifact.harmonic_degrees, dtype=np.int64)
    return {
        **_HARMONIC_BASIS_CONVENTION,
        "coefficient_count": int(np.asarray(artifact.harmonic_coefficients).size),
        "minimum_degree": int(np.min(degrees)),
        "maximum_degree": int(np.max(degrees)),
        "component_labels": ["zonal", "cos", "sin"],
        "centering_measure_array": "quadrature_masses",
        "reference_measure_array": "reference_masses",
    }



def _float_vector(name: str, value: Any) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values



def _integer_vector(name: str, value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integers")
    return raw.astype(np.int64, copy=False)



def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")

