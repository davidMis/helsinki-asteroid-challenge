"""Submission core extracted from the developed HAC implementation.

Only the final algorithm and its dependencies are retained. See
``provenance/source_extraction.json`` for original source and symbol hashes.
"""

from __future__ import annotations
import csv
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON from ``path``."""
    return json.loads(path.read_text(encoding="utf-8"))



def write_json(path: Path, payload: Any, *, default: Any | None = None) -> None:
    """Atomically write strict, pretty UTF-8 JSON to ``path``.

    ``NaN`` and infinities are rejected because they are not JSON values and
    make experiment manifests silently incompatible with strict parsers.
    """
    serialized = json.dumps(
        payload,
        indent=2,
        default=default,
        allow_nan=False,
    )
    _atomic_write_text(Path(path), serialized)



def write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    """Atomically write strict mapping rows to CSV.

    Field names must be unique strings and numeric cells must be finite. Key
    order is preserved by first appearance when ``fieldnames`` is omitted.
    """
    output_fields = list(fieldnames) if fieldnames is not None else row_fieldnames(rows)
    _validate_csv_rows(rows, output_fields)

    path = Path(path)
    if not output_fields:
        _atomic_write_text(path, "")
        return
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
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=output_fields,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        _match_output_permissions(temporary_path, path)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)



def row_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return field names in first-seen order across all rows."""
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                names.append(str(key))
                seen.add(str(key))
    return names



def file_provenance(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    """Return path, byte size, and SHA-256 provenance for one input file."""
    path = Path(path)
    digest, size_bytes = _sha256_and_size(path, chunk_size=1024 * 1024)
    display_path = path
    if relative_to is not None:
        try:
            display_path = path.resolve().relative_to(Path(relative_to).resolve())
        except ValueError:
            display_path = path.resolve()
    return {
        "path": str(display_path),
        "size_bytes": size_bytes,
        "sha256": digest,
    }



def _sha256_and_size(path: Path, *, chunk_size: int) -> tuple[str, int]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not path.is_file():
        raise ValueError(f"provenance path is not a regular file: {path}")
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes



def _atomic_write_text(path: Path, text: str) -> None:
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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _match_output_permissions(temporary_path, path)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)



def _match_output_permissions(temporary_path: Path, destination: Path) -> None:
    mode = destination.stat().st_mode & 0o777 if destination.exists() else 0o644
    temporary_path.chmod(mode)



def _validate_csv_rows(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    if not fieldnames and rows:
        raise ValueError("CSV fieldnames must not be empty")
    if any(not isinstance(field, str) or not field for field in fieldnames):
        raise ValueError("CSV fieldnames must be non-empty strings")
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError("CSV fieldnames must be unique")
    allowed = set(fieldnames)
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"CSV row {row_index} is not a mapping")
        for key, value in row.items():
            if not isinstance(key, str):
                raise TypeError(f"CSV row {row_index} has a non-string field name")
            if key not in allowed:
                raise ValueError(
                    f"CSV row {row_index} contains field {key!r} outside fieldnames"
                )
            if isinstance(value, numbers.Number):
                if not isinstance(value, numbers.Real):
                    raise TypeError(
                        f"CSV row {row_index} field {key!r} contains a "
                        "non-real numeric value"
                    )
                numeric_value: Any = value
                finite = math.isfinite(numeric_value)
                if not finite:
                    raise ValueError(
                        f"CSV row {row_index} field {key!r} contains a "
                        "non-finite number"
                    )

