#!/usr/bin/env python3
"""Import only the ten checksummed intensity tables from an HAC folder or ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def prepare(source: Path, destination: Path, models: list[int]) -> list[dict]:
    manifest = json.loads((ROOT / "data_manifest/real_intensity_20260907.json").read_text())
    if not models or len(models) != len(set(models)) or not set(models) <= set(range(1, 11)):
        raise ValueError("models must be distinct IDs between 1 and 10")
    rows = [row for row in manifest["files"] if row["model_id"] in models]
    contents = {}
    if source.is_dir():
        for row in rows:
            matches = sorted(source.rglob(Path(row["path"]).name))
            if len(matches) != 1:
                raise ValueError(
                    f"expected exactly one {Path(row['path']).name}, found {len(matches)}"
                )
            contents[row["model_id"]] = matches[0].read_bytes()
    else:
        with zipfile.ZipFile(source) as archive:
            for row in rows:
                matches = [
                    name for name in archive.namelist() if Path(name).name == Path(row["path"]).name
                ]
                if len(matches) != 1:
                    raise ValueError(f"expected exactly one {Path(row['path']).name} in archive")
                info = archive.getinfo(matches[0])
                if info.file_size != row["size_bytes"]:
                    raise ValueError(f"released input size mismatch: {matches[0]}")
                contents[row["model_id"]] = archive.read(matches[0])
    # Validate the entire requested set before writing any destination file.
    for row in rows:
        content = contents[row["model_id"]]
        if (
            len(content) != row["size_bytes"]
            or hashlib.sha256(content).hexdigest() != row["sha256"]
        ):
            raise ValueError(f"input differs from the September 7 release: {row['path']}")
        path = destination / row["path"]
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite different data: {path}")
    for row in rows:
        path = destination / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(contents[row["model_id"]])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="downloaded HAC folder or complete ZIP")
    parser.add_argument("--destination", type=Path, default=ROOT / "data")
    parser.add_argument("--models", default="1,2,3,4,5,6,7,8,9,10")
    args = parser.parse_args()
    rows = prepare(args.source, args.destination, [int(x) for x in args.models.split(",")])
    print(
        json.dumps(
            {
                "destination": str(args.destination),
                "verified_models": [row["model_id"] for row in rows],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
