"""Locate the release files, which ship gzipped (.csv.gz) but may be extracted (.csv)."""
from __future__ import annotations

from pathlib import Path


def data_file(data: Path, name: str) -> Path:
    """Return data/<name>.csv or data/<name>.csv.gz, whichever exists."""
    for suffix in (".csv", ".csv.gz"):
        path = Path(data) / f"{name}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"{name}.csv or {name}.csv.gz not found in {data}")
