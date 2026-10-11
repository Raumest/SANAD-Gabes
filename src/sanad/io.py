"""File I/O helpers. Every written file carries the fictional-plant disclaimer."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

DISCLAIMER = (
    "Fictional process plant in the Gabès industrial zone. All parameters are simulation "
    "assumptions. No measurement originates from a real company."
)


def _resolve(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        return gz
    raise FileNotFoundError(path)


def write_csv(df: pd.DataFrame, path: Path, compress: bool = False) -> Path:
    """Write a CSV with a leading '# disclaimer' comment line (gzip if ``compress``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        path = path.with_name(path.name + ".gz")
        # mtime=0 keeps gzip output byte-identical across runs (reproducibility)
        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(f"# {DISCLAIMER}\n".encode())
            gz.write(df.to_csv(index=False, lineterminator="\n").encode())
    else:
        other = path.with_name(path.name + ".gz")
        if other.exists():
            other.unlink()
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(f"# {DISCLAIMER}\n")
            df.to_csv(fh, index=False, lineterminator="\n")
    return path


def read_csv(path: Path, **kw) -> pd.DataFrame:
    """Read a CSV (or its .gz twin) skipping the disclaimer comment line."""
    return pd.read_csv(_resolve(path), comment="#", **kw)


def exists(path: Path) -> bool:
    try:
        _resolve(path)
        return True
    except FileNotFoundError:
        return False


def write_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
        fh.write("\n")
    return path


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_measurements(run_dir: Path) -> pd.DataFrame:
    df = read_csv(Path(run_dir) / "measurements.csv", low_memory=False)
    for c in ("timestamp_gw_utc", "timestamp_dev_utc"):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    return df


def read_events(run_dir: Path) -> pd.DataFrame:
    df = read_csv(Path(run_dir) / "events.csv")
    for c in ("start_utc", "end_utc"):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    df["compound_with"] = df["compound_with"].fillna("")
    return df
