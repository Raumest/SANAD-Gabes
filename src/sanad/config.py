"""Configuration loading.

Configs are YAML files. A config may declare ``extends: <file>`` to inherit from another
config in the same folder; mappings are deep-merged (child wins). All paths in the config
are relative to the repository root, which is located from this file (no absolute paths).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Return ``base`` deep-merged with ``override`` (override wins, base untouched)."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a config file, resolving ``extends`` chains."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    cfg["_config_path"] = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
    return cfg


def repo_path(rel: str | Path) -> Path:
    """Resolve a repository-relative path."""
    return REPO_ROOT / rel


def generated_dir(cfg: dict) -> Path:
    return repo_path(cfg["paths"]["generated_dir"]) / cfg["run_id"]


def processed_dir(cfg: dict) -> Path:
    return repo_path(cfg["paths"]["processed_dir"]) / cfg["run_id"]


def reports_dir(cfg: dict) -> Path:
    return repo_path(cfg["paths"]["reports_dir"])


def time_index(cfg: dict) -> pd.DatetimeIndex:
    """The 1-minute UTC grid of the run."""
    t = cfg["time"]
    start = pd.Timestamp(t["start_utc"])
    n = int(t["duration_days"] * 24 * 60 // t["resolution_min"])
    return pd.date_range(start, periods=n, freq=f"{t['resolution_min']}min", tz="UTC")


def block_bounds(cfg: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Start/end (end exclusive) of the normal/train/val/test blocks."""
    s = cfg["split"]
    start = pd.Timestamp(cfg["time"]["start_utc"])
    out, cur = {}, start
    for name in ("normal", "train", "val", "test"):
        end = cur + pd.Timedelta(days=s[f"{name}_days"])
        out[name] = (cur, end)
        cur = end
    return out


def usable_bounds(cfg: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Block bounds with the embargo removed from the start of every block after the first."""
    emb = pd.Timedelta(hours=cfg["split"]["embargo_hours"])
    out = {}
    for name, (a, b) in block_bounds(cfg).items():
        out[name] = (a if name == "normal" else a + emb, b)
    return out


def block_of(cfg: dict, ts: pd.Series) -> pd.Series:
    """Label each timestamp with its block; embargo minutes get 'embargo'."""
    lab = pd.Series("embargo", index=ts.index, dtype=object)
    for name, (a, b) in usable_bounds(cfg).items():
        lab[(ts >= a) & (ts < b)] = name
    return lab


def rng_for(cfg: dict, *keys: str | int) -> "np.random.Generator":
    """Deterministic child RNG for a named stream (same seed + keys -> same stream)."""
    import hashlib

    import numpy as np

    h = hashlib.sha256("/".join(str(k) for k in keys).encode()).digest()
    return np.random.default_rng([int(cfg["seed"]), int.from_bytes(h[:8], "little")])
