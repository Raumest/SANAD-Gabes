"""Data-quality injectors.

* ``apply_baseline``: natural imperfections in ALL data (incl. normal): sparse MCAR
  missing values, single-minute spikes, device timestamp jitter and occasional DELAYED
  rows — so that "NaN = anomaly" cannot be learned.
* ``apply_events``: DQ_MISSING events — MCAR at an elevated rate inside a window plus
  1-3 bursts where every value of the equipment is empty (comm_status stays OK).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanad.config import rng_for
from sanad.inject.events import Event
from sanad.schema import VALUE_COLUMNS

NO_SPIKE = {"kwh_register", "ambient_temp_c"}


def _applicable(df: pd.DataFrame) -> list[str]:
    return [c for c in VALUE_COLUMNS if df[c].notna().any()]


def apply_baseline(cfg: dict, frames: dict[str, pd.DataFrame], idx: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    b = cfg["baseline_dq"]
    n = len(idx)
    dev_offset = {}
    for eq, df in frames.items():
        rng = rng_for(cfg, "baseline_dq", eq)
        cols = _applicable(df)
        for c in cols:
            x = df[c].to_numpy(dtype=float).copy()
            if c not in NO_SPIKE:
                sp = rng.random(n) < b["spike_rate"]
                mag = rng.uniform(*b["spike_rel"], n) * rng.choice([-1, 1], n)
                x = np.where(sp, x * (1 + mag), x)
            x[rng.random(n) < b["mcar_missing_rate"]] = np.nan
            df[c] = x
        offset = rng.uniform(-0.3, 0.3) + rng.normal(0, b["jitter_s"], n)
        delayed = rng.random(n) < b["delayed_rate"]
        offset = np.where(delayed, -rng.uniform(5, 30, n), offset)
        df.loc[delayed, "comm_status"] = "DELAYED"
        dev_offset[eq] = offset
    return dev_offset


def apply_events(cfg: dict, frames: dict[str, pd.DataFrame], events: list[Event]) -> None:
    for e in events:
        if e.label != "DQ_MISSING":
            continue
        df = frames[e.equipment_id]
        rng = rng_for(cfg, "dq_event", e.k0, e.equipment_id)
        cols = _applicable(df)
        dur = e.k1 - e.k0
        for c in cols:
            col = df[c].to_numpy(dtype=float).copy()
            mask = rng.random(dur) < e.params["mcar_rate"]
            seg = col[e.k0:e.k1]
            seg[mask] = np.nan
            for off, bl in e.params["bursts"]:
                seg[off:off + bl] = np.nan
            col[e.k0:e.k1] = seg
            df[c] = col
