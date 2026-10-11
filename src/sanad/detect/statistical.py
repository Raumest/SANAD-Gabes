"""EWMA and two-sided CUSUM on normalised residuals (catch slow drift).

Thresholds are learned on the normal-only block: threshold = margin x the maximum
statistic observed there (floored), so the false-alarm rate on normal data is controlled
empirically even though residuals are autocorrelated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RESET_GAP = 60  # minutes without evidence after which EWMA/CUSUM restart from zero


def ewma(z: pd.Series, lam: float) -> np.ndarray:
    """EWMA that holds its state through short NaN gaps and resets after long ones."""
    x = z.to_numpy(dtype=float).tolist()
    out = np.full(len(x), np.nan)
    s, gap = 0.0, 0
    for k, v in enumerate(x):
        if v == v:
            s = (1 - lam) * s + lam * v
            gap = 0
            out[k] = s
        else:
            gap += 1
            if gap > RESET_GAP:
                s = 0.0
    return out


def cusum(z: pd.Series, k: float, cap: float) -> np.ndarray:
    """Two-sided CUSUM; signed statistic (+S_hi or -S_lo, whichever larger), capped.

    NaN where there is no evidence; the state resets after RESET_GAP minutes without data.
    """
    x = z.to_numpy(dtype=float).tolist()
    hi = lo = 0.0
    gap = 0
    out = np.full(len(x), np.nan)
    for i, v in enumerate(x):
        if v == v:
            hi = min(cap, max(0.0, hi + v - k))
            lo = min(cap, max(0.0, lo - v - k))
            gap = 0
            out[i] = hi if hi >= lo else -lo
        else:
            gap += 1
            if gap > RESET_GAP:
                hi = lo = 0.0
    return out


def fit_thresholds(zf: dict[str, pd.DataFrame], masks: dict[str, np.ndarray], rels: dict[str, list[str]], cfg: dict) -> dict:
    """Learn EWMA and CUSUM thresholds per (equipment, relation) on the normal block."""
    det = cfg["detection"]
    big_cap = 1e9
    th = {}
    for eq, z in zf.items():
        th[eq] = {}
        for rel in rels[eq]:
            m = masks[eq]
            col = z[rel].where(pd.Series(m, index=z.index))
            e = np.abs(ewma(col, det["ewma_lambda"]))
            c = np.abs(cusum(col, det["cusum_k"], big_cap))
            e_max = float(np.nanmax(e[m])) if np.isfinite(e[m]).any() else 0.0
            c_max = float(np.nanmax(c[m])) if np.isfinite(c[m]).any() else 0.0
            h = max(det["cusum_min_h"], det["cusum_margin"] * c_max)
            th[eq][rel] = {"ewma": max(det["ewma_min"], det["ewma_margin"] * e_max), "cusum_h": h,
                           "cusum_cap": h + det["cusum_cap_extra"]}
    return th
