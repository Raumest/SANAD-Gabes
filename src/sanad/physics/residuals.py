"""Residual normalisation: z = (r - median_normal) / sigma_normal (robust, per equipment).

Sigma is 1.4826 * MAD on the normal-only block, floored to the physical resolution of the
relation so that rounding noise cannot produce huge z-scores. A causal median-of-3 filter
removes single-minute spikes (natural baseline imperfections) before thresholds,
EWMA and CUSUM are applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanad.physics.relations import APPLIES

SIGMA_FLOOR = {
    "R1": 2e-4, "R2": 5e-4, "R3": 2e-3, "R4": 0.5, "R5": 3e-3, "R5B": 3e-3, "R6": 0.2, "R7": 0.5,
    "R8": 5e-4, "R9": 2e-3, "R10": 0.01, "R10E_DE": 3e-3, "R10E_NDE": 3e-3,
}
Z_CLIP = 30.0


def fit_normalisation(raw: dict[str, pd.DataFrame], masks: dict[str, np.ndarray]) -> dict:
    norm = {}
    for eq, r in raw.items():
        norm[eq] = {}
        for rel in APPLIES[eq]:
            x = r[rel][masks[eq]].dropna()
            if len(x) < 50:
                continue
            med = float(x.median())
            mad = float((x - med).abs().median()) * 1.4826
            norm[eq][rel] = {"median": med, "sigma": max(mad, SIGMA_FLOOR[rel])}
    return norm


def normalise(raw: dict[str, pd.DataFrame], norm: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for eq, r in raw.items():
        z = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
        for rel, p in norm[eq].items():
            z[rel] = (r[rel] - p["median"]) / p["sigma"]
        out[eq] = z
    return out


def despike(z: pd.DataFrame) -> pd.DataFrame:
    """Causal median of the last 3 minutes, clipped (removes single-minute spikes)."""
    return z.rolling(3, min_periods=2).median().clip(-Z_CLIP, Z_CLIP)
