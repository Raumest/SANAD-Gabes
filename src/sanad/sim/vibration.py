"""Parametric vibration model (documented assumption, synthetic mode).

v_DE  = (v0 + k_L * L) * h(t)
v_NDE = (v0 + k_L * L) * r_NDE * (1 + c * (h(t) - 1))      (c = 0.6: NDE sees part of a DE fault)
h(t) >= 1 is the bearing health multiplier (1 = healthy). Units: g RMS.
When the machine is off a small background level remains.
"""

from __future__ import annotations

import numpy as np

NDE_COUPLING = 0.6
BACKGROUND_G = 0.004


def vibration(L, running, health, v0, k_l, nde_ratio, start=None):
    base = v0 + k_l * np.asarray(L)
    if start is not None:
        base = np.where(start, base * 1.5, base)
    de = np.where(running, base * health, BACKGROUND_G)
    nde = np.where(running, base * nde_ratio * (1 + NDE_COUPLING * (health - 1)), BACKGROUND_G)
    return de, nde
