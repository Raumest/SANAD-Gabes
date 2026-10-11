"""First-order winding thermal model.

T_ss = T_amb + ΔT_rated * (I / I_rated)^2 ;  T[k+1] = T[k] + (Δt/τ) * (T_ss[k] - T[k]).
The same function is used by the simulator (true current, true ambient) and by relation
R6 (relay current, measured ambient), so the relation is exact on clean data.
"""

from __future__ import annotations

import numpy as np


def thermal_response(i_pu: np.ndarray, t_amb: np.ndarray, tau_min: float, dtheta_rated: float,
                     dt_min: float = 1.0, t0: float | None = None, t_reset: np.ndarray | None = None,
                     reset_after: int = 5) -> np.ndarray:
    """Simulate winding temperature. NaN inputs hold the previous input value.

    If ``t_reset`` is given, the state is re-initialised to ``t_reset[k]`` at the first
    valid current sample after a gap of >= ``reset_after`` minutes (warm restart: the
    model cannot know what happened while its input was missing).
    """
    i_pu = np.asarray(i_pu, dtype=float)
    t_amb = np.asarray(t_amb, dtype=float)
    n = len(i_pu)
    out = np.empty(n)
    a = dt_min / tau_min
    last_i = 0.0 if np.isnan(i_pu[0]) else i_pu[0]
    last_a = float(np.nanmean(t_amb[:60])) if np.isnan(t_amb[0]) else t_amb[0]
    T = last_a + dtheta_rated * last_i**2 if t0 is None else t0
    ip = i_pu.tolist()
    ta = t_amb.tolist()
    tr = t_reset.tolist() if t_reset is not None else None
    gap = 0
    for k in range(n):
        x = ip[k]
        if x == x:  # not NaN
            if tr is not None and gap >= reset_after and tr[k] == tr[k]:
                T = tr[k]
            gap = 0
            last_i = x
        else:
            gap += 1
        y = ta[k]
        if y == y:
            last_a = y
        out[k] = T
        T = T + a * (last_a + dtheta_rated * last_i * last_i - T)
    return out
