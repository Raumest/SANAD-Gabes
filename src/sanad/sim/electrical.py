"""Electrical models of the feeders (all assumptions in configs/default.yaml).

Motor: P_in(L) = L*P_rated + P_loss_rated*(c0 + (1-c0)*L^2), so eta(L) = L*P_rated/P_in.
PF(L) = 0.88 - 0.33*exp(-4.5*(L-0.2)) (≈0.55 at 20 % load, ≈0.87 at 100 %), clipped.
Q = P*tan(acos PF), S = P/PF, balanced I = P/(√3*V_LL*PF).
"""

from __future__ import annotations

import numpy as np

SQRT3 = np.sqrt(3.0)


def motor_input_kw(L: np.ndarray, p_rated: float, eta_rated: float, c0: float) -> np.ndarray:
    loss_rated = p_rated * (1.0 / eta_rated - 1.0)
    return L * p_rated + loss_rated * (c0 + (1.0 - c0) * L**2)


def motor_efficiency(L, p_rated, eta_rated, c0):
    L = np.asarray(L, dtype=float)
    return np.where(L > 0, L * p_rated / motor_input_kw(L, p_rated, eta_rated, c0), 0.0)


def motor_pf(L: np.ndarray) -> np.ndarray:
    return np.clip(0.88 - 0.33 * np.exp(-4.5 * (np.asarray(L, dtype=float) - 0.2)), 0.12, 0.90)


def rated_values(eq_cfg: dict, v_ll: float) -> dict[str, float]:
    """Rated input power, PF, efficiency and current computed from the ratings."""
    if eq_cfg["type"] == "motor":
        p_in = float(motor_input_kw(np.array(1.0), eq_cfg["p_rated_kw"], eq_cfg["eta_rated"], eq_cfg["no_load_loss_share"]))
        pf = float(motor_pf(np.array(1.0)))
        eta = eq_cfg["p_rated_kw"] / p_in
    else:
        p_in, pf, eta = eq_cfg["p_rated_kw"], eq_cfg["pf"], 1.0
    i_rated = p_in * 1000.0 / (SQRT3 * v_ll * pf)
    return {"p_in_rated_kw": p_in, "pf_rated": pf, "eta_rated": eta, "i_rated_a": i_rated}


def sync_speed(f_hz, poles: int):
    return 120.0 * np.asarray(f_hz) / poles


def motor_speed(f_hz, poles: int, slip_rated: float, L, v_pu=1.0):
    """n = (120 f / p) * (1 - s), s = s_rated * L / v_pu^2 (slip rises under low voltage)."""
    v_pu = np.maximum(np.asarray(v_pu, dtype=float), 0.3)
    return sync_speed(f_hz, poles) * (1.0 - slip_rated * np.asarray(L) / v_pu**2)
