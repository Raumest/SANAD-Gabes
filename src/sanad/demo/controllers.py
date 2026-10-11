"""Naive vs SANAD-protected load-limit controller (offline what-if, demo only).

A downstream controller (NOT part of SANAD — SANAD never actuates) keeps the combined
current of PUMP1 + COMP1 below a group limit by letting COMP1 take the headroom:
    allowed COMP1 current = limit x (1 - safety margin) - PUMP1 current reading.
* naive: reads the raw PUMP1 meter current.
* SANAD-protected: reads the trusted stream; when the meter is quarantined it uses the
  substituted (relay-based) estimate, and when a value is SUSPECT it takes the
  conservative max(meter, relay).
The *true* group current is then PUMP1's true current + the COMP1 current the controller
allowed. If the PUMP1 meter CT reads low, the naive controller over-loads the group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanad.sim import electrical as el


def comp_current_curve(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    p = cfg["plant"]["equipment"]["COMP1"]
    L = np.linspace(0.0, 1.4, 281)
    p_in = el.motor_input_kw(L, p["p_rated_kw"], p["eta_rated"], p["no_load_loss_share"])
    i = p_in * 1000.0 / (np.sqrt(3) * cfg["plant"]["v_ll_nominal"] * el.motor_pf(L))
    return L, i


def simulate(cfg: dict, truth_pump: pd.DataFrame, meas_pump: pd.DataFrame, trusted_pump: pd.DataFrame) -> pd.DataFrame:
    """All frames are PUMP1 rows on the same minute grid (same order)."""
    d = cfg["demo"]
    L_grid, i_grid = comp_current_curve(cfg)
    lo, hi = np.interp([d["comp_min_load"], d["comp_max_load"]], L_grid, i_grid)
    limit = d["group_current_limit_a"]
    target = limit * (1.0 - d.get("safety_margin_frac", 0.0))
    phases = ["i_l1_a", "i_l2_a", "i_l3_a"]
    true_pump = truth_pump[phases].mean(axis=1).to_numpy()
    naive = meas_pump[phases].mean(axis=1).ffill().to_numpy()
    tr = trusted_pump[phases].mean(axis=1)
    relay = trusted_pump["i_relay_a"]
    suspect = trusted_pump["trust_meter"].isin(["SUSPECT", "QUARANTINED"]) & tr.notna()
    prot = tr.where(~suspect, np.maximum(tr, relay)).ffill().to_numpy()
    out = {"timestamp_gw_utc": trusted_pump["timestamp_gw_utc"].to_numpy(), "limit_a": limit,
           "pump_true_a": true_pump, "pump_reading_naive_a": naive, "pump_reading_protected_a": prot}
    for name, reading in (("naive", naive), ("protected", prot)):
        comp_i = np.clip(target - reading, lo, hi)
        out[f"comp_load_{name}"] = np.interp(comp_i, i_grid, L_grid)
        out[f"true_total_{name}_a"] = true_pump + comp_i
    df = pd.DataFrame(out)
    running = truth_pump["op_state"].isin(["RUN_LOW", "RUN_NORMAL", "RUN_HIGH"]).to_numpy()
    for c in [c for c in df.columns if c.startswith(("true_total", "comp_load", "pump_"))]:
        df.loc[~running, c] = np.nan
    return df


def summary(df: pd.DataFrame, t0, t1) -> dict:
    w = df[(df.timestamp_gw_utc >= t0) & (df.timestamp_gw_utc < t1)]
    out = {}
    for name in ("naive", "protected"):
        exc = w[f"true_total_{name}_a"] - w["limit_a"]
        out[name] = {"minutes_over_limit": int((exc > 0).sum()), "max_excess_a": round(float(np.nanmax(exc.clip(lower=0))), 1)}
    return out
