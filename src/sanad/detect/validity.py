"""Validity checks: NaN, range, rate of change, flatline, timestamps, comm status.

All checks are per equipment and per minute and read only measurements + nameplate data.
State context (op_state, maint_mode) is applied later by the detector to suppress alarms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanad.physics.relations import V_LN_NOM
from sanad.schema import MOTORS

FLAT_GROUPS = {"meter": "p_kw", "relay": "i_relay_a", "rtd": "winding_temp_c", "speed": "speed_rpm",
               "acc_de": "vib_de_g", "acc_nde": "vib_nde_g"}


def flat_run(x: pd.Series) -> np.ndarray:
    """Length of the current run of identical, non-NaN values (in minutes)."""
    v = x.to_numpy(dtype=float)
    same = np.zeros(len(v), dtype=bool)
    same[1:] = (v[1:] == v[:-1]) & ~np.isnan(v[1:])
    grp = np.cumsum(~same)
    run = pd.Series(1, index=x.index).groupby(grp).cumsum().to_numpy()
    return np.where(np.isnan(v), 0, run)


def validity(d: pd.DataFrame, eq: str, e: pd.Series, cfg: dict, r6_raw: pd.Series | None) -> pd.DataFrame:
    det = cfg["detection"]
    run = d["running"].to_numpy()
    out = pd.DataFrame(index=d.index)
    cols = [c for c in e["applicable_columns"].split(";") if c not in ("pf", "thd_i_pct")]
    out["nan_frac"] = d[cols].isna().mean(axis=1)
    out["lost"] = (d["comm_status"] == "LOST").to_numpy()
    out["delayed"] = (d["comm_status"] == "DELAYED").to_numpy()
    skew = (d["timestamp_dev_utc"] - d["timestamp_gw_utc"]).dt.total_seconds().abs()
    out["time_skew"] = (skew > 5.0).to_numpy() & ~out["delayed"].to_numpy()
    v_pu = d["v_avg"] / V_LN_NOM
    out["v_pu"] = v_pu
    out["undervoltage"] = ((v_pu < det["undervoltage_pu"]) & (v_pu >= det["interruption_pu"])).to_numpy()
    out["interruption"] = (v_pu < det["interruption_pu"]).to_numpy()
    out["overvoltage"] = (v_pu > 1.10).to_numpy()
    i_pu = d["i_avg"] / e["i_rated_a"]
    if eq in MOTORS:
        i_pu = pd.concat([i_pu, d["i_relay_a"] / e["i_rated_a"]], axis=1).max(axis=1)
    out["i_pu"] = i_pu
    out["overcurrent"] = ((i_pu > det["overcurrent_pu"]) & run).to_numpy()
    out["pf_invalid"] = ((d["pf"] > 1.0) | (d["pf"] < 0.0)).to_numpy()
    out["kwh_decrease"] = (d["kwh_register"].diff() < -0.01).to_numpy()
    out["temp_range"] = ((d["winding_temp_c"] > 155) | (d["winding_temp_c"] < -10)).to_numpy() if eq in MOTORS else False
    # temporal plausibility: the winding cannot move faster than tau allows -> a step in R6
    if eq in MOTORS and r6_raw is not None:
        out["temp_step"] = (r6_raw.diff().abs() > 2.0).to_numpy()
    else:
        out["temp_step"] = False
    for g, col in FLAT_GROUPS.items():
        if eq not in MOTORS and g != "meter":
            out[f"flat_{g}"] = False
            continue
        out[f"flat_{g}"] = (flat_run(d[col]) >= det["flatline_min"]) & run
    return out
