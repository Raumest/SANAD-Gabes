"""Physics relations R1-R10 (the "Trust Fingerprint") as raw residuals.

Inputs are measurements only (+ nameplate data from equipment.csv and calibration
constants learned on the normal-only block). Residuals are later normalised to z-scores
by ``physics.residuals``.

| ID  | Residual (raw)                                             | Independent paths |
|-----|------------------------------------------------------------|-------------------|
| R1  | (S - sqrt(P^2+Q^2)) / S_rated                              | no                |
| R2  | (P - 3 V_ln I PF) / P_rated                                | no                |
| R3  | (I_meter - I_relay) / I_relay                              | yes               |
| R4  | P_bus - (a + b * sum P_feeders)   [kW]                     | yes               |
| R5  | window mean of clip((dkWh - P dt) / (P dt), +/-3 %)        | partly            |
| R5B | window mean of (dkWh_bus - a - b sum dkWh_f) / mean dkWh_bus | partly           |
| R6  | T_winding - (thermal model(I_relay, T_amb) + c0 + c1 dT_model) [K] | yes       |

R5/R5B use window *means* (not medians): register increments are quantised (0.01 kWh) and
a median of quantised values is biased, while their mean is not. Minutes where P is a
single-minute spike (|P - centred median of 3| large) are excluded from R5.
| R7  | speed - 120 f / p (1 - s_r L / v^2)   [rpm]                | yes               |
| R8  | V_meter/V_n - offset - median_others(V/V_n - offset)        | yes               |
| R9  | f_meter - median f of the other meters   [Hz]              | yes               |
| R10 | log(vib_DE / vib_NDE)                                      | yes               |
| R10E_DE / R10E_NDE | vibration - load envelope (learned)  [g]    | yes               |
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanad.schema import EQUIPMENT_ORDER, FEEDERS, MOTORS
from sanad.sim.thermal import thermal_response

RELATIONS = ["R1", "R2", "R3", "R4", "R5", "R5B", "R6", "R7", "R8", "R9", "R10", "R10E_DE", "R10E_NDE"]
APPLIES = {
    "BUS1": ["R1", "R2", "R4", "R5B", "R8", "R9"],
    "PUMP1": ["R1", "R2", "R3", "R5", "R6", "R7", "R8", "R9", "R10", "R10E_DE", "R10E_NDE"],
    "COMP1": ["R1", "R2", "R3", "R5", "R6", "R7", "R8", "R9", "R10", "R10E_DE", "R10E_NDE"],
    "AUX1": ["R1", "R2", "R5", "R8", "R9"],
}
RUN_STATES = ("RUN_LOW", "RUN_NORMAL", "RUN_HIGH")
V_LN_NOM = 400.0 / np.sqrt(3.0)


def split_by_equipment(meas: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    """Wide per-equipment frames on a shared positional index (one row per timestamp)."""
    ts = pd.DatetimeIndex(sorted(meas["timestamp_gw_utc"].unique()))
    frames = {}
    for eq in EQUIPMENT_ORDER:
        d = meas[meas.equipment_id == eq].sort_values("timestamp_gw_utc").reset_index(drop=True)
        if len(d) != len(ts):  # rows missing entirely: re-index on the grid
            d = d.set_index("timestamp_gw_utc").reindex(ts).rename_axis("timestamp_gw_utc").reset_index()
            d["equipment_id"] = eq
            d["comm_status"] = d["comm_status"].fillna("LOST")
        d["v_avg"] = d[["v_l1_v", "v_l2_v", "v_l3_v"]].mean(axis=1, skipna=False)
        d["i_avg"] = d[["i_l1_a", "i_l2_a", "i_l3_a"]].mean(axis=1, skipna=False)
        d["running"] = d["op_state"].isin(RUN_STATES).to_numpy()
        frames[eq] = d
    return ts, frames


def load_estimate(d: pd.DataFrame, eq_row: pd.Series) -> pd.Series:
    """Load fraction estimated from the meter's own input power (nameplate only)."""
    return (d["p_kw"] * eq_row["eta_rated"] / eq_row["p_rated_kw"]).clip(0, 1.6)


def _window_ratio(num: pd.Series, den: pd.Series, valid: pd.Series, window: int) -> pd.Series:
    mp = max(5, window // 3)
    n = num.where(valid).rolling(window, min_periods=mp).mean()
    d = den.where(valid).rolling(window, min_periods=mp).mean()
    return n / d


def spike_minutes(x: pd.Series, rel: float = 0.05, abs_: float = 1.0) -> pd.Series:
    """Single-minute spikes: far from the centred median of 3 (uses one minute look-ahead)."""
    med = x.rolling(3, center=True, min_periods=2).median()
    return (x - med).abs() > rel * med.abs() + abs_


def thermal_model(d: pd.DataFrame, e: pd.Series) -> np.ndarray:
    """Winding temperature predicted from the relay CT current and measured ambient.

    Single-minute spikes of the relay current are removed first (a 1-min spike would
    otherwise bias the model for ~tau minutes).
    """
    run = d["running"]
    steady = run & run.shift(1, fill_value=False) & run.shift(-1, fill_value=False)
    i_r = d["i_relay_a"].mask(spike_minutes(d["i_relay_a"], 0.08, 2.0) & steady).ffill()
    return thermal_response((i_r / e["i_rated_a"]).to_numpy(), d["ambient_temp_c"].to_numpy(),
                            e["tau_thermal_min"], e["dtheta_rated_k"])


def _leave_one_out_median(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: df.drop(columns=c).median(axis=1) for c in df.columns})


def raw_residuals(frames: dict[str, pd.DataFrame], eqt: pd.DataFrame, calib: dict, cfg: dict) -> dict[str, pd.DataFrame]:
    """Raw (un-normalised) relation residuals per equipment."""
    w = cfg["detection"]["r5_window_min"]
    n = len(frames["BUS1"])
    voff = calib.get("v_offset", {})
    v_all = pd.concat({eq: frames[eq]["v_avg"] / V_LN_NOM - voff.get(eq, 0.0) for eq in EQUIPMENT_ORDER}, axis=1)
    f_all = pd.concat({eq: frames[eq]["freq_hz"] for eq in EQUIPMENT_ORDER}, axis=1)
    v_ref = _leave_one_out_median(v_all)
    f_ref = _leave_one_out_median(f_all)
    p_feed = sum(frames[eq]["p_kw"] for eq in FEEDERS)
    a, b = calib.get("r4_a", 0.0), calib.get("r4_b", 1.0)
    out = {}
    for eq in EQUIPMENT_ORDER:
        d = frames[eq]
        e = eqt.loc[eq]
        r = pd.DataFrame(index=d.index, columns=RELATIONS, dtype=float)
        s_rated = e["s_rated_kva"]
        r["R1"] = (d["s_kva"] - np.sqrt(d["p_kw"] ** 2 + d["q_kvar"] ** 2)) / s_rated
        r["R2"] = (d["p_kw"] - 3 * d["v_avg"] * d["i_avg"] * d["pf"] / 1000.0) / e["p_in_rated_kw"]
        r["R8"] = v_all[eq] - v_ref[eq]
        r["R9"] = f_all[eq] - f_ref[eq]
        # R5: register increments vs the meter's own power (relative, robust rolling median)
        dk = d["kwh_register"].diff()
        e_int = d["p_kw"] / 60.0
        valid = dk.notna() & e_int.notna() & (e_int > 0.02 * e["p_in_rated_kw"] / 60.0)
        if eq == "BUS1":
            dk_f = sum(frames[f]["kwh_register"].diff() for f in FEEDERS)
            num = dk * 60.0 - a - b * dk_f * 60.0
            r["R5B"] = _window_ratio(num, dk * 60.0, valid & dk_f.notna(), w)
            r["R4"] = d["p_kw"] - (a + b * p_feed)
        else:
            # per-minute mismatch winsorised at +/-3 %: a display spike moves the window mean by
            # < 0.05 %, while falsification (>= 10 % under-reporting) saturates at -3 %
            mism = ((dk - e_int) / e_int).clip(-0.03, 0.03)
            r["R5"] = mism.where(valid & ~spike_minutes(d["p_kw"])).rolling(w, min_periods=max(5, w // 3)).mean()
        if eq in MOTORS:
            run = d["running"]
            i_r = d["i_relay_a"]
            r["R3"] = ((d["i_avg"] - i_r) / i_r).where(run & (i_r > 0.1 * e["i_rated_a"]))
            model = thermal_model(d, e)
            c0, c1 = calib.get("r6", {}).get(eq, (0.0, 0.0))
            r["R6"] = d["winding_temp_c"] - (model + c0 + c1 * (model - d["ambient_temp_c"].to_numpy()))
            L = load_estimate(d, e)
            v_pu = (d["v_avg"] / V_LN_NOM).clip(lower=0.3)
            n_exp = 120.0 * d["freq_hz"] / e["poles"] * (1 - e["slip_rated"] * L / v_pu**2)
            r["R7"] = (d["speed_rpm"] - n_exp).where(run)
            r["R10"] = np.log(d["vib_de_g"].clip(lower=1e-3) / d["vib_nde_g"].clip(lower=1e-3)).where(run)
            env = calib.get("vib_env", {}).get(eq)
            if env:
                r["R10E_DE"] = (d["vib_de_g"] - (env["de"][0] + env["de"][1] * L)).where(run)
                r["R10E_NDE"] = (d["vib_nde_g"] - (env["nde"][0] + env["nde"][1] * L)).where(run)
        drop = [c for c in RELATIONS if c not in APPLIES[eq]]
        r[drop] = np.nan
        out[eq] = r
    return out


def fit_structural(frames: dict[str, pd.DataFrame], eqt: pd.DataFrame, mask: np.ndarray) -> dict:
    """Constants learned on the normal-only block: bus balance (a, b), vibration envelopes."""
    p_feed = sum(frames[eq]["p_kw"] for eq in FEEDERS)
    p_bus = frames["BUS1"]["p_kw"]
    ok = mask & p_feed.notna().to_numpy() & p_bus.notna().to_numpy()
    b, a = np.polyfit(p_feed[ok], p_bus[ok], 1)
    # robust refit after dropping gross outliers (natural spikes)
    res = p_bus[ok] - (a + b * p_feed[ok])
    keep = (res - res.median()).abs() < 5 * 1.4826 * (res - res.median()).abs().median() + 1e-6
    b, a = np.polyfit(p_feed[ok][keep], p_bus[ok][keep], 1)
    v_all = pd.concat({eq: frames[eq]["v_avg"] / V_LN_NOM for eq in EQUIPMENT_ORDER}, axis=1)[mask]
    dev = v_all.sub(v_all.mean(axis=1), axis=0)
    v_offset = {eq: float(dev[eq].median()) for eq in EQUIPMENT_ORDER}
    env, r6 = {}, {}
    for eq in MOTORS:
        d = frames[eq]
        model = thermal_model(d, eqt.loc[eq])
        x = model - d["ambient_temp_c"].to_numpy()
        y = d["winding_temp_c"].to_numpy() - model
        s2 = mask & np.isfinite(x) & np.isfinite(y)
        s2[:180] = False  # thermal model warm-up
        c1, c0 = np.polyfit(x[s2], y[s2], 1)
        r6[eq] = (float(c0), float(c1))
        d = frames[eq]
        L = load_estimate(d, eqt.loc[eq])
        sel = mask & d["running"].to_numpy() & L.notna().to_numpy()
        env[eq] = {}
        for side in ("de", "nde"):
            y = d[f"vib_{side}_g"]
            s2 = sel & y.notna().to_numpy()
            slope, icpt = np.polyfit(L[s2], y[s2], 1)
            env[eq][side] = [float(icpt), float(slope)]
    return {"r4_a": float(a), "r4_b": float(b), "vib_env": env, "r6": r6, "v_offset": v_offset}
