"""Sensor path models (golden rules 1 and 3).

Every measurement path has its own static gain/offset error (drawn once, seeded) and its
own independent noise realisation. A feeder *meter* senses V and I through its own VT/CT
and then computes P, Q, S, PF and integrates kWh internally (``meter_compute``), so a CT
gain error propagates consistently into P/Q/S/kWh of that meter (R1/R2 still hold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanad.config import rng_for
from sanad.schema import EQUIPMENT_ORDER, MOTORS

PATHS_MOTOR = ["METER_CT", "RELAY_CT", "RTD", "ACC_DE", "ACC_NDE", "SPEED"]


def sensor_registry(cfg: dict) -> pd.DataFrame:
    """sensors.csv content: ids, paths, accuracy classes, install & calibration dates."""
    s = cfg["sensors"]
    rng = rng_for(cfg, "sensor_dates")
    start = pd.Timestamp(cfg["time"]["start_utc"])
    rows = []
    cls = {"METER_CT": s["meter"]["class"], "BUS_CT": s["bus_meter"]["class"], "RELAY_CT": s["relay_ct"]["class"],
           "RTD": s["rtd"]["class"], "ACC_DE": s["accelerometer"]["class"], "ACC_NDE": s["accelerometer"]["class"],
           "SPEED": s["speed"]["class"], "AMBIENT": s["ambient"]["class"]}
    for eq in EQUIPMENT_ORDER:
        paths = ["BUS_CT", "AMBIENT"] if eq == "BUS1" else (PATHS_MOTOR if eq in MOTORS else ["METER_CT"])
        for path in paths:
            sid = f"{eq}_{'AMB' if path == 'AMBIENT' else path}"
            inst = start - pd.Timedelta(days=int(rng.integers(400, 2500)))
            cal = start - pd.Timedelta(days=int(rng.integers(30, 360)))
            rows.append({"sensor_id": sid, "equipment_id": eq, "path": path, "accuracy_class": cls[path],
                         "install_date": inst.date().isoformat(), "calibration_date": cal.date().isoformat()})
    return pd.DataFrame(rows)


def sensor_params(cfg: dict) -> dict[str, dict[str, float]]:
    """Static per-path errors (ground truth; saved only in generation_config.yaml)."""
    s = cfg["sensors"]
    rng = rng_for(cfg, "sensor_params")
    out = {}
    for eq in EQUIPMENT_ORDER:
        m = s["bus_meter"] if eq == "BUS1" else s["meter"]
        out[f"{eq}_meter"] = {"gi": float(rng.normal(0, m["gain_sigma"])), "gv": float(rng.normal(0, m["v_gain_sigma"]))}
        if eq in MOTORS:
            out[f"{eq}_relay"] = {"g": float(rng.normal(0, s["relay_ct"]["gain_sigma"]))}
            out[f"{eq}_rtd"] = {"off": float(rng.normal(0, s["rtd"]["offset_sigma_k"]))}
            out[f"{eq}_acc_de"] = {"g": float(rng.normal(0, s["accelerometer"]["gain_sigma"]))}
            out[f"{eq}_acc_nde"] = {"g": float(rng.normal(0, s["accelerometer"]["gain_sigma"]))}
    out["ambient"] = {"off": float(rng.normal(0, 0.2))}
    return out


def sense_raw(cfg: dict, eq: str, truth: pd.DataFrame, params: dict, amb_meas: np.ndarray) -> dict[str, np.ndarray]:
    """Raw sensed signals (before the meter's internal computation)."""
    s = cfg["sensors"]
    m = s["bus_meter"] if eq == "BUS1" else s["meter"]
    rng = rng_for(cfg, "sense", eq)
    n = len(truth)
    mp = params[f"{eq}_meter"]
    raw: dict[str, np.ndarray] = {}
    for k in (1, 2, 3):
        v = truth[f"v_l{k}_v"].to_numpy()
        raw[f"v_l{k}_v"] = v * (1 + mp["gv"]) + rng.normal(0, m["v_noise_v"], n) * (v > 1)
        i = truth[f"i_l{k}_a"].to_numpy()
        raw[f"i_l{k}_a"] = np.abs(i * (1 + mp["gi"]) * (1 + rng.normal(0, m["i_noise_rel"], n)) + rng.normal(0, 0.01, n) * (i > 0))
    raw["pf"] = np.clip(truth["pf"].to_numpy() + rng.normal(0, m["pf_noise"], n), 0.0, 1.0)
    raw["freq_hz"] = truth["freq_hz"].to_numpy() + rng.normal(0, m["f_noise_hz"], n)
    raw["thd_v_pct"] = np.abs(truth["thd_v_pct"].to_numpy() + rng.normal(0, 0.08, n))
    raw["thd_i_pct"] = np.abs(truth["thd_i_pct"].to_numpy() + rng.normal(0, 0.15, n))
    raw["ambient_temp_c"] = amb_meas
    if eq in MOTORS:
        r = rng_for(cfg, "sense_motor", eq)
        ir = truth["i_relay_a"].to_numpy()
        raw["i_relay_a"] = np.abs(ir * (1 + params[f"{eq}_relay"]["g"]) * (1 + r.normal(0, s["relay_ct"]["i_noise_rel"], n)))
        raw["winding_temp_c"] = truth["winding_temp_c"].to_numpy() + params[f"{eq}_rtd"]["off"] + r.normal(0, s["rtd"]["noise_k"], n)
        for side in ("de", "nde"):
            v = truth[f"vib_{side}_g"].to_numpy()
            raw[f"vib_{side}_g"] = np.abs(v * (1 + params[f"{eq}_acc_{side}"]["g"]) + r.normal(0, s["accelerometer"]["noise_g"], n))
        raw["speed_rpm"] = np.maximum(truth["speed_rpm"].to_numpy() + r.normal(0, s["speed"]["noise_rpm"], n) * (truth["speed_rpm"].to_numpy() > 0), 0.0)
    else:
        for c in ("i_relay_a", "winding_temp_c", "vib_de_g", "vib_nde_g", "speed_rpm"):
            raw[c] = np.full(n, np.nan)
    return raw


def meter_compute(raw: dict[str, np.ndarray], kwh0: float, s_min_kva: float) -> dict[str, np.ndarray]:
    """What the meter computes from its own V and CT signals (and integrates into kWh)."""
    V = np.stack([raw["v_l1_v"], raw["v_l2_v"], raw["v_l3_v"]], axis=1)
    I = np.stack([raw["i_l1_a"], raw["i_l2_a"], raw["i_l3_a"]], axis=1)
    s = (V * I).sum(axis=1) / 1000.0
    pf = np.where(s > s_min_kva, raw["pf"], np.nan)
    p = s * np.nan_to_num(pf)
    q = np.sqrt(np.maximum(s**2 - p**2, 0.0))
    out = dict(raw)
    out.update({"s_kva": s, "p_kw": p, "q_kvar": q, "pf": pf})
    out["freq_hz"] = np.where(V.mean(axis=1) > 20.0, raw["freq_hz"], np.nan)
    out["kwh_register"] = kwh0 + np.cumsum(np.nan_to_num(p)) / 60.0
    return out
