"""Synthetic dataset generator.

Pipeline (CLAUDE.md §6):
true_state -> physical event library -> sensor path models (gain/offset/noise)
-> data-quality & comm injectors -> write files.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sanad import io
from sanad.config import generated_dir, rng_for
from sanad.inject.events import events_frame
from sanad.schema import EQUIPMENT_ORDER, MEASUREMENT_COLUMNS, MOTORS, VALUE_COLUMNS, applicable_columns
from sanad.sim import electrical as el
from sanad.sim.plant import build_drivers, compute_true_state
from sanad.sim.sensors import meter_compute, sense_raw, sensor_params, sensor_registry

ROUND_MAP = {
    "v_l1_v": "v", "v_l2_v": "v", "v_l3_v": "v", "i_l1_a": "i", "i_l2_a": "i", "i_l3_a": "i", "i_relay_a": "i",
    "p_kw": "p", "q_kvar": "p", "s_kva": "p", "pf": "pf", "freq_hz": "f", "kwh_register": "kwh",
    "thd_v_pct": "thd", "thd_i_pct": "thd", "speed_rpm": "speed", "winding_temp_c": "temp",
    "vib_de_g": "vib", "vib_nde_g": "vib", "ambient_temp_c": "temp",
}


def equipment_table(cfg: dict) -> pd.DataFrame:
    v_ll = cfg["plant"]["v_ll_nominal"]
    reg = sensor_registry(cfg)
    rows = [{
        "equipment_id": "BUS1", "type": "incomer", "description": "Incomer meter, 1 MVA 30/0.4 kV transformer",
        "p_rated_kw": cfg["plant"]["transformer_kva"], "p_in_rated_kw": cfg["plant"]["transformer_kva"],
        "s_rated_kva": cfg["plant"]["transformer_kva"],
        "i_rated_a": cfg["plant"]["transformer_kva"] * 1000 / (np.sqrt(3) * v_ll), "pf_rated": np.nan, "eta_rated": np.nan,
        "poles": np.nan, "n_sync_rpm": np.nan, "slip_rated": np.nan, "tau_thermal_min": np.nan, "dtheta_rated_k": np.nan,
    }]
    for eq, p in cfg["plant"]["equipment"].items():
        rv = el.rated_values(p, v_ll)
        motor = p["type"] == "motor"
        rows.append({
            "equipment_id": eq, "type": p["type"], "description": p["description"], "p_rated_kw": p["p_rated_kw"],
            "p_in_rated_kw": rv["p_in_rated_kw"], "s_rated_kva": rv["p_in_rated_kw"] / rv["pf_rated"],
            "i_rated_a": rv["i_rated_a"], "pf_rated": rv["pf_rated"], "eta_rated": rv["eta_rated"],
            "poles": p.get("poles", np.nan), "n_sync_rpm": 120 * cfg["plant"]["f_nominal"] / p["poles"] if motor else np.nan,
            "slip_rated": p.get("slip_rated", np.nan), "tau_thermal_min": p.get("tau_thermal_min", np.nan),
            "dtheta_rated_k": p.get("dtheta_rated_k", np.nan),
        })
    df = pd.DataFrame(rows)
    df["v_ll_nominal"] = v_ll
    df["applicable_columns"] = [";".join(applicable_columns("motor" if e in MOTORS else "other")) for e in df["equipment_id"]]
    df["sensor_ids"] = [";".join(reg.loc[reg.equipment_id == e, "sensor_id"]) for e in df["equipment_id"]]
    return df.round(4)


def _fmt_ts(idx: pd.DatetimeIndex, ms: bool = False) -> np.ndarray:
    if ms:
        return np.asarray(idx.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z", dtype=object)
    return np.asarray(idx.strftime("%Y-%m-%dT%H:%M:%SZ"), dtype=object)


def _round(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    r = cfg["sensors"]["rounding"]
    for c, key in ROUND_MAP.items():
        if c in df:
            dec = max(0, int(round(-np.log10(r[key]))))
            df[c] = df[c].round(dec)
    return df


def _long(frames: dict[str, pd.DataFrame], idx: pd.DatetimeIndex, cfg: dict, dev_offset_s: dict | None,
          extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Stack per-equipment frames into one long table sorted by (timestamp, equipment)."""
    gw = _fmt_ts(idx)
    parts = []
    for order, eq in enumerate(EQUIPMENT_ORDER):
        df = frames[eq].copy()
        df.insert(0, "equipment_id", eq)
        df.insert(0, "node_id", cfg["plant"]["node_id"])
        if dev_offset_s is None:
            df.insert(0, "timestamp_dev_utc", gw)
        else:
            dev = idx + pd.to_timedelta(dev_offset_s[eq], unit="s")
            df.insert(0, "timestamp_dev_utc", _fmt_ts(dev, ms=True))
        df.insert(0, "timestamp_gw_utc", gw)
        df["_k"] = np.arange(len(idx))
        df["_o"] = order
        parts.append(df)
    out = pd.concat(parts, ignore_index=True).sort_values(["_k", "_o"], kind="stable")
    cols = MEASUREMENT_COLUMNS + (extra_cols or [])
    return out[cols].reset_index(drop=True)


def generate(cfg: dict, out_dir: Path | None = None) -> Path:
    """Generate a full synthetic run and write all output files."""
    from sanad.inject import comm, data_quality, falsification, physical, scheduler, sensor

    out_dir = Path(out_dir) if out_dir else generated_dir(cfg)
    compress = bool(cfg["paths"].get("compress", False))
    drivers = build_drivers(cfg)
    idx = drivers["idx"]
    n = len(idx)

    events = scheduler.schedule_events(cfg, drivers)
    physical.apply(cfg, drivers, events)
    truth = compute_true_state(cfg, drivers)
    events += scheduler.natural_starts(cfg, truth, events)
    scheduler.finalise(events)

    params = sensor_params(cfg)
    amb_rng = rng_for(cfg, "ambient_sensor")
    amb_meas = truth["BUS1"]["ambient_temp_c"].to_numpy() + params["ambient"]["off"] + amb_rng.normal(0, cfg["sensors"]["ambient"]["noise_k"], n)
    raw = {eq: sense_raw(cfg, eq, truth[eq], params, amb_meas) for eq in EQUIPMENT_ORDER}
    sensor.apply_raw(cfg, raw, events)
    eqt = equipment_table(cfg).set_index("equipment_id")
    meas = {}
    for eq in EQUIPMENT_ORDER:
        kwh0 = float(truth[eq]["kwh_register"].iloc[0] - truth[eq]["p_kw"].iloc[0] / 60.0)
        meas[eq] = meter_compute(raw[eq], kwh0, 0.005 * eqt.loc[eq, "p_in_rated_kw"])
    sensor.apply_meter(cfg, meas, events)
    falsification.apply(cfg, meas, events)

    frames = {}
    for eq in EQUIPMENT_ORDER:
        df = pd.DataFrame({c: meas[eq][c] for c in VALUE_COLUMNS})
        for c in ("op_state", "run_cmd", "maint_mode", "comm_status"):
            df[c] = truth[eq][c].to_numpy()
        app = set(eqt.loc[eq, "applicable_columns"].split(";"))
        for c in VALUE_COLUMNS:
            if c not in app:
                df[c] = np.nan
        frames[eq] = _round(df, cfg)

    dev_offset = data_quality.apply_baseline(cfg, frames, idx)
    data_quality.apply_events(cfg, frames, events)
    comm.apply(cfg, frames, events, dev_offset)

    measurements = _long(frames, idx, cfg, dev_offset)
    truth_frames = {eq: _round(truth[eq].copy(), cfg) for eq in EQUIPMENT_ORDER}
    truth_long = _long(truth_frames, idx, cfg, None, ["load_fraction", "health", "irradiance_wm2"])

    out_dir.mkdir(parents=True, exist_ok=True)
    io.write_csv(measurements, out_dir / "measurements.csv", compress)
    io.write_csv(truth_long, out_dir / "truth_clean.csv", compress)
    io.write_csv(events_frame(events, idx, cfg["taxonomy"]), out_dir / "events.csv", False)
    io.write_csv(equipment_table(cfg), out_dir / "equipment.csv", False)
    io.write_csv(sensor_registry(cfg), out_dir / "sensors.csv", False)
    frozen = copy.deepcopy(cfg)
    frozen["_sensor_params_ground_truth"] = params
    with open(out_dir / "generation_config.yaml", "w", encoding="utf-8") as fh:
        fh.write(f"# {io.DISCLAIMER}\n# Frozen copy of the configuration used for this run (incl. seeds).\n")
        yaml.safe_dump(frozen, fh, sort_keys=False, allow_unicode=True)
    return out_dir
