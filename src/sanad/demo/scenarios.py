"""Scripted ~3-minute demo scenario (CLAUDE.md §13), fully precomputed and deterministic.

1. Normal operation + COMP1 motor start -> not flagged.
2. Real overload on COMP1 -> genuine overload, operator alerted.
3. CT drift on PUMP1 -> R3/R4 light up, SENSOR_FAULT/DRIFT, meter quarantined, estimate
   substituted; naive vs SANAD-protected load-limit controller side panel.
4. kWh falsification on AUX1 -> UNCERTAIN (only one independent witness), escalated;
   a scripted operator override with a reason is logged.
5. Audit -> verify chain -> edit one entry -> chain breaks.
All numbers in the narrative are computed from the scenario outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sanad import io
from sanad.audit.log import AuditLog, verify_chain
from sanad.config import generated_dir, load_config, processed_dir
from sanad.demo import controllers
from sanad.pipeline import run_pipeline
from sanad.sim.generator import generate

SCENARIO_CONFIG = "configs/scenario.yaml"
KEEP = ["timestamp_gw_utc", "equipment_id", "i_l1_a", "i_relay_a", "p_kw", "kwh_register", "winding_temp_c", "v_l1_v",
        "speed_rpm", "op_state"]


def _iso(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def _overlap(cases: pd.DataFrame, eq: str, a, b) -> pd.DataFrame:
    return cases[(cases.equipment_id == eq) & (cases.start_utc < b) & (cases.end_utc > a)]


def build_demo(_cfg: dict | None = None) -> Path:
    cfg = load_config(SCENARIO_CONFIG)
    g = generate(cfg)
    out = run_pipeline(cfg)
    ev = io.read_events(g)
    cases = io.read_csv(out / "cases.csv")
    for c in ("start_utc", "end_utc"):
        cases[c] = pd.to_datetime(cases[c], utc=True)
    meas = io.read_measurements(g)
    truth = io.read_csv(g / "truth_clean.csv")
    trusted = io.read_csv(out / "trusted_stream.csv", low_memory=False)
    trusted["timestamp_gw_utc"] = pd.to_datetime(trusted["timestamp_gw_utc"], utc=True)

    by = {lab: ev[ev.subtype == lab].iloc[0] for lab in ("PHYS_OVERLOAD", "SENS_DRIFT", "FALS_ENERGY")}
    starts = ev[(ev.subtype == "NORMAL_START") & (ev.equipment_id == "COMP1") & (ev.start_utc >= pd.Timestamp(cfg["time"]["start_utc"]) + pd.Timedelta(days=3))]
    st0 = starts.iloc[0]

    # ---- controllers (PUMP1 rows) ----
    pm = meas[meas.equipment_id == "PUMP1"].reset_index(drop=True)
    pt = truth[truth.equipment_id == "PUMP1"].reset_index(drop=True)
    ptr = trusted[trusted.equipment_id == "PUMP1"].reset_index(drop=True)
    ctrl = controllers.simulate(cfg, pt, pm, ptr)
    ctrl_out = ctrl.copy()
    num = ctrl_out.columns.drop("timestamp_gw_utc")
    ctrl_out[num] = ctrl_out[num].round(3)
    ctrl_out["timestamp_gw_utc"] = ctrl.timestamp_gw_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    io.write_csv(ctrl_out, out / "controllers.csv", compress=False)
    drift = by["SENS_DRIFT"]
    csum = controllers.summary(ctrl, drift.start_utc, drift.end_utc)

    # ---- step case selection ----
    c_start = _overlap(cases, "COMP1", st0.start_utc, st0.end_utc + pd.Timedelta(minutes=30))
    c_ov = _overlap(cases, "COMP1", by["PHYS_OVERLOAD"].start_utc, by["PHYS_OVERLOAD"].end_utc)
    c_dr = _overlap(cases, "PUMP1", drift.start_utc, drift.end_utc + pd.Timedelta(minutes=60))
    c_dr = c_dr[c_dr.families.str.contains("electrical")]
    c_fa = _overlap(cases, "AUX1", by["FALS_ENERGY"].start_utc, by["FALS_ENERGY"].end_utc)
    c_fa = c_fa[c_fa.families.str.contains("energy")]

    # ---- scripted operator override on the falsification case (step 4) ----
    if len(c_fa):
        fc = c_fa.iloc[0]
        ts = _iso(fc.end_utc + pd.Timedelta(minutes=30))
        payload = {"case_id": fc.case_id, "action": "OVERRIDE", "reason": cfg["demo"]["override_reason"],
                   "operator": "operator-demo (scripted)", "verdict": fc.verdict, "label": "SENSOR_FAULT"}
        AuditLog(out / cfg["audit"]["log_name"]).append("OPERATOR_ACTION", ts, payload)
        (out / "feedback.jsonl").write_text(json.dumps({"ts_utc": ts, **payload}) + "\n", encoding="utf-8")
    ok, _, chain_msg = verify_chain(out / cfg["audit"]["log_name"])

    # ---- measured window for plotting raw vs trusted ----
    t_a = pd.Timestamp(cfg["time"]["start_utc"]) + pd.Timedelta(days=3)
    mw = meas[KEEP].copy()
    # energy integrated from the meter's own power, anchored to the register at the window start (for step 4)
    mw["kwh_from_power"] = mw.groupby("equipment_id")["p_kw"].transform(lambda p: p.fillna(0).cumsum() / 60.0)
    first = mw[mw.timestamp_gw_utc >= t_a].groupby("equipment_id").first()
    mw["kwh_from_power"] += mw.equipment_id.map(first["kwh_register"] - first["kwh_from_power"])
    mw = mw[mw.timestamp_gw_utc >= t_a]
    mw["timestamp_gw_utc"] = mw.timestamp_gw_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    io.write_csv(mw, out / "measured_window.csv", compress=True)

    def fmt_cases(df: pd.DataFrame) -> str:
        if not len(df):
            return "no case"
        return "; ".join(f"{r.case_id} {r.verdict}" + ("" if r.verdict == "UNCERTAIN" else f"/{r.subtype}") +
                         f" (confidence {r.confidence:.2f}, {r.tier})" for r in df.itertuples())

    pad = pd.Timedelta(hours=1)
    steps = [
        {"title": "Normal operation and a COMP1 motor start",
         "t0": _iso(st0.start_utc - 2 * pad), "t1": _iso(st0.start_utc + 2 * pad),
         "signals": [["COMP1", "i_l1_a", "measured"], ["COMP1", "i_relay_a", "measured"]], "y_title": "A",
         "narrative": (f"COMP1 starts at {st0.start_utc:%a %H:%M} UTC. The 1-min average start-up current is higher than "
                       f"the running current, op_state = STARTING. Start-up transients are expected behaviour: SANAD "
                       f"suppresses alarms during start/stop. Result in this run: **{fmt_cases(c_start)}** on COMP1 around the start."),
         "case_ids": c_start.case_id.tolist(),
         "expect_no_case": "No case raised: the start is recognised as normal." if not len(c_start) else ""},
        {"title": "Real overload on COMP1",
         "t0": _iso(by["PHYS_OVERLOAD"].start_utc - pad), "t1": _iso(by["PHYS_OVERLOAD"].end_utc + pad),
         "signals": [["COMP1", "i_l1_a", "measured"], ["COMP1", "i_relay_a", "measured"]], "y_title": "A",
         "narrative": ("The compressor load is raised to 1.25× rating for 40 min (scripted, illustrative). Both independent "
                       "current paths (meter CT and relay CT) rise together, the winding temperature follows the thermal model and "
                       f"the bus balance holds — the fingerprint is clean. SANAD: **{fmt_cases(c_ov)}**."),
         "case_ids": c_ov.case_id.tolist()},
        {"title": "CT drift on PUMP1 — naive vs SANAD-protected controller",
         "t0": _iso(drift.start_utc - pad), "t1": _iso(drift.end_utc + pad),
         "signals": [["PUMP1", "i_l1_a", "measured"], ["PUMP1", "i_relay_a", "measured"], ["PUMP1", "i_l1_a", "trusted"]],
         "y_title": "A",
         "narrative": ("The PUMP1 feeder-meter CT drifts low by 6 %/day (scripted, illustrative). Inside the meter, P = √3·V·I·PF "
                       "still holds (R1/R2 clean); only independent paths reveal it: R3 (meter vs relay CT) and R4 (bus balance). "
                       f"SANAD: **{fmt_cases(c_dr)}**. When confidence ≥ 0.85 the meter CT is QUARANTINED and the trusted "
                       "stream carries a relay-based ESTIMATED current (third trace)."),
         "case_ids": c_dr.case_id.tolist(),
         "controller": (f"Over the drift window, the naive controller (raw meter) exceeds the group limit for "
                        f"**{csum['naive']['minutes_over_limit']} min** (max {csum['naive']['max_excess_a']} A over); the "
                        f"SANAD-protected controller for **{csum['protected']['minutes_over_limit']} min** "
                        f"(max {csum['protected']['max_excess_a']} A). Before SANAD's case opens, both controllers "
                        "see the same (drifting) value.")},
        {"title": "kWh falsification on AUX1 — SANAD abstains and escalates",
         "t0": _iso(by["FALS_ENERGY"].start_utc - pad), "t1": _iso(by["FALS_ENERGY"].end_utc + pad),
         "signals": [["AUX1", "kwh_register", "measured"], ["AUX1", "kwh_from_power", "measured"]], "y_title": "kWh",
         "narrative": ("The AUX1 kWh register under-reports 25 % of its increments for 18 h (scripted, illustrative). The meter's "
                       "own power (second trace: ∫P dt) and the bus energy balance disagree with the register (R5, R5B). AUX1 has a single meter, so only "
                       f"one independent witness exists. SANAD: **{fmt_cases(c_fa)}**. The data is held as SUSPECT and an operator "
                       f"decides; SANAD never accuses. Scripted operator action: OVERRIDE with reason “{cfg['demo']['override_reason']}”."),
         "case_ids": c_fa.case_id.tolist()},
        {"title": "Audit — verify the chain, then tamper",
         "t0": _iso(t_a), "t1": _iso(ctrl.timestamp_gw_utc.max()), "signals": [],
         "narrative": (f"Every verdict and the operator override are entries of an append-only, SHA-256 hash-chained log. "
                       f"Verification of the stored log: **{chain_msg}**. Editing one entry (below, in memory) breaks the chain."),
         "audit": True},
    ]
    sc = {"intro": ("Scripted, deterministic scenario on a fictional plant (config `configs/scenario.yaml`, seed "
                    f"{cfg['seed']}). Days 1-3 are normal operation used to fit the detector; the scripted events follow. "
                    "Use **Next ▶** to replay it step by step."),
          "steps": steps, "controller_summary": csum, "chain_ok": ok}
    io.write_json(sc, out / "scenario.json")
    return out
