"""End-to-end SANAD pipeline.

stream -> detect (Stage A) -> diagnose (Stage B) -> confidence + abstention (Stage C)
-> governance routing -> explanation -> hash-chained audit log -> trusted data stream.

Inputs: measurements.csv, equipment.csv, sensors.csv (never truth_clean.csv). events.csv
is read ONLY to fit the isotonic confidence calibration on validation-block cases.
Outputs go to data/processed/<run_id>/ and are what the dashboard reads.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sanad import io
from sanad.audit.log import AuditLog, canonical
from sanad.confidence.abstain import abstain_reasons
from sanad.confidence.calibration import Calibrator
from sanad.confidence.scoring import raw_confidence
from sanad.config import block_of, generated_dir, processed_dir
from sanad.detect.detector import StageA
from sanad.diagnose.episodes import build_episodes
from sanad.diagnose.signatures import FLAT, FLAT_PATH, episode_features, hypotheses_for, score
from sanad.diagnose.voting import bus_implied_current, fit_baselines, path_table
from sanad.evaluate.matching import match_cases
from sanad.explain.templates import explain, held_relations, top_factors
from sanad.governance.router import escalate_concurrent, route
from sanad.governance.trust import read_feedback, second_look, tightened_classes, track_record
from sanad.physics.relations import RELATIONS, split_by_equipment, thermal_model
from sanad.schema import EQUIPMENT_ORDER, MOTORS, VALUE_COLUMNS

GROUPS = {
    "meter": ["v_l1_v", "v_l2_v", "v_l3_v", "i_l1_a", "i_l2_a", "i_l3_a", "p_kw", "q_kvar", "s_kva", "pf",
              "freq_hz", "thd_v_pct", "thd_i_pct"],
    "energy": ["kwh_register"], "relay": ["i_relay_a"], "rtd": ["winding_temp_c"], "speed": ["speed_rpm"],
    "acc_de": ["vib_de_g"], "acc_nde": ["vib_nde_g"], "ambient": ["ambient_temp_c"],
}
PATH_GROUP = {"METER_CT": "meter", "BUS_CT": "meter", "RELAY_CT": "relay", "RTD": "rtd", "SPEED": "speed",
              "ACC_DE": "acc_de", "ACC_NDE": "acc_nde"}


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _severity(cls: str, sub: str, f: dict) -> str:
    if cls == "GRID_EVENT":
        return "S3" if sub == "INTERRUPTION" or (f.get("v_pu_min", 1) == f.get("v_pu_min", 1) and f.get("v_pu_min", 1) < 0.6) else "S2"
    if cls == "PHYSICAL_EQUIPMENT_FAULT":
        return "S3" if f.get("i_pu_max", 0) >= 1.25 else "S2"
    if cls == "CORRUPTION_FALSIFICATION":
        return "S3"
    if cls == "COMMUNICATION_PROBLEM":
        return "S2" if f["dur_min"] >= 30 else "S1"
    return "S2"


def _sensor_path(eq: str, top: dict, f: dict) -> str:
    h = top["hyp"]
    if h.subtype == "STUCK":
        g = max(FLAT, key=lambda c: f[f"{c}_frac"])
        return FLAT_PATH[g]
    if h.path == "CT":
        return f["vote"] if f["vote"] in ("METER_CT", "RELAY_CT") else "METER_CT"
    if h.path == "ACC":
        return "ACC_DE" if f["viol_R10E_DE"] >= f["viol_R10E_NDE"] else "ACC_NDE"
    if h.path == "METER_CT" and eq == "BUS1":
        return "BUS_CT"
    return h.path


class Pipeline:
    """Holds the fitted stages so that the dashboard/demo can reuse them."""

    def __init__(self, cfg: dict, gen_dir: Path | None = None):
        self.cfg = cfg
        self.gen_dir = Path(gen_dir) if gen_dir else generated_dir(cfg)
        self.timing: dict[str, float] = {}

    # ------------------------------------------------------------------ stage A
    def detect(self) -> None:
        t0 = time.perf_counter()
        meas = io.read_measurements(self.gen_dir)
        self.eqt = io.read_csv(self.gen_dir / "equipment.csv").set_index("equipment_id")
        self.ts, self.frames = split_by_equipment(meas)
        self.stage_a = StageA(self.cfg, self.eqt).fit(self.ts, self.frames)
        self.res = self.stage_a.run(self.ts, self.frames)
        self.paths = path_table(self.frames, bus_implied_current(self.frames, self.stage_a.calib))
        self.vote_base = fit_baselines(self.paths, self.stage_a._masks(self.ts, self.frames))
        self.timing["stage_a_s"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ stage B
    def diagnose(self) -> list[dict]:
        cfg = self.cfg
        episodes = build_episodes(self.res, cfg)
        blocks = block_of(cfg, pd.Series(self.ts)).to_numpy()
        cases = []
        for i, ep in enumerate(episodes):
            t0 = time.perf_counter()
            f = episode_features(ep, self.res, self.paths, self.vote_base, cfg, self.eqt)
            ranked = raw_confidence(score(f, hypotheses_for(ep.equipment_id, ep.kind, ep.families)), cfg["diagnosis"]["null_evidence"])
            dt = time.perf_counter() - t0
            cases.append({
                "case_id": f"C{i + 1:05d}", "equipment_id": ep.equipment_id, "kind": ep.kind,
                "k0": ep.k0, "k1": ep.k1, "start_utc": self.ts[ep.k0], "end_utc": self.ts[ep.k1 - 1] + pd.Timedelta(minutes=1),
                "duration_min": ep.k1 - ep.k0, "block": blocks[ep.k0], "features": f, "ranked": ranked,
                "families": ",".join(sorted(ep.families)) or "grid",
                "leading_class": ranked[0]["main_class"], "leading_subtype": ranked[0]["subtype"],
                "raw_conf": ranked[0]["conf"], "_diag_s": dt,
            })
        return cases

    # ------------------------------------------------------------------ stage C
    def calibrate(self, cases: list[dict]) -> Calibrator:
        cal = Calibrator(self.cfg["confidence"]["calibration"])
        ext = self.cfg["confidence"].get("calibration_file")
        if ext:  # reuse a calibration fitted on another run's validation block (e.g. scenario)
            from sanad.config import repo_path

            if repo_path(ext).exists():
                cal = Calibrator.from_json(io.read_json(repo_path(ext)))
                cal.info["source"] = str(ext)
                return cal
        if not io.exists(self.gen_dir / "events.csv"):
            return cal
        val = [c for c in cases if c["block"] == "val"]
        if not val:
            return cal
        df = pd.DataFrame([{k: c[k] for k in ("case_id", "equipment_id", "kind", "start_utc", "end_utc", "leading_class", "raw_conf")} for c in val])
        m = match_cases(df, io.read_events(self.gen_dir))
        return cal.fit(m["raw_conf"].to_numpy(), (m["leading_class"] == m["truth_class"]).to_numpy())

    def decide(self, cases: list[dict], cal: Calibrator, tightened: set[str]) -> None:
        cfg = self.cfg
        for c in cases:
            t0 = time.perf_counter()
            f, ranked = c["features"], c["ranked"]
            top, runner = ranked[0], (ranked[1] if len(ranked) > 1 else None)
            conf = float(cal.transform([top["conf"]])[0])
            eq = c["equipment_id"]
            reasons = abstain_reasons(f, top, runner, conf, self.stage_a.iforest.thresholds[eq], cfg)
            c.update({
                "confidence": round(conf, 4), "raw_conf": round(top["conf"], 4),
                "runner_up": runner["key"] if runner else "", "runner_up_conf": round(runner["conf"], 4) if runner else 0.0,
                "margin": round(top["conf"] - (runner["conf"] if runner else 0.0), 4),
                "abstain_reasons": reasons, "verdict": "UNCERTAIN" if reasons else top["main_class"],
                "subtype": top["subtype"], "sensor_path": _sensor_path(eq, top, f),
                "severity": _severity(top["main_class"], top["subtype"], f), "i_pu_max": f["i_pu_max"],
                "rules_fired": [r["rule"] for r in top["rules"]],
                "evidence": top["rules"], "top_factors": top_factors(f), "held_relations": held_relations(f),
            })
            c["sensor_id"] = f"{eq}_{c['sensor_path']}" if c["sensor_path"] else ""
            c.update(route(c, cfg, tightened))
            c["explanation"] = explain(c, f)
            c["second_look"] = c["tier"] == "AUTO" and second_look(c["case_id"], cfg["governance"]["second_look_rate"], cfg["seed"])
            c["_decide_s"] = time.perf_counter() - t0
        escalate_concurrent(cases)

    # ------------------------------------------------------------------ trusted stream
    def trusted_stream(self, cases: list[dict]) -> pd.DataFrame:
        cfg = self.cfg
        short = int(cfg["governance"]["short_gap_max_min"])
        parts = []
        calib = self.stage_a.calib
        ts_str = np.asarray(self.ts.strftime("%Y-%m-%dT%H:%M:%SZ"), dtype=object)
        for eq in EQUIPMENT_ORDER:
            d = self.frames[eq]
            vals = d[VALUE_COLUMNS].copy()
            app = set(self.eqt.loc[eq, "applicable_columns"].split(";"))
            labels = {}
            for g, cols in GROUPS.items():
                cols = [c for c in cols if c in app]
                if not cols:
                    labels[g] = np.full(len(d), "", dtype=object)
                    continue
                present = vals[cols[0] if g != "meter" else "p_kw"].notna().to_numpy()
                lab = np.where(present, "VERIFIED", "QUARANTINED").astype(object)
                filled = vals[cols].interpolate(limit=short, limit_area="inside")
                newly = (~present) & filled[cols[0] if g != "meter" else "p_kw"].notna().to_numpy()
                vals[cols] = filled
                lab[newly] = "ESTIMATED"
                labels[g] = lab
            for c in [c for c in cases if c["equipment_id"] == eq or (c["kind"] == "grid")]:
                sl = slice(c["k0"], c["k1"])
                path_g = PATH_GROUP.get(c["sensor_path"], "")
                if c["kind"] == "grid" or not path_g:
                    if c["provenance"] == "SUSPECT" and c["equipment_id"] == eq:
                        for g in ("meter", "energy"):
                            labels[g][sl] = np.where(labels[g][sl] == "VERIFIED", "SUSPECT", labels[g][sl])
                    continue
                groups = [path_g] + (["energy"] if c["subtype"] in ("STUCK", "ENERGY_UNDERREPORT") and path_g == "meter" else [])
                if c["subtype"] == "ENERGY_UNDERREPORT":
                    groups = ["energy"]
                if c["sensor_state"] == "QUARANTINED":
                    for g in groups:
                        est = self._estimate(eq, g, c, calib)
                        if est is not None:
                            vals.loc[vals.index[sl], est.columns] = est.to_numpy()
                            labels[g][sl] = "ESTIMATED"
                        else:
                            vals.loc[vals.index[sl], [x for x in GROUPS[g] if x in app]] = np.nan
                            labels[g][sl] = "QUARANTINED"
                elif c["provenance"] == "SUSPECT":
                    for g in groups:
                        labels[g][sl] = np.where(labels[g][sl] == "VERIFIED", "SUSPECT", labels[g][sl])
            out = pd.DataFrame({"timestamp_gw_utc": ts_str, "equipment_id": eq}, index=d.index)
            out = pd.concat([out, vals.round(4)], axis=1)
            for g in GROUPS:
                out[f"trust_{g}"] = labels[g]
            parts.append(out)
        df = pd.concat(parts, ignore_index=True)
        df["_k"] = np.tile(np.arange(len(self.ts)), len(EQUIPMENT_ORDER))
        df["_o"] = np.repeat(np.arange(len(EQUIPMENT_ORDER)), len(self.ts))
        return df.sort_values(["_k", "_o"]).drop(columns=["_k", "_o"]).reset_index(drop=True)

    def _estimate(self, eq: str, group: str, c: dict, calib: dict) -> pd.DataFrame | None:
        """Substitute estimate for a quarantined group from independent paths."""
        d = self.frames[eq].iloc[c["k0"]:c["k1"]]
        base = self.vote_base.get(eq, {})
        if group == "meter" and eq in MOTORS:
            i_est = d["i_relay_a"] * np.exp(base.get("METER_CT/RELAY_CT", 0.0))
            ratio = d[["i_l1_a", "i_l2_a", "i_l3_a"]].div(d["i_avg"], axis=0).median()
            out = pd.DataFrame({f"i_l{k}_a": i_est * ratio.get(f"i_l{k}_a", 1.0) for k in (1, 2, 3)})
            p = 3 * d["v_avg"] * i_est * d["pf"] / 1000.0
            s = 3 * d["v_avg"] * i_est / 1000.0
            out["p_kw"], out["s_kva"], out["q_kvar"] = p, s, np.sqrt(np.maximum(s**2 - p**2, 0))
            return out
        if group == "relay":
            return pd.DataFrame({"i_relay_a": d["i_avg"] * np.exp(-self.vote_base.get(eq, {}).get("METER_CT/RELAY_CT", 0.0))})
        if group == "rtd":
            full = self.frames[eq]
            model = thermal_model(full, self.eqt.loc[eq])
            c0, c1 = calib["r6"][eq]
            est = model + c0 + c1 * (model - full["ambient_temp_c"].to_numpy())
            return pd.DataFrame({"winding_temp_c": est[c["k0"]:c["k1"]]}, index=d.index)
        if group == "speed":
            e = self.eqt.loc[eq]
            L = (d["p_kw"] * e["eta_rated"] / e["p_rated_kw"]).clip(0, 1.6)
            return pd.DataFrame({"speed_rpm": 120 * d["freq_hz"] / e["poles"] * (1 - e["slip_rated"] * L)})
        if group == "acc_de":
            return pd.DataFrame({"vib_de_g": d["vib_nde_g"] * np.exp(self.stage_a.norm[eq].get("R10", {}).get("median", 0.0))})
        return None

    # ------------------------------------------------------------------ outputs
    def fingerprint(self) -> pd.DataFrame:
        """5-min max |z| per relation and equipment (Trust Fingerprint heatmap)."""
        parts = []
        for eq in EQUIPMENT_ORDER:
            z = self.res[eq][[f"z_{r}" for r in RELATIONS]].abs().clip(upper=30)
            z.columns = RELATIONS
            z["alarm"] = self.res[eq]["alarm"].astype(int)
            z.index = self.ts
            agg = z.resample("5min").max().round(1)
            agg.insert(0, "equipment_id", eq)
            parts.append(agg.reset_index(names="timestamp_utc"))
        df = pd.concat(parts, ignore_index=True)
        df["timestamp_utc"] = df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return df


CASE_COLUMNS = [
    "case_id", "equipment_id", "kind", "families", "start_utc", "end_utc", "duration_min", "block", "verdict", "leading_class",
    "subtype", "confidence", "raw_conf", "runner_up", "runner_up_conf", "margin", "abstain_reasons", "severity",
    "tier", "actions", "provenance", "sensor_state", "sensor_path", "sensor_id", "explanation", "rules_fired",
    "top_factors", "held_relations", "second_look", "k0", "k1",
]


def _case_row(c: dict) -> dict:
    row = {k: c[k] for k in CASE_COLUMNS}
    row["start_utc"], row["end_utc"] = _iso(c["start_utc"]), _iso(c["end_utc"])
    for k in ("abstain_reasons", "actions", "rules_fired", "top_factors", "held_relations"):
        row[k] = json.dumps(c[k], ensure_ascii=False)
    return row


def run_pipeline(cfg: dict, gen_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else processed_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    p = Pipeline(cfg, gen_dir)
    p.detect()
    t0 = time.perf_counter()
    cases = p.diagnose()
    cal = p.calibrate(cases)
    fb = read_feedback(out / "feedback.jsonl")
    tightened: set[str] = set()
    if len(fb):
        prev = io.read_csv(out / "cases.csv") if io.exists(out / "cases.csv") else pd.DataFrame(columns=["case_id", "verdict"])
        tightened = tightened_classes(track_record(prev, fb), cfg, cal.info.get("ece_after"))
    p.decide(cases, cal, tightened)
    p.timing["stage_bc_s"] = time.perf_counter() - t0

    # audit log: rebuilt from scratch for every run (operator actions are appended later)
    log_path = out / cfg["audit"]["log_name"]
    if log_path.exists():
        log_path.unlink()
    log = AuditLog(log_path)
    log.append("RUN_START", _iso(p.ts[0]), {
        "run_id": cfg["run_id"], "config": cfg.get("_config_path", ""), "seed": cfg["seed"],
        "n_minutes": len(p.ts), "stage_a_calibration": "stage_a_calibration.json", "confidence_calibration": cal.info.get("method"),
    })
    for c in cases:
        log.append("VERDICT", _iso(c["end_utc"]), {k: c[k] for k in (
            "case_id", "equipment_id", "verdict", "subtype", "confidence", "runner_up", "runner_up_conf", "abstain_reasons",
            "tier", "actions", "rules_fired", "explanation", "sensor_id", "severity")} | {
            "start_utc": _iso(c["start_utc"]), "end_utc": _iso(c["end_utc"]),
            "top_factors": [t["relation"] for t in c["top_factors"]]})

    cases_df = pd.DataFrame([_case_row(c) for c in cases], columns=CASE_COLUMNS)
    io.write_csv(cases_df, out / "cases.csv")
    with open(out / "case_details.jsonl", "w", encoding="utf-8") as fh:
        for c in cases:
            det = {"case_id": c["case_id"], "features": {k: v for k, v in c["features"].items()},
                   "hypotheses": [{"key": h["key"], "score": round(h["score"], 4), "conf": round(h["conf"], 4), "rules": h["rules"]}
                                  for h in c["ranked"][:5]], "evidence": c["evidence"]}
            fh.write(canonical(det) + "\n")
    t1 = time.perf_counter()
    io.write_csv(p.trusted_stream(cases), out / "trusted_stream.csv", compress=True)
    io.write_csv(p.fingerprint(), out / "fingerprint.csv", compress=True)
    p.timing["outputs_s"] = time.perf_counter() - t1
    a = p.stage_a
    io.write_json({"structural": a.calib, "normalisation": a.norm, "thresholds": a.thresholds,
                   "iforest_thresholds": a.iforest.thresholds, "vote_baselines": p.vote_base}, out / "stage_a_calibration.json")
    io.write_json(cal.to_json(), out / "confidence_calibration.json")
    lat = np.array([(c["_diag_s"] + c["_decide_s"]) * 1000 for c in cases]) if cases else np.array([0.0])
    io.write_json({
        "note": "wall-clock timings; vary between machines and runs",
        "stage_a_total_s": round(p.timing["stage_a_s"], 3),
        "stage_a_per_minute_cycle_ms": round(p.timing["stage_a_s"] * 1000 / len(p.ts), 4),
        "decision_ms_p50": round(float(np.percentile(lat, 50)), 3), "decision_ms_p95": round(float(np.percentile(lat, 95)), 3),
        "stage_bc_total_s": round(p.timing["stage_bc_s"], 3), "outputs_s": round(p.timing["outputs_s"], 3),
        "n_cases": len(cases),
    }, out / "latency.json")
    gs = {"tightened_classes": sorted(tightened), "second_look_rate": cfg["governance"]["second_look_rate"],
          "override_rate_limit": cfg["governance"]["override_rate_limit"], "calibration": cal.info}
    io.write_json(gs, out / "governance_state.json")
    for f in ("equipment.csv", "sensors.csv"):
        shutil.copy(p.gen_dir / f, out / f)
    return out
