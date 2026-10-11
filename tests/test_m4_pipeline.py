"""M4: voting, abstention, governance, explanations, audit chain and the end-to-end pipeline."""

import json

import numpy as np
import pandas as pd
import pytest

from sanad import io
from sanad.audit.log import AuditLog, read_entries, verify_chain
from sanad.config import load_config
from sanad.diagnose.voting import vote
from sanad.governance.router import route


def test_vote_identifies_odd_path():
    n = 50
    base = np.full(n, 100.0)
    p = pd.DataFrame({"METER_CT": base * 0.95, "RELAY_CT": base, "BUS_IMPLIED": base * 1.001})
    assert vote(p, {}, 0.012, 10)["faulty_path"] == "METER_CT"
    p = pd.DataFrame({"METER_CT": base, "RELAY_CT": base * 1.04, "BUS_IMPLIED": base})
    assert vote(p, {}, 0.012, 10)["faulty_path"] == "RELAY_CT"
    p = pd.DataFrame({"METER_CT": base, "RELAY_CT": base, "BUS_IMPLIED": base})
    assert vote(p, {}, 0.012, 10)["faulty_path"] == "NONE"
    p = pd.DataFrame({"METER_CT": base, "RELAY_CT": np.nan, "BUS_IMPLIED": base})
    assert vote(p, {}, 0.012, 10)["faulty_path"] in ("INSUFFICIENT", "NONE_2PATH")


def test_audit_chain_and_tamper(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(5):
        log.append("VERDICT", f"2025-01-01T00:0{i}:00Z", {"case": i, "verdict": "NORMAL"})
    ok, bad, _ = verify_chain(tmp_path / "a.jsonl")
    assert ok and bad is None
    entries = read_entries(tmp_path / "a.jsonl")
    entries[2]["payload"]["verdict"] = "SENSOR_FAULT"  # tamper
    ok, bad, msg = verify_chain(entries)
    assert not ok and bad == 2 and "modified" in msg
    # re-opening the log continues the chain
    AuditLog(tmp_path / "a.jsonl").append("OPERATOR_ACTION", "2025-01-01T01:00:00Z", {"x": 1})
    assert verify_chain(tmp_path / "a.jsonl")[0]


def _case(**kw):
    c = {"verdict": "SENSOR_FAULT", "subtype": "DRIFT", "confidence": 0.9, "leading_class": "SENSOR_FAULT",
         "duration_min": 100, "sensor_path": "METER_CT", "i_pu_max": 0.8}
    c.update(kw)
    return c


def test_governance_policy():
    cfg = load_config()
    r = route(_case(), cfg)
    assert r["sensor_state"] == "QUARANTINED" and r["tier"] == "APPROVAL"  # recalibration needs approval
    assert any(a["tier"] == "AUTO" and "quarantine" in a["action"] for a in r["actions"])
    r = route(_case(confidence=0.7), cfg)
    assert r["provenance"] == "SUSPECT" and r["tier"] == "APPROVAL"
    r = route(_case(verdict="CORRUPTION_FALSIFICATION", subtype="ENERGY_UNDERREPORT", leading_class="CORRUPTION_FALSIFICATION"), cfg)
    assert r["provenance"] == "SUSPECT" and r["tier"] == "APPROVAL"
    r = route(_case(verdict="COMMUNICATION_PROBLEM", subtype="DATA_GAP", duration_min=5), cfg)
    assert r["tier"] == "AUTO" and r["provenance"] == "ESTIMATED"
    r = route(_case(verdict="PHYSICAL_EQUIPMENT_FAULT", subtype="OVERLOAD", duration_min=45), cfg)
    assert r["tier"] == "IMMEDIATE" and r["provenance"] == "VERIFIED"
    r = route(_case(verdict="UNCERTAIN"), cfg)
    assert r["tier"] == "APPROVAL"
    r = route(_case(verdict="COMMUNICATION_PROBLEM", subtype="DATA_GAP", duration_min=5), cfg, {"COMMUNICATION_PROBLEM"})
    assert r["tier"] == "APPROVAL"  # adaptive autonomy tightened the class


@pytest.fixture(scope="module")
def demo_out(tmp_path_factory):
    from sanad.config import generated_dir
    from sanad.pipeline import run_pipeline
    from sanad.sim.generator import generate

    cfg = load_config("configs/demo.yaml")
    g = generated_dir(cfg)
    if not io.exists(g / "measurements.csv"):
        g = generate(cfg, tmp_path_factory.mktemp("gen"))
    return run_pipeline(cfg, g, tmp_path_factory.mktemp("proc"))


def test_pipeline_outputs(demo_out):
    for f in ("cases.csv", "audit_log.jsonl", "trusted_stream.csv.gz", "fingerprint.csv.gz", "latency.json",
              "governance_state.json", "stage_a_calibration.json", "case_details.jsonl"):
        assert (demo_out / f).exists(), f
    assert verify_chain(demo_out / "audit_log.jsonl")[0]
    cases = io.read_csv(demo_out / "cases.csv")
    assert len(cases) > 0
    assert set(cases.verdict) <= {"NORMAL", "PHYSICAL_EQUIPMENT_FAULT", "GRID_EVENT", "SENSOR_FAULT", "BIAS_MISCALIBRATION",
                                  "CORRUPTION_FALSIFICATION", "COMMUNICATION_PROBLEM", "UNKNOWN", "UNCERTAIN"}
    # every verdict is traceable to named rules (or is UNCERTAIN/UNKNOWN with reasons)
    for _, c in cases.iterrows():
        assert json.loads(c.rules_fired) or c.verdict in ("UNCERTAIN", "UNKNOWN")
        assert isinstance(c.explanation, str) and len(c.explanation) > 20
    assert ((cases.verdict == "UNCERTAIN") == (cases.abstain_reasons != "[]")).all()


def test_trusted_stream_labels(demo_out):
    ts = io.read_csv(demo_out / "trusted_stream.csv", nrows=20000)
    labels = set()
    for c in [c for c in ts.columns if c.startswith("trust_")]:
        labels |= set(ts[c].dropna().unique())
    assert labels <= {"VERIFIED", "ESTIMATED", "SUSPECT", "QUARANTINED", ""}
    assert "VERIFIED" in labels


def test_pipeline_is_deterministic(demo_out, tmp_path):
    from sanad.config import generated_dir
    from sanad.pipeline import run_pipeline

    cfg = load_config("configs/demo.yaml")
    out2 = run_pipeline(cfg, generated_dir(cfg) if io.exists(generated_dir(cfg) / "measurements.csv") else None, tmp_path)
    a = read_entries(demo_out / "audit_log.jsonl")
    b = read_entries(out2 / "audit_log.jsonl")
    assert [e["hash"] for e in a] == [e["hash"] for e in b]
