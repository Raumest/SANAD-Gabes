"""M7: scripted demo scenario and the naive vs SANAD-protected controller."""

import json

import numpy as np
import pandas as pd

from sanad import io
from sanad.config import load_config
from sanad.demo import controllers
from sanad.noosphere.pattern_card import JSON_SCHEMA, example_card
from sanad.config import repo_path


def test_controller_uses_trusted_estimate():
    cfg = load_config("configs/scenario.yaml")
    n = 100
    ts = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    true = pd.DataFrame({"i_l1_a": 120.0, "i_l2_a": 120.0, "i_l3_a": 120.0, "op_state": "RUN_NORMAL"}, index=range(n))
    meas = true.copy()
    meas[["i_l1_a", "i_l2_a", "i_l3_a"]] = 108.0  # meter CT reads 10 % low
    trusted = true.copy()
    trusted["timestamp_gw_utc"] = ts
    trusted["i_relay_a"] = 120.0
    trusted["trust_meter"] = "ESTIMATED"
    df = controllers.simulate(cfg, true, meas, trusted)
    limit = cfg["demo"]["group_current_limit_a"]
    assert (df.true_total_naive_a > limit).all()
    assert (df.true_total_protected_a <= limit).all()


def test_pattern_card_schema():
    card = json.loads(example_card().to_json())
    for k in JSON_SCHEMA["required"]:
        assert k in card
    assert len(card["signature"]) == 64


def test_committed_scenario_is_consistent():
    sc_dir = repo_path("data/processed/demo_scenario")
    sc = io.read_json(sc_dir / "scenario.json")
    assert len(sc["steps"]) == 5 and sc["chain_ok"]
    cases = io.read_csv(sc_dir / "cases.csv")
    assert set(sum((s.get("case_ids", []) for s in sc["steps"]), [])) <= set(cases.case_id)
    s = sc["controller_summary"]
    assert s["protected"]["minutes_over_limit"] <= s["naive"]["minutes_over_limit"]
