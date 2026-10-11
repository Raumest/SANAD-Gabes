"""M5: evaluation with time-block split writes metrics.json / metrics.md from code."""

import json

import pandas as pd

from sanad import io
from sanad.config import load_config, processed_dir
from sanad.evaluate.matching import match_cases
from sanad.evaluate.metrics import block_metrics
from sanad.evaluate.report import evaluate_run


def test_matching_and_metrics_on_toy_data():
    t = pd.Timestamp("2025-01-01", tz="UTC")
    ev = pd.DataFrame({"event_id": ["E1", "E2"], "main_class": ["SENSOR_FAULT", "GRID_EVENT"],
                       "subtype": ["SENS_DRIFT", "GRID_SAG"], "origin": ["MEASUREMENT", "PHYSICAL"],
                       "equipment_id": ["PUMP1", "BUS1"], "start_utc": [t, t + pd.Timedelta(hours=5)],
                       "end_utc": [t + pd.Timedelta(hours=2), t + pd.Timedelta(hours=5, minutes=2)],
                       "gt_detectability": ["HIGH", "HIGH"], "block": ["test", "test"]})
    cases = pd.DataFrame({"case_id": ["C1", "C2", "C3"], "equipment_id": ["PUMP1", "BUS1", "AUX1"],
                          "kind": ["equipment", "grid", "equipment"],
                          "start_utc": [t + pd.Timedelta(minutes=30), t + pd.Timedelta(hours=5), t + pd.Timedelta(hours=9)],
                          "end_utc": [t + pd.Timedelta(hours=2), t + pd.Timedelta(hours=5, minutes=3), t + pd.Timedelta(hours=10)],
                          "leading_class": ["SENSOR_FAULT", "GRID_EVENT", "BIAS_MISCALIBRATION"],
                          "subtype": ["DRIFT", "SUSTAINED_UNDERVOLTAGE", "VOLTAGE"],
                          "verdict": ["SENSOR_FAULT", "GRID_EVENT", "BIAS_MISCALIBRATION"]})
    m = match_cases(cases, ev)
    assert list(m.matched_event_id) == ["E1", "E2", ""]
    res = block_metrics(m, ev, days=1.0)
    assert res["event_detection_rate"] == 1.0 and res["event_accuracy"] == 1.0
    assert res["false_positive_cases"] == 1 and res["false_positives_per_day"] == 1.0
    assert res["per_class"]["SENSOR_FAULT"]["delay_min_median"] == 30.0


def test_evaluate_demo_run(tmp_path):
    cfg = load_config("configs/demo.yaml")
    if not io.exists(processed_dir(cfg) / "cases.csv"):
        from sanad.pipeline import run_pipeline
        run_pipeline(cfg)
    out = evaluate_run(cfg, tmp_path)
    res = json.loads((out / "metrics.json").read_text())
    assert set(res["blocks"]) == {"train", "val", "test"}
    assert "confusion_matrix" in res["blocks"]["test"]
    md = (out / "metrics.md").read_text()
    assert "Event counts per label and block" in md and "Fictional process plant" in md
