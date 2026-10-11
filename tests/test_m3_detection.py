"""M3: relations, validity checks, EWMA/CUSUM and Isolation Forest on the demo dataset."""

import json

import numpy as np
import pandas as pd
import pytest

from sanad import io
from sanad.config import block_of, generated_dir, load_config
from sanad.detect.detector import StageA
from sanad.detect.statistical import cusum, ewma
from sanad.detect.validity import flat_run
from sanad.physics.relations import split_by_equipment


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    from sanad.sim.generator import generate

    cfg = load_config("configs/demo.yaml")
    g = generated_dir(cfg)
    if not io.exists(g / "measurements.csv"):
        g = generate(cfg, tmp_path_factory.mktemp("demo"))
    meas = io.read_measurements(g)
    eqt = io.read_csv(g / "equipment.csv").set_index("equipment_id")
    ts, frames = split_by_equipment(meas)
    a = StageA(cfg, eqt).fit(ts, frames)
    return {"cfg": cfg, "ts": ts, "res": a.run(ts, frames), "ev": io.read_events(g), "a": a}


def _window(stage, e, pad=30):
    ts = stage["ts"]
    return (ts >= e.start_utc) & (ts < e.end_utc + pd.Timedelta(minutes=pad))


def test_unit_statistics():
    z = pd.Series([0.0] * 50 + [3.0] * 50)
    c = cusum(z, 0.5, 1e9)
    assert c[49] == 0 and c[-1] == pytest.approx(125.0)
    e = ewma(pd.Series([1.0] * 200), 0.1)
    assert e[-1] == pytest.approx(1.0, abs=1e-6)
    gap = pd.Series([5.0] * 10 + [np.nan] * 100 + [0.0])
    assert abs(cusum(gap, 0.5, 1e9)[-1]) == 0  # reset after a long gap
    assert list(flat_run(pd.Series([1, 1, 1, 2, 2, np.nan, 3]))) == [1, 2, 3, 1, 2, 0, 1]


def test_normal_block_is_quiet(stage):
    blk = block_of(stage["cfg"], pd.Series(stage["ts"])).to_numpy()
    for eq, r in stage["res"].items():
        assert r["alarm"].to_numpy()[blk == "normal"].mean() < 0.002, eq


def test_every_injected_event_raises_an_alarm(stage):
    ev = stage["ev"]
    for _, e in ev[ev.subtype != "NORMAL_START"].iterrows():
        eqs = list(stage["res"]) if e.subtype.startswith("GRID") else [e.equipment_id]
        assert any(stage["res"][q]["alarm"].to_numpy()[_window(stage, e)].any() for q in eqs), e.event_id


def test_drift_breaks_r3_not_r2(stage):
    ev = stage["ev"]
    for _, e in ev[ev.subtype == "SENS_DRIFT"].iterrows():
        r = stage["res"][e.equipment_id][_window(stage, e, 0)]
        assert r["flag_R3"].any()
        assert r["flag_R2"].mean() < 0.05


def test_falsification_breaks_r5(stage):
    ev = stage["ev"]
    for _, e in ev[ev.subtype == "FALS_ENERGY"].iterrows():
        r = stage["res"][e.equipment_id][_window(stage, e, 0)]
        assert r["flag_R5"].mean() > 0.3
        assert r["z_R5"].median() < 0


def test_overload_is_coherent(stage):
    ev = stage["ev"]
    for _, e in ev[ev.subtype == "PHYS_OVERLOAD"].iterrows():
        r = stage["res"][e.equipment_id][_window(stage, e, 0)]
        assert r["v_overcurrent"].any()
        assert r["flag_R3"].mean() < 0.2  # both current paths agree: the plant really is overloaded


def test_iforest_scores_exist(stage):
    for eq, r in stage["res"].items():
        assert r["if_score"].notna().all()
        assert eq in stage["a"].iforest.thresholds
