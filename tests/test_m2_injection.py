"""M2: injection library, events.csv and leakage-safe outputs."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from sanad import io
from sanad.config import load_config
from sanad.schema import FORBIDDEN_IN_MEASUREMENTS, MEASUREMENT_COLUMNS
from sanad.sim.generator import generate


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    cfg = load_config("configs/demo.yaml")
    cfg["paths"]["compress"] = False
    out = generate(cfg, tmp_path_factory.mktemp("demo"))
    ev = io.read_events(out)
    ev["block"] = ev.params_json.map(lambda s: json.loads(s)["block"])
    return {"cfg": cfg, "dir": out, "m": io.read_measurements(out), "t": io.read_csv(out / "truth_clean.csv"), "ev": ev}


def _rows(df, ev):
    d = df[df.equipment_id == ev.equipment_id].copy()
    ts = pd.to_datetime(d.timestamp_gw_utc, utc=True)
    return d[(ts >= ev.start_utc) & (ts < ev.end_utc)]


def test_measurements_have_no_ground_truth(demo_run):
    m = demo_run["m"]
    assert list(m.columns) == MEASUREMENT_COLUMNS
    for c in m.columns:
        assert not any(tok == c or c.startswith(tok + "_") for tok in FORBIDDEN_IN_MEASUREMENTS), c


def test_event_counts_per_block(demo_run):
    ev, cfg = demo_run["ev"], demo_run["cfg"]
    per = cfg["injection"]["events_per_block"]
    inj = ev[ev.subtype != "NORMAL_START"]
    for label in cfg["taxonomy"]:
        if label == "NORMAL_START":
            continue
        for block in ("train", "val", "test"):
            n = ((inj.subtype == label) & (inj.block == block)).sum()
            assert n >= per[block], (label, block, n)
    # no injected event in the normal-only block
    assert not (inj.block == "normal").any()


def test_onsets_not_round(demo_run):
    inj = demo_run["ev"][demo_run["ev"].subtype != "NORMAL_START"]
    assert (inj.start_utc.dt.minute % 15 != 0).all()


def test_physical_overload_changes_truth(demo_run):
    ev = demo_run["ev"]
    e = ev[ev.subtype == "PHYS_OVERLOAD"].iloc[0]
    t = _rows(demo_run["t"], e)
    m = _rows(demo_run["m"], e)
    assert t.load_fraction.iloc[5:].min() > 1.05
    # both independent current paths see the real overload
    assert m.i_relay_a.iloc[5:].median() > 0.98 * t.i_relay_a.iloc[5:].median()


def test_drift_only_measured_channel(demo_run):
    ev = demo_run["ev"]
    e = ev[ev.subtype == "SENS_DRIFT"].iloc[0]
    rate = json.loads(e.params_json)["rate_pct_per_day"]
    m = _rows(demo_run["m"], e)
    t = _rows(demo_run["t"], e)
    run = m.op_state.isin(["RUN_LOW", "RUN_NORMAL", "RUN_HIGH"]).to_numpy()
    tail = slice(int(len(m) * 0.8), None)
    meter = m[["i_l1_a", "i_l2_a", "i_l3_a"]].mean(axis=1).to_numpy()[tail][run[tail]]
    relay = m.i_relay_a.to_numpy()[tail][run[tail]]
    truth = t[["i_l1_a", "i_l2_a", "i_l3_a"]].mean(axis=1).to_numpy()[tail][run[tail]]
    ratio = np.nanmedian(meter / relay) - 1
    assert np.sign(ratio) == np.sign(rate) and abs(ratio) > 0.01
    assert abs(np.nanmedian(relay / truth) - 1) < 0.02  # relay unaffected
    # R2 still holds inside the drifting meter (golden rule 3)
    v = m[["v_l1_v", "v_l2_v", "v_l3_v"]].mean(axis=1)
    i = m[["i_l1_a", "i_l2_a", "i_l3_a"]].mean(axis=1)
    r2 = (m.p_kw - 3 * v * i * m.pf / 1000).abs() / 95
    assert r2.quantile(0.95) < 0.01


def test_falsification_under_reports_register(demo_run):
    ev = demo_run["ev"]
    e = ev[ev.subtype == "FALS_ENERGY"].iloc[0]
    m = _rows(demo_run["m"], e).dropna(subset=["kwh_register"])
    t = _rows(demo_run["t"], e)
    dm = m.kwh_register.iloc[-1] - m.kwh_register.iloc[0]
    dt = t.kwh_register.iloc[-1] - t.kwh_register.iloc[0]
    assert dm < 0.95 * dt


def test_comm_dropout_rows_lost(demo_run):
    ev = demo_run["ev"]
    e = ev[ev.subtype == "COMM_DROPOUT"].iloc[0]
    m = _rows(demo_run["m"], e)
    assert (m.comm_status == "LOST").all()
    assert m[["p_kw", "v_l1_v", "kwh_register"]].isna().all().all()


def test_grid_sag_on_all_meters(demo_run):
    ev = demo_run["ev"]
    e = ev[ev.subtype == "GRID_SAG"].iloc[0]
    vf = json.loads(e.params_json)["v_factor"]
    t = demo_run["t"]
    ts = pd.to_datetime(t.timestamp_gw_utc, utc=True)
    sel = t[(ts >= e.start_utc) & (ts < e.end_utc)]
    assert np.allclose(sel.v_l1_v / 230.94, vf, atol=0.03)


def test_baseline_imperfections_in_normal_block(demo_run):
    m = demo_run["m"]
    ts = m.timestamp_gw_utc
    normal = m[ts < ts.min() + pd.Timedelta(days=demo_run["cfg"]["split"]["normal_days"])]
    assert normal.p_kw.isna().sum() > 0  # NaN also occurs in normal data
    assert (normal.comm_status == "DELAYED").sum() > 0


def test_reproducible_with_injections(demo_run, tmp_path):
    out = generate(demo_run["cfg"], tmp_path / "again")
    for f in ("measurements.csv", "events.csv"):
        assert hashlib.sha256((out / f).read_bytes()).hexdigest() == hashlib.sha256((demo_run["dir"] / f).read_bytes()).hexdigest()
