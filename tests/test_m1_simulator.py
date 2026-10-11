"""M1: the clean simulator obeys the physics relations R1-R10 (Trust Fingerprint)."""

import hashlib

import numpy as np
import pandas as pd

from sanad.sim import electrical as el
from sanad.sim.generator import generate
from sanad.sim.thermal import thermal_response
from tests.conftest import small_cfg


def _eq(df, eq):
    return df[df.equipment_id == eq].reset_index(drop=True)


def _running(d):
    return d.op_state.isin(["RUN_LOW", "RUN_NORMAL", "RUN_HIGH"]).to_numpy()


def test_pf_curve_and_ratings():
    assert abs(el.motor_pf(np.array(0.2)) - 0.55) < 0.01
    assert abs(el.motor_pf(np.array(1.0)) - 0.87) < 0.01
    rv = el.rated_values({"type": "motor", "p_rated_kw": 90, "eta_rated": 0.945, "no_load_loss_share": 0.3}, 400)
    assert 150 < rv["i_rated_a"] < 165


def test_files_and_disclaimer(clean_run):
    for f in ("measurements.csv", "truth_clean.csv", "events.csv", "equipment.csv", "sensors.csv", "generation_config.yaml"):
        p = clean_run["dir"] / f
        assert p.exists(), f
        assert "Fictional process plant" in p.read_text(encoding="utf-8").splitlines()[0]


def test_r1_r2_within_meter(clean_run):
    m = clean_run["m"]
    for eq in ("BUS1", "PUMP1", "COMP1", "AUX1"):
        d = _eq(m, eq).dropna(subset=["pf"])
        p_rated = clean_run["eq"].loc[eq, "p_in_rated_kw"]
        r1 = d.s_kva - np.sqrt(d.p_kw**2 + d.q_kvar**2)
        assert r1.abs().max() < 0.05
        v = d[["v_l1_v", "v_l2_v", "v_l3_v"]].mean(axis=1)
        i = d[["i_l1_a", "i_l2_a", "i_l3_a"]].mean(axis=1)
        r2 = (d.p_kw - 3 * v * i * d.pf / 1000) / p_rated
        assert r2.abs().max() < 0.002


def test_r3_meter_vs_relay(clean_run):
    m = clean_run["m"]
    for eq in ("PUMP1", "COMP1"):
        d = _eq(m, eq)
        run = _running(d)
        rel = (d[["i_l1_a", "i_l2_a", "i_l3_a"]].mean(axis=1) - d.i_relay_a) / clean_run["eq"].loc[eq, "i_rated_a"]
        assert rel[run].abs().quantile(0.99) < 0.03


def test_r4_bus_balance_truth(clean_run):
    t = clean_run["t"]
    bus = _eq(t, "BUS1").p_kw
    feeders = sum(_eq(t, e).p_kw for e in ("PUMP1", "COMP1", "AUX1"))
    resid = bus - 1.015 * feeders - 2.0
    assert resid.abs().max() < 3.0


def test_r5_energy_integration(clean_run):
    t = clean_run["t"]
    m = clean_run["m"]
    for eq in ("PUMP1", "AUX1"):
        d = _eq(t, eq)
        dk = d.kwh_register.diff().iloc[1:]
        assert (dk - d.p_kw.iloc[1:] / 60).abs().max() < 0.02
        mm = _eq(m, eq)
        hourly = mm.kwh_register.diff(60) - mm.p_kw.rolling(60).sum() / 60
        assert hourly.abs().max() < 0.05


def test_r6_thermal_model(clean_run):
    t, eqt = clean_run["t"], clean_run["eq"]
    for eq in ("PUMP1", "COMP1"):
        d = _eq(t, eq)
        i_pu = d[["i_l1_a", "i_l2_a", "i_l3_a"]].mean(axis=1) / eqt.loc[eq, "i_rated_a"]
        model = thermal_response(i_pu.to_numpy(), d.ambient_temp_c.to_numpy(), eqt.loc[eq, "tau_thermal_min"], eqt.loc[eq, "dtheta_rated_k"])
        assert np.abs(model - d.winding_temp_c)[120:].max() < 1.0


def test_r7_speed(clean_run):
    t = clean_run["t"]
    for eq, s in (("PUMP1", 0.012), ("COMP1", 0.010)):
        d = _eq(t, eq)
        run = _running(d)
        exp = 120 * d.freq_hz / 4 * (1 - s * d.load_fraction)
        assert (d.speed_rpm - exp)[run].abs().max() < 1.0


def test_r8_r9_cross_meter_agreement(clean_run):
    m = clean_run["m"]
    v = m.assign(v=m[["v_l1_v", "v_l2_v", "v_l3_v"]].mean(axis=1)).pivot(index="timestamp_gw_utc", columns="equipment_id", values="v")
    assert (v.max(axis=1) - v.min(axis=1)).max() / 230.9 < 0.01
    f = m.pivot(index="timestamp_gw_utc", columns="equipment_id", values="freq_hz")
    assert (f.max(axis=1) - f.min(axis=1)).max() < 0.03


def test_r10_de_nde(clean_run):
    t = clean_run["t"]
    d = _eq(t, "PUMP1")
    run = _running(d)
    ratio = (d.vib_nde_g / d.vib_de_g)[run]
    assert abs(ratio.median() - 0.85) < 0.02


def test_non_applicable_columns_empty(clean_run):
    m = clean_run["m"]
    aux = _eq(m, "AUX1")
    assert aux[["i_relay_a", "speed_rpm", "winding_temp_c", "vib_de_g", "vib_nde_g"]].isna().all().all()


def test_reproducible_same_seed(tmp_path):
    cfg = small_cfg(1, clean=False)
    cfg["injection"]["enabled"] = False  # baseline noise/jitter on, no scheduled events
    a = generate(cfg, tmp_path / "a")
    b = generate(cfg, tmp_path / "b")
    for f in ("measurements.csv", "truth_clean.csv", "events.csv"):
        ha = hashlib.sha256((a / f).read_bytes()).hexdigest()
        hb = hashlib.sha256((b / f).read_bytes()).hexdigest()
        assert ha == hb, f
