"""TRUE plant state (golden rule 1: physical events act here and propagate everywhere).

``build_drivers`` creates the exogenous drivers (grid, climate, commanded schedules).
Physical events (``sanad.inject.physical``) modify those drivers (load, bearing health,
voltage factor, interruption). ``compute_true_state`` then derives every dependent
variable (currents, power, speed, temperature, vibration, bus balance, energy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanad.config import rng_for, time_index
from sanad.schema import FEEDERS
from sanad.sim import climate, electrical as el, load, thermal, vibration
from sanad.sim.noise import ar1

V_LN_NOM = 400.0 / np.sqrt(3.0)
BUS_UNBALANCE = np.array([0.004, -0.002, -0.002])


def build_drivers(cfg: dict) -> dict:
    """Exogenous drivers before any event: grid voltage/frequency, climate, schedules."""
    idx = time_index(cfg)
    n = len(idx)
    g = cfg["plant"]["grid"]
    rng = rng_for(cfg, "grid")
    h = climate.local_hours(idx, cfg["time"]["utc_offset_h"])
    v_rel = 1.0 + g["v_daily_amp"] * np.sin(2 * np.pi * (h - 9.0) / 24.0) + ar1(rng, n, g["v_ar_phi"], g["v_ar_sigma"])
    v_base = V_LN_NOM * v_rel[:, None] * (1.0 + np.array(g["phase_unbalance"]))[None, :]
    f = cfg["plant"]["f_nominal"] + ar1(rng, n, g["f_ar_phi"], g["f_ar_sigma"])
    amb = climate.ambient_temperature(cfg, idx, rng_for(cfg, "climate"))
    return {
        "idx": idx,
        "ambient": amb,
        "irradiance": climate.irradiance(cfg, idx),
        "v_base": v_base,
        "f": f,
        "v_factor": np.ones(n),
        "interrupt": np.zeros(n, dtype=bool),
        "coast_tau_min": 1.0,
        "eq": load.build_schedules(cfg, idx, amb),
    }


def _coast_speed(speed_run: np.ndarray, interrupt: np.ndarray, tau: float) -> np.ndarray:
    """Exponential coast-down of speed during interruptions (1-min averages)."""
    out = speed_run.copy()
    k, n = 0, len(interrupt)
    while k < n:
        if interrupt[k]:
            j = k
            while j < n and interrupt[j]:
                j += 1
            n0 = out[k - 1] if k > 0 else 0.0
            out[k:j] = n0 * np.exp(-(np.arange(1, j - k + 1) - 0.5) / tau)
            k = j
        else:
            k += 1
    return out


def _feeder_state(cfg: dict, eq: str, d: dict, V: np.ndarray, f: np.ndarray, interrupt: np.ndarray) -> dict:
    p = cfg["plant"]["equipment"][eq]
    rv = el.rated_values(p, cfg["plant"]["v_ll_nominal"])
    s = d["eq"][eq]
    n = len(f)
    running = s["on"] & ~interrupt
    prev_int = np.concatenate([[False], interrupt[:-1]])
    start = running & (s["start"] | prev_int)
    stop = running & s["stop"]
    L = np.where(running, s["L"], 0.0)
    v_avg = V.mean(axis=1)
    v_pu = v_avg / V_LN_NOM
    safe_v = np.maximum(v_avg, 1.0)
    if p["type"] == "motor":
        p_in = np.where(running, el.motor_input_kw(L, p["p_rated_kw"], p["eta_rated"], p["no_load_loss_share"]), 0.0)
        pf = el.motor_pf(L)
        i_avg = p_in * 1000.0 / (3.0 * safe_v * pf)
        i_start = i_avg + (p["inrush_multiple"] * rv["i_rated_a"] - i_avg) * p["inrush_decay_s"] / 60.0
        i_avg = np.where(start, i_start, i_avg)
        pf = np.where(start, 0.45, pf)
        i_avg = np.where(stop, 0.5 * i_avg, i_avg)
    else:
        pf = np.full(n, p["pf"]) - 0.02 * (1 - np.clip(L, 0, 1))
        p_in = np.where(running, L * p["p_rated_kw"], 0.0)
        i_avg = p_in * 1000.0 / (3.0 * safe_v * pf)
    i_avg = np.where(running, i_avg, 0.0)
    I = i_avg[:, None] * (1.0 + np.array(p["current_unbalance"]))[None, :]
    s_kva = (V * I).sum(axis=1) / 1000.0
    p_kw = s_kva * pf
    q = np.sqrt(np.maximum(s_kva**2 - p_kw**2, 0.0))
    pf_out = np.where(s_kva > 0.005 * rv["p_in_rated_kw"], pf, np.nan)
    out = {
        "v": V, "i": I, "i_avg": i_avg, "p_kw": p_kw, "q_kvar": q, "s_kva": s_kva, "pf": pf_out,
        "freq_hz": f, "thd_i_pct": np.where(running, 3.0 + 5.0 * (1 - np.clip(L, 0, 1)) if p["type"] == "motor" else 11.0, 0.0),
        "running": running, "start": start, "stop": stop, "L": L, "rated": rv,
    }
    if p["type"] == "motor":
        n_run = el.motor_speed(f, p["poles"], p["slip_rated"], L, v_pu)
        speed = np.where(running, n_run, 0.0)
        speed = np.where(start, 0.7 * n_run, speed)
        speed = np.where(stop, 0.6 * n_run, speed)
        speed = _coast_speed(speed, interrupt & s["on"], d["coast_tau_min"])
        temp = thermal.thermal_response(i_avg / rv["i_rated_a"], d["ambient"], p["tau_thermal_min"], p["dtheta_rated_k"])
        de, nde = vibration.vibration(L, running, s["health"], p["vib_v0_g"], p["vib_kl_g"], p["vib_nde_ratio"], start)
        out.update({"speed_rpm": speed, "winding_temp_c": temp, "vib_de_g": de, "vib_nde_g": nde, "i_relay_a": i_avg})
    return out


def compute_true_state(cfg: dict, d: dict) -> dict[str, pd.DataFrame]:
    """Return the true (noise-free, uncorrupted) state per equipment as DataFrames."""
    idx = d["idx"]
    n = len(idx)
    interrupt = d["interrupt"]
    vf = np.where(interrupt, 0.0, d["v_factor"])
    V = d["v_base"] * vf[:, None]
    f = np.where(interrupt, np.nan, d["f"])
    b = cfg["plant"]["bus"]
    rng = rng_for(cfg, "bus")
    feeders = {eq: _feeder_state(cfg, eq, d, V, f, interrupt) for eq in FEEDERS}

    alive = ~interrupt
    p_sum = sum(feeders[e]["p_kw"] for e in FEEDERS)
    q_sum = sum(feeders[e]["q_kvar"] for e in FEEDERS)
    unmet = (b["unmetered_kw"] + ar1(rng, n, 0.9, b["unmetered_noise_kw"] * np.sqrt(1 - 0.81))) * alive
    p_bus = (1 + b["loss_frac"]) * p_sum + unmet
    q_bus = (1 + b["loss_frac_q"]) * q_sum + b["unmetered_kvar"] * alive
    s_bus = np.sqrt(p_bus**2 + q_bus**2)
    i_bus_avg = np.where(alive, s_bus * 1000.0 / (3.0 * np.maximum(V.mean(axis=1), 1.0)), 0.0)
    # small residual phase unbalance on the bus, scaled so that sum(V_k I_k) = S_bus
    w = (1.0 + BUS_UNBALANCE)[None, :]
    I_bus = i_bus_avg[:, None] * w
    scale = np.where(alive, s_bus * 1000.0 / np.maximum((V * I_bus).sum(axis=1), 1e-9), 0.0)
    I_bus = I_bus * scale[:, None]
    pf_bus = np.where(s_bus > 1.0, p_bus / np.maximum(s_bus, 1e-9), np.nan)

    kw0 = rng_for(cfg, "kwh0")
    h = climate.local_hours(idx, cfg["time"]["utc_offset_h"])
    thd_v = cfg["plant"]["grid"]["thd_v_mean"] + 0.4 * np.sin(2 * np.pi * (h - 20) / 24.0)
    thd_v = np.where(alive, thd_v, np.nan)
    out: dict[str, pd.DataFrame] = {}
    sched_bus_state = np.where(p_bus > 0, "RUN_NORMAL", "OFF")

    def frame(eq, st, v, i, p_kw, q, s_kva, pf, thd_i, sched):
        df = pd.DataFrame({
            "v_l1_v": v[:, 0], "v_l2_v": v[:, 1], "v_l3_v": v[:, 2],
            "i_l1_a": i[:, 0], "i_l2_a": i[:, 1], "i_l3_a": i[:, 2],
            "i_relay_a": st.get("i_relay_a", np.full(n, np.nan)),
            "p_kw": p_kw, "q_kvar": q, "s_kva": s_kva, "pf": pf, "freq_hz": f,
            "kwh_register": float(kw0.uniform(1e4, 9e4)) + np.cumsum(p_kw) / 60.0,
            "thd_v_pct": thd_v, "thd_i_pct": np.where(alive, thd_i, np.nan),
            "speed_rpm": st.get("speed_rpm", np.full(n, np.nan)),
            "winding_temp_c": st.get("winding_temp_c", np.full(n, np.nan)),
            "vib_de_g": st.get("vib_de_g", np.full(n, np.nan)),
            "vib_nde_g": st.get("vib_nde_g", np.full(n, np.nan)),
            "ambient_temp_c": d["ambient"],
        })
        df["op_state"] = sched["op_state"]
        df["run_cmd"] = sched["run_cmd"]
        df["maint_mode"] = sched["maint_mode"]
        df["comm_status"] = "OK"
        df["load_fraction"] = st.get("L", np.full(n, np.nan))
        df["health"] = sched.get("health", np.full(n, np.nan))
        df["irradiance_wm2"] = d["irradiance"]
        return df

    bus_sched = {"op_state": sched_bus_state.astype(object), "run_cmd": np.ones(n, dtype=int),
                 "maint_mode": np.full(n, "NONE", dtype=object)}
    out["BUS1"] = frame("BUS1", {}, V, I_bus, p_bus, q_bus, s_bus, pf_bus, np.full(n, 5.0), bus_sched)
    for eq in FEEDERS:
        st = feeders[eq]
        sched = dict(d["eq"][eq])
        state = np.array(sched["op_state"], dtype=object)
        state[st["start"]] = "STARTING"  # restarts after an interruption
        sched["op_state"] = state
        out[eq] = frame(eq, st, st["v"], st["i"], st["p_kw"], st["q_kvar"], st["s_kva"], st["pf"], st["thd_i_pct"], sched)
    return out
