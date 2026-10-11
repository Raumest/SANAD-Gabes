"""Load envelope and operating schedules (commanded state, as the PLC sees it).

* Envelope: shift-based industrial profile (night low, morning ramp, weekday high,
  weekend low) from ``load.weekday_profile``; weekend multiplied by ``weekend_factor``.
* L(t) per machine = envelope x share x (1 + AR(1) noise), clipped to ``load.clip``.
* PUMP1 runs continuously with one nightly process pause (natural restart).
* COMP1 runs weekdays 05:xx-23:xx local with load/unload cycling; off at weekends.
* AUX1 (lighting/HVAC) always on; HVAC share follows ambient temperature.
* Planned shutdowns / inspections set ``maint_mode``.

The commanded state (op_state, run_cmd) is computed *before* any physical event, so a
real overload is not visible in op_state (no label leakage through the PLC state).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanad.config import rng_for
from sanad.sim.climate import local_hours
from sanad.sim.noise import ar1


def envelope(cfg: dict, idx: pd.DatetimeIndex) -> np.ndarray:
    lc = cfg["load"]
    h = local_hours(idx, cfg["time"]["utc_offset_h"])
    prof = np.array(lc["weekday_profile"], dtype=float)
    env = np.interp(h, prof[:, 0], prof[:, 1])
    loc = idx + pd.Timedelta(hours=cfg["time"]["utc_offset_h"])
    weekend = loc.dayofweek.to_numpy() >= 5
    return np.where(weekend, env * lc["weekend_factor"], env)


def _intervals_to_mask(n: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in intervals:
        m[max(0, a):max(0, min(n, b))] = True
    return m


def _pump_on(cfg: dict, idx: pd.DatetimeIndex, rng) -> np.ndarray:
    n = len(idx)
    on = np.ones(n, dtype=bool)
    loc = idx + pd.Timedelta(hours=cfg["time"]["utc_offset_h"])
    day0 = loc[0].normalize()
    ndays = int(np.ceil(n / 1440)) + 1
    off = []
    for d in range(ndays):
        start_local = day0 + pd.Timedelta(days=d, minutes=int(rng.integers(60, 270)))
        k = int((start_local - loc[0]).total_seconds() // 60)
        off.append((k, k + int(rng.integers(20, 61))))
    return on & ~_intervals_to_mask(n, off)


def _comp_on(cfg: dict, idx: pd.DatetimeIndex, rng) -> np.ndarray:
    c = cfg["plant"]["equipment"]["COMP1"]
    n = len(idx)
    loc = idx + pd.Timedelta(hours=cfg["time"]["utc_offset_h"])
    day0 = loc[0].normalize()
    ndays = int(np.ceil(n / 1440)) + 1
    on_iv = []
    for d in range(ndays):
        day = day0 + pd.Timedelta(days=d)
        if day.dayofweek >= 5:
            continue
        a = day + pd.Timedelta(hours=c["night_off"][1], minutes=int(rng.integers(-20, 25)))
        b = day + pd.Timedelta(hours=c["night_off"][0], minutes=int(rng.integers(-25, 20)))
        ka = int((a - loc[0]).total_seconds() // 60)
        kb = int((b - loc[0]).total_seconds() // 60)
        on_iv.append((ka, kb))
    return _intervals_to_mask(n, on_iv)


def _comp_loaded(cfg: dict, n: int, rng) -> np.ndarray:
    c = cfg["plant"]["equipment"]["COMP1"]
    loaded = np.zeros(n, dtype=bool)
    k, state = 0, True
    while k < n:
        lo, hi = c["cycle_load_min"] if state else c["cycle_unload_min"]
        d = int(rng.integers(lo, hi + 1))
        loaded[k:k + d] = state
        k += d
        state = not state
    return loaded


def _maintenance(cfg: dict, idx, on: np.ndarray, rng) -> np.ndarray:
    """One planned shutdown (4-8 h, daytime weekday) per motor and per 2 weeks."""
    n = len(idx)
    maint = np.full(n, "NONE", dtype=object)
    period = 14 * 1440
    for p0 in range(0, n, period):
        span = min(period, n - p0)
        if span < 3 * 1440:
            continue
        k = p0 + int(rng.integers(1440, span - 1440))
        # align to 09:xx local of that day
        loc = idx[k] + pd.Timedelta(hours=cfg["time"]["utc_offset_h"])
        k = k - (loc.hour * 60 + loc.minute) + 9 * 60 + int(rng.integers(0, 50))
        dur = int(rng.integers(240, 481))
        maint[k:k + dur] = "PLANNED_SHUTDOWN"
    return maint


def op_state_from(L: np.ndarray, on: np.ndarray, thresholds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Commanded op_state plus start/stop minute flags."""
    n = len(on)
    prev = np.concatenate([[on[0]], on[:-1]])
    nxt = np.concatenate([on[1:], [on[-1]]])
    start = on & ~prev
    stop = on & ~nxt
    lo, hi = thresholds
    st = np.where(L < lo, "RUN_LOW", np.where(L < hi, "RUN_NORMAL", "RUN_HIGH")).astype(object)
    st[~on] = "OFF"
    st[stop] = "STOPPING"
    st[start] = "STARTING"
    return st, start, stop


def build_schedules(cfg: dict, idx: pd.DatetimeIndex, ambient: np.ndarray) -> dict[str, dict]:
    """Commanded schedules and base load fraction for every feeder."""
    lc = cfg["load"]
    env = envelope(cfg, idx)
    n = len(idx)
    out: dict[str, dict] = {}
    for eq, p in cfg["plant"]["equipment"].items():
        rng = rng_for(cfg, "load", eq)
        noise = ar1(rng, n, lc["ar_phi"], lc["ar_sigma"])
        maint = np.full(n, "NONE", dtype=object)
        if eq == "PUMP1":
            on = _pump_on(cfg, idx, rng)
            L = env * p["share"] * (1 + noise)
        elif eq == "COMP1":
            on = _comp_on(cfg, idx, rng)
            loaded = _comp_loaded(cfg, n, rng)
            L = np.where(loaded, env * p["share"], p["unload_fraction"]) * (1 + noise)
        else:
            on = np.ones(n, dtype=bool)
            hvac = p["hvac_gain"] * np.clip((ambient - 25.0) / 15.0, -0.5, 1.5)
            L = (p["share"] * (0.55 + 0.45 * env) + hvac) * (1 + noise)
        if p["type"] == "motor":
            maint = _maintenance(cfg, idx, on, rng)
            on = on & (maint != "PLANNED_SHUTDOWN")
        L = np.clip(L, *lc["clip"])
        L = np.where(on, L, 0.0)
        state, start, stop = op_state_from(L, on, lc["op_state_thresholds"])
        out[eq] = {
            "L": L, "on": on, "start": start, "stop": stop, "op_state": state,
            "run_cmd": on.astype(int), "maint_mode": maint,
            "health": np.ones(n),
        }
    return out
