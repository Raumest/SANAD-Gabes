"""Event scheduler: randomised onsets/parameters, time-block aware, no round timestamps.

* Weeks of the ``normal`` block receive no injected event (only natural motor starts).
* For every later block (train/val/test) and every label, ``events_per_block`` events are
  placed inside the usable block (after the embargo), avoiding overlap with other events
  on the same equipment track and with maintenance windows.
* Compound pairs (real overload during a meter CT drift) are placed in the test block only.
"""

from __future__ import annotations

import numpy as np

from sanad.config import rng_for, usable_bounds
from sanad.inject.events import Event

LABEL_ORDER = [
    "PHYS_BEARING_DEGRADATION", "SENS_DRIFT", "FALS_ENERGY", "SENS_BIAS", "SENS_STUCK",
    "DQ_MISSING", "PHYS_OVERLOAD", "COMM_DROPOUT", "GRID_INTERRUPTION", "GRID_SAG",
]
# Which occupancy track an event uses. Events on different tracks of the same equipment
# may overlap (e.g. a slow bearing degradation and a short data gap).
TRACK = {"PHYS_BEARING_DEGRADATION": "mech", "FALS_ENERGY": "register"}
ALL_EQ = ["BUS1", "PUMP1", "COMP1", "AUX1"]


class Occupancy:
    """Per (equipment, track) occupancy masks with a safety gap."""

    def __init__(self, n: int, gap: int):
        self.n, self.gap, self.masks = n, gap, {}

    def _m(self, eq, track):
        return self.masks.setdefault((eq, track), np.zeros(self.n, dtype=bool))

    def free(self, eqs, track, k0, k1) -> bool:
        a, b = max(0, k0 - self.gap), min(self.n, k1 + self.gap)
        return not any(self._m(e, track)[a:b].any() for e in eqs)

    def take(self, eqs, track, k0, k1):
        for e in eqs:
            self._m(e, track)[k0:k1] = True


def _minutes(rng, lo, hi) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _days(rng, lo, hi) -> int:
    return int(rng.uniform(lo, hi) * 1440)


def _draw(label: str, cfg: dict, rng) -> tuple[str, int, dict, str]:
    """Draw (equipment, duration_min, params, sensor_id) for one event of ``label``."""
    c = cfg["injection"][label]
    if label == "PHYS_OVERLOAD":
        eq = str(rng.choice(c["targets"]))
        return eq, _minutes(rng, *c["duration_min"]), {"load": round(float(rng.uniform(*c["load"])), 3)}, ""
    if label == "PHYS_BEARING_DEGRADATION":
        eq = str(rng.choice(c["targets"]))
        return eq, _days(rng, *c["duration_days"]), {"h_max": round(float(rng.uniform(*c["h_max"])), 3)}, f"{eq}_ACC_DE"
    if label == "GRID_SAG":
        return "BUS1", _minutes(rng, *c["duration_min"]), {"v_factor": round(float(rng.uniform(*c["v_factor"])), 3)}, ""
    if label == "GRID_INTERRUPTION":
        return "BUS1", _minutes(rng, *c["duration_min"]), {}, ""
    if label == "SENS_DRIFT":
        eq = str(rng.choice(c["targets"]))
        r = float(rng.uniform(*c["rate_pct_per_day"])) * float(rng.choice(c["sign"]))
        return eq, _days(rng, *c["duration_days"]), {"rate_pct_per_day": round(r, 3), "path": "METER_CT"}, f"{eq}_METER_CT"
    if label == "SENS_BIAS":
        kind = str(rng.choice(["VOLTAGE", "TEMPERATURE", "CURRENT"]))
        dur = _minutes(rng, *c["duration_min"])
        sign = float(rng.choice([-1, 1]))
        if kind == "VOLTAGE":
            eq = str(rng.choice(c["targets_voltage"]))
            sid = f"{eq}_BUS_CT" if eq == "BUS1" else f"{eq}_METER_CT"
            return eq, dur, {"channel": "VOLTAGE", "offset_v": round(sign * float(rng.uniform(*c["voltage_v"])), 2)}, sid
        if kind == "TEMPERATURE":
            eq = str(rng.choice(c["targets_temperature"]))
            return eq, dur, {"channel": "TEMPERATURE", "offset_k": round(sign * float(rng.uniform(*c["temperature_k"])), 2)}, f"{eq}_RTD"
        eq = str(rng.choice(c["targets_current"]))
        path = str(rng.choice(["METER_CT", "RELAY_CT"]))
        return eq, dur, {"channel": "CURRENT", "path": path,
                         "offset_pct_rated": round(sign * float(rng.uniform(*c["current_pct_rated"])), 2)}, f"{eq}_{path}"
    if label == "SENS_STUCK":
        eq = str(rng.choice(c["targets"]))
        ch = str(rng.choice(c["channels"]))
        sid = {"METER": f"{eq}_METER_CT", "RTD": f"{eq}_RTD", "SPEED": f"{eq}_SPEED", "ACC_DE": f"{eq}_ACC_DE"}[ch]
        return eq, _minutes(rng, *c["duration_min"]), {"channel": ch}, sid
    if label == "DQ_MISSING":
        eq = str(rng.choice(c["targets"]))
        dur = _minutes(rng, *c["window_min"])
        nb = _minutes(rng, *c["bursts"])
        bursts = []
        for _ in range(nb):
            bl = min(_minutes(rng, *c["burst_min"]), max(1, dur // 2))
            bursts.append([int(rng.integers(0, max(1, dur - bl))), bl])
        return eq, dur, {"mcar_rate": round(float(rng.uniform(*c["mcar_rate"])), 4), "bursts": bursts}, ""
    if label == "FALS_ENERGY":
        eq = str(rng.choice(c["targets"]))
        return eq, _days(rng, *c["duration_days"]), {"underreport": round(float(rng.uniform(*c["underreport"])), 3)}, f"{eq}_METER_CT"
    if label == "COMM_DROPOUT":
        eq = str(rng.choice(c["targets"]))
        return eq, _minutes(rng, *c["duration_min"]), {}, ""
    raise KeyError(label)


def _running_ok(label: str, params: dict, eq: str, drivers: dict, k0: int, k1: int) -> bool:
    """Placement constraints that depend on the commanded schedule."""
    eqs = ALL_EQ if eq == "BUS1" else [eq]
    for e in eqs:
        if e == "BUS1":
            continue
        if (drivers["eq"][e]["maint_mode"][k0:k1] != "NONE").any():
            return False
    if eq == "BUS1" or eq == "AUX1":
        return True
    on = drivers["eq"][eq]["on"][k0:k1]
    st = drivers["eq"][eq]["start"][k0:k1] | drivers["eq"][eq]["stop"][k0:k1]
    if label == "PHYS_OVERLOAD":
        return bool(on.all() and not st.any())
    if label in ("SENS_DRIFT", "SENS_BIAS", "SENS_STUCK", "PHYS_BEARING_DEGRADATION", "FALS_ENERGY"):
        return float(on.mean()) >= 0.7 and bool(on[: min(30, len(on))].all())
    return True


def _place(label, cfg, drivers, occ, rng, a, b, block, eq_fixed=None, window=None):
    gap = cfg["injection"]["min_gap_min"]
    for _ in range(4000):
        eq, dur, params, sid = _draw(label, cfg, rng)
        if eq_fixed:
            eq = eq_fixed
            sid = sid.replace(sid.split("_")[0], eq, 1) if sid else sid
        lo, hi = (a, b - dur) if window is None else (window[0], window[1] - dur)
        if hi <= lo:
            continue
        k0 = int(rng.integers(lo, hi))
        if k0 % 15 == 0:  # no round timestamps
            k0 += int(rng.integers(1, 14))
        k1 = k0 + dur
        if k1 > b:
            continue
        eqs = ALL_EQ if eq == "BUS1" and label.startswith("GRID") else [eq]
        track = TRACK.get(label, "main")
        if window is None and not occ.free(eqs, track, k0, k1):
            continue
        if not _running_ok(label, params, eq, drivers, k0, k1):
            continue
        if window is None:
            occ.take(eqs, track, k0, k1)
            if label.startswith("GRID"):
                # a grid event also disturbs the mechanical and register tracks
                occ.take(eqs, "register", k0, k1)
        return Event(label, eq, k0, k1, params, sid, block)
    raise RuntimeError(f"could not place {label} in block {block}; relax counts or durations")


def schedule_events(cfg: dict, drivers: dict) -> list[Event]:
    """Randomly schedule the Phase 1 injection library over train/val/test blocks."""
    inj = cfg["injection"]
    if not inj.get("enabled", True):
        return []
    idx = drivers["idx"]
    n = len(idx)
    occ = Occupancy(n, inj["min_gap_min"])
    rng = rng_for(cfg, "scheduler")
    t0 = idx[0]
    events: list[Event] = []
    bounds = {k: (int((a - t0).total_seconds() // 60), int((b - t0).total_seconds() // 60))
              for k, (a, b) in usable_bounds(cfg).items()}
    # Compound pairs first (test block only): meter CT drift with a real overload inside it.
    a, b = bounds["test"]
    for _ in range(int(inj.get("compound_in_test", 0))):
        drift = _place("SENS_DRIFT", cfg, drivers, occ, rng, a, b, "test")
        lo = drift.k0 + (drift.k1 - drift.k0) // 3
        ov = _place("PHYS_OVERLOAD", cfg, drivers, occ, rng, a, b, "test", eq_fixed=drift.equipment_id, window=(lo, drift.k1))
        drift.params["compound"] = True
        ov.params["compound"] = True
        events += [drift, ov]
        drift.compound_with, ov.compound_with = "pending", "pending"
    for block in ("train", "val", "test"):
        a, b = bounds[block]
        count = int(inj["events_per_block"][block])
        for label in LABEL_ORDER:
            for _ in range(count):
                events.append(_place(label, cfg, drivers, occ, rng, a, b, block))
    return events


def natural_starts(cfg: dict, truth: dict, injected: list[Event]) -> list[Event]:
    """Normal motor starts (NORMAL_START) found in the true state — not anomalies."""
    out = []
    outages = [(e.k0, e.k1 + 2) for e in injected if e.label == "GRID_INTERRUPTION"]
    for eq in ("PUMP1", "COMP1"):
        st = truth[eq]["op_state"].to_numpy()
        maint = truth[eq]["maint_mode"].to_numpy()
        ks = np.flatnonzero(st == "STARTING")
        for k in ks:
            if maint[k] != "NONE" or any(a <= k <= b for a, b in outages):
                continue  # restarts after a grid interruption belong to that event
            out.append(Event("NORMAL_START", eq, int(k), int(k) + 2, {}, f"{eq}_METER_CT", ""))
    return out


def finalise(events: list[Event]) -> None:
    """Sort, assign ids, link compound pairs, set severity and detectability."""
    events.sort(key=lambda e: (e.k0, e.equipment_id, e.label))
    for i, e in enumerate(events, 1):
        e.event_id = f"EV{i:04d}"
    comp = [e for e in events if e.compound_with == "pending"]
    for d in [e for e in comp if e.label == "SENS_DRIFT"]:
        ov = next(o for o in comp if o.label == "PHYS_OVERLOAD" and o.equipment_id == d.equipment_id and d.k0 <= o.k0 < d.k1)
        d.compound_with, ov.compound_with = ov.event_id, d.event_id
    for e in events:
        e.severity, e.detectability = _severity(e)


def _severity(e: Event) -> tuple[str, str]:
    p = e.params
    dur = e.k1 - e.k0
    if e.label == "NORMAL_START":
        return "S1", "NONE"
    if e.label == "PHYS_OVERLOAD":
        return ("S3" if p["load"] >= 1.25 else "S2"), ("HIGH" if p["load"] >= 1.15 else "MEDIUM")
    if e.label == "PHYS_BEARING_DEGRADATION":
        return "S2", ("HIGH" if p["h_max"] >= 2.5 else "MEDIUM")
    if e.label == "GRID_SAG":
        return ("S3" if p["v_factor"] < 0.6 else "S2"), "HIGH"
    if e.label == "GRID_INTERRUPTION":
        return "S3", "HIGH"
    if e.label == "SENS_DRIFT":
        final = abs(p["rate_pct_per_day"]) * dur / 1440
        return "S2", ("HIGH" if final >= 3 else "MEDIUM" if final >= 1.5 else "LOW")
    if e.label == "SENS_BIAS":
        mag = abs(p.get("offset_v", 0)) / 8 + abs(p.get("offset_k", 0)) / 10 + abs(p.get("offset_pct_rated", 0)) / 3
        return "S2", ("HIGH" if mag >= 0.7 else "MEDIUM")
    if e.label == "SENS_STUCK":
        return "S2", ("MEDIUM" if dur < 30 else "HIGH")
    if e.label == "DQ_MISSING":
        return "S1", "HIGH"
    if e.label == "FALS_ENERGY":
        return "S3", ("HIGH" if p["underreport"] >= 0.2 else "MEDIUM")
    if e.label == "COMM_DROPOUT":
        return ("S2" if dur >= 30 else "S1"), "HIGH"
    return "S1", "MEDIUM"
