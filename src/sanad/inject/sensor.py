"""Sensor fault injectors — modify only the MEASURED channel (golden rule 1).

``apply_raw`` acts on the raw sensed signals *before* the meter computes P/Q/S/kWh, so a
CT drift or VT bias propagates consistently inside that meter (golden rule 3).
``apply_meter`` acts on the meter outputs (a frozen meter holds all its registers).
"""

from __future__ import annotations

import numpy as np

from sanad.inject.events import Event
from sanad.sim import electrical as el

METER_FIELDS = ["v_l1_v", "v_l2_v", "v_l3_v", "i_l1_a", "i_l2_a", "i_l3_a", "p_kw", "q_kvar", "s_kva",
                "pf", "freq_hz", "kwh_register", "thd_v_pct", "thd_i_pct"]
STUCK_RAW = {"RTD": ["winding_temp_c"], "SPEED": ["speed_rpm"], "ACC_DE": ["vib_de_g"]}


def _hold(arr: np.ndarray, k0: int, k1: int) -> None:
    arr[k0:k1] = arr[k0 - 1] if k0 > 0 else arr[k0]


def apply_raw(cfg: dict, raw: dict[str, dict], events: list[Event]) -> None:
    for e in events:
        r = raw.get(e.equipment_id)
        sl = slice(e.k0, e.k1)
        dur = e.k1 - e.k0
        if e.label == "SENS_DRIFT":
            gain = 1.0 + e.params["rate_pct_per_day"] / 100.0 * np.arange(dur) / 1440.0
            for k in (1, 2, 3):
                r[f"i_l{k}_a"][sl] *= gain
        elif e.label == "SENS_BIAS":
            ch = e.params["channel"]
            if ch == "VOLTAGE":
                for k in (1, 2, 3):
                    v = r[f"v_l{k}_v"][sl]
                    r[f"v_l{k}_v"][sl] = np.where(v > 1.0, v + e.params["offset_v"], v)
            elif ch == "TEMPERATURE":
                r["winding_temp_c"][sl] += e.params["offset_k"]
            else:
                i_r = el.rated_values(cfg["plant"]["equipment"][e.equipment_id], cfg["plant"]["v_ll_nominal"])["i_rated_a"]
                off = e.params["offset_pct_rated"] / 100.0 * i_r
                cols = ["i_relay_a"] if e.params["path"] == "RELAY_CT" else ["i_l1_a", "i_l2_a", "i_l3_a"]
                for c in cols:
                    r[c][sl] = np.maximum(r[c][sl] + off, 0.0)
        elif e.label == "SENS_STUCK" and e.params["channel"] in STUCK_RAW:
            for c in STUCK_RAW[e.params["channel"]]:
                _hold(r[c], e.k0, e.k1)


def apply_meter(cfg: dict, meas: dict[str, dict], events: list[Event]) -> None:
    for e in events:
        if e.label == "SENS_STUCK" and e.params["channel"] == "METER":
            m = meas[e.equipment_id]
            for c in METER_FIELDS:
                _hold(m[c], e.k0, e.k1)
