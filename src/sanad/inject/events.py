"""Ground-truth event records (written to events.csv, never to measurements.csv)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from sanad.schema import EVENT_COLUMNS

ORIGIN = {
    "NORMAL_START": "PHYSICAL", "PHYS_OVERLOAD": "PHYSICAL", "PHYS_BEARING_DEGRADATION": "PHYSICAL",
    "GRID_SAG": "PHYSICAL", "GRID_INTERRUPTION": "PHYSICAL", "SENS_DRIFT": "MEASUREMENT",
    "SENS_BIAS": "MEASUREMENT", "SENS_STUCK": "MEASUREMENT", "DQ_MISSING": "MEASUREMENT",
    "FALS_ENERGY": "CYBER_FALSIFICATION", "COMM_DROPOUT": "COMMUNICATION",
}


@dataclass
class Event:
    """One injected (or natural) event on the 1-min grid: minutes [k0, k1)."""

    label: str
    equipment_id: str
    k0: int
    k1: int
    params: dict = field(default_factory=dict)
    sensor_id: str = ""
    block: str = ""
    event_id: str = ""
    compound_with: str = ""
    severity: str = "S1"
    detectability: str = "MEDIUM"


def events_frame(events: list[Event], idx: pd.DatetimeIndex, taxonomy: dict) -> pd.DataFrame:
    rows = []
    for e in events:
        main_class, _sub = taxonomy[e.label]
        end = idx[min(e.k1, len(idx)) - 1] + (idx[1] - idx[0])
        rows.append({
            "event_id": e.event_id, "main_class": main_class, "subtype": e.label,
            "origin": ORIGIN[e.label], "severity": e.severity, "equipment_id": e.equipment_id,
            "sensor_id": e.sensor_id, "start_utc": idx[e.k0].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "gt_detectability": e.detectability,
            "params_json": json.dumps({"block": e.block, **e.params}, sort_keys=True),
            "compound_with": e.compound_with,
        })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
