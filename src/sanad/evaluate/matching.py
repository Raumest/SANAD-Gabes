"""Match SANAD cases to ground-truth events (evaluation and confidence calibration only).

A case matches an event when their time intervals overlap (event window extended by a
tolerance after its end, to allow for detector latency/recovery) and the equipment is
compatible: same equipment; grid cases <-> GRID_* events; a BUS1 case may witness a
feeder event (the bus sees every feeder). Among candidates, an event of the same class
as the case's leading hypothesis is preferred, then the largest overlap.
"""

from __future__ import annotations

import pandas as pd

TOL_MIN = 60
SUBTYPE_LABEL = {
    "OVERLOAD": "PHYS_OVERLOAD", "BEARING_DEGRADATION": "PHYS_BEARING_DEGRADATION", "DRIFT": "SENS_DRIFT",
    "STUCK": "SENS_STUCK", "VOLTAGE": "SENS_BIAS", "CURRENT": "SENS_BIAS", "TEMPERATURE": "SENS_BIAS",
    "DATA_GAP": "DQ_MISSING", "DROPOUT": "COMM_DROPOUT", "ENERGY_UNDERREPORT": "FALS_ENERGY",
    "INTERRUPTION": "GRID_INTERRUPTION", "SUSTAINED_UNDERVOLTAGE": "GRID_SAG",
}


def _compatible(case, ev) -> bool:
    if case["kind"] == "grid":
        return ev["subtype"].startswith("GRID")
    if ev["subtype"].startswith("GRID"):
        return True  # side effects of a grid event outside the absorbed window
    if ev["equipment_id"] == case["equipment_id"]:
        return True
    return case["equipment_id"] == "BUS1"


def match_cases(cases: pd.DataFrame, events: pd.DataFrame, tol_min: int = TOL_MIN) -> pd.DataFrame:
    """Return cases with matched_event_id, truth_class, truth_label (NORMAL if unmatched)."""
    out = cases.copy()
    tol = pd.Timedelta(minutes=tol_min)
    ev = events.copy()
    mids, tclass, tlabel = [], [], []
    for _, c in out.iterrows():
        cs, ce = c["start_utc"], c["end_utc"]
        cand = ev[(ev.start_utc - pd.Timedelta(minutes=5) < ce) & (cs < ev.end_utc + tol)]
        cand = cand[[_compatible(c, e) for _, e in cand.iterrows()]] if len(cand) else cand
        if len(cand) == 0:
            mids.append("")
            tclass.append("NORMAL")
            tlabel.append("")
            continue
        ov = (cand.end_utc.clip(upper=ce) - cand.start_utc.clip(lower=cs)).dt.total_seconds()
        same_eq = (cand.equipment_id == c["equipment_id"]).astype(int)
        same_cls = (cand.main_class == c["leading_class"]).astype(int) + \
            (cand.subtype == SUBTYPE_LABEL.get(c.get("subtype", ""), "?")).astype(int)
        not_start = (cand.subtype != "NORMAL_START").astype(int)
        meas_case = c["leading_class"] not in ("PHYSICAL_EQUIPMENT_FAULT", "GRID_EVENT")
        same_origin = ((cand.origin != "PHYSICAL") == meas_case).astype(int)
        key = pd.DataFrame({"cls": same_cls, "org": same_origin, "eq": same_eq, "ns": not_start, "ov": ov})
        best = key.sort_values(["cls", "ns", "org", "eq", "ov"], ascending=False).index[0]
        e = cand.loc[best]
        mids.append(e.event_id)
        tclass.append(e.main_class)
        tlabel.append(e.subtype)
    out["matched_event_id"] = mids
    out["truth_class"] = tclass
    out["truth_label"] = tlabel
    return out
