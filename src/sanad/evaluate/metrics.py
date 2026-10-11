"""Event-level evaluation metrics (CLAUDE.md §12).

* Event-level: an injected event is DETECTED if at least one case matches it. Its
  predicted class is the verdict of the first non-UNCERTAIN matched case (UNCERTAIN if all
  matched cases abstained, MISSED if none matched).
* Precision per class is case-level: matched-to-same-class cases / cases with that verdict.
* False positives per day: cases matched to no injected event (NORMAL truth, incl. normal
  motor starts) divided by the block length in days.
* Detection delay: first matched case start - event start (minutes, >= 0).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sanad.schema import CLASSES


def event_table(cases: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    inj = events[events.subtype != "NORMAL_START"].copy()
    rows = []
    for _, e in inj.iterrows():
        m = cases[cases.matched_event_id == e.event_id].sort_values("start_utc")
        decided = m[m.verdict != "UNCERTAIN"]
        if len(m) == 0:
            pred, delay = "MISSED", np.nan
        else:
            pred = decided.iloc[0].verdict if len(decided) else "UNCERTAIN"
            delay = max(0.0, (m.iloc[0].start_utc - e.start_utc).total_seconds() / 60.0)
        rows.append({"event_id": e.event_id, "label": e.subtype, "truth_class": e.main_class, "block": e.block,
                     "equipment_id": e.equipment_id, "detectability": e.gt_detectability, "detected": len(m) > 0,
                     "predicted": pred, "correct": pred == e.main_class, "delay_min": delay, "n_cases": len(m)})
    return pd.DataFrame(rows)


def block_metrics(cases: pd.DataFrame, events: pd.DataFrame, days: float) -> dict:
    et = event_table(cases, events)
    out: dict = {"n_cases": int(len(cases)), "n_events": int(len(et))}
    inj_cases = cases[cases.truth_label.ne("") & cases.truth_label.ne("NORMAL_START")]
    fp = cases[~cases.index.isin(inj_cases.index)]
    out["false_positive_cases"] = int(len(fp))
    out["false_positives_per_day"] = round(len(fp) / days, 3) if days else None
    out["false_positive_on_normal_starts"] = int((cases.truth_label == "NORMAL_START").sum())
    out["abstention_rate"] = round(float((cases.verdict == "UNCERTAIN").mean()), 4) if len(cases) else 0.0
    out["event_detection_rate"] = round(float(et.detected.mean()), 4) if len(et) else None
    out["event_accuracy"] = round(float(et.correct.mean()), 4) if len(et) else None
    per = {}
    for cls in CLASSES:
        if cls in ("NORMAL", "UNKNOWN"):
            continue
        ev_c = et[et.truth_class == cls]
        pred_c = cases[cases.verdict == cls]
        tp_cases = int((pred_c.truth_class == cls).sum())
        prec = tp_cases / len(pred_c) if len(pred_c) else float("nan")
        rec = float(ev_c.correct.mean()) if len(ev_c) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) > 0 else float("nan")
        per[cls] = {"events": int(len(ev_c)), "detected": int(ev_c.detected.sum()) if len(ev_c) else 0,
                    "recall": _r(rec), "cases_predicted": int(len(pred_c)), "precision": _r(prec), "f1": _r(f1),
                    "abstained_events": int((ev_c.predicted == "UNCERTAIN").sum()) if len(ev_c) else 0,
                    "delay_min_median": _r(float(ev_c.delay_min.median())) if ev_c.detected.any() else None}
    out["per_class"] = per
    labels = sorted(et.label.unique()) if len(et) else []
    out["per_label"] = {lab: {"events": int((et.label == lab).sum()), "detected": int(et[et.label == lab].detected.sum()),
                              "correct": int(et[et.label == lab].correct.sum()),
                              "uncertain": int((et[et.label == lab].predicted == "UNCERTAIN").sum()),
                              "delay_min_median": _r(float(et[et.label == lab].delay_min.median()))}
                        for lab in labels}
    cols = [c for c in CLASSES if c != "NORMAL"] + ["UNCERTAIN", "MISSED"]
    cm = pd.crosstab(et.truth_class, et.predicted).reindex(columns=cols, fill_value=0) if len(et) else pd.DataFrame()
    out["confusion_matrix"] = {r: {c: int(cm.loc[r, c]) for c in cm.columns} for r in cm.index}
    return out


def _r(x, n: int = 4):
    return None if x is None or x != x else round(float(x), n)


def to_json(obj) -> str:
    return json.dumps(obj, indent=2)
