"""Trust controls: override-rate monitor, second-look sampling, per-class track record and
adaptive autonomy (tighten one tier for a class whose override rate or calibration error
crosses its limit)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def second_look(case_id: str, rate: float, seed: int) -> bool:
    """Deterministic forced second-look sampling of auto-resolved cases."""
    h = int(hashlib.sha256(f"{seed}/{case_id}".encode()).hexdigest()[:8], 16)
    return (h % 10000) / 10000.0 < rate


def read_feedback(path: Path) -> pd.DataFrame:
    cols = ["ts_utc", "case_id", "action", "verdict", "label", "reason", "operator"]
    if not Path(path).exists():
        return pd.DataFrame(columns=cols)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def track_record(cases: pd.DataFrame, feedback: pd.DataFrame) -> pd.DataFrame:
    """Per verdict class: cases, reviewed, approvals, overrides, override rate."""
    rows = []
    fb = feedback.drop_duplicates("case_id", keep="last") if len(feedback) else feedback
    for cls, grp in cases.groupby("verdict"):
        f = fb[fb.case_id.isin(grp.case_id)] if len(fb) else fb
        n_rev = len(f)
        n_ov = int((f.action == "OVERRIDE").sum()) if n_rev else 0
        n_ap = int((f.action == "APPROVE").sum()) if n_rev else 0
        rows.append({"verdict": cls, "cases": len(grp), "reviewed": n_rev, "approved": n_ap, "overridden": n_ov,
                     "override_rate": (n_ov / n_rev) if n_rev else 0.0})
    return pd.DataFrame(rows)


def tightened_classes(record: pd.DataFrame, cfg: dict, calib_error: float | None = None) -> set[str]:
    g = cfg["governance"]
    out = set()
    for _, r in record.iterrows():
        if r["reviewed"] >= g["min_cases_for_trust"] and r["override_rate"] > g["override_rate_limit"]:
            out.add(r["verdict"])
    if calib_error is not None and calib_error == calib_error and calib_error > g["calibration_error_limit"]:
        out |= {"SENSOR_FAULT", "BIAS_MISCALIBRATION"}
    return out
