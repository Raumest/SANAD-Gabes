"""2-of-3 voting on current: meter CT vs relay CT vs bus-implied current.

The bus-implied current of feeder i comes from the bus balance learned on the normal
block: P_i = (P_bus - a) / b - sum_{j != i} P_j, converted with the feeder's own V and PF.
Each path is compared pairwise (relative, baseline-corrected on the normal block).
The odd one out is the faulty path; if fewer than three paths are available the vote is
INSUFFICIENT (-> too few redundant paths -> UNCERTAIN downstream).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanad.schema import FEEDERS, MOTORS


def bus_implied_current(frames: dict[str, pd.DataFrame], calib: dict) -> dict[str, pd.Series]:
    a, b = calib["r4_a"], calib["r4_b"]
    p_bus = frames["BUS1"]["p_kw"]
    total = (p_bus - a) / b
    out = {}
    for eq in FEEDERS:
        others = sum(frames[o]["p_kw"] for o in FEEDERS if o != eq)
        p_i = total - others
        d = frames[eq]
        out[eq] = p_i * 1000.0 / (3.0 * d["v_avg"] * d["pf"])
    return out


def path_table(frames, bus_i: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    """Per-minute current of each available path (A)."""
    out = {}
    for eq in FEEDERS:
        d = frames[eq]
        out[eq] = pd.DataFrame({
            "METER_CT": d["i_avg"],
            "RELAY_CT": d["i_relay_a"] if eq in MOTORS else np.nan,
            "BUS_IMPLIED": bus_i[eq],
        })
    return out


def fit_baselines(paths: dict[str, pd.DataFrame], masks: dict[str, np.ndarray]) -> dict:
    """Median log-ratios between paths on the normal block (static gain differences)."""
    base = {}
    for eq, p in paths.items():
        m = masks[eq]
        base[eq] = {}
        for x, y in (("METER_CT", "RELAY_CT"), ("METER_CT", "BUS_IMPLIED"), ("RELAY_CT", "BUS_IMPLIED")):
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.log(p[x] / p[y]).replace([np.inf, -np.inf], np.nan)[m].dropna()
            base[eq][f"{x}/{y}"] = float(r.median()) if len(r) > 100 else 0.0
    return base


def vote(p: pd.DataFrame, base: dict, tol: float, i_min: float) -> dict:
    """Vote over an episode slice ``p`` (rows = minutes, columns = paths)."""
    ok = p.where(p > i_min)  # also drops non-physical (negative) bus-implied currents
    avail = [c for c in ("METER_CT", "RELAY_CT", "BUS_IMPLIED") if ok[c].notna().mean() > 0.3]
    dis = {}
    for x, y in (("METER_CT", "RELAY_CT"), ("METER_CT", "BUS_IMPLIED"), ("RELAY_CT", "BUS_IMPLIED")):
        if x in avail and y in avail:
            r = np.log(ok[x] / ok[y]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(r) >= 3:
                dis[f"{x}/{y}"] = float(np.expm1(r.median() - base.get(f"{x}/{y}", 0.0)))
    res = {"paths": avail, "disagreement": dis, "faulty_path": "INSUFFICIENT", "n_paths": len(avail)}
    if len(avail) < 3 or len(dis) < 3:
        if len(dis) == 1 and abs(next(iter(dis.values()))) <= tol:
            res["faulty_path"] = "NONE_2PATH"
        return res
    big = {k: abs(v) > tol for k, v in dis.items()}
    mr, mb, rb = big["METER_CT/RELAY_CT"], big["METER_CT/BUS_IMPLIED"], big["RELAY_CT/BUS_IMPLIED"]
    if not (mr or mb or rb):
        res["faulty_path"] = "NONE"
    elif mr and mb and not rb:
        res["faulty_path"] = "METER_CT"
    elif mr and rb and not mb:
        res["faulty_path"] = "RELAY_CT"
    elif mb and rb and not mr:
        res["faulty_path"] = "BUS_IMPLIED"
    else:
        res["faulty_path"] = "AMBIGUOUS"
    return res
