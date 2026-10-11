"""Group per-minute alarms into episodes (cases) and build plant-level grid episodes.

* Equipment episodes: runs of ``alarm`` merged across gaps <= ``episode_merge_gap_min``.
* Grid episodes: minutes where >= 3 of the 4 meters see under-voltage / interruption at
  the same time. Equipment episodes mostly inside a grid episode are absorbed into it.
* Bus-balance episodes (BUS1 alarms only from R4/R5B) overlapping a feeder episode are
  absorbed into that feeder case (the bus is the independent witness, not the culprit).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sanad.schema import EQUIPMENT_ORDER, MOTORS


@dataclass
class Episode:
    equipment_id: str
    k0: int
    k1: int  # exclusive
    kind: str = "equipment"  # equipment | grid | bus_balance
    absorbed: list = field(default_factory=list)
    families: set = field(default_factory=set)


LONG_MIN = 720  # episodes longer than this are never merged with other families


def runs(mask: np.ndarray, merge_gap: int = 0, min_len: int = 1, evidence: np.ndarray | None = None) -> list[tuple[int, int]]:
    """[k0, k1) runs of True, merging gaps <= merge_gap, dropping runs < min_len.

    If ``evidence`` is given, only gap minutes with evidence count towards the gap (a motor
    stop in the middle of a slow drift does not split the case).
    """
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    cum = None if evidence is None else np.concatenate([[0], np.cumsum(evidence.astype(int))])
    out = []
    a = b = idx[0]
    for k in idx[1:]:
        gap = (k - b - 1) if cum is None else int(cum[k] - cum[b + 1])
        if gap <= merge_gap:
            b = k
        else:
            out.append((a, b + 1))
            a = b = k
    out.append((a, b + 1))
    return [(a, b) for a, b in out if b - a >= min_len]


COUPLED = [{"electrical", "energy"}, {"electrical", "thermal"}, {"electrical", "speed"}, {"thermal", "speed"}]
DATA_SHADOW = 60  # minutes after a data gap whose side effects (window relations) belong to it


def _coupled(a: set, b: set) -> bool:
    """Families that one physical/sensor root cause can light up together."""
    a, b = a - {"other"}, b - {"other"}
    if not a or not b:
        return True
    u = a | b
    if "mech" in u or "data" in u:
        return a == b
    return all({x, y} in COUPLED for x in u for y in u if x != y)


def _merge_short(eps: list[Episode], gap: int) -> list[Episode]:
    """Merge overlapping short episodes of coupled families (same root cause).

    Data-gap episodes absorb short episodes of other families that they cause (window
    relations with missing inputs) instead of merging with them.
    """
    eps = sorted(eps, key=lambda e: e.k0)
    data = [e for e in eps if e.families == {"data"}]
    eps = [e for e in eps if e.families == {"data"} or e.k1 - e.k0 > LONG_MIN or not any(
        d.k0 - 2 <= e.k0 and e.k1 <= d.k1 + DATA_SHADOW for d in data)]
    # short episodes of a coupled family inside a long episode are its side effects
    longs = [e for e in eps if e.k1 - e.k0 > LONG_MIN and e.families != {"data"}]
    eps = [e for e in eps if e in longs or e.families == {"data"} or not any(
        L.k0 <= e.k0 and e.k1 <= L.k1 + DATA_SHADOW and _coupled(L.families, e.families) and L.families != e.families
        for L in longs)]
    out: list[Episode] = []
    for e in eps:
        host = next((o for o in out if o.k1 - o.k0 <= LONG_MIN and e.k1 - e.k0 <= LONG_MIN
                     and e.k0 < o.k1 + gap and o.k0 < e.k1 + gap and _coupled(o.families, e.families)), None)
        if host is None:
            out.append(e)
        else:
            host.k0, host.k1 = min(host.k0, e.k0), max(host.k1, e.k1)
            host.families |= e.families
    # "other" (Isolation Forest only) episodes inside a long episode add nothing new
    return [e for e in out if e.families != {"other"} or not any(
        o is not e and o.k0 <= e.k0 and e.k1 <= o.k1 for o in out)]


def build_episodes(res: dict[str, pd.DataFrame], cfg: dict) -> list[Episode]:
    det = cfg["detection"]
    gap, min_len = det["episode_merge_gap_min"], det["episode_min_len_min"]
    uv = sum(res[eq]["v_undervoltage"].to_numpy().astype(int) + res[eq]["v_interruption"].to_numpy().astype(int) > 0
             for eq in EQUIPMENT_ORDER)
    grid_mask = uv >= 3
    grid = [Episode("BUS1", a, b, "grid") for a, b in runs(grid_mask, 5, 1)]
    post = 20  # minutes after a grid event attributed to it (restarts, thermal settling)
    covered = np.zeros(len(grid_mask), dtype=bool)
    for g in grid:
        covered[max(0, g.k0 - 2):g.k1 + post] = True

    eq_eps: list[Episode] = []
    fams = [c[4:] for c in res["BUS1"].columns if c.startswith("fam_")]
    for eq in EQUIPMENT_ORDER:
        r = res[eq]
        lost = r["v_lost"].to_numpy()
        fam_eps: list[Episode] = []
        for fam in fams:
            ev = r["running"].to_numpy() if fam in ("electrical", "thermal", "speed", "mech") and eq in MOTORS else None
            for a, b in runs(r[f"fam_{fam}"].to_numpy(), gap, 1, ev):
                if b - a < min_len and not lost[a:b].any():
                    continue
                if covered[a:b].mean() >= 0.5:
                    g = next(x for x in grid if x.k0 - 2 <= a + int(np.argmax(covered[a:b])) < x.k1 + post)
                    if eq not in g.absorbed:
                        g.absorbed.append(eq)
                    continue
                fam_eps.append(Episode(eq, a, b, "equipment", families={fam}))
        eq_eps += _merge_short(fam_eps, gap)
    for e in eq_eps:
        if e.equipment_id == "BUS1" and e.families <= {"electrical", "energy"}:
            r = res["BUS1"].iloc[e.k0:e.k1]
            rels = [c for c in r.columns if c.startswith("flag_") and c not in ("flag_R4", "flag_R5B")]
            val = [c for c in r.columns if c.startswith("v_") and c != "v_pu"]
            if r[rels + val].to_numpy().any(axis=1).mean() < 0.1:
                e.kind = "bus_balance"
    feeders = [e for e in eq_eps if e.equipment_id != "BUS1"]
    kept = []
    for e in eq_eps:
        if e.kind == "bus_balance":
            host = [f for f in feeders if f.k0 < e.k1 + gap and e.k0 < f.k1 + gap]
            if host:
                max(host, key=lambda f: min(f.k1, e.k1) - max(f.k0, e.k0)).absorbed.append("BUS1")
                continue
        kept.append(e)
    return sorted(grid + kept, key=lambda e: (e.k0, EQUIPMENT_ORDER.index(e.equipment_id)))
