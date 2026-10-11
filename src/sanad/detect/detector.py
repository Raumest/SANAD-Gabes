"""Stage A — Detect: validity checks + relation residual thresholds + EWMA/CUSUM + IForest.

``StageA.fit`` learns everything from the normal-only block (weeks 1-2 by default):
structural constants (bus balance, vibration envelopes), residual normalisation,
EWMA/CUSUM thresholds and the Isolation Forest. ``StageA.run`` produces, per equipment
and per minute, z-scores, statistics, flags and a combined ``alarm`` with state-context
suppression (maintenance, start-up/stop transients).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanad.config import block_of
from sanad.detect import statistical as st
from sanad.detect.iforest import IForestDetector, features
from sanad.detect.validity import validity
from sanad.physics import relations as rl
from sanad.physics import residuals as rs
from sanad.schema import EQUIPMENT_ORDER, MOTORS

TRIGGERS = {eq: [r for r in rl.APPLIES[eq]] for eq in EQUIPMENT_ORDER}
# Evidence families: episodes are built per family so that a slow multi-day event (e.g.
# bearing wear on the vibration channels) does not swallow short unrelated events.
FAMILIES = {
    "data": ["v_lost", "v_dq_gap", "v_dq_rate", "v_time_skew"],
    "electrical": ["flag_R1", "flag_R2", "flag_R3", "flag_R4", "flag_R8", "flag_R9", "v_overcurrent", "v_undervoltage",
                   "v_interruption", "v_pf_invalid", "v_flat_meter", "v_flat_relay"],
    "energy": ["flag_R5", "flag_R5B", "v_kwh_decrease"],
    "thermal": ["flag_R6", "v_temp_range", "v_flat_rtd"],
    "speed": ["flag_R7", "v_flat_speed"],
    "mech": ["flag_R10", "flag_R10E_DE", "flag_R10E_NDE", "v_flat_acc_de", "v_flat_acc_nde"],
    "other": ["if_flag"],
}
STAT_CLIP = 8.0
NO_PERSIST = ("lost", "undervoltage", "interruption")  # corroborated plant-wide / by comm status
VALIDITY_TRIGGERS = ["lost", "dq_gap", "dq_rate", "undervoltage", "interruption", "overcurrent", "pf_invalid",
                     "kwh_decrease", "temp_range", "time_skew", "flat_meter", "flat_relay", "flat_rtd",
                     "flat_speed", "flat_acc_de", "flat_acc_nde"]


def context_ok(d: pd.DataFrame) -> np.ndarray:
    """False during maintenance and start/stop transients (+2 min after a start)."""
    trans = d["op_state"].isin(["STARTING", "STOPPING"]).to_numpy()
    after = np.zeros_like(trans)
    for s in (1, 2):
        after[s:] |= trans[:-s]
    return (d["maint_mode"].fillna("NONE") == "NONE").to_numpy() & ~trans & ~after


def _persist(x: np.ndarray, k: int = 2, w: int = 3) -> np.ndarray:
    return pd.Series(x.astype(int)).rolling(w, min_periods=1).sum().to_numpy() >= k


def _min_len(x: np.ndarray, n: int) -> np.ndarray:
    """Keep only runs of True of length >= n."""
    s = pd.Series(x)
    grp = (s != s.shift()).cumsum()
    length = s.groupby(grp).transform("size")
    return (s & (length >= n)).to_numpy()


class StageA:
    def __init__(self, cfg: dict, eqt: pd.DataFrame):
        self.cfg = cfg
        self.eqt = eqt
        self.calib: dict = {}
        self.norm: dict = {}
        self.thresholds: dict = {}
        self.iforest = IForestDetector(cfg, int(cfg["seed"]) % (2**31))

    def _masks(self, ts, frames) -> dict[str, np.ndarray]:
        normal = (block_of(self.cfg, pd.Series(ts)) == "normal").to_numpy()
        return {eq: normal & context_ok(d) & (d["comm_status"] != "LOST").to_numpy() for eq, d in frames.items()}

    def fit(self, ts: pd.DatetimeIndex, frames: dict[str, pd.DataFrame]) -> "StageA":
        masks = self._masks(ts, frames)
        self.calib = rl.fit_structural(frames, self.eqt, masks["BUS1"])
        raw = rl.raw_residuals(frames, self.eqt, self.calib, self.cfg)
        self.norm = rs.fit_normalisation(raw, masks)
        zf = {eq: rs.despike(z) for eq, z in rs.normalise(raw, self.norm).items()}
        self.thresholds = st.fit_thresholds(zf, masks, {eq: list(self.norm[eq]) for eq in zf}, self.cfg)
        for eq in EQUIPMENT_ORDER:
            val = self._validity(eq, frames[eq], raw[eq])
            X = features(zf[eq], rl.APPLIES[eq], val)
            m = masks[eq] & frames[eq]["running"].to_numpy()
            self.iforest.fit(eq, X[m])
        return self

    def _validity(self, eq, d, raw_eq):
        v = validity(d, eq, self.eqt.loc[eq], self.cfg, raw_eq["R6"] if eq in MOTORS else None)
        # missing values while the link is up (rows with voltage ~0 are expected to lack f/PF)
        has_nan = (v["nan_frac"] > 0).to_numpy() & ~v["lost"].to_numpy() & ~v["interruption"].to_numpy()
        v["dq_gap"] = (v["nan_frac"] >= 0.99).to_numpy() & ~v["lost"].to_numpy()
        rate = pd.Series(has_nan.astype(float)).rolling(60, min_periods=20).mean().to_numpy()
        v["dq_rate"] = (rate > 0.10) & has_nan
        return v

    def run(self, ts: pd.DatetimeIndex, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        det = self.cfg["detection"]
        raw = rl.raw_residuals(frames, self.eqt, self.calib, self.cfg)
        z = rs.normalise(raw, self.norm)
        out = {}
        for eq in EQUIPMENT_ORDER:
            d = frames[eq]
            zf = rs.despike(z[eq])
            val = self._validity(eq, d, raw[eq])
            ctx = context_ok(d)
            res: dict[str, np.ndarray] = {}
            any_rel = np.zeros(len(d), dtype=bool)
            for rel in rl.RELATIONS:
                res[f"z_{rel}"] = zf[rel].to_numpy()
                res[f"raw_{rel}"] = raw[eq][rel].to_numpy()
            for rel, th in self.thresholds[eq].items():
                zs = zf[rel].clip(-STAT_CLIP, STAT_CLIP)  # bound the influence of one outlier
                e = st.ewma(zs, det["ewma_lambda"])
                c = st.cusum(zs, det["cusum_k"], th["cusum_cap"])
                f = (np.abs(zf[rel].to_numpy()) > det["z_threshold"]) | (np.abs(e) > th["ewma"]) | (np.abs(c) > th["cusum_h"])
                f = np.nan_to_num(f, nan=False).astype(bool)
                res[f"ewma_{rel}"] = e
                res[f"cusum_{rel}"] = c / th["cusum_h"]
                res[f"flag_{rel}"] = f
                if rel in TRIGGERS[eq]:
                    any_rel |= f
            for c in val.columns:
                res[c] = val[c].to_numpy()
            vflag = np.zeros(len(d), dtype=bool)
            for c in VALIDITY_TRIGGERS:
                x = val[c].to_numpy().astype(bool)
                x = x if c in NO_PERSIST + ("dq_rate",) else _persist(x)
                res[f"v_{c}"] = x
                vflag |= x
            X = features(zf, rl.APPLIES[eq], val)
            score = self.iforest.score(eq, X)
            iff = _min_len(score < self.iforest.thresholds[eq], det["iforest"]["min_len_min"])
            res["if_score"] = score
            res["if_flag"] = iff
            res["context_ok"] = ctx
            # LOST rows are always reported; other alarms are suppressed outside normal context
            res["alarm"] = res["v_lost"] | (ctx & (any_rel | vflag | iff))
            for fam, cols in FAMILIES.items():
                m = np.zeros(len(d), dtype=bool)
                for c in cols:
                    if c in res and (not c.startswith("flag_") or c[5:] in TRIGGERS[eq]):
                        m |= np.asarray(res[c], dtype=bool)
                res[f"fam_{fam}"] = (res["v_lost"] if fam == "data" else False) | (ctx & m)
            out[eq] = pd.DataFrame(res, index=d.index)
        return out
