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
        v["dq_gap"] = (v["nan_frac"] >= 0.99).to_numpy() & ~v["lost"].to_numpy()
        v["dq_rate"] = pd.Series((v["nan_frac"] > 0).to_numpy() & ~v["lost"].to_numpy()).rolling(60, min_periods=20).mean().to_numpy() > 0.10
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
                e = st.ewma(zf[rel], det["ewma_lambda"])
                c = st.cusum(zf[rel], det["cusum_k"], th["cusum_cap"])
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
                x = x if c in ("lost",) else _persist(x)
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
            out[eq] = pd.DataFrame(res, index=d.index)
        return out
