"""Isolation Forest on residual and state features (unknown-pattern detector + OOD score).

Trained per equipment on the normal-only block. A row is flagged when its score falls
below the minimum training score minus a margin (i.e. beyond the training range).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def features(zf: pd.DataFrame, rels: list[str], val: pd.DataFrame) -> np.ndarray:
    X = zf[rels].clip(-10, 10).fillna(0.0)
    X = X.assign(nan_frac=val["nan_frac"].to_numpy(), i_pu=val["i_pu"].fillna(0).clip(0, 3).to_numpy(),
                 v_pu=val["v_pu"].fillna(0).clip(0, 1.3).to_numpy())
    return X.to_numpy(dtype=float)


class IForestDetector:
    def __init__(self, cfg: dict, seed: int):
        p = cfg["detection"]["iforest"]
        self.p = p
        self.seed = seed
        self.models: dict[str, IsolationForest] = {}
        self.thresholds: dict[str, float] = {}

    def fit(self, eq: str, X: np.ndarray) -> None:
        m = IsolationForest(n_estimators=self.p["n_estimators"], max_samples=min(self.p["max_samples"], len(X)),
                            random_state=self.seed, n_jobs=1)
        m.fit(X)
        s = m.score_samples(X)
        self.models[eq] = m
        self.thresholds[eq] = float(np.quantile(s, 0.0005)) - self.p["ood_margin"]

    def score(self, eq: str, X: np.ndarray) -> np.ndarray:
        return self.models[eq].score_samples(X)
