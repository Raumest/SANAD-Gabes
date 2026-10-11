"""Isotonic calibration of the top-hypothesis confidence on the validation block.

Target: P(top hypothesis class == ground-truth class of the matched event). Fitted only
on validation-block cases (ground truth from events.csv is used for calibration only,
never as a model input). Falls back to identity if too few validation cases.

With few validation cases a pure isotonic map is jumpy (it can map to exactly 0 or 1), so
the calibrated value is shrunk toward the raw score: c = w*iso + (1-w)*raw with
w = n / (n + PRIOR_CASES).
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

MIN_CASES = 15
PRIOR_CASES = 20


class Calibrator:
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.model: IsotonicRegression | None = None
        self.info: dict = {"method": "identity", "n": 0}

    def fit(self, raw: np.ndarray, correct: np.ndarray) -> "Calibrator":
        raw, correct = np.asarray(raw, float), np.asarray(correct, float)
        self.info = {"method": "identity", "n": int(len(raw))}
        if self.method == "isotonic" and len(raw) >= MIN_CASES and 0 < correct.mean() < 1:
            self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, correct)
            self.info = {"method": "isotonic", "n": int(len(raw)), "accuracy": float(correct.mean()),
                         "ece_before": expected_calibration_error(raw, correct),
                         "ece_after": expected_calibration_error(self.transform(raw), correct)}
        return self

    def transform(self, raw) -> np.ndarray:
        raw = np.asarray(raw, float)
        if self.model is None:
            return raw
        n = self.info.get("n", 0)
        w = n / (n + PRIOR_CASES)
        return w * self.model.predict(raw) + (1 - w) * raw

    def to_json(self) -> dict:
        d = dict(self.info)
        if self.model is not None:
            d["x"] = [float(x) for x in self.model.X_thresholds_]
            d["y"] = [float(y) for y in self.model.y_thresholds_]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Calibrator":
        c = cls()
        c.info = d
        if d.get("method") == "isotonic":
            m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            m.fit(d["x"], d["y"])
            c.model = m
        return c


def expected_calibration_error(conf, correct, bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & ((conf < b) if b < 1 else (conf <= b))
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)
