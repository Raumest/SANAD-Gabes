"""Paderborn bearing dataset adapter (Phase 2 stub).

Source: ``data/raw/paderborn/{K003,KA01,KA04,KI01}/*.mat`` — kHz vibration, CC BY-NC 4.0.
Only *relative statistics* (RMS, kurtosis, crest factor, healthy-vs-damaged ratios) are
ever derived. Raw kHz signals are never merged with minute data. Never commit .mat files.
"""

from __future__ import annotations

import pandas as pd

from sanad.adapters.base import DataSourceAdapter

SETS = {"K003": "healthy", "KA01": "outer_race_artificial", "KA04": "outer_race_real", "KI01": "inner_race_artificial"}


class PaderbornAdapter(DataSourceAdapter):
    name = "paderborn"
    licence = "CC BY-NC 4.0"

    def available(self) -> bool:
        return all((self.raw_dir / "paderborn" / s).exists() for s in SETS)

    def load(self) -> pd.DataFrame:
        # TODO(Phase 2): read *.mat with scipy.io.loadmat, compute per-file RMS, kurtosis,
        # crest factor; return one row per file (set, condition, stats). No raw signals.
        raise NotImplementedError("Paderborn adapter is a Phase 2 stub")

    def health_ratios(self) -> dict[str, float]:
        # TODO(Phase 2): damaged/healthy RMS ratios -> calibrate h_max of the vibration model.
        raise NotImplementedError("Paderborn adapter is a Phase 2 stub")

    def to_drivers(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        raise NotImplementedError("Paderborn only provides relative statistics, not a time series")
