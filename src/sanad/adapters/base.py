"""Common adapter interface for real data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DataSourceAdapter(ABC):
    """A read-only adapter over a raw dataset in ``data/raw/``.

    Adapters never write to ``data/raw/``. They return tidy frames that a Phase 2
    driver will map onto the simulator drivers (load envelope, climate, vibration
    health) so that the generated files keep the synthetic schema.
    """

    name: str = "base"
    licence: str = ""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)

    @abstractmethod
    def available(self) -> bool:
        """True if the raw files exist locally."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load and parse the raw files into a tidy frame (UTC timestamps)."""

    @abstractmethod
    def to_drivers(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Resample/map to simulator drivers on the 1-min grid ``index``."""
