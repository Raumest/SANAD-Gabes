"""UCI Steel Industry Energy Consumption adapter (Phase 2 stub).

Source: ``data/raw/steel/Steel_industry_data.csv`` — 15-min, 2018, CC BY 4.0.
Provides kWh, lagging/leading kVarh, PF, load type (Light/Medium/Maximum), week status.
Intended use: replace the synthetic load envelope L(t) and LIGHT/MEDIUM/MAXIMUM states.
"""

from __future__ import annotations

import pandas as pd

from sanad.adapters.base import DataSourceAdapter

EXPECTED_COLUMNS = [
    "date", "Usage_kWh", "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh", "CO2(tCO2)", "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor", "NSM", "WeekStatus", "Day_of_week", "Load_Type",
]


class SteelAdapter(DataSourceAdapter):
    name = "uci_steel"
    licence = "CC BY 4.0"

    def path(self):
        return self.raw_dir / "steel" / "Steel_industry_data.csv"

    def available(self) -> bool:
        return self.path().exists()

    def load(self) -> pd.DataFrame:
        # TODO(Phase 2): parse 'date' (dd/mm/yyyy HH:MM, local time) -> UTC, validate
        # EXPECTED_COLUMNS, keep kWh, kVarh, PF, Load_Type, WeekStatus.
        raise NotImplementedError("Steel adapter is a Phase 2 stub")

    def to_drivers(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        # TODO(Phase 2): normalise Usage_kWh to a 0..1 envelope, map Load_Type to
        # LIGHT/MEDIUM/MAXIMUM, interpolate 15 min -> 1 min onto ``index``.
        raise NotImplementedError("Steel adapter is a Phase 2 stub")
