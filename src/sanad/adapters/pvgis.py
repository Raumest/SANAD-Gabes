"""PVGIS hourly Gabès 2023 adapter (Phase 2 stub).

Source: ``data/raw/pvgis/PVGIS_Gabes_2023.csv`` — hourly, UTC, with header metadata lines
and footer legend lines. Columns: time, Gb(i), Gd(i), Gr(i), H_sun, T2m, WS10m, Int.
Intended use: replace the synthetic ambient temperature (T2m) and irradiance.
"""

from __future__ import annotations

import pandas as pd

from sanad.adapters.base import DataSourceAdapter

EXPECTED_COLUMNS = ["time", "Gb(i)", "Gd(i)", "Gr(i)", "H_sun", "T2m", "WS10m", "Int"]


class PVGISAdapter(DataSourceAdapter):
    name = "pvgis_gabes"
    licence = "PVGIS / EU JRC terms"

    def path(self):
        return self.raw_dir / "pvgis" / "PVGIS_Gabes_2023.csv"

    def available(self) -> bool:
        return self.path().exists()

    def load(self) -> pd.DataFrame:
        # TODO(Phase 2): skip header metadata until the 'time,' row, stop at the first
        # blank/legend line, parse time 'YYYYMMDD:HHMM' as UTC.
        raise NotImplementedError("PVGIS adapter is a Phase 2 stub")

    def to_drivers(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        # TODO(Phase 2): ambient_temp_c <- T2m, irradiance <- Gb+Gd+Gr, interpolate to 1 min.
        raise NotImplementedError("PVGIS adapter is a Phase 2 stub")
