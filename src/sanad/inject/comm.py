"""Communication dropout: the gateway receives nothing from a device for 1-120 min.

At 1-min resolution a dropout shorter than a minute rounds up to one row. The gateway
keeps a placeholder row with every value empty and ``comm_status = LOST``. The device
kept integrating energy, so the first kWh register after recovery "catches up" (jump).
The first row after recovery arrives late (``DELAYED``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanad.inject.events import Event
from sanad.schema import VALUE_COLUMNS


def apply(cfg: dict, frames: dict[str, pd.DataFrame], events: list[Event], dev_offset: dict) -> None:
    for e in events:
        if e.label != "COMM_DROPOUT":
            continue
        df = frames[e.equipment_id]
        rows = df.index[e.k0:e.k1]
        df.loc[rows, VALUE_COLUMNS] = np.nan
        df.loc[rows, "comm_status"] = "LOST"
        if e.k1 < len(df):
            df.loc[df.index[e.k1], "comm_status"] = "DELAYED"
            dev_offset[e.equipment_id][e.k1] = -float(min(55.0, 5.0 + (e.k1 - e.k0)))
