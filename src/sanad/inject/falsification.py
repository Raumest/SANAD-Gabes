"""Energy falsification: the kWh register under-reports its increments (10-40 %).

Only the register is tampered with; the meter's instantaneous P/Q/S stay honest, so the
signature is R5 (ΔkWh vs ∫P, and bus vs feeders) breaking while sensor-health cues are fine.
The deficit accumulated during the event persists afterwards (the register stays low).
"""

from __future__ import annotations

import numpy as np

from sanad.inject.events import Event


def apply(cfg: dict, meas: dict[str, dict], events: list[Event]) -> None:
    for e in events:
        if e.label != "FALS_ENERGY":
            continue
        kwh = meas[e.equipment_id]["kwh_register"]
        inc = np.diff(kwh, prepend=kwh[0])
        deficit = np.zeros_like(kwh)
        deficit[e.k0:e.k1] = e.params["underreport"] * inc[e.k0:e.k1]
        meas[e.equipment_id]["kwh_register"] = kwh - np.cumsum(deficit)
