"""Physical event library — modifies the TRUE plant drivers (golden rule 1).

The consequences (current, power, speed, temperature, vibration, bus, kWh) are derived
afterwards by ``sim.plant.compute_true_state`` so they propagate to every variable.
"""

from __future__ import annotations

import numpy as np

from sanad.config import rng_for
from sanad.inject.events import Event


def apply(cfg: dict, drivers: dict, events: list[Event]) -> None:
    rng = rng_for(cfg, "physical")
    for e in events:
        sl = slice(e.k0, e.k1)
        dur = e.k1 - e.k0
        if e.label == "PHYS_OVERLOAD":
            d = drivers["eq"][e.equipment_id]
            ramp = np.minimum(1.0, (np.arange(dur) + 1) / 3.0)  # 3-min ramp-up
            target = e.params["load"] * (1 + rng.normal(0, 0.01, dur))
            d["L"][sl] = d["L"][sl] + (target - d["L"][sl]) * ramp
        elif e.label == "PHYS_BEARING_DEGRADATION":
            # Time-compressed (days instead of months): h rises 1 -> h_max quadratically,
            # bearing replaced at the end of the event (h back to 1).
            x = np.arange(dur) / max(1, dur - 1)
            drivers["eq"][e.equipment_id]["health"][sl] = 1.0 + (e.params["h_max"] - 1.0) * x**2
        elif e.label == "GRID_SAG":
            # 1-min resolution: a sustained under-voltage (>= 1 min), not a sub-second sag.
            drivers["v_factor"][sl] = e.params["v_factor"]
        elif e.label == "GRID_INTERRUPTION":
            drivers["interrupt"][sl] = True
