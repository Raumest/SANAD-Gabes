"""Gabès-like climate: ambient temperature and clear-sky irradiance (synthetic mode).

Assumptions (configs/default.yaml -> climate): ambient = annual sinusoid (peak ~ day 205)
+ daily sinusoid (peak 15:00 local) + AR(1) weather noise. Early-summer afternoons reach
~38-42 °C and winter nights ~6-10 °C with the default parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanad.sim.noise import ar1


def local_hours(idx: pd.DatetimeIndex, utc_offset_h: float) -> np.ndarray:
    loc = idx + pd.Timedelta(hours=utc_offset_h)
    return loc.hour.to_numpy() + loc.minute.to_numpy() / 60.0


def ambient_temperature(cfg: dict, idx: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    c = cfg["climate"]
    doy = idx.dayofyear.to_numpy() + local_hours(idx, cfg["time"]["utc_offset_h"]) / 24.0
    h = local_hours(idx, cfg["time"]["utc_offset_h"])
    annual = c["t_annual_mean_c"] + c["t_annual_amp_c"] * np.cos(2 * np.pi * (doy - 205) / 365.25)
    daily = c["t_daily_amp_c"] * np.cos(2 * np.pi * (h - 15.0) / 24.0)
    return annual + daily + ar1(rng, len(idx), c["t_ar_phi"], c["t_ar_sigma"])


def irradiance(cfg: dict, idx: pd.DatetimeIndex) -> np.ndarray:
    """Clear-sky-like bell curve (W/m2); day length follows the season (Gabès ~33.9°N)."""
    h = local_hours(idx, cfg["time"]["utc_offset_h"])
    doy = idx.dayofyear.to_numpy()
    daylen = 12.0 + 2.2 * np.cos(2 * np.pi * (doy - 172) / 365.25)
    noon = 12.6
    x = (h - (noon - daylen / 2)) / daylen
    bell = np.where((x > 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)) ** 1.5, 0.0)
    season = 0.85 + 0.15 * np.cos(2 * np.pi * (doy - 172) / 365.25)
    return cfg["climate"]["irradiance_peak_wm2"] * season * bell
