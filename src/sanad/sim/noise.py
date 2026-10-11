"""Seeded noise processes."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Stationary AR(1): x[k] = phi*x[k-1] + e[k], e ~ N(0, sigma^2)."""
    e = rng.normal(0.0, sigma, n)
    e[0] = rng.normal(0.0, sigma / np.sqrt(max(1e-9, 1 - phi**2)))
    return lfilter([1.0], [1.0, -phi], e)
