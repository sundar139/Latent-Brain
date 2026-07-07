from __future__ import annotations

import numpy as np


def summarize_rate_predictions(rates_hz: np.ndarray) -> dict[str, float]:
    """Summarize finite positive rate predictions without making benchmark claims."""
    rates = np.asarray(rates_hz, dtype=np.float64)
    if rates.size == 0:
        msg = "rates_hz must not be empty"
        raise ValueError(msg)
    if not np.all(np.isfinite(rates)):
        msg = "rates_hz must be finite"
        raise ValueError(msg)
    if np.any(rates <= 0.0):
        msg = "rates_hz must be positive"
        raise ValueError(msg)
    return {
        "mean_rate_hz": float(np.mean(rates)),
        "min_rate_hz": float(np.min(rates)),
        "max_rate_hz": float(np.max(rates)),
    }
