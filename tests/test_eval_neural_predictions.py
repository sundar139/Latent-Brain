from __future__ import annotations

import numpy as np

from latentbrain.eval.neural_predictions import summarize_rate_predictions


def test_summarize_rate_predictions_returns_safe_summary() -> None:
    summary = summarize_rate_predictions(np.array([1.0, 2.0, 3.0]))

    assert summary == {"mean_rate_hz": 2.0, "min_rate_hz": 1.0, "max_rate_hz": 3.0}
