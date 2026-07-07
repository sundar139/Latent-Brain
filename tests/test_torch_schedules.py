from __future__ import annotations

import pytest

from latentbrain.torch.schedules import linear_warmup


def test_warmup_starts_low() -> None:
    assert linear_warmup(epoch=0, warmup_epochs=4) == pytest.approx(0.0)


def test_warmup_reaches_max() -> None:
    assert linear_warmup(epoch=4, warmup_epochs=4, max_value=2.0) == pytest.approx(2.0)
    assert linear_warmup(epoch=10, warmup_epochs=4, max_value=2.0) == pytest.approx(2.0)


def test_warmup_with_zero_epochs_returns_max() -> None:
    assert linear_warmup(epoch=0, warmup_epochs=0, max_value=0.7) == pytest.approx(0.7)
