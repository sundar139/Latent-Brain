from __future__ import annotations

import numpy as np
import pytest

from latentbrain.randomness import seed_everything


def test_same_seed_produces_same_numpy_values() -> None:
    seed_everything(1337)
    first = np.random.random(5)

    seed_everything(1337)
    second = np.random.random(5)

    np.testing.assert_allclose(first, second)


def test_negative_seed_raises_value_error() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)
