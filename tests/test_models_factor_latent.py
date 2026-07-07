from __future__ import annotations

import numpy as np
import pytest

from latentbrain.models.factor_latent import FactorLatentModel


def _features() -> np.ndarray:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(40, 2))
    weights = np.array([[1.0, 0.2, -0.4, 0.5], [0.1, 0.8, 0.3, -0.2]])
    return base @ weights + rng.normal(scale=0.05, size=(40, 4))


def test_factor_model_rejects_non_2d_features() -> None:
    with pytest.raises(ValueError, match="rank 2"):
        FactorLatentModel(latent_dim=2, random_state=1, max_iter=100, tol=1e-4).fit(
            np.zeros((2, 3, 4))
        )


def test_factor_model_rejects_nan_or_inf() -> None:
    features = _features()
    features[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        FactorLatentModel(latent_dim=2, random_state=1, max_iter=100, tol=1e-4).fit(features)


def test_factor_model_rejects_latent_dim_at_least_feature_dim() -> None:
    with pytest.raises(ValueError, match="latent_dim"):
        FactorLatentModel(latent_dim=4, random_state=1, max_iter=100, tol=1e-4).fit(_features())


def test_fit_transform_shape_and_exposed_parameters() -> None:
    model = FactorLatentModel(latent_dim=2, random_state=1, max_iter=200, tol=1e-4)

    latents = model.fit_transform(_features())

    assert latents.shape == (40, 2)
    assert model.components_.shape == (2, 4)
    assert model.noise_variance_.shape == (4,)


def test_deterministic_transform_with_same_random_state() -> None:
    features = _features()
    model_a = FactorLatentModel(latent_dim=2, random_state=9, max_iter=200, tol=1e-4).fit(features)
    model_b = FactorLatentModel(latent_dim=2, random_state=9, max_iter=200, tol=1e-4).fit(features)

    np.testing.assert_allclose(model_a.transform(features), model_b.transform(features))
