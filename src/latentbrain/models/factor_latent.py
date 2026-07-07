from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.decomposition import FactorAnalysis  # type: ignore[import-untyped]


def _as_2d_features(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        msg = f"features must have rank 2; got shape {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "features must be finite"
        raise ValueError(msg)
    return array


@dataclass(slots=True)
class FactorLatentModel:
    latent_dim: int
    random_state: int
    max_iter: int
    tol: float
    _model: Any = field(init=False, default=None, repr=False)

    def fit(self, train_features: np.ndarray) -> FactorLatentModel:
        features = _as_2d_features(train_features)
        if self.latent_dim <= 0:
            msg = "latent_dim must be positive"
            raise ValueError(msg)
        if self.latent_dim >= features.shape[1]:
            msg = "latent_dim must be less than the feature dimension"
            raise ValueError(msg)
        if self.max_iter <= 0:
            msg = "max_iter must be positive"
            raise ValueError(msg)
        self._model = FactorAnalysis(
            n_components=self.latent_dim,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        ).fit(features)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            msg = "FactorLatentModel must be fit before transform"
            raise ValueError(msg)
        return np.asarray(self._model.transform(_as_2d_features(features)), dtype=np.float64)

    def fit_transform(self, train_features: np.ndarray) -> np.ndarray:
        self.fit(train_features)
        return self.transform(train_features)

    @property
    def components_(self) -> np.ndarray:
        if self._model is None:
            msg = "FactorLatentModel must be fit before components_ is available"
            raise ValueError(msg)
        return np.asarray(self._model.components_, dtype=np.float64)

    @property
    def noise_variance_(self) -> np.ndarray:
        if self._model is None:
            msg = "FactorLatentModel must be fit before noise_variance_ is available"
            raise ValueError(msg)
        return np.asarray(self._model.noise_variance_, dtype=np.float64)
