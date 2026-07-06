from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def _as_2d(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        msg = f"{name} must have rank 2; got shape {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return array


def standardize_train_apply(
    train_values: np.ndarray,
    values: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute train-only standardization stats and apply them to values."""
    train = _as_2d("train_values", train_values)
    array = _as_2d("values", values)
    if train.shape[1] != array.shape[1]:
        msg = "train_values and values must have the same feature dimension"
        raise ValueError(msg)
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    stats = {"mean": mean, "std": np.maximum(std, eps)}
    return apply_standardization(array, stats), stats


def apply_standardization(values: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Apply precomputed standardization stats."""
    array = _as_2d("values", values)
    return np.asarray((array - stats["mean"]) / stats["std"], dtype=np.float64)


def fit_ridge_decoder(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    fit_intercept: bool = True,
) -> dict[str, np.ndarray]:
    """Fit a closed-form ridge decoder. Intercept is not regularized."""
    if alpha < 0:
        msg = "alpha must be non-negative"
        raise ValueError(msg)
    x = _as_2d("features", features)
    y = _as_2d("targets", targets)
    if x.shape[0] != y.shape[0]:
        msg = "features and targets must have the same sample count"
        raise ValueError(msg)
    design = np.column_stack([np.ones(x.shape[0]), x]) if fit_intercept else x
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    if fit_intercept:
        penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    if fit_intercept:
        intercept = weights[0]
        coefficients = weights[1:]
    else:
        intercept = np.zeros(y.shape[1], dtype=np.float64)
        coefficients = weights
    return {"coefficients": coefficients, "intercept": intercept}


def predict_ridge_decoder(features: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    """Predict targets from ridge decoder coefficients."""
    x = _as_2d("features", features)
    return np.asarray(x @ model["coefficients"] + model["intercept"], dtype=np.float64)


def r2_score_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
) -> np.ndarray | float:
    """Compute R2 with NaN for constant targets."""
    true = _as_2d("y_true", y_true)
    pred = _as_2d("y_pred", y_pred)
    if true.shape != pred.shape:
        msg = "y_true and y_pred must have the same shape"
        raise ValueError(msg)
    ss_res = np.sum((true - pred) ** 2, axis=0)
    ss_tot = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
    r2 = np.full(true.shape[1], np.nan, dtype=np.float64)
    mask = ss_tot > 0
    r2[mask] = 1.0 - ss_res[mask] / ss_tot[mask]
    if multioutput == "raw_values":
        return r2
    if multioutput == "uniform_average":
        return float(np.nanmean(r2))
    msg = "multioutput must be raw_values or uniform_average"
    raise ValueError(msg)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> pd.DataFrame:
    """Compute per-target regression metrics."""
    true = _as_2d("y_true", y_true)
    pred = _as_2d("y_pred", y_pred)
    if true.shape != pred.shape:
        msg = "y_true and y_pred must have the same shape"
        raise ValueError(msg)
    if true.shape[1] != len(target_names):
        msg = "target_names length must match target dimension"
        raise ValueError(msg)
    r2 = np.asarray(r2_score_numpy(true, pred), dtype=np.float64)
    return pd.DataFrame(
        {
            "target_name": target_names,
            "r2": r2,
            "mse": np.mean((true - pred) ** 2, axis=0),
            "mae": np.mean(np.abs(true - pred), axis=0),
            "target_variance": np.var(true, axis=0),
        }
    )
