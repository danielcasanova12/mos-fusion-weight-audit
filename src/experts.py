"""Fold-safe preprocessing and ridge-expert fitting.

All learned transforms are fit on the supplied training indices.  The module
operates on already extracted representations and never reads private audio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


@dataclass
class FoldTransform:
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(x, dtype=float), self.lower, self.upper)
        return (clipped - self.mean) / self.scale


def fit_fold_transform(
    x_train: np.ndarray,
    winsor_percentiles: tuple[float, float] | None = None,
) -> FoldTransform:
    """Fit optional winsorization and z-normalization on training data only."""
    x = np.asarray(x_train, dtype=float)
    if x.ndim != 2:
        raise ValueError("x_train must be a 2-D array")
    if winsor_percentiles is None:
        lower = np.full(x.shape[1], -np.inf)
        upper = np.full(x.shape[1], np.inf)
    else:
        lo, hi = winsor_percentiles
        if not 0 <= lo < hi <= 100:
            raise ValueError("winsor percentiles must satisfy 0 <= lo < hi <= 100")
        lower, upper = np.percentile(x, [lo, hi], axis=0)
    clipped = np.clip(x, lower, upper)
    mean = clipped.mean(axis=0)
    scale = clipped.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return FoldTransform(lower=lower, upper=upper, mean=mean, scale=scale)


def fit_ridge_expert(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    alpha: float = 10.0,
    winsor_percentiles: tuple[float, float] | None = None,
) -> tuple[np.ndarray, Ridge, FoldTransform]:
    """Fit one ridge expert and return predictions plus fitted objects."""
    transform = fit_fold_transform(x_train, winsor_percentiles)
    model = Ridge(alpha=alpha)
    model.fit(transform.apply(x_train), np.asarray(y_train, dtype=float))
    return model.predict(transform.apply(x_eval)), model, transform


def grouped_oof_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float = 10.0,
    n_splits: int = 5,
    winsor_percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """Generate grouped out-of-fold predictions with fold-local transforms."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if len(x) != len(y) or len(y) != len(groups):
        raise ValueError("x, y, and groups must have equal length")
    if np.unique(groups).size < n_splits:
        raise ValueError("n_splits exceeds the number of unique systems")
    pred = np.full(len(y), np.nan)
    for train, valid in GroupKFold(n_splits=n_splits).split(x, y, groups):
        pred[valid], _, _ = fit_ridge_expert(
            x[train], y[train], x[valid], alpha=alpha,
            winsor_percentiles=winsor_percentiles,
        )
    if not np.isfinite(pred).all():
        raise RuntimeError("OOF generation left non-finite predictions")
    return pred
