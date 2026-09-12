"""Cluster-bootstrap helpers with explicit deterministic seeds."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def resample_group_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    chunks = [np.flatnonzero(groups == group) for group in sampled]
    return np.concatenate(chunks)


def grouped_bootstrap(
    groups: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20260909,
) -> np.ndarray:
    """Evaluate a statistic on indices from group-level resamples."""
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    return np.asarray([
        statistic(resample_group_indices(groups, rng))
        for _ in range(n_bootstrap)
    ], dtype=float)


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(np.asarray(values, dtype=float), [tail, 1.0 - tail])
    return float(low), float(high)
