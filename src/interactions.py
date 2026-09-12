"""Pairwise Shapley interactions and prediction-redundancy diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy.stats import pearsonr, spearmanr


def pairwise_shapley_interactions(
    values: Mapping[int, float], n_players: int
) -> np.ndarray:
    """Compute the symmetric Shapley interaction index for every pair."""
    if set(values) != set(range(1 << n_players)):
        raise ValueError("a complete game including the empty coalition is required")
    result = np.zeros((n_players, n_players), dtype=float)
    denom = math.factorial(n_players - 1)
    for i in range(n_players):
        for j in range(i + 1, n_players):
            bits = (1 << i) | (1 << j)
            total = 0.0
            for mask in range(1 << n_players):
                if mask & bits:
                    continue
                size = mask.bit_count()
                weight = math.factorial(size) * math.factorial(n_players - size - 2) / denom
                second = values[mask | bits] - values[mask | (1 << i)] - values[mask | (1 << j)] + values[mask]
                total += weight * second
            result[i, j] = result[j, i] = total
    return result


def prediction_correlations(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Pearson and Spearman matrices for expert predictions."""
    p = np.asarray(predictions, dtype=float)
    if p.ndim != 2:
        raise ValueError("predictions must be two-dimensional")
    n = p.shape[1]
    pearson = np.eye(n)
    spearman = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            pearson[i, j] = pearson[j, i] = pearsonr(p[:, i], p[:, j]).statistic
            spearman[i, j] = spearman[j, i] = spearmanr(p[:, i], p[:, j]).statistic
    return pearson, spearman
