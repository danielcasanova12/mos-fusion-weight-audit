"""Ranking rules used inside the system-disjoint subset benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def top_k_mask(scores: Sequence[float], k: int) -> int:
    scores = np.asarray(scores, dtype=float)
    if not 1 <= k <= scores.size:
        raise ValueError("k must be between 1 and the number of players")
    order = np.lexsort((np.arange(scores.size), -scores))[:k]
    return int(sum(1 << int(index) for index in order))


def best_validation_mask(
    validation_mse: Mapping[int, float], k: int, n_players: int
) -> int:
    candidates = [mask for mask in range(1, 1 << n_players) if mask.bit_count() == k]
    missing = set(candidates).difference(validation_mse)
    if missing:
        raise ValueError(f"validation lattice lacks {len(missing)} size-{k} coalitions")
    return min(candidates, key=lambda mask: (validation_mse[mask], mask))


def selection_cost(n_players: int, method: str) -> int:
    costs = {
        "gate": 1,
        "singleton": n_players,
        "shapley": (1 << n_players) - 1,
        "inner_search": (1 << n_players) - 1,
    }
    if method not in costs:
        raise ValueError(f"unknown selection method: {method}")
    return costs[method]
