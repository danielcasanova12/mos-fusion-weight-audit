"""Exact Shapley and grouped-player utilities for complete coalition games."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def exact_shapley(values: Mapping[int, float], n_players: int) -> np.ndarray:
    """Return exact Shapley values for a game including empty mask 0."""
    expected = set(range(1 << n_players))
    missing = expected.difference(values)
    if missing:
        raise ValueError(f"coalition game is incomplete; missing {len(missing)} masks")
    result = np.zeros(n_players, dtype=float)
    factorial = math.factorial
    normalizer = factorial(n_players)
    for player in range(n_players):
        bit = 1 << player
        for mask in range(1 << n_players):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(n_players - size - 1) / normalizer
            result[player] += weight * (values[mask | bit] - values[mask])
    return result


def merge_players(
    values: Mapping[int, float],
    groups: Sequence[Sequence[int]],
    n_players: int,
) -> dict[int, float]:
    """Create the induced game when source players are merged into groups."""
    flat = [player for group in groups for player in group]
    if sorted(flat) != list(range(n_players)):
        raise ValueError("groups must partition all source players exactly once")
    merged: dict[int, float] = {}
    for group_mask in range(1 << len(groups)):
        source_mask = 0
        for group_index, group in enumerate(groups):
            if group_mask & (1 << group_index):
                for player in group:
                    source_mask |= 1 << player
        merged[group_mask] = float(values[source_mask])
    return merged
