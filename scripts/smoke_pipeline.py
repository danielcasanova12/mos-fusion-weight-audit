#!/usr/bin/env python3
"""Run a small synthetic end-to-end smoke test of the public modules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import grouped_bootstrap, percentile_interval
from src.coalitions import evaluate_coalitions
from src.experts import grouped_oof_predictions
from src.interactions import pairwise_shapley_interactions
from src.nested_selection import best_validation_mask, selection_cost, top_k_mask
from src.rashomon import coordinate_ranges
from src.shapley import exact_shapley


def main() -> None:
    rng = np.random.default_rng(20260912)
    n, players = 72, 3
    groups = np.repeat(np.arange(6), n // 6)
    latent = rng.normal(size=n)
    y = 0.7 * latent + rng.normal(scale=0.35, size=n)
    features = [
        np.column_stack([latent + rng.normal(scale=scale, size=n), rng.normal(size=n)])
        for scale in (0.25, 0.45, 0.85)
    ]
    oof = np.column_stack([
        grouped_oof_predictions(x, y, groups, n_splits=3) for x in features
    ])
    rows = evaluate_coalitions(oof, oof, y, y, ["a", "b", "c"])
    values = {0: 0.0}
    values.update({int(row["mask"]): float(row["value"]) for row in rows})
    shapley = exact_shapley(values, players)
    interactions = pairwise_shapley_interactions(values, players)
    ranges = coordinate_ranges(oof, y, relative_tolerance=0.01)
    lattice_mse = {int(row["mask"]): float(row["test_mse"]) for row in rows}
    best = best_validation_mask(lattice_mse, k=2, n_players=players)
    boot = grouped_bootstrap(groups, lambda idx: float(np.mean(y[idx])), n_bootstrap=50)
    assert len(rows) == 7 and np.isfinite(shapley).all()
    assert np.isclose(shapley.sum(), values[(1 << players) - 1] - values[0], atol=1e-10)
    assert interactions.shape == (players, players) and len(ranges) == players
    assert np.allclose(interactions, interactions.T) and np.allclose(np.diag(interactions), 0)
    assert best.bit_count() == 2 and top_k_mask(shapley, 2).bit_count() == 2
    assert selection_cost(players, "singleton") == 3
    assert percentile_interval(boot)[0] <= percentile_interval(boot)[1]
    print("OK: synthetic expert -> OOF -> gate -> coalition -> attribution pipeline")


if __name__ == "__main__":
    main()
