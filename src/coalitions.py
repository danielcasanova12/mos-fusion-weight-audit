"""Exhaustive coalition enumeration and evaluation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from .fusion import fit_simplex_gate, mse


def coalition_members(mask: int, n_players: int) -> tuple[int, ...]:
    if mask < 0 or mask >= 1 << n_players:
        raise ValueError("coalition mask is outside the player set")
    return tuple(index for index in range(n_players) if mask & (1 << index))


def nonempty_coalitions(n_players: int) -> Iterator[int]:
    if n_players <= 0:
        raise ValueError("n_players must be positive")
    yield from range(1, 1 << n_players)


def evaluate_coalitions(
    oof_predictions: np.ndarray,
    test_predictions: np.ndarray,
    y_oof: np.ndarray,
    y_test: np.ndarray,
    player_names: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Fit each gate on OOF predictions and evaluate once on held-out items."""
    oof = np.asarray(oof_predictions, dtype=float)
    test = np.asarray(test_predictions, dtype=float)
    if oof.ndim != 2 or test.ndim != 2 or oof.shape[1] != test.shape[1]:
        raise ValueError("OOF and test prediction matrices must share columns")
    n_players = oof.shape[1]
    names = list(player_names or [f"player_{i}" for i in range(n_players)])
    if len(names) != n_players:
        raise ValueError("player_names length does not match prediction columns")
    rows: list[dict[str, object]] = []
    null_mse = mse(y_test, np.full(len(y_test), np.mean(y_oof)))
    for mask in nonempty_coalitions(n_players):
        members = coalition_members(mask, n_players)
        weights, oof_mse = fit_simplex_gate(oof[:, members], y_oof)
        test_mse = mse(y_test, test[:, members] @ weights)
        rows.append({
            "mask": mask,
            "n_features": len(members),
            "features": "+".join(names[i] for i in members),
            "oof_mse": oof_mse,
            "test_mse": test_mse,
            "value": null_mse - test_mse,
            "weights": weights,
        })
    return rows
