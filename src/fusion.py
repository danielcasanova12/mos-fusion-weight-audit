"""Nonnegative simplex fusion of frozen expert predictions."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def project_simplex(vector: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum(w) = 1}."""
    v = np.asarray(vector, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("vector must be non-empty and one-dimensional")
    ordered = np.sort(v)[::-1]
    cssv = np.cumsum(ordered) - 1.0
    rho = np.flatnonzero(ordered - cssv / np.arange(1, v.size + 1) > 0)
    theta = cssv[rho[-1]] / float(rho[-1] + 1)
    return np.maximum(v - theta, 0.0)


def mse(y: np.ndarray, prediction: np.ndarray) -> float:
    residual = np.asarray(y, dtype=float) - np.asarray(prediction, dtype=float)
    return float(np.mean(residual * residual))


def fit_simplex_gate(
    predictions: np.ndarray,
    y: np.ndarray,
    *,
    ftol: float = 1e-12,
    maxiter: int = 5000,
) -> tuple[np.ndarray, float]:
    """Fit a deterministic MSE-minimizing convex mixture."""
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.ndim != 2 or p.shape[0] != y.size:
        raise ValueError("predictions must have shape (n_items, n_experts)")
    if p.shape[1] == 1:
        return np.ones(1), mse(y, p[:, 0])
    objective = lambda w: mse(y, p @ w)
    initial = np.full(p.shape[1], 1.0 / p.shape[1])
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * p.shape[1],
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"ftol": ftol, "maxiter": maxiter, "disp": False},
    )
    weights = project_simplex(result.x)
    if not result.success:
        raise RuntimeError(f"simplex optimization failed: {result.message}")
    return weights, objective(weights)


def predict_fusion(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    p = np.asarray(predictions, dtype=float)
    w = np.asarray(weights, dtype=float)
    if p.ndim != 2 or p.shape[1] != w.size:
        raise ValueError("prediction columns must match the number of weights")
    return p @ w
