"""Coordinatewise gate ranges inside a relative-loss Rashomon set."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .fusion import fit_simplex_gate, mse


def coordinate_ranges(
    predictions: np.ndarray,
    y: np.ndarray,
    *,
    relative_tolerance: float = 0.005,
    ftol: float = 1e-12,
) -> list[dict[str, float | bool | str]]:
    """Minimize/maximize each gate coordinate under a loss tolerance."""
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(y, dtype=float)
    optimum, optimal_mse = fit_simplex_gate(p, y, ftol=ftol)
    threshold = optimal_mse * (1.0 + relative_tolerance)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: threshold - mse(y, p @ w)},
    ]
    rows: list[dict[str, float | bool | str]] = []
    for index in range(p.shape[1]):
        extrema = []
        for direction in (1.0, -1.0):
            result = minimize(
                lambda w, d=direction, i=index: d * w[i],
                optimum,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * p.shape[1],
                constraints=constraints,
                options={"ftol": ftol, "maxiter": 5000, "disp": False},
            )
            extrema.append(result)
        minimum, maximum = extrema
        rows.append({
            "player": index,
            "optimal_weight": float(optimum[index]),
            "weight_min": float(minimum.x[index]),
            "weight_max": float(maximum.x[index]),
            "mse_threshold": threshold,
            "success_min": bool(minimum.success),
            "success_max": bool(maximum.success),
            "message_min": str(minimum.message),
            "message_max": str(maximum.message),
        })
    return rows
