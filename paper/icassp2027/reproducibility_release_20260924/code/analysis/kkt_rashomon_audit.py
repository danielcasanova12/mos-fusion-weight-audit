#!/usr/bin/env python3
"""Recompute Rashomon extrema and audit primal/KKT residuals.

This is a reviewer-facing numerical certificate for the convex coordinate
range problems used in the ICASSP manuscript. It does not modify model data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER))

import run_final_icasp_controls as controls  # noqa: E402
import run_reviewer_priority_cpu as reviewer  # noqa: E402


def kkt_residuals(
    predictions: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    feature_index: int,
    maximize: bool,
    active_tolerance: float,
) -> dict[str, float | int]:
    """Estimate KKT multipliers on the numerical active set."""
    n = len(weights)
    objective_gradient = np.zeros(n, dtype=np.float64)
    objective_gradient[feature_index] = -1.0 if maximize else 1.0
    loss, loss_gradient = reviewer.loss_and_gradient(predictions, target, weights)

    columns = [np.ones(n, dtype=np.float64)]
    lower_bounds = [-np.inf]
    upper_bounds = [np.inf]
    multiplier_kinds: list[tuple[str, int | None]] = [("equality", None)]

    loss_active = abs(loss - threshold) <= active_tolerance
    if loss_active:
        columns.append(loss_gradient)
        lower_bounds.append(0.0)
        upper_bounds.append(np.inf)
        multiplier_kinds.append(("loss", None))

    for index, value in enumerate(weights):
        if value <= active_tolerance:
            column = np.zeros(n, dtype=np.float64)
            column[index] = -1.0
            columns.append(column)
            lower_bounds.append(0.0)
            upper_bounds.append(np.inf)
            multiplier_kinds.append(("lower", index))
        if 1.0 - value <= active_tolerance:
            column = np.zeros(n, dtype=np.float64)
            column[index] = 1.0
            columns.append(column)
            lower_bounds.append(0.0)
            upper_bounds.append(np.inf)
            multiplier_kinds.append(("upper", index))

    matrix = np.column_stack(columns)
    dual = lsq_linear(
        matrix,
        -objective_gradient,
        bounds=(np.asarray(lower_bounds), np.asarray(upper_bounds)),
        lsmr_tol="auto",
        max_iter=1000,
    )
    stationarity = objective_gradient + matrix @ dual.x

    complementarity = [0.0]
    for multiplier, (kind, index) in zip(dual.x, multiplier_kinds):
        if kind == "loss":
            complementarity.append(abs(multiplier * (loss - threshold)))
        elif kind == "lower" and index is not None:
            complementarity.append(abs(multiplier * weights[index]))
        elif kind == "upper" and index is not None:
            complementarity.append(abs(multiplier * (weights[index] - 1.0)))

    primal = max(
        abs(float(weights.sum()) - 1.0),
        max(0.0, float(-weights.min())),
        max(0.0, float(weights.max() - 1.0)),
        max(0.0, float(loss - threshold)),
    )
    return {
        "loss": float(loss),
        "loss_active": int(loss_active),
        "n_active_lower_bounds": int(np.sum(weights <= active_tolerance)),
        "primal_residual": float(primal),
        "stationarity_residual": float(np.max(np.abs(stationarity))),
        "complementarity_residual": float(max(complementarity)),
        "dual_fit_cost": float(dual.cost),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path("results/paper_submission_final_cpu/primary_oof"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/paper_reviewer_priority_cpu/rashomon_kkt_audit.csv"),
    )
    parser.add_argument("--active-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    root = args.root.resolve()
    primary = args.primary if args.primary.is_absolute() else root / args.primary
    output = args.output if args.output.is_absolute() else root / args.output

    rows: list[dict[str, float | int | str]] = []
    for dataset in reviewer.DATASETS:
        bundle = reviewer.load_bundle(primary, dataset)
        predictions = np.asarray(bundle["oof"], dtype=np.float64)
        target = np.asarray(bundle["y_train"], dtype=np.float64)
        optimum = controls.fit_simplex(predictions, target)
        optimum_loss, _ = reviewer.loss_and_gradient(predictions, target, optimum)
        for tolerance in (0.001, 0.005, 0.01):
            threshold = optimum_loss * (1.0 + tolerance)
            for feature_index, feature in enumerate(reviewer.FEATURES):
                for label, maximize in (("min", False), ("max", True)):
                    weights, result = reviewer.rashomon_extreme(
                        predictions,
                        target,
                        optimum,
                        threshold,
                        feature_index,
                        maximize,
                    )
                    audit = kkt_residuals(
                        predictions,
                        target,
                        weights,
                        threshold,
                        feature_index,
                        maximize,
                        args.active_tolerance,
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "feature": feature,
                            "relative_mse_tolerance": tolerance,
                            "extreme": label,
                            "solver_success": int(bool(result.success)),
                            "weight": float(weights[feature_index]),
                            "mse_threshold": float(threshold),
                            **audit,
                        }
                    )

    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "rows": len(frame),
        "solver_failures": int((frame["solver_success"] == 0).sum()),
        "max_primal_residual": float(frame["primal_residual"].max()),
        "max_stationarity_residual": float(frame["stationarity_residual"].max()),
        "max_complementarity_residual": float(frame["complementarity_residual"].max()),
    }
    print(summary)


if __name__ == "__main__":
    main()
