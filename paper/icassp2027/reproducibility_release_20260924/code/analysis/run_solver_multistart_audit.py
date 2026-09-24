"""Same-data SLSQP multi-start audit for the ten-player full gates.

The audit reuses saved OOF predictions and changes only the simplex
initialization. It intentionally does not apply the projected-gradient
fallback, so SLSQP convergence is recorded rather than hidden.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE.parent / "review_v2_snapshot"
OUT = HERE / "review_evidence" / "solver_multistart"
DATASETS = ["brspeech", "bvcc", "singmos", "tmhintqi"]
FEATURES = [
    "whisper",
    "contentvec12",
    "wavlm",
    "beats",
    "auditory_erb",
    "speaker",
    "rmvpe_cont",
    "rmvpe_quant",
    "ced",
    "egemaps",
]
N_RANDOM = 100
SEED = 20260915
FTOL = 1e-12
MAXITER = 1000
ZERO_THRESHOLD = 1e-6


def fit_gate(predictions: np.ndarray, target: np.ndarray, initial: np.ndarray):
    predictions = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    k = predictions.shape[1]

    def objective(weights: np.ndarray) -> float:
        return float(np.mean((target - predictions @ weights) ** 2))

    def gradient(weights: np.ndarray) -> np.ndarray:
        return (-2.0 / len(target)) * predictions.T @ (target - predictions @ weights)

    return minimize(
        objective,
        np.asarray(initial, dtype=np.float64),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
            "jac": lambda weights: np.ones(k, dtype=np.float64),
        },
        options={"maxiter": MAXITER, "ftol": FTOL, "disp": False},
    )


def support(weights: np.ndarray) -> str:
    return "+".join(
        feature for feature, weight in zip(FEATURES, weights) if weight >= ZERO_THRESHOLD
    ) or "<none>"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        path = SNAPSHOT / f"{dataset}_alpha10p0_folds5.npz"
        with np.load(path, allow_pickle=False) as saved:
            predictions = np.asarray(saved["oof"], dtype=np.float64)
            target = np.asarray(saved["y_train"], dtype=np.float64)
        n = predictions.shape[1]
        rng = np.random.default_rng(SEED + dataset_index)
        initials = [("uniform", 0, np.full(n, 1.0 / n, dtype=np.float64))]
        initials.extend(
            ("dirichlet", index + 1, draw)
            for index, draw in enumerate(rng.dirichlet(np.ones(n), size=N_RANDOM))
        )
        for kind, init_id, initial in initials:
            result = fit_gate(predictions, target, initial)
            weights = np.asarray(result.x, dtype=np.float64)
            loss = float(np.mean((target - predictions @ weights) ** 2))
            rows.append(
                {
                    "dataset": dataset,
                    "init_kind": kind,
                    "init_id": init_id,
                    "success": bool(result.success),
                    "status": int(result.status),
                    "message": str(result.message),
                    "loss": loss,
                    "sum_error": float(abs(weights.sum() - 1.0)),
                    "min_weight": float(weights.min()),
                    "support": support(weights),
                    **{f"weight_{feature}": float(weight) for feature, weight in zip(FEATURES, weights)},
                }
            )
        group = pd.DataFrame([row for row in rows if row["dataset"] == dataset])
        successful = group[group.success].copy()
        uniform = successful[successful.init_kind == "uniform"].iloc[0]
        modal_support = successful.support.value_counts().idxmax()
        modal_count = int((successful.support == modal_support).sum())
        uniform_loss = float(uniform.loss)
        summaries.append(
            {
                "dataset": dataset,
                "n_initializations": int(len(group)),
                "n_random_initializations": N_RANDOM,
                "n_successful": int(len(successful)),
                "n_failed": int(len(group) - len(successful)),
                "uniform_support": str(uniform.support),
                "modal_support": str(modal_support),
                "modal_support_frequency": float(modal_count / len(successful)),
                "n_distinct_supports": int(successful.support.nunique()),
                "uniform_loss": uniform_loss,
                "max_abs_loss_difference_vs_uniform": float(
                    np.max(np.abs(successful.loss.to_numpy(float) - uniform_loss))
                ),
                "min_loss": float(successful.loss.min()),
                "max_loss": float(successful.loss.max()),
                "max_weight_range": float(
                    max(
                        successful[f"weight_{feature}"].max()
                        - successful[f"weight_{feature}"].min()
                        for feature in FEATURES
                    )
                ),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "solver_multistart_runs.csv", index=False)
    (OUT / "solver_multistart_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "seed": SEED,
                "n_random_initializations_per_corpus": N_RANDOM,
                "ftol": FTOL,
                "maxiter": MAXITER,
                "zero_threshold": ZERO_THRESHOLD,
                "initializations": "uniform plus symmetric Dirichlet(1,...,1)",
                "summary": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
