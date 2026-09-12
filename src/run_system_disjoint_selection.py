#!/usr/bin/env python3
"""Nested system-disjoint comparison of representation-selection scores.

The experiment answers a decision question: under the same cardinality budget,
which score selects a useful set of frozen representation extractors for MOS on
systems that were absent from every fitted component?

For each outer GroupKFold split, ridge experts are refit from cached embeddings
using only outer-training systems. Inner GroupKFold predictions on those systems
are then used to compute four train-only selection rules:

* full-gate weight ranking;
* exact Shapley ranking from the complete 10-player coalition lattice;
* singleton validation-loss ranking;
* the subset with the lowest inner loss at each cardinality (search baseline).

The selected subset's simplex gate is calibrated on the inner cross-fitted
expert predictions and applied once to outer-held-system predictions. eGeMAPS
winsorization, standardization, ridge fitting, selection, and gate calibration
are all contained inside the outer split. Outputs are fold-resumable and include
item predictions, aggregate metrics, paired system-bootstrap intervals, solver
parity diagnostics, progress, and errors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


HERE = Path(__file__).resolve().parent
REMOTE_PAPER = HERE.parent if (HERE.parent / "run_missing_experiments.py").exists() else HERE
sys.path.insert(0, str(REMOTE_PAPER))

import run_final_icasp_controls as controls  # noqa: E402
import run_missing_experiments as base  # noqa: E402


FEATURES = list(base.FEATURES)
DATASETS = list(base.DATASETS)
N_FEATURES = len(FEATURES)
FULL_MASK = (1 << N_FEATURES) - 1
METHODS = ("gate", "shapley", "singleton", "inner_search")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def record_error(out: Path, task: str, error: BaseException) -> None:
    payload = {
        "task": task,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }
    with (out / "errors.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def mask_names(mask: int) -> list[str]:
    return [feature for index, feature in enumerate(FEATURES) if mask & (1 << index)]


def load_arrays(root: Path, cache: str, dataset: str, split: str) -> dict[str, np.ndarray]:
    return {feature: base.load_x(root, cache, dataset, split, feature) for feature in FEATURES}


def transform_feature(
    feature: str,
    raw_fit: np.ndarray,
    raw_evaluate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fit = np.asarray(raw_fit, dtype=np.float64)
    evaluate = np.asarray(raw_evaluate, dtype=np.float64)
    if feature == "egemaps":
        low, high = np.quantile(fit, (0.001, 0.999), axis=0)
        fit = np.clip(fit, low, high)
        evaluate = np.clip(evaluate, low, high)
    mean = fit.mean(axis=0)
    scale = fit.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return (fit - mean) / scale, (evaluate - mean) / scale


def fit_experts(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    fit_indices: np.ndarray,
    evaluate_indices: np.ndarray,
    alpha: float,
) -> np.ndarray:
    prediction = np.empty((len(evaluate_indices), N_FEATURES), dtype=np.float64)
    for feature_index, feature in enumerate(FEATURES):
        fit_x, evaluate_x = transform_feature(
            feature,
            arrays[feature][fit_indices],
            arrays[feature][evaluate_indices],
        )
        model = Ridge(
            alpha=float(alpha),
            fit_intercept=True,
            solver="lsqr",
            tol=1e-10,
            max_iter=10000,
        )
        model.fit(fit_x, y[fit_indices])
        prediction[:, feature_index] = model.predict(evaluate_x)
    return prediction


def inner_cross_fitted_predictions(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    alpha: float,
    inner_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    local_groups = groups[outer_train]
    n_splits = min(int(inner_folds), len(np.unique(local_groups)))
    if n_splits < 2:
        raise ValueError("At least two outer-training systems are required")
    prediction = np.full((len(outer_train), N_FEATURES), np.nan, dtype=np.float64)
    null_prediction = np.full(len(outer_train), np.nan, dtype=np.float64)
    split = GroupKFold(n_splits=n_splits).split(
        np.zeros(len(outer_train)), y[outer_train], local_groups
    )
    for fit_local, hold_local in split:
        fit_absolute = outer_train[fit_local]
        hold_absolute = outer_train[hold_local]
        prediction[hold_local] = fit_experts(arrays, y, fit_absolute, hold_absolute, alpha)
        null_prediction[hold_local] = float(np.mean(y[fit_absolute]))
    if not np.isfinite(prediction).all() or not np.isfinite(null_prediction).all():
        raise RuntimeError("Incomplete inner cross-fitted predictions")
    return prediction, null_prediction


def coalition_game(prediction: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    losses = np.full(1 << N_FEATURES, np.nan, dtype=np.float64)
    weights: list[np.ndarray] = [np.empty(0, dtype=np.float64) for _ in range(1 << N_FEATURES)]
    for mask in range(1, 1 << N_FEATURES):
        indices = [i for i in range(N_FEATURES) if mask & (1 << i)]
        gate = controls.fit_simplex(prediction[:, indices], y)
        losses[mask] = float(np.mean((y - prediction[:, indices] @ gate) ** 2))
        weights[mask] = gate
    return losses, weights


def exact_shapley(losses: np.ndarray, empty_loss: float) -> np.ndarray:
    utility = np.empty_like(losses)
    utility[0] = 0.0
    utility[1:] = empty_loss - losses[1:]
    result = np.zeros(N_FEATURES, dtype=np.float64)
    denominator = math.factorial(N_FEATURES)
    for feature in range(N_FEATURES):
        bit = 1 << feature
        for subset in range(1 << N_FEATURES):
            if subset & bit:
                continue
            size = subset.bit_count()
            coefficient = (
                math.factorial(size)
                * math.factorial(N_FEATURES - size - 1)
                / denominator
            )
            result[feature] += coefficient * (utility[subset | bit] - utility[subset])
    return result


def top_mask(score: np.ndarray, k: int, higher_is_better: bool = True) -> int:
    order = np.argsort(-score if higher_is_better else score, kind="stable")
    return sum(1 << int(index) for index in order[:k])


def best_mask(losses: np.ndarray, k: int) -> int:
    candidates = [mask for mask in range(1, 1 << N_FEATURES) if mask.bit_count() == k]
    return min(candidates, key=lambda mask: (float(losses[mask]), mask))


def run_fold(
    root: Path,
    cache: str,
    out: Path,
    dataset: str,
    fold: int,
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    alpha: float,
    inner_folds: int,
) -> None:
    fold_predictions = out / "folds" / f"{dataset}_fold{fold}_predictions.csv"
    fold_selection = out / "folds" / f"{dataset}_fold{fold}_selection.csv"
    if fold_predictions.exists() and fold_selection.exists():
        log(f"skip completed {dataset} outer fold {fold}")
        return

    started = time.time()
    inner_prediction, null_prediction = inner_cross_fitted_predictions(
        arrays, y, groups, outer_train, alpha, inner_folds
    )
    outer_prediction = fit_experts(arrays, y, outer_train, outer_test, alpha)
    losses, coalition_weights = coalition_game(inner_prediction, y[outer_train])
    empty_loss = float(np.mean((y[outer_train] - null_prediction) ** 2))
    shapley = exact_shapley(losses, empty_loss)
    full_gate = np.zeros(N_FEATURES, dtype=np.float64)
    full_gate[:] = coalition_weights[FULL_MASK]
    singleton_losses = np.asarray([losses[1 << i] for i in range(N_FEATURES)])

    method_masks: dict[tuple[str, int], int] = {}
    for k in range(1, N_FEATURES + 1):
        method_masks[("gate", k)] = top_mask(full_gate, k, True)
        method_masks[("shapley", k)] = top_mask(shapley, k, True)
        method_masks[("singleton", k)] = top_mask(singleton_losses, k, False)
        method_masks[("inner_search", k)] = best_mask(losses, k)

    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for method in METHODS:
        for k in range(1, N_FEATURES + 1):
            mask = method_masks[(method, k)]
            indices = [i for i in range(N_FEATURES) if mask & (1 << i)]
            gate = coalition_weights[mask]
            held_prediction = outer_prediction[:, indices] @ gate
            squared_error = (y[outer_test] - held_prediction) ** 2
            selection_rows.append(
                {
                    "dataset": dataset,
                    "outer_fold": fold,
                    "method": method,
                    "k": k,
                    "mask": mask,
                    "features": "+".join(mask_names(mask)),
                    "inner_mse": float(losses[mask]),
                    "outer_utterance_mse": float(np.mean(squared_error)),
                    "n_outer_systems": int(len(np.unique(groups[outer_test]))),
                    "n_outer_items": int(len(outer_test)),
                    "weights_json": json.dumps(dict(zip(mask_names(mask), gate.tolist()))),
                }
            )
            prediction_rows.extend(
                {
                    "dataset": dataset,
                    "outer_fold": fold,
                    "method": method,
                    "k": k,
                    "mask": mask,
                    "features": "+".join(mask_names(mask)),
                    "item_index": int(item),
                    "system_id": str(groups[item]),
                    "mos": float(y[item]),
                    "prediction": float(prediction),
                    "squared_error": float(error),
                }
                for item, prediction, error in zip(outer_test, held_prediction, squared_error)
            )

    atomic_csv(fold_selection, pd.DataFrame(selection_rows))
    atomic_csv(fold_predictions, pd.DataFrame(prediction_rows))
    atomic_json(
        out / "folds" / f"{dataset}_fold{fold}_diagnostics.json",
        {
            "dataset": dataset,
            "outer_fold": fold,
            "n_outer_train_items": int(len(outer_train)),
            "n_outer_test_items": int(len(outer_test)),
            "n_outer_train_systems": int(len(np.unique(groups[outer_train]))),
            "n_outer_test_systems": int(len(np.unique(groups[outer_test]))),
            "empty_inner_mse": empty_loss,
            "full_inner_mse": float(losses[FULL_MASK]),
            "shapley_efficiency_error": float(abs(shapley.sum() - (empty_loss - losses[FULL_MASK]))),
            "full_gate": dict(zip(FEATURES, full_gate.tolist())),
            "shapley": dict(zip(FEATURES, shapley.tolist())),
            "elapsed_seconds": time.time() - started,
        },
    )
    log(f"complete {dataset} outer fold {fold} in {time.time() - started:.1f}s")


def solver_parity(
    root: Path,
    cache: str,
    primary: Path,
    datasets: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        arrays = load_arrays(root, cache, dataset, "train")
        test_arrays = load_arrays(root, cache, dataset, "test")
        y = np.asarray(base.load_y(root, cache, dataset, "train"), dtype=np.float64)
        fit_indices = np.arange(len(y))
        # Keep this independent from fit_experts because train/test live in separate mappings.
        with np.load(primary / f"{dataset}.npz", allow_pickle=False) as saved:
            reference = np.asarray(saved["test"], dtype=np.float64)
        for feature_index, feature in enumerate(FEATURES):
            fit_x, test_x = transform_feature(feature, arrays[feature], test_arrays[feature])
            model = Ridge(alpha=10.0, fit_intercept=True, solver="lsqr", tol=1e-10, max_iter=10000)
            model.fit(fit_x, y[fit_indices])
            prediction = model.predict(test_x)
            difference = prediction - reference[:, feature_index]
            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "max_absolute_prediction_difference": float(np.max(np.abs(difference))),
                    "prediction_difference_rmse": float(np.sqrt(np.mean(difference**2))),
                }
            )
    return pd.DataFrame(rows)


def aggregate(out: Path, datasets: list[str], bootstrap: int, seed: int) -> None:
    prediction_files = sorted((out / "folds").glob("*_predictions.csv"))
    selection_files = sorted((out / "folds").glob("*_selection.csv"))
    predictions = pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True)
    selections = pd.concat([pd.read_csv(path) for path in selection_files], ignore_index=True)
    predictions = predictions[predictions.dataset.isin(datasets)].copy()
    selections = selections[selections.dataset.isin(datasets)].copy()
    atomic_csv(out / "outer_fold_selection.csv", selections)
    atomic_csv(out / "outer_system_predictions.csv", predictions)

    metric_rows: list[dict[str, object]] = []
    system_rows: list[pd.DataFrame] = []
    for (dataset, method, k), group in predictions.groupby(["dataset", "method", "k"]):
        system = group.groupby("system_id", as_index=False).agg(
            system_mse=("squared_error", "mean"),
            mos=("mos", "mean"),
            prediction=("prediction", "mean"),
            n_items=("item_index", "size"),
        )
        system.insert(0, "k", int(k))
        system.insert(0, "method", method)
        system.insert(0, "dataset", dataset)
        system_rows.append(system)
        rho = spearmanr(system.mos, system.prediction).statistic if len(system) > 1 else np.nan
        metric_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "k": int(k),
                "n_systems": len(system),
                "n_items": len(group),
                "utterance_mse": float(group.squared_error.mean()),
                "equal_system_mse": float(system.system_mse.mean()),
                "system_srcc": float(rho),
            }
        )
    system_errors = pd.concat(system_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    atomic_csv(out / "outer_system_errors.csv", system_errors)
    atomic_csv(out / "selection_metrics.csv", metrics)

    rng = np.random.default_rng(seed)
    bootstrap_rows: list[dict[str, object]] = []
    primary_budgets = [1, 2, 3, 5]
    for dataset in datasets:
        for k in primary_budgets:
            pivot = system_errors[
                (system_errors.dataset == dataset) & (system_errors.k == k)
            ].pivot(index="system_id", columns="method", values="system_mse")
            n_systems = len(pivot)
            draw = rng.integers(0, n_systems, size=(bootstrap, n_systems))
            for method in ("gate", "shapley", "singleton"):
                delta_system = pivot[method].to_numpy() - pivot["inner_search"].to_numpy()
                samples = delta_system[draw].mean(axis=1)
                point = float(delta_system.mean())
                low, high = np.quantile(samples, (0.025, 0.975))
                bootstrap_rows.append(
                    {
                        "dataset": dataset,
                        "k": k,
                        "method": method,
                        "reference": "inner_search",
                        "delta_equal_system_mse": point,
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                        "probability_delta_lt_0": float(np.mean(samples < 0)),
                        "n_systems": n_systems,
                        "n_bootstrap": bootstrap,
                    }
                )
    atomic_csv(out / "paired_system_bootstrap.csv", pd.DataFrame(bootstrap_rows))

    primary = metrics[metrics.k.isin(primary_budgets)].copy()
    full = metrics[metrics.k == N_FEATURES][["dataset", "method", "equal_system_mse"]]
    full = full.groupby("dataset", as_index=False).equal_system_mse.mean().rename(
        columns={"equal_system_mse": "full_equal_system_mse"}
    )
    primary = primary.merge(full, on="dataset", how="left")
    primary["mse_ratio_to_full"] = primary.equal_system_mse / primary.full_equal_system_mse
    primary["rank_within_dataset_budget"] = primary.groupby(["dataset", "k"])[
        "equal_system_mse"
    ].rank(method="min")
    summary = primary.groupby("method", as_index=False).agg(
        mean_mse_ratio_to_full=("mse_ratio_to_full", "mean"),
        median_mse_ratio_to_full=("mse_ratio_to_full", "median"),
        mean_rank=("rank_within_dataset_budget", "mean"),
        wins=("rank_within_dataset_budget", lambda values: int(np.sum(values == 1))),
        comparisons=("rank_within_dataset_budget", "size"),
    )
    summary["nonempty_coalitions_scored"] = summary.method.map(
        {"gate": 1, "singleton": N_FEATURES, "shapley": FULL_MASK, "inner_search": FULL_MASK}
    )
    atomic_csv(out / "selection_method_summary.csv", summary)

    feature_rows: list[dict[str, object]] = []
    for (dataset, method, k), group in selections[
        selections.k.isin(primary_budgets)
    ].groupby(["dataset", "method", "k"]):
        for feature in FEATURES:
            count = int(group.features.fillna("").str.split("+").apply(lambda x: feature in x).sum())
            feature_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "k": int(k),
                    "feature": feature,
                    "selected_outer_folds": count,
                    "n_outer_folds": len(group),
                    "selection_frequency": count / len(group),
                }
            )
    atomic_csv(out / "selection_feature_frequency.csv", pd.DataFrame(feature_rows))

    # Macro inference preserves paired methods/budgets while resampling systems
    # independently inside each corpus. Deltas are normalized by each sampled
    # corpus's full-model MSE before averaging over corpora and primary budgets.
    macro_rng = np.random.default_rng(seed + 1)
    macro_samples = {method: np.empty(bootstrap, dtype=np.float64) for method in METHODS[:-1]}
    point_deltas: dict[str, list[float]] = {method: [] for method in METHODS[:-1]}
    for dataset in datasets:
        dataset_errors = system_errors[system_errors.dataset == dataset]
        full_point = float(
            dataset_errors[
                (dataset_errors.k == N_FEATURES) & (dataset_errors.method == "inner_search")
            ].system_mse.mean()
        )
        for k in primary_budgets:
            pivot = dataset_errors[dataset_errors.k == k].pivot(
                index="system_id", columns="method", values="system_mse"
            )
            for method in METHODS[:-1]:
                point_deltas[method].append(
                    float((pivot[method].mean() - pivot["inner_search"].mean()) / full_point)
                )
    bootstrap_arrays: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        dataset_errors = system_errors[system_errors.dataset == dataset]
        full = dataset_errors[
            (dataset_errors.k == N_FEATURES) & (dataset_errors.method == "inner_search")
        ].set_index("system_id").system_mse.sort_index()
        systems = full.index.to_numpy()
        deltas: dict[tuple[int, str], np.ndarray] = {}
        for k in primary_budgets:
            pivot = dataset_errors[dataset_errors.k == k].pivot(
                index="system_id", columns="method", values="system_mse"
            ).loc[systems]
            for method in METHODS[:-1]:
                deltas[(k, method)] = (
                    pivot[method].to_numpy() - pivot["inner_search"].to_numpy()
                )
        bootstrap_arrays[dataset] = {
            "full": full.to_numpy(),
            "deltas": deltas,
        }
    for replicate in range(bootstrap):
        replicate_deltas: dict[str, list[float]] = {method: [] for method in METHODS[:-1]}
        for dataset in datasets:
            cached = bootstrap_arrays[dataset]
            full_values = cached["full"]
            sampled_indices = macro_rng.integers(0, len(full_values), size=len(full_values))
            sampled_full = max(float(full_values[sampled_indices].mean()), 1e-12)
            delta_values = cached["deltas"]
            for k in primary_budgets:
                for method in METHODS[:-1]:
                    delta = float(delta_values[(k, method)][sampled_indices].mean())
                    replicate_deltas[method].append(delta / sampled_full)
        for method in METHODS[:-1]:
            macro_samples[method][replicate] = float(np.mean(replicate_deltas[method]))
    macro_rows: list[dict[str, object]] = []
    for method in METHODS[:-1]:
        low, high = np.quantile(macro_samples[method], (0.025, 0.975))
        macro_rows.append(
            {
                "method": method,
                "reference": "inner_search",
                "mean_normalized_mse_delta": float(np.mean(point_deltas[method])),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "probability_delta_lt_0": float(np.mean(macro_samples[method] < 0)),
                "n_corpora": len(datasets),
                "n_budgets": len(primary_budgets),
                "n_bootstrap": bootstrap,
            }
        )
    for method in ("gate", "shapley"):
        contrast_samples = macro_samples[method] - macro_samples["singleton"]
        point = float(np.mean(point_deltas[method]) - np.mean(point_deltas["singleton"]))
        low, high = np.quantile(contrast_samples, (0.025, 0.975))
        macro_rows.append(
            {
                "method": method,
                "reference": "singleton",
                "mean_normalized_mse_delta": point,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "probability_delta_lt_0": float(np.mean(contrast_samples < 0)),
                "n_corpora": len(datasets),
                "n_budgets": len(primary_budgets),
                "n_bootstrap": bootstrap,
            }
        )
    # Report a two-sided bootstrap sign p-value and Holm adjustment across
    # the five displayed macro contrasts.  The adjustment is intentionally
    # conservative: these contrasts are an exploratory family, so a CI that
    # excludes zero before adjustment must not be described as confirmatory
    # evidence after searching several rules.
    raw_p = []
    for row in macro_rows:
        p = 2.0 * min(row["probability_delta_lt_0"],
                      1.0 - row["probability_delta_lt_0"])
        row["p_two_sided_bootstrap"] = float(min(1.0, p))
        raw_p.append(float(min(1.0, p)))
    order = np.argsort(np.asarray(raw_p))
    adjusted = np.ones(len(raw_p), dtype=np.float64)
    running = 0.0
    m = len(raw_p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * raw_p[idx]))
        adjusted[idx] = running
    for row, p_holm in zip(macro_rows, adjusted):
        row["p_holm_all_five"] = float(p_holm)
    atomic_csv(out / "macro_method_bootstrap.csv", pd.DataFrame(macro_rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--cache", default="results/explainability_ridge_full/cache")
    parser.add_argument("--primary", type=Path, default=Path("results/paper_submission_final_cpu/primary_oof"))
    parser.add_argument("--output", type=Path, default=Path("results/paper_system_disjoint_selection"))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--skip-parity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    primary = (root / args.primary).resolve() if not args.primary.is_absolute() else args.primary
    out.mkdir(parents=True, exist_ok=True)
    (out / "folds").mkdir(parents=True, exist_ok=True)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    atomic_json(
        out / "protocol.json",
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "datasets": datasets,
            "features": FEATURES,
            "ridge_alpha": args.alpha,
            "ridge_solver": "sklearn Ridge lsqr tol=1e-10 max_iter=10000",
            "outer_split": f"GroupKFold({args.outer_folds}) by training system_id",
            "inner_split": f"GroupKFold({args.inner_folds}) within outer-training systems",
            "preprocessing": "fold-only standardization; eGeMAPS fold-only 0.1/99.9% winsorization",
            "selection_methods": METHODS,
            "budgets": list(range(1, N_FEATURES + 1)),
            "primary_budgets": [1, 2, 3, 5],
            "test_use": "each outer-held system is used once, after selection and gate calibration",
            "primary_metric": "equal-system MSE",
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
    )

    if not args.skip_parity:
        parity_path = out / "solver_parity.csv"
        if not parity_path.exists():
            log("auditing CPU LSQR parity against saved final expert predictions")
            parity = solver_parity(root, args.cache, primary, datasets)
            atomic_csv(parity_path, parity)
            max_difference = float(parity.max_absolute_prediction_difference.max())
            max_rmse = float(parity.prediction_difference_rmse.max())
            # LSQR and the saved direct float32 solve need not be bit-identical.
            # Reject only a material disagreement; both diagnostics remain public.
            if max_difference > 5e-3 or max_rmse > 1e-3:
                raise RuntimeError(
                    f"Material LSQR parity error: max={max_difference:.6g}, rmse={max_rmse:.6g}"
                )

    completed = 0
    total = 0
    for dataset in datasets:
        arrays = load_arrays(root, args.cache, dataset, "train")
        y = np.asarray(base.load_y(root, args.cache, dataset, "train"), dtype=np.float64)
        groups = base.load_metadata(root, dataset, "train").system_id.astype(str).to_numpy()
        n_splits = min(args.outer_folds, len(np.unique(groups)))
        outer_split = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
        total += len(outer_split)
        for fold, (outer_train, outer_test) in enumerate(outer_split):
            task = f"{dataset}_fold{fold}"
            try:
                run_fold(
                    root, args.cache, out, dataset, fold, arrays, y, groups,
                    np.asarray(outer_train), np.asarray(outer_test), args.alpha, args.inner_folds,
                )
                completed += 1
                atomic_json(
                    out / "progress.json",
                    {"status": "running", "completed_folds": completed, "known_total_folds": total},
                )
            except Exception as error:
                record_error(out, task, error)
                atomic_json(
                    out / "progress.json",
                    {"status": "failed", "task": task, "completed_folds": completed, "error": str(error)},
                )
                raise

    aggregate(out, datasets, args.bootstrap, args.seed)
    max_efficiency_error = max(
        json.loads(path.read_text(encoding="utf-8"))["shapley_efficiency_error"]
        for path in (out / "folds").glob("*_diagnostics.json")
    )
    predictions = pd.read_csv(out / "outer_system_predictions.csv")
    expected_items = {
        dataset: len(base.load_y(root, args.cache, dataset, "train")) for dataset in datasets
    }
    coverage = predictions.groupby(["dataset", "method", "k"]).agg(
        rows=("item_index", "size"), unique_items=("item_index", "nunique")
    ).reset_index()
    complete_coverage = all(
        int(row.rows) == expected_items[str(row.dataset)]
        and int(row.unique_items) == expected_items[str(row.dataset)]
        for row in coverage.itertuples(index=False)
    )
    one_outer_fold_per_system = bool(
        predictions.groupby(["dataset", "system_id"]).outer_fold.nunique().eq(1).all()
    )
    parity = pd.read_csv(out / "solver_parity.csv")
    audit = {
        "complete_item_coverage_every_method_budget": complete_coverage,
        "one_outer_fold_per_system": one_outer_fold_per_system,
        "selection_row_count": int(len(pd.read_csv(out / "outer_fold_selection.csv"))),
        "expected_selection_row_count": int(completed * len(METHODS) * N_FEATURES),
        "max_shapley_efficiency_error": max_efficiency_error,
        "max_solver_prediction_difference": float(parity.max_absolute_prediction_difference.max()),
        "max_solver_prediction_difference_rmse": float(parity.prediction_difference_rmse.max()),
        "errors_file_exists": (out / "errors.jsonl").exists(),
    }
    if not complete_coverage or not one_outer_fold_per_system:
        raise RuntimeError(f"Outer prediction coverage audit failed: {audit}")
    if audit["selection_row_count"] != audit["expected_selection_row_count"]:
        raise RuntimeError(f"Selection row-count audit failed: {audit}")
    atomic_json(out / "audit.json", audit)
    atomic_json(
        out / "completion.json",
        {
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "datasets": datasets,
            "outer_folds": completed,
            "max_shapley_efficiency_error": max_efficiency_error,
            "errors_file_exists": (out / "errors.jsonl").exists(),
            "audit": "audit.json",
        },
    )
    atomic_json(out / "progress.json", {"status": "complete", "completed_folds": completed})
    log("all nested system-disjoint selection experiments complete")


if __name__ == "__main__":
    main()
