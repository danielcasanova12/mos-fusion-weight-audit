#!/usr/bin/env python3
"""CPU experiments requested by the ICASSP MOS reviewer.

The runner is deliberately split into two stages:

``fast``
    Uses the final frozen OOF expert predictions.  It computes a system
    bootstrap of the gate conditional on those experts, Rashomon weight ranges,
    exact pairwise Shapley interactions, alternative importance definitions,
    system-level paired inference, and fusion baselines.

``alpha``
    Refits the representation experts from the cached embeddings for
    alpha in {1e-2,...,1e7}, plus an intercept-only limit, selects alpha per representation using grouped OOF
    error, and recomputes the complete 1,023-coalition game for alpha in
    {1,10,100} and for the per-representation selected bundle.

Every expensive unit is written atomically and can be resumed.  Progress,
errors, and a manifest distinguishing conditional from refitted analyses are
persisted in the output directory.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_final_icasp_controls as fc  # noqa: E402
import run_icasp_followups_v2 as v2  # noqa: E402
import run_missing_experiments as base  # noqa: E402


FEATURES = base.FEATURES
DATASETS = base.DATASETS
BITS = base.BITS
N_FEATURES = len(FEATURES)
FULL_MASK = (1 << N_FEATURES) - 1
DEFAULT_ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
SENSITIVITY_ALPHAS = [1.0, 10.0, 100.0]
CONFIRMATORY = {
    "full_minus_basic": (fc.MASKS["full"], fc.MASKS["basic"]),
    "basic_ced_minus_basic": (fc.MASKS["basic_plus_ced"], fc.MASKS["basic"]),
    "core_ced_minus_core": (fc.MASKS["core_plus_ced"], fc.MASKS["core"]),
    "core_beats_minus_core": (fc.MASKS["core_plus_beats"], fc.MASKS["core"]),
}


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


def state(out: Path, task: str, status: str, **extra: object) -> None:
    atomic_json(
        out / "state" / f"{task}.json",
        {"task": task, "status": status, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra},
    )


def record_error(out: Path, task: str, error: BaseException) -> None:
    path = out / "errors.jsonl"
    payload = {
        "task": task,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    state(out, task, "failed", error_type=type(error).__name__, message=str(error))


def names(mask: int) -> list[str]:
    return [feature for i, feature in enumerate(FEATURES) if mask & (1 << i)]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_datasets(value: str) -> list[str]:
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return datasets


def load_bundle(primary: Path, dataset: str) -> dict[str, np.ndarray]:
    path = primary / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return {
        "mse": float(np.mean((y - prediction) ** 2)),
        "mae": float(np.mean(np.abs(y - prediction))),
        "pearson": float(pearsonr(y, prediction).statistic),
        "spearman": float(spearmanr(y, prediction).statistic),
    }


def fit_gate(bundle: dict[str, np.ndarray], mask: int, train_indices: np.ndarray | None = None) -> np.ndarray:
    indices = [FEATURES.index(feature) for feature in names(mask)]
    predictions = bundle["oof"][:, indices]
    target = bundle["y_train"]
    if train_indices is not None:
        predictions = predictions[train_indices]
        target = target[train_indices]
    return fc.fit_simplex(predictions, target)


def predict_mask(bundle: dict[str, np.ndarray], mask: int, split: str, weights: np.ndarray | None = None) -> np.ndarray:
    indices = [FEATURES.index(feature) for feature in names(mask)]
    if weights is None:
        weights = fit_gate(bundle, mask)
    return np.asarray(bundle[split][:, indices] @ weights, dtype=np.float64)


def stable_seed(dataset: str, offset: int = 0) -> int:
    return 20260910 + 1009 * DATASETS.index(dataset) + offset


def task_conditional_gate_bootstrap(args: argparse.Namespace, out: Path, datasets: list[str]) -> None:
    task = "01_conditional_gate_bootstrap"
    destination = out / "conditional_gate_bootstrap_replicates.csv"
    if destination.exists() and not args.force:
        log(f"skip {task}: output exists")
        return
    state(out, task, "running", n_bootstrap=args.gate_bootstrap)
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        bundle = load_bundle(args.primary, dataset)
        groups = base.load_metadata(args.root, dataset, "train")["system_id"].astype(str).to_numpy()
        if len(groups) != len(bundle["y_train"]):
            raise ValueError(f"Training group length mismatch for {dataset}")
        unique = np.unique(groups)
        rng = np.random.default_rng(stable_seed(dataset, 1))
        full_indices = [FEATURES.index(feature) for feature in FEATURES]
        for replicate in range(args.gate_bootstrap):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            sampled_indices = np.concatenate([np.flatnonzero(groups == system) for system in sampled])
            weights = fit_gate(bundle, FULL_MASK, sampled_indices)
            prediction_sample = bundle["oof"][sampled_indices][:, full_indices] @ weights
            prediction_oof = bundle["oof"][:, full_indices] @ weights
            prediction_test = bundle["test"][:, full_indices] @ weights
            active = [FEATURES[i] for i, value in enumerate(weights) if value >= 1e-6]
            common = {
                "dataset": dataset,
                "bootstrap": replicate,
                "n_system_draws": len(sampled),
                "n_unique_systems_drawn": len(np.unique(sampled)),
                "n_active": len(active),
                "active_set": "+".join(active),
                "sample_mse": float(np.mean((bundle["y_train"][sampled_indices] - prediction_sample) ** 2)),
                "original_oof_mse": float(np.mean((bundle["y_train"] - prediction_oof) ** 2)),
                "test_mse": float(np.mean((bundle["y_test"] - prediction_test) ** 2)),
            }
            rows.extend({**common, "feature": feature, "weight": float(weights[i])} for i, feature in enumerate(FEATURES))
            if (replicate + 1) % 20 == 0 or replicate + 1 == args.gate_bootstrap:
                log(f"conditional gate bootstrap {dataset}: {replicate + 1}/{args.gate_bootstrap}")
                state(out, task, "running", dataset=dataset, completed=replicate + 1, total=args.gate_bootstrap)
    frame = pd.DataFrame(rows)
    atomic_csv(destination, frame)
    summary = (
        frame.groupby(["dataset", "feature"], as_index=False)
        .agg(
            median_weight=("weight", "median"),
            ci95_low=("weight", lambda x: float(np.quantile(x, 0.025))),
            ci95_high=("weight", lambda x: float(np.quantile(x, 0.975))),
            probability_zero=("weight", lambda x: float(np.mean(np.asarray(x) < 1e-6))),
            active_frequency=("weight", lambda x: float(np.mean(np.asarray(x) >= 1e-6))),
        )
    )
    atomic_csv(out / "conditional_gate_bootstrap_summary.csv", summary)
    active = (
        frame.drop_duplicates(["dataset", "bootstrap"])
        .groupby("dataset", as_index=False)
        .agg(
            median_active=("n_active", "median"),
            unique_active_sets=("active_set", "nunique"),
            oof_mse_sd=("original_oof_mse", "std"),
            test_mse_sd=("test_mse", "std"),
            test_mse_min=("test_mse", "min"),
            test_mse_max=("test_mse", "max"),
        )
    )
    atomic_csv(out / "conditional_gate_bootstrap_model_summary.csv", active)
    state(
        out,
        task,
        "complete",
        rows=len(frame),
        limitation="System bootstrap refits the gate conditional on frozen final OOF experts; it is not a full expert refit.",
    )


def loss_and_gradient(predictions: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    residual = predictions @ weights - target
    return float(np.mean(residual**2)), (2.0 / len(target)) * predictions.T @ residual


def rashomon_extreme(
    predictions: np.ndarray,
    target: np.ndarray,
    optimum: np.ndarray,
    threshold: float,
    feature_index: int,
    maximize: bool,
) -> tuple[np.ndarray, object]:
    sign = -1.0 if maximize else 1.0

    def objective(weights: np.ndarray) -> float:
        return sign * float(weights[feature_index])

    def objective_jac(weights: np.ndarray) -> np.ndarray:
        gradient = np.zeros_like(weights)
        gradient[feature_index] = sign
        return gradient

    def feasible(weights: np.ndarray) -> float:
        loss, _ = loss_and_gradient(predictions, target, weights)
        return threshold - loss

    def feasible_jac(weights: np.ndarray) -> np.ndarray:
        _, gradient = loss_and_gradient(predictions, target, weights)
        return -gradient

    result = minimize(
        objective,
        optimum,
        jac=objective_jac,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * predictions.shape[1],
        constraints=[
            {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones_like(w)},
            {"type": "ineq", "fun": feasible, "jac": feasible_jac},
        ],
        options={"maxiter": 3000, "ftol": 1e-12, "disp": False},
    )
    weights = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    weights /= weights.sum()
    return weights, result


def task_rashomon(args: argparse.Namespace, out: Path, datasets: list[str]) -> None:
    task = "02_rashomon_ranges"
    destination = out / "rashomon_weight_ranges.csv"
    if destination.exists() and not args.force:
        log(f"skip {task}: output exists")
        return
    state(out, task, "running", tolerances=args.rashomon_tolerances)
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        bundle = load_bundle(args.primary, dataset)
        predictions = np.asarray(bundle["oof"], dtype=np.float64)
        target = np.asarray(bundle["y_train"], dtype=np.float64)
        optimum = fc.fit_simplex(predictions, target)
        optimum_loss, _ = loss_and_gradient(predictions, target, optimum)
        for tolerance in args.rashomon_tolerances:
            threshold = optimum_loss * (1.0 + tolerance)
            for feature_index, feature in enumerate(FEATURES):
                extremes: dict[str, float | bool | str] = {}
                for label, maximize in (("min", False), ("max", True)):
                    weights, result = rashomon_extreme(
                        predictions, target, optimum, threshold, feature_index, maximize
                    )
                    achieved_loss, _ = loss_and_gradient(predictions, target, weights)
                    extremes[f"weight_{label}"] = float(weights[feature_index])
                    extremes[f"loss_{label}"] = achieved_loss
                    extremes[f"success_{label}"] = bool(result.success and achieved_loss <= threshold + 1e-9)
                    extremes[f"message_{label}"] = str(result.message)
                rows.append(
                    {
                        "dataset": dataset,
                        "feature": feature,
                        "relative_mse_tolerance": tolerance,
                        "optimal_weight": float(optimum[feature_index]),
                        "optimal_oof_mse": optimum_loss,
                        "mse_threshold": threshold,
                        **extremes,
                    }
                )
            log(f"Rashomon {dataset}: tolerance={100*tolerance:.1f}%")
            state(out, task, "running", dataset=dataset, tolerance=tolerance)
    frame = pd.DataFrame(rows)
    frame["range_width"] = frame["weight_max"] - frame["weight_min"]
    atomic_csv(destination, frame)
    state(out, task, "complete", rows=len(frame), all_success=bool(frame[["success_min", "success_max"]].all().all()))


def value_lookup(lattice: pd.DataFrame, bundle: dict[str, np.ndarray], split: str = "test_mse") -> dict[int, float]:
    values = {int(row.mask): -float(getattr(row, split)) for row in lattice.itertuples(index=False)}
    empty_prediction = np.full(len(bundle["y_test"]), float(np.mean(bundle["y_train"])))
    values[0] = -float(np.mean((bundle["y_test"] - empty_prediction) ** 2))
    return values


def exact_shapley(values: dict[int, float]) -> np.ndarray:
    result = np.zeros(N_FEATURES, dtype=np.float64)
    denominator = math.factorial(N_FEATURES)
    for i in range(N_FEATURES):
        bit = 1 << i
        for subset in range(1 << N_FEATURES):
            if subset & bit:
                continue
            size = subset.bit_count()
            coefficient = math.factorial(size) * math.factorial(N_FEATURES - size - 1) / denominator
            result[i] += coefficient * (values[subset | bit] - values[subset])
    return result


def exact_interaction(values: dict[int, float], i: int, j: int) -> float:
    bit_i, bit_j = 1 << i, 1 << j
    denominator = math.factorial(N_FEATURES - 1)
    total = 0.0
    for subset in range(1 << N_FEATURES):
        if subset & (bit_i | bit_j):
            continue
        size = subset.bit_count()
        coefficient = math.factorial(size) * math.factorial(N_FEATURES - size - 2) / denominator
        second_difference = (
            values[subset | bit_i | bit_j]
            - values[subset | bit_i]
            - values[subset | bit_j]
            + values[subset]
        )
        total += coefficient * second_difference
    return float(total)


def task_interactions(args: argparse.Namespace, out: Path, datasets: list[str]) -> None:
    task = "03_shapley_interactions"
    destination = out / "shapley_interactions_and_expert_correlations.csv"
    if destination.exists() and not args.force:
        log(f"skip {task}: output exists")
        return
    state(out, task, "running")
    lattice_all = pd.read_csv(args.lattice)
    rows: list[dict[str, object]] = []
    conditioning: list[dict[str, object]] = []
    for dataset in datasets:
        bundle = load_bundle(args.primary, dataset)
        lattice = lattice_all[lattice_all["dataset"] == dataset]
        values = value_lookup(lattice, bundle)
        standardized = (bundle["oof"] - bundle["oof"].mean(0)) / np.maximum(bundle["oof"].std(0), 1e-12)
        correlation = np.corrcoef(standardized, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(correlation)
        conditioning.append(
            {
                "dataset": dataset,
                "correlation_condition_number": float(np.linalg.cond(correlation)),
                "minimum_eigenvalue": float(eigenvalues.min()),
                "maximum_eigenvalue": float(eigenvalues.max()),
            }
        )
        for i, j in itertools.combinations(range(N_FEATURES), 2):
            x, y = bundle["oof"][:, i], bundle["oof"][:, j]
            rows.append(
                {
                    "dataset": dataset,
                    "feature_i": FEATURES[i],
                    "feature_j": FEATURES[j],
                    "oof_pearson": float(pearsonr(x, y).statistic),
                    "oof_spearman": float(spearmanr(x, y).statistic),
                    "shapley_interaction_mse_utility": exact_interaction(values, i, j),
                }
            )
        log(f"interactions complete: {dataset}")
    frame = pd.DataFrame(rows)
    atomic_csv(destination, frame)
    atomic_csv(out / "expert_prediction_conditioning.csv", pd.DataFrame(conditioning))
    relation: list[dict[str, object]] = []
    for dataset, group in frame.groupby("dataset"):
        relation.append(
            {
                "dataset": dataset,
                "spearman_abs_correlation_vs_interaction": float(
                    spearmanr(np.abs(group["oof_spearman"]), group["shapley_interaction_mse_utility"]).statistic
                ),
                "spearman_abs_correlation_vs_negative_interaction": float(
                    spearmanr(np.abs(group["oof_spearman"]), -group["shapley_interaction_mse_utility"]).statistic
                ),
                "n_pairs": len(group),
            }
        )
    atomic_csv(out / "correlation_interaction_relation.csv", pd.DataFrame(relation))
    state(out, task, "complete", rows=len(frame))


def task_importance_comparison(args: argparse.Namespace, out: Path, datasets: list[str]) -> None:
    task = "04_importance_comparison"
    destination = out / "importance_methods.csv"
    if destination.exists() and not args.force:
        log(f"skip {task}: output exists")
        return
    state(out, task, "running")
    lattice_all = pd.read_csv(args.lattice)
    rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for dataset in datasets:
        bundle = load_bundle(args.primary, dataset)
        lattice = lattice_all[lattice_all["dataset"] == dataset]
        values = value_lookup(lattice, bundle)
        shapley = exact_shapley(values)
        full_row = lattice[lattice["mask"] == FULL_MASK].iloc[0]
        full_weights = json.loads(full_row["weights_json"])
        dataset_rows: list[dict[str, object]] = []
        for i, feature in enumerate(FEATURES):
            row = {
                "dataset": dataset,
                "feature": feature,
                "gate_weight": float(full_weights.get(feature, 0.0)),
                "singleton_mse_utility": float(values[1 << i] - values[0]),
                "leave_one_out_mse_utility": float(values[FULL_MASK] - values[FULL_MASK ^ (1 << i)]),
                "exact_shapley_mse_utility": float(shapley[i]),
            }
            dataset_rows.append(row)
        frame = pd.DataFrame(dataset_rows)
        methods = ["gate_weight", "singleton_mse_utility", "leave_one_out_mse_utility", "exact_shapley_mse_utility"]
        for method in methods:
            frame[f"rank_{method}"] = frame[method].rank(ascending=False, method="average")
            top = frame.nlargest(3, method)
            top_rows.append({"dataset": dataset, "method": method, "top_3": "+".join(top["feature"])})
        for first, second in itertools.combinations(methods, 2):
            rank_rows.append(
                {
                    "dataset": dataset,
                    "method_a": first,
                    "method_b": second,
                    "spearman_rank_correlation": float(spearmanr(frame[first], frame[second]).statistic),
                }
            )
        rows.extend(frame.to_dict("records"))
        log(f"importance comparison complete: {dataset}")
    result = pd.DataFrame(rows)
    atomic_csv(destination, result)
    atomic_csv(out / "importance_rank_correlations.csv", pd.DataFrame(rank_rows))
    atomic_csv(out / "importance_top3.csv", pd.DataFrame(top_rows))
    disagreement = result[
        (result["leave_one_out_mse_utility"] <= 1e-6)
        & (result["exact_shapley_mse_utility"] > 0)
    ]
    atomic_csv(out / "loo_zero_shapley_positive.csv", disagreement)
    state(out, task, "complete", rows=len(result), loo_zero_shapley_positive=len(disagreement))


def cluster_bootstrap_interval(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_p(values: np.ndarray, rng: np.random.Generator, n_permutations: int) -> tuple[float, str, int]:
    observed = abs(float(np.mean(values)))
    n = len(values)
    if n <= 20:
        signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=n)), dtype=np.float64)
        statistics = np.abs((signs * values).mean(axis=1))
        return float(np.mean(statistics >= observed - 1e-15)), "exact", len(statistics)
    batch = 10_000
    exceed = 0
    completed = 0
    while completed < n_permutations:
        size = min(batch, n_permutations - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, n))
        statistics = np.abs((signs * values).mean(axis=1))
        exceed += int(np.sum(statistics >= observed - 1e-15))
        completed += size
    return float((exceed + 1) / (n_permutations + 1)), "monte_carlo", n_permutations


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm family-wise adjusted p-values, returned in the original order."""
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def task_system_inference(args: argparse.Namespace, out: Path, datasets: list[str]) -> None:
    task = "05_system_level_inference"
    destination = out / "system_level_contrasts.csv"
    if destination.exists() and not args.force:
        log(f"skip {task}: output exists")
        return
    state(out, task, "running", permutations=args.permutations, bootstrap=args.system_bootstrap)
    system_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for dataset in datasets:
        bundle = load_bundle(args.primary, dataset)
        metadata = base.load_metadata(args.root, dataset, "test")
        groups = metadata["system_id"].astype(str).to_numpy()
        if len(groups) != len(bundle["y_test"]):
            raise ValueError(f"Test group length mismatch for {dataset}")
        masks = sorted({mask for pair in CONFIRMATORY.values() for mask in pair})
        predictions = {mask: predict_mask(bundle, mask, "test") for mask in masks}
        per_system: dict[tuple[str, int], float] = {}
        for system in np.unique(groups):
            selected = groups == system
            y_system = bundle["y_test"][selected]
            for mask in masks:
                prediction = predictions[mask][selected]
                mse = float(np.mean((y_system - prediction) ** 2))
                per_system[(system, mask)] = mse
                system_rows.append(
                    {"dataset": dataset, "system_id": system, "mask": mask, "features": "+".join(names(mask)), "n_items": int(selected.sum()), "mse": mse}
                )
        rng = np.random.default_rng(stable_seed(dataset, 2))
        systems = np.unique(groups)
        for comparison, (candidate, reference) in CONFIRMATORY.items():
            # Legacy names identify candidate/reference order. The numeric
            # effect is reference MSE minus candidate MSE (positive = gain).
            differences = np.asarray([per_system[(system, reference)] - per_system[(system, candidate)] for system in systems])
            low, high = cluster_bootstrap_interval(differences, rng, args.system_bootstrap)
            p_value, method, n_used = sign_flip_p(differences, rng, args.permutations)
            contrast_rows.append(
                {
                    "dataset": dataset,
                    "comparison": comparison,
                    "n_systems": len(systems),
                    "mean_system_mse_improvement": float(differences.mean()),
                    "effect_definition": "mean_system(MSE_reference - MSE_candidate)",
                    "median_system_mse_improvement": float(np.median(differences)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "sign_flip_p_two_sided": p_value,
                    "sign_flip_method": method,
                    "n_sign_flips": n_used,
                    "exploratory": dataset == "brspeech",
                }
            )
        for mask in masks:
            system_table = pd.DataFrame(
                {
                    "system": systems,
                    "mos": [float(bundle["y_test"][groups == system].mean()) for system in systems],
                    "prediction": [float(predictions[mask][groups == system].mean()) for system in systems],
                }
            )
            metric_rows.append(
                {
                    "dataset": dataset,
                    "mask": mask,
                    "features": "+".join(names(mask)),
                    "n_systems": len(systems),
                    "system_mse_equal_weight": float(np.mean([per_system[(system, mask)] for system in systems])),
                    "system_level_srcc": float(spearmanr(system_table["mos"], system_table["prediction"]).statistic),
                    "utterance_mse": metrics(bundle["y_test"], predictions[mask])["mse"],
                    "utterance_srcc": metrics(bundle["y_test"], predictions[mask])["spearman"],
                }
            )
        log(f"system inference complete: {dataset}")
    contrast_frame = pd.DataFrame(contrast_rows)
    contrast_frame["p_holm_16"] = holm_adjust(
        contrast_frame["sign_flip_p_two_sided"].to_numpy()
    )
    atomic_csv(out / "per_system_mse.csv", pd.DataFrame(system_rows))
    atomic_csv(destination, contrast_frame)
    atomic_csv(out / "system_level_metrics.csv", pd.DataFrame(metric_rows))
    state(out, task, "complete", contrasts=len(contrast_rows), per_system_rows=len(system_rows))


def task_fusion_baselines(args: argparse.Namespace, out: Path, datasets: list[str]) -> None:
    task = "06_fusion_baselines"
    destination = out / "fusion_baselines.csv"
    if destination.exists() and not args.force:
        log(f"skip {task}: output exists")
        return
    state(out, task, "running")
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        bundle = load_bundle(args.primary, dataset)
        train_groups = base.load_metadata(args.root, dataset, "train")["system_id"].astype(str).to_numpy()
        splits = list(GroupKFold(n_splits=min(5, len(np.unique(train_groups)))).split(bundle["oof"], groups=train_groups))
        singleton_mse = np.mean((bundle["oof"] - bundle["y_train"][:, None]) ** 2, axis=0)
        best = int(np.argmin(singleton_mse))
        candidates: list[tuple[str, np.ndarray, float, dict[str, float | str]]] = []
        one_hot = np.zeros(N_FEATURES); one_hot[best] = 1.0
        candidates.append(("best_singleton", one_hot, 0.0, {"selected_feature": FEATURES[best]}))
        candidates.append(("uniform_mean", np.full(N_FEATURES, 1.0 / N_FEATURES), 0.0, {}))
        candidates.append(("simplex_nonnegative", fc.fit_simplex(bundle["oof"], bundle["y_train"]), 0.0, {}))

        alpha_grid = np.logspace(-4, 4, 17)
        cv_scores = []
        for alpha in alpha_grid:
            fold_losses = []
            for fit_indices, hold_indices in splits:
                model = Ridge(alpha=float(alpha), fit_intercept=True)
                model.fit(bundle["oof"][fit_indices], bundle["y_train"][fit_indices])
                prediction = model.predict(bundle["oof"][hold_indices])
                fold_losses.append(float(np.mean((bundle["y_train"][hold_indices] - prediction) ** 2)))
            cv_scores.append(float(np.mean(fold_losses)))
        selected_alpha = float(alpha_grid[int(np.argmin(cv_scores))])
        ridge = Ridge(alpha=selected_alpha, fit_intercept=True).fit(bundle["oof"], bundle["y_train"])
        candidates.append(("ridge_stacking", np.asarray(ridge.coef_, dtype=float), float(ridge.intercept_), {"selected_alpha": selected_alpha}))
        nnls = LinearRegression(positive=True, fit_intercept=True).fit(bundle["oof"], bundle["y_train"])
        candidates.append(("nnls_with_intercept", np.asarray(nnls.coef_, dtype=float), float(nnls.intercept_), {}))

        for method, weights, intercept, extra in candidates:
            prediction = bundle["test"] @ weights + intercept
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "intercept": intercept,
                    "weights_json": json.dumps(dict(zip(FEATURES, weights.tolist()))),
                    **extra,
                    **{f"test_{key}": value for key, value in metrics(bundle["y_test"], prediction).items()},
                }
            )
        log(f"fusion baselines complete: {dataset}")
    atomic_csv(destination, pd.DataFrame(rows))
    state(out, task, "complete", rows=len(rows))


def bootstrap_refit_bundle(
    args: argparse.Namespace,
    dataset: str,
    train_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    groups: np.ndarray,
    sampled_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Refit fold preprocessing and all experts on a cluster-bootstrap sample."""
    sampled_groups = groups[sampled_indices]
    sampled_y = np.asarray(y_train[sampled_indices], dtype=np.float32)
    sampled_train = {
        feature: np.asarray(train_arrays[feature][sampled_indices], dtype=np.float32)
        for feature in FEATURES
    }
    unique_groups = np.unique(sampled_groups)
    if len(unique_groups) < 2:
        raise ValueError(f"Bootstrap draw for {dataset} has fewer than two unique systems")
    splits = GroupKFold(n_splits=min(5, len(unique_groups))).split(
        np.zeros(len(sampled_y)), groups=sampled_groups
    )
    oof = np.full((len(sampled_y), N_FEATURES), np.nan, dtype=np.float32)
    device = base.device_arg("cpu")
    for fit_indices, hold_indices in splits:
        fit, hold, _ = fc.transformed_arrays(
            {feature: sampled_train[feature][fit_indices] for feature in FEATURES},
            {feature: sampled_train[feature][hold_indices] for feature in FEATURES},
            "egemaps",
            (0.001, 0.999),
        )
        oof[hold_indices] = v2.solve_np(
            args.root, fit, hold, sampled_y[fit_indices], device, args.bootstrap_alpha
        )
    if not np.isfinite(oof).all():
        raise RuntimeError(f"Non-finite refitted OOF predictions for {dataset}")
    full_train, transformed_test, _ = fc.transformed_arrays(
        sampled_train, test_arrays, "egemaps", (0.001, 0.999)
    )
    test_predictions = v2.solve_np(
        args.root, full_train, transformed_test, sampled_y, device, args.bootstrap_alpha
    )
    return {
        "oof": oof,
        "test": test_predictions,
        "y_train": sampled_y,
        "y_test": np.asarray(y_test, dtype=np.float32),
    }


def task_full_train_bootstrap(args: argparse.Namespace, out: Path, datasets: list[str]) -> None:
    task = "09_full_train_system_bootstrap"
    replicate_root = out / "full_train_bootstrap_replicates"
    state(
        out,
        task,
        "running",
        n_bootstrap=args.train_bootstrap,
        shapley_replicates=args.shapley_bootstrap,
        alpha=args.bootstrap_alpha,
    )
    masks = {
        "full": fc.MASKS["full"],
        "core": fc.MASKS["core"],
        "core_plus_ced": fc.MASKS["core_plus_ced"],
        "core_plus_beats": fc.MASKS["core_plus_beats"],
        "basic": fc.MASKS["basic"],
    }
    failures: list[dict[str, object]] = []
    for dataset in datasets:
        train_arrays = {feature: base.load_x(args.root, args.cache, dataset, "train", feature) for feature in FEATURES}
        test_arrays = {feature: base.load_x(args.root, args.cache, dataset, "test", feature) for feature in FEATURES}
        y_train = base.load_y(args.root, args.cache, dataset, "train")
        y_test = base.load_y(args.root, args.cache, dataset, "test")
        groups = base.load_metadata(args.root, dataset, "train")["system_id"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        dataset_dir = replicate_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for replicate in range(args.train_bootstrap):
            destination = dataset_dir / f"replicate_{replicate:03d}.json"
            if destination.exists() and not args.force:
                continue
            try:
                rng = np.random.default_rng(stable_seed(dataset, 9) + 104729 * replicate)
                for _ in range(100):
                    sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                    if len(np.unique(sampled)) >= 2:
                        break
                else:
                    raise RuntimeError("Could not draw at least two unique systems")
                sampled_indices = np.concatenate(
                    [np.flatnonzero(groups == system) for system in sampled]
                )
                bundle = bootstrap_refit_bundle(
                    args,
                    dataset,
                    train_arrays,
                    test_arrays,
                    y_train,
                    y_test,
                    groups,
                    sampled_indices,
                )
                result: dict[str, object] = {
                    "dataset": dataset,
                    "bootstrap": replicate,
                    "n_system_draws": len(sampled),
                    "n_unique_systems_drawn": len(np.unique(sampled)),
                    "alpha": args.bootstrap_alpha,
                    "specs": {},
                }
                for spec, mask in masks.items():
                    weights = fit_gate(bundle, mask)
                    prediction = predict_mask(bundle, mask, "test", weights)
                    result["specs"][spec] = {
                        "mask": mask,
                        "weights": dict(zip(names(mask), weights.tolist())),
                        **metrics(bundle["y_test"], prediction),
                    }
                if replicate < args.shapley_bootstrap:
                    _, attribution = coalition_game(
                        bundle, dataset, f"train_bootstrap_{replicate:03d}"
                    )
                    result["shapley"] = dict(
                        zip(attribution["feature"], attribution["shapley_mse_utility"])
                    )
                atomic_json(destination, result)
            except BaseException as error:
                failure = {
                    "dataset": dataset,
                    "bootstrap": replicate,
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                with (out / "full_train_bootstrap_failures.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            if (replicate + 1) % 5 == 0 or replicate + 1 == args.train_bootstrap:
                completed = len(list(dataset_dir.glob("replicate_*.json")))
                log(f"full train bootstrap {dataset}: {completed}/{args.train_bootstrap}")
                state(
                    out,
                    task,
                    "running",
                    dataset=dataset,
                    completed=completed,
                    total=args.train_bootstrap,
                    failures=len(failures),
                )

    weight_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    shapley_rows: list[dict[str, object]] = []
    for path in sorted(replicate_root.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        common = {
            "dataset": payload["dataset"],
            "bootstrap": payload["bootstrap"],
            "n_system_draws": payload["n_system_draws"],
            "n_unique_systems_drawn": payload["n_unique_systems_drawn"],
            "alpha": payload["alpha"],
        }
        for spec, values in payload["specs"].items():
            metric_rows.append(
                {
                    **common,
                    "spec": spec,
                    **{key: values[key] for key in ("mse", "mae", "pearson", "spearman")},
                }
            )
            for feature, weight in values["weights"].items():
                weight_rows.append({**common, "spec": spec, "feature": feature, "weight": weight})
        for feature, utility in payload.get("shapley", {}).items():
            shapley_rows.append({**common, "feature": feature, "shapley_mse_utility": utility})
    weights = pd.DataFrame(weight_rows)
    metric_frame = pd.DataFrame(metric_rows)
    shapley_frame = pd.DataFrame(shapley_rows)
    atomic_csv(out / "full_train_bootstrap_weights.csv", weights)
    atomic_csv(out / "full_train_bootstrap_metrics.csv", metric_frame)
    atomic_csv(out / "full_train_bootstrap_shapley.csv", shapley_frame)
    if not weights.empty:
        summary = (
            weights.groupby(["dataset", "spec", "feature"], as_index=False)
            .agg(
                median_weight=("weight", "median"),
                ci95_low=("weight", lambda x: float(np.quantile(x, 0.025))),
                ci95_high=("weight", lambda x: float(np.quantile(x, 0.975))),
                probability_zero=("weight", lambda x: float(np.mean(np.asarray(x) < 1e-6))),
            )
        )
        atomic_csv(out / "full_train_bootstrap_weight_summary.csv", summary)
    expected = len(datasets) * args.train_bootstrap
    completed = len(list(replicate_root.glob("*/*.json")))
    status = "complete" if completed == expected and not failures else "complete_with_failures"
    state(
        out,
        task,
        status,
        expected=expected,
        completed=completed,
        failures=len(failures),
        shapley_rows=len(shapley_frame),
    )
    if completed != expected or failures:
        raise RuntimeError(
            f"Full train bootstrap incomplete: completed={completed}/{expected}, failures={len(failures)}"
        )


def alpha_token(alpha: float) -> str:
    return f"{alpha:.8g}".replace(".", "p").replace("-", "m").replace("+", "p")


def alpha_bundle_path(out: Path, dataset: str, alpha: float) -> Path:
    return out / "alpha_bundles" / f"{dataset}_alpha_{alpha_token(alpha)}.npz"


def task_alpha_bundles(args: argparse.Namespace, out: Path, datasets: list[str]) -> None:
    task = "07_alpha_bundles"
    state(out, task, "running", alphas=args.alphas)
    device = base.device_arg("cpu")
    for dataset in datasets:
        for alpha in args.alphas:
            destination = alpha_bundle_path(out, dataset, alpha)
            if destination.exists() and not args.force:
                log(f"reuse alpha bundle: {dataset} alpha={alpha:g}")
                continue
            if math.isclose(alpha, 10.0) and (args.primary / f"{dataset}.npz").exists():
                bundle = load_bundle(args.primary, dataset)
            else:
                bundle = fc.robust_oof(args.root, args.cache, dataset, alpha, device, (0.001, 0.999), 5)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, **bundle)
            os.replace(temporary, destination)
            log(f"alpha bundle complete: {dataset} alpha={alpha:g}")
            state(out, task, "running", dataset=dataset, alpha=alpha)
    expected = len(datasets) * len(args.alphas)
    present = sum(alpha_bundle_path(out, dataset, alpha).exists() for dataset in datasets for alpha in args.alphas)
    state(out, task, "complete", expected=expected, present=present)


def load_alpha_bundle(out: Path, dataset: str, alpha: float) -> dict[str, np.ndarray]:
    path = alpha_bundle_path(out, dataset, alpha)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def selected_alpha_bundle(
    root: Path, out: Path, dataset: str, alphas: list[float]
) -> tuple[dict[str, np.ndarray], list[float], list[float]]:
    bundles = [load_alpha_bundle(out, dataset, alpha) for alpha in alphas]
    # Explicit alpha -> infinity candidate.  Its OOF values use the mean of the
    # corresponding training fold, rather than the global mean, so the null
    # expert is evaluated under the same grouped cross-fitting discipline.
    reference = bundles[0]
    groups = base.load_metadata(root, dataset, "train")["system_id"].astype(str).to_numpy()
    splits = GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(
        np.zeros(len(groups)), groups=groups
    )
    null_oof = np.empty(len(reference["y_train"]), dtype=np.float32)
    for fit_indices, hold_indices in splits:
        null_oof[hold_indices] = float(np.mean(reference["y_train"][fit_indices]))
    full_mean = float(np.mean(reference["y_train"]))
    null_bundle = {
        "oof": np.tile(null_oof[:, None], (1, N_FEATURES)),
        "val": np.full((len(reference["y_val"]), N_FEATURES), full_mean, dtype=np.float32),
        "test": np.full((len(reference["y_test"]), N_FEATURES), full_mean, dtype=np.float32),
        "y_train": reference["y_train"],
        "y_val": reference["y_val"],
        "y_test": reference["y_test"],
    }
    candidate_bundles = bundles + [null_bundle]
    candidate_alphas = list(alphas) + [float("inf")]
    oof_losses = np.asarray(
        [
            np.mean((bundle["oof"] - bundle["y_train"][:, None]) ** 2, axis=0)
            for bundle in candidate_bundles
        ]
    )
    selected_indices = np.argmin(oof_losses, axis=0)
    selected_alphas = [float(candidate_alphas[index]) for index in selected_indices]
    result: dict[str, np.ndarray] = {}
    for split in ("oof", "val", "test"):
        result[split] = np.column_stack(
            [candidate_bundles[selected_indices[i]][split][:, i] for i in range(N_FEATURES)]
        )
    for target in ("y_train", "y_val", "y_test"):
        result[target] = bundles[0][target]
    selected_losses = [float(oof_losses[selected_indices[i], i]) for i in range(N_FEATURES)]
    return result, selected_alphas, selected_losses


def coalition_game(bundle: dict[str, np.ndarray], dataset: str, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    values: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    empty_prediction = np.full(len(bundle["y_test"]), float(np.mean(bundle["y_train"])))
    values[0] = -float(np.mean((bundle["y_test"] - empty_prediction) ** 2))
    for mask in range(1, 1 << N_FEATURES):
        selected = [FEATURES.index(feature) for feature in names(mask)]
        weights = fc.fit_simplex(bundle["oof"][:, selected], bundle["y_train"])
        prediction = bundle["test"][:, selected] @ weights
        test_mse = float(np.mean((bundle["y_test"] - prediction) ** 2))
        values[mask] = -test_mse
        if mask == FULL_MASK:
            full_weights = dict(zip(names(mask), weights.tolist()))
        rows.append({"dataset": dataset, "protocol": protocol, "mask": mask, "test_mse": test_mse})
    shapley = exact_shapley(values)
    attribution = pd.DataFrame(
        {
            "dataset": dataset,
            "protocol": protocol,
            "feature": FEATURES,
            "shapley_mse_utility": shapley,
            "full_gate_weight": [float(full_weights[feature]) for feature in FEATURES],
        }
    )
    attribution["shapley_rank"] = attribution["shapley_mse_utility"].rank(ascending=False, method="average")
    attribution["gate_rank"] = attribution["full_gate_weight"].rank(ascending=False, method="average")
    return pd.DataFrame(rows), attribution


def task_alpha_analysis(args: argparse.Namespace, out: Path, datasets: list[str]) -> None:
    task = "08_alpha_analysis"
    destination = out / "alpha_shapley_and_gate.csv"
    if destination.exists() and not args.force:
        log(f"skip {task}: output exists")
        return
    state(out, task, "running")
    attribution_frames: list[pd.DataFrame] = []
    lattice_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    for dataset in datasets:
        protocols: list[tuple[str, dict[str, np.ndarray]]] = []
        for alpha in SENSITIVITY_ALPHAS:
            if alpha not in args.alphas:
                raise ValueError(f"Sensitivity alpha {alpha} missing from --alphas")
            protocols.append((f"common_alpha_{alpha:g}", load_alpha_bundle(out, dataset, alpha)))
        selected, selected_alphas, selected_losses = selected_alpha_bundle(
            args.root, out, dataset, args.alphas
        )
        protocols.append(("per_representation_grouped_oof_selected", selected))
        selection_rows.extend(
            {
                "dataset": dataset,
                "feature": feature,
                "selected_alpha": selected_alphas[i],
                "selected_oof_mse": selected_losses[i],
            }
            for i, feature in enumerate(FEATURES)
        )
        for protocol, bundle in protocols:
            lattice, attribution = coalition_game(bundle, dataset, protocol)
            lattice_frames.append(lattice)
            attribution_frames.append(attribution)
            log(f"alpha coalition game complete: {dataset} {protocol}")
            state(out, task, "running", dataset=dataset, protocol=protocol)
    attributions = pd.concat(attribution_frames, ignore_index=True)
    lattices = pd.concat(lattice_frames, ignore_index=True)
    atomic_csv(destination, attributions)
    atomic_csv(out / "alpha_coalition_lattices.csv", lattices)
    atomic_csv(out / "per_representation_selected_alpha.csv", pd.DataFrame(selection_rows))

    correlations: list[dict[str, object]] = []
    counts: list[dict[str, object]] = []
    for dataset, group in attributions.groupby("dataset"):
        protocols = list(group["protocol"].unique())
        for first, second in itertools.combinations(protocols, 2):
            a = group[group["protocol"] == first].set_index("feature")
            b = group[group["protocol"] == second].set_index("feature")
            correlations.append(
                {
                    "dataset": dataset,
                    "protocol_a": first,
                    "protocol_b": second,
                    "shapley_rank_spearman": float(spearmanr(a["shapley_mse_utility"], b["shapley_mse_utility"]).statistic),
                    "gate_rank_spearman": float(spearmanr(a["full_gate_weight"], b["full_gate_weight"]).statistic),
                    "top3_overlap": len(set(a.nlargest(3, "shapley_mse_utility").index) & set(b.nlargest(3, "shapley_mse_utility").index)),
                }
            )
        for protocol, part in group.groupby("protocol"):
            counts.append(
                {
                    "dataset": dataset,
                    "protocol": protocol,
                    "zero_gate_positive_shapley": int(np.sum((part["full_gate_weight"] < 1e-6) & (part["shapley_mse_utility"] > 0))),
                    "top3_shapley": "+".join(part.nlargest(3, "shapley_mse_utility")["feature"]),
                    "gate_shapley_spearman": float(spearmanr(part["full_gate_weight"], part["shapley_mse_utility"]).statistic),
                }
            )
    atomic_csv(out / "alpha_rank_stability.csv", pd.DataFrame(correlations))
    atomic_csv(out / "alpha_conclusion_stability.csv", pd.DataFrame(counts))
    state(out, task, "complete", attribution_rows=len(attributions), lattice_rows=len(lattices))


def write_manifest(args: argparse.Namespace, out: Path) -> None:
    atomic_json(
        out / "manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "device": "cpu",
            "features": FEATURES,
            "datasets": parse_datasets(args.datasets),
            "primary_bundles": str(args.primary),
            "final_lattice": str(args.lattice),
            "conditional_analysis": [
                "gate system bootstrap reuses the frozen final OOF expert predictions",
                "Rashomon ranges use the frozen final OOF expert predictions",
                "fusion baselines use the frozen final OOF expert predictions",
            ],
            "refitted_analysis": [
                "alpha bundles refit preprocessing and ridge experts from cached embeddings",
                "alpha sensitivity recomputes all 1,023 coalitions per corpus and protocol",
            ],
            "random_seed_base": 20260910,
            "gate_bootstrap": args.gate_bootstrap,
            "full_train_system_bootstrap": args.train_bootstrap,
            "full_train_shapley_bootstrap": args.shapley_bootstrap,
            "full_train_bootstrap_alpha": args.bootstrap_alpha,
            "system_bootstrap": args.system_bootstrap,
            "sign_flip_permutations": args.permutations,
            "rashomon_relative_mse_tolerances": args.rashomon_tolerances,
            "alpha_grid": args.alphas,
            "alpha_null_candidate": "group-cross-fitted intercept-only limit (alpha -> infinity)",
        },
    )


def run_task(out: Path, name: str, function, *arguments) -> bool:
    try:
        function(*arguments)
        return True
    except BaseException as error:  # persist failures and continue the queue
        log(f"FAILED {name}: {type(error).__name__}: {error}")
        record_error(out, name, error)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fast", "alpha", "train-bootstrap", "all"])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--cache", default="results/explainability_ridge_full/cache")
    parser.add_argument("--primary", type=Path, default=Path("results/paper_submission_final_cpu/primary_oof"))
    parser.add_argument("--lattice", type=Path, default=Path("results/paper_submission_final_cpu/converged_oof_lattice.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/paper_reviewer_priority_cpu"))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--gate-bootstrap", type=int, default=200)
    parser.add_argument("--train-bootstrap", type=int, default=100)
    parser.add_argument("--shapley-bootstrap", type=int, default=30)
    parser.add_argument("--bootstrap-alpha", type=float, default=10.0)
    parser.add_argument("--system-bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--rashomon-tolerances", default="0.001,0.005,0.01")
    parser.add_argument("--alphas", default=",".join(f"{alpha:g}" for alpha in DEFAULT_ALPHAS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    args.primary = (args.root / args.primary).resolve() if not args.primary.is_absolute() else args.primary.resolve()
    args.lattice = (args.root / args.lattice).resolve() if not args.lattice.is_absolute() else args.lattice.resolve()
    args.output = (args.root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    args.rashomon_tolerances = parse_float_list(args.rashomon_tolerances)
    args.alphas = parse_float_list(args.alphas)
    datasets = parse_datasets(args.datasets)
    args.output.mkdir(parents=True, exist_ok=True)
    write_manifest(args, args.output)
    state(args.output, "queue", "running", stage=args.stage)

    successes: list[bool] = []
    if args.stage in ("fast", "all"):
        tasks = [
            ("01_conditional_gate_bootstrap", task_conditional_gate_bootstrap),
            ("02_rashomon_ranges", task_rashomon),
            ("03_shapley_interactions", task_interactions),
            ("04_importance_comparison", task_importance_comparison),
            ("05_system_level_inference", task_system_inference),
            ("06_fusion_baselines", task_fusion_baselines),
        ]
        for name, function in tasks:
            successes.append(run_task(args.output, name, function, args, args.output, datasets))
    if args.stage in ("train-bootstrap", "all"):
        successes.append(
            run_task(
                args.output,
                "09_full_train_system_bootstrap",
                task_full_train_bootstrap,
                args,
                args.output,
                datasets,
            )
        )
    if args.stage in ("alpha", "all"):
        successes.append(run_task(args.output, "07_alpha_bundles", task_alpha_bundles, args, args.output, datasets))
        if successes[-1]:
            successes.append(run_task(args.output, "08_alpha_analysis", task_alpha_analysis, args, args.output, datasets))

    status = "complete" if all(successes) else "complete_with_errors"
    state(args.output, "queue", status, stage=args.stage, tasks=len(successes), successful=sum(successes))
    log(f"queue finished: status={status}, successful={sum(successes)}/{len(successes)}")
    return 0 if all(successes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
