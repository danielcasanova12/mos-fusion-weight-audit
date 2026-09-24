#!/usr/bin/env python3
"""Validation-discovery / test-confirmation audit for zero-gate cases.

The official attribution lattice is fitted without using the validation or
test targets: each coalition gate is fitted on the saved system-grouped OOF
predictions.  This script applies those frozen gates to validation, discovers
the cases satisfying ``w < threshold`` and positive Shapley utility, freezes
that list, and only then evaluates the same cases on the official test split.

The test split is never used to define the case list.  Test confidence
intervals are system-cluster bootstrap intervals for the fixed-gate Shapley
contribution.  All numbers use the utterance-level MSE used by the common
alpha=10 attribution audit; the system bootstrap is reported separately as an
uncertainty summary.

Example (from the project root)::

    python paper/icassp2027/audit_discovery_confirmation.py \
        --root . --input results/paper_submission_final_cpu \
        --output paper/icassp2027/review_evidence/attribution_discovery_confirmation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_missing_experiments as base  # noqa: E402


FEATURES = list(base.FEATURES)
DATASETS = list(base.DATASETS)
N = len(FEATURES)
FULL = (1 << N) - 1
PROTOCOL = "common_alpha_10_final_fixed_oof_gates_validation_discovery_test_confirmation"
THRESHOLD = 1e-6


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def exact_shapley(losses: dict[int, float], empty_loss: float) -> np.ndarray:
    """Return MSE reduction Shapley values for a ten-player loss game."""
    utility = {0: 0.0}
    utility.update({mask: float(empty_loss - loss) for mask, loss in losses.items()})
    values = np.zeros(N, dtype=np.float64)
    denominator = math.factorial(N)
    for feature in range(N):
        bit = 1 << feature
        for mask in range(1 << N):
            if mask & bit:
                continue
            size = mask.bit_count()
            coefficient = math.factorial(size) * math.factorial(N - size - 1) / denominator
            values[feature] += coefficient * (utility[mask | bit] - utility[mask])
    return values


def load_bundle(input_dir: Path, dataset: str) -> dict[str, np.ndarray]:
    path = input_dir / "primary_oof" / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def load_fixed_predictions(bundle: dict[str, np.ndarray], lattice: pd.DataFrame,
                           split: str) -> dict[int, np.ndarray]:
    """Apply the saved OOF-fitted gate for every coalition to one split."""
    result: dict[int, np.ndarray] = {0: np.full(
        len(bundle[f"y_{split}"]), float(np.mean(bundle["y_train"])), dtype=np.float64
    )}
    for row in lattice.itertuples(index=False):
        mask = int(row.mask)
        names = base.names(mask)
        indices = [FEATURES.index(name) for name in names]
        weights = np.asarray(list(json.loads(row.weights_json).values()), dtype=np.float64)
        if len(weights) != len(indices):
            raise ValueError(f"Gate length mismatch for {row.dataset} mask {mask}")
        result[mask] = np.asarray(bundle[split][:, indices], dtype=np.float64) @ weights
    if len(result) != (1 << N):
        raise ValueError(f"Expected {1 << N} predictions, got {len(result)}")
    return result


def system_bootstrap(values: np.ndarray, groups: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    """Bootstrap the utterance-weighted mean by resampling complete systems."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    sums = np.asarray([values[idx].sum() for idx in members], dtype=np.float64)
    sizes = np.asarray([len(idx) for idx in members], dtype=np.float64)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)), size=n)
    boot = (counts @ sums) / np.maximum(counts @ sizes, 1.0)
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def item_shapley(loss_arrays: dict[int, np.ndarray], empty_loss_items: np.ndarray) -> np.ndarray:
    """Per-item Shapley MSE reductions, used for system-cluster intervals."""
    values = np.zeros((N, len(empty_loss_items)), dtype=np.float64)
    denominator = math.factorial(N)
    for feature in range(N):
        bit = 1 << feature
        for mask in range(1 << N):
            if mask & bit:
                continue
            size = mask.bit_count()
            coefficient = math.factorial(size) * math.factorial(N - size - 1) / denominator
            if mask == 0:
                without = empty_loss_items
            else:
                without = loss_arrays[mask]
            values[feature] += coefficient * (without - loss_arrays[mask | bit])
    return values


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.root).expanduser().resolve()
    input_dir = (root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lattice_all = pd.read_csv(input_dir / "converged_oof_lattice.csv")
    required = {"dataset", "mask", "weights_json", "val_mse", "test_mse"}
    missing = required.difference(lattice_all.columns)
    if missing:
        raise ValueError(f"Missing lattice columns: {sorted(missing)}")

    # Phase 1: validation-only discovery.  No test value is read in this loop.
    discovery_rows: list[dict[str, object]] = []
    all_validation_rows: list[dict[str, object]] = []
    fixed_by_dataset: dict[str, pd.DataFrame] = {}
    bundles: dict[str, dict[str, np.ndarray]] = {}
    for dataset in DATASETS:
        bundle = load_bundle(input_dir, dataset)
        bundles[dataset] = bundle
        lattice = lattice_all[lattice_all.dataset.eq(dataset)].copy()
        if len(lattice) != (1 << N) - 1 or set(lattice["mask"].astype(int)) != set(range(1, 1 << N)):
            raise ValueError(f"Incomplete lattice for {dataset}")
        if "gate" in lattice and not lattice["gate"].eq("converged_simplex").all():
            raise ValueError(f"Unexpected gate provenance for {dataset}")
        fixed_by_dataset[dataset] = lattice
        y_val = np.asarray(bundle["y_val"], dtype=np.float64)
        groups_val = base.load_metadata(root, dataset, "val").system_id.to_numpy()
        pred_val = load_fixed_predictions(bundle, lattice, "val")
        losses_val = {mask: float(np.mean((y_val - pred) ** 2)) for mask, pred in pred_val.items() if mask}
        empty_val = float(np.mean((y_val - pred_val[0]) ** 2))
        phi_val = exact_shapley(losses_val, empty_val)
        full_weights = np.asarray(list(json.loads(lattice[lattice["mask"].eq(FULL)].iloc[0].weights_json).values()), dtype=np.float64)
        if len(full_weights) != N:
            raise ValueError(f"Full gate length mismatch for {dataset}")
        # Recompute the stored val losses from the same frozen gates as a provenance check.
        max_loss_error = float(np.max([abs(losses_val[int(row.mask)] - float(row.val_mse)) for row in lattice.itertuples(index=False)]))
        if max_loss_error > 1e-10:
            raise ValueError(f"Validation lattice mismatch for {dataset}: {max_loss_error:g}")
        for index, feature in enumerate(FEATURES):
            row = {
                "dataset": dataset,
                "feature": feature,
                "gate_weight": float(full_weights[index]),
                "validation_shapley_mse_reduction": float(phi_val[index]),
                "validation_empty_mse": empty_val,
                "validation_full_mse": losses_val[FULL],
                "n_validation_items": len(y_val),
                "n_validation_systems": int(len(np.unique(groups_val.astype(str)))),
                "validation_discovery_rule": f"gate_weight < {THRESHOLD:g} and validation_shapley_mse_reduction > 0",
                "discovered_in_validation": bool(full_weights[index] < THRESHOLD and phi_val[index] > 0),
                "protocol": PROTOCOL,
            }
            all_validation_rows.append(row)
            if row["discovered_in_validation"]:
                discovery_rows.append(dict(row))

    # Freeze the validation list before any test computation.
    discovered_keys = {(str(row["dataset"]), str(row["feature"])) for row in discovery_rows}
    validation = pd.DataFrame(all_validation_rows)
    discovered = pd.DataFrame(discovery_rows)
    discovered.to_csv(output / "validation_discovered_cases.csv", index=False)
    validation.to_csv(output / "validation_all_cases.csv", index=False)

    # Phase 2: test-only confirmation of the frozen list.
    confirmation_rows: list[dict[str, object]] = []
    all_test_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        bundle = bundles[dataset]
        lattice = fixed_by_dataset[dataset]
        y_test = np.asarray(bundle["y_test"], dtype=np.float64)
        groups_test = base.load_metadata(root, dataset, "test").system_id.to_numpy()
        pred_test = load_fixed_predictions(bundle, lattice, "test")
        losses_test = {mask: float(np.mean((y_test - pred) ** 2)) for mask, pred in pred_test.items() if mask}
        empty_test = float(np.mean((y_test - pred_test[0]) ** 2))
        phi_test = exact_shapley(losses_test, empty_test)
        item_losses = {mask: (y_test - pred) ** 2 for mask, pred in pred_test.items() if mask}
        item_phi = item_shapley(item_losses, (y_test - pred_test[0]) ** 2)
        max_loss_error = float(np.max([abs(losses_test[int(row.mask)] - float(row.test_mse)) for row in lattice.itertuples(index=False)]))
        if max_loss_error > 1e-10:
            raise ValueError(f"Test lattice mismatch for {dataset}: {max_loss_error:g}")
        for index, feature in enumerate(FEATURES):
            key = (dataset, feature)
            discovered_flag = key in discovered_keys
            row = {
                "dataset": dataset,
                "feature": feature,
                "test_shapley_mse_reduction": float(phi_test[index]),
                "test_empty_mse": empty_test,
                "test_full_mse": losses_test[FULL],
                "n_test_items": len(y_test),
                "n_test_systems": int(len(np.unique(groups_test.astype(str)))),
                "validation_discovered": discovered_flag,
                "test_confirmation_rule": "validation_discovered list frozen before test evaluation",
                "protocol": PROTOCOL,
            }
            all_test_rows.append(row)
            if discovered_flag:
                estimate, low, high = system_bootstrap(
                    item_phi[index], groups_test, args.bootstrap, args.seed + 1000 * DATASETS.index(dataset) + index
                )
                row.update({
                    "test_confirmation_shapley_mse_reduction": estimate,
                    "test_ci95_low": low,
                    "test_ci95_high": high,
                    "test_ci_bootstrap_unit": "system_id",
                    "test_ci_bootstrap_n": int(args.bootstrap),
                    "positive_in_test_point_estimate": bool(phi_test[index] > 0),
                    "positive_in_test_ci": bool(low > 0),
                })
                confirmation_rows.append(row)

    pd.DataFrame(all_test_rows).to_csv(output / "test_all_cases_confirmation_audit.csv", index=False)
    confirmations = pd.DataFrame(confirmation_rows)
    confirmations.to_csv(output / "test_confirmation_of_validation_cases.csv", index=False)

    # Join the frozen list to make the principal table easy to cite in the paper.
    if len(discovered):
        principal = discovered.merge(
            confirmations,
            on=["dataset", "feature", "protocol"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_test"),
        )
    else:
        principal = discovered.copy()
    principal.to_csv(output / "discovery_confirmation_principal.csv", index=False)
    summary = {
        "status": "completed",
        "protocol": PROTOCOL,
        "input": str(input_dir),
        "output": str(output),
        "n_validation_discovered": int(len(discovered)),
        "n_test_confirmed_positive_point": int(confirmations.positive_in_test_point_estimate.sum()) if len(confirmations) else 0,
        "n_test_confirmed_ci_low_above_zero": int(confirmations.positive_in_test_ci.sum()) if len(confirmations) else 0,
        "n_test_cases_evaluated_for_confirmation": int(len(confirmations)),
        "threshold": THRESHOLD,
        "loss_metric": "utterance_level_MSE_with_training_mean_empty_baseline",
        "gate_source": "converged_oof_lattice.csv; gates fitted on system-grouped OOF predictions",
        "validation_used_for_gate_fit": False,
        "test_used_for_gate_fit": False,
        "bootstrap": int(args.bootstrap),
        "test_was_used_to_define_discovery": False,
        "validation_cases": [f"{d}/{f}" for d, f in sorted(discovered_keys)],
    }
    atomic_json(output / "discovery_confirmation_protocol.json", summary)
    (output / "README.md").write_text(
        "# Validation-discovery / test-confirmation attribution audit\n\n"
        "The frozen coalition gates were fitted only on system-grouped OOF predictions. "
        "Cases were discovered using validation Shapley utility and full-gate weight, "
        "then the list was frozen before computing any test confirmation value. "
        "The test table therefore contains no post-hoc search for new positive cases.\n\n"
        f"Discovery rule: `w < {THRESHOLD:g}` and `phi_val > 0`. "
        f"The run found {len(discovered)} validation cases; "
        f"{summary['n_test_confirmed_positive_point']} retained positive point utility on test and "
        f"{summary['n_test_confirmed_ci_low_above_zero']} had a positive 95% system-bootstrap lower bound.\n\n"
        "Files: `validation_discovered_cases.csv` (frozen list), "
        "`test_confirmation_of_validation_cases.csv` (only frozen cases with intervals), "
        "`discovery_confirmation_principal.csv` (joined paper table), and "
        "`test_all_cases_confirmation_audit.csv` (audit-only complete test values; do not use it to redefine the list).\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path, default=Path("results/paper_submission_final_cpu"))
    parser.add_argument("--output", type=Path, default=HERE / "review_evidence/attribution_discovery_confirmation")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
