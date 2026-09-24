#!/usr/bin/env python3
"""Test-correlation and system-bootstrap audit for interaction redundancy.

The final local bundle does not contain the post-winsorized eGeMAPS expert
prediction vector.  The other nine cached experts reproduce the final singleton
test MSEs to numerical precision.  We therefore form an exact reduced
nine-player game that excludes eGeMAPS, using its saved final coalition gates.

For each test-system bootstrap replicate, both the test expert-correlation
matrix and all pairwise interaction indices are recomputed.  Fitted experts and
gates remain fixed, so intervals quantify test-system sampling only.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FEATURES_10 = [
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
FEATURES = FEATURES_10[:-1]
BITS_10 = {feature: 1 << index for index, feature in enumerate(FEATURES_10)}
DATASETS = ["brspeech", "bvcc", "singmos", "tmhintqi"]


def mask9_to_mask10(mask9: int) -> int:
    return sum(BITS_10[feature] for index, feature in enumerate(FEATURES) if mask9 & (1 << index))


def interaction_coefficients(n_players: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    pairs = list(combinations(range(n_players), 2))
    coefficients = np.zeros((1 << n_players, len(pairs)), dtype=np.float64)
    denominator = math.factorial(n_players - 1)
    for pair_index, (left, right) in enumerate(pairs):
        left_bit = 1 << left
        right_bit = 1 << right
        for subset in range(1 << n_players):
            if subset & (left_bit | right_bit):
                continue
            size = subset.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(n_players - size - 2)
                / denominator
            )
            coefficients[subset, pair_index] += weight
            coefficients[subset | left_bit, pair_index] -= weight
            coefficients[subset | right_bit, pair_index] -= weight
            coefficients[subset | left_bit | right_bit, pair_index] += weight
    return coefficients, pairs


def read_metadata(path: Path) -> pd.DataFrame:
    chunks = []
    columns = ["dataset", "spec", "seed", "item_index", "system_id", "mos"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=200_000):
        selected = chunk[(chunk["spec"] == "full") & (chunk["seed"] == 42)]
        if len(selected):
            chunks.append(selected)
    metadata = pd.concat(chunks, ignore_index=True)
    duplicated = metadata.duplicated(["dataset", "item_index"]).sum()
    if duplicated:
        raise RuntimeError(f"Metadata contains {duplicated} duplicate items")
    return metadata


def pair_values(matrix: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([matrix[left, right] for left, right in pairs], dtype=float)


def rank_correlation_matrix(predictions: np.ndarray) -> np.ndarray:
    result = spearmanr(predictions, axis=0).statistic
    return np.asarray(result, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lattice",
        type=Path,
        default=Path("../submission_final_cpu_review/converged_oof_lattice.csv"),
    )
    parser.add_argument(
        "--importance",
        type=Path,
        default=Path("../paper_reviewer_priority_cpu/importance_methods.csv"),
    )
    parser.add_argument(
        "--interactions",
        type=Path,
        default=Path("../paper_reviewer_priority_cpu/shapley_interactions_and_expert_correlations.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("../paper_missing_remote/item_predictions.csv"),
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("../review_v2_snapshot"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output-dir", type=Path, default=Path("review_evidence"))
    args = parser.parse_args()

    lattice = pd.read_csv(args.lattice)
    importance = pd.read_csv(args.importance)
    interactions = pd.read_csv(args.interactions)
    metadata = read_metadata(args.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coefficients, pairs = interaction_coefficients(len(FEATURES))
    rng = np.random.default_rng(args.seed)
    pair_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        bundle = np.load(args.bundle_dir / f"{dataset}_alpha10p0_folds5.npz")
        predictions = np.asarray(bundle["test"], dtype=np.float64)[:, : len(FEATURES)]
        target = np.asarray(bundle["y_test"], dtype=np.float64)
        dataset_metadata = metadata[metadata["dataset"] == dataset].sort_values("item_index")
        if not np.array_equal(dataset_metadata["item_index"].to_numpy(), np.arange(len(target))):
            raise RuntimeError(f"Item order mismatch for {dataset}")
        if not np.allclose(dataset_metadata["mos"].to_numpy(dtype=float), target, atol=1e-6):
            raise RuntimeError(f"MOS mismatch for {dataset}")

        by_mask = lattice[lattice["dataset"] == dataset].set_index("mask")
        dataset_importance = importance[importance["dataset"] == dataset].set_index("feature")
        empty_mse = np.mean(
            [
                float(by_mask.loc[BITS_10[feature], "test_mse"])
                + float(dataset_importance.loc[feature, "singleton_mse_utility"])
                for feature in FEATURES
            ]
        )

        losses = np.empty((len(target), 1 << len(FEATURES)), dtype=np.float64)
        losses[:, 0] = (target - float(np.asarray(bundle["y_train"]).mean())) ** 2
        reconstruction_errors = [abs(float(losses[:, 0].mean()) - empty_mse)]
        for mask9 in range(1, 1 << len(FEATURES)):
            mask10 = mask9_to_mask10(mask9)
            row = by_mask.loc[mask10]
            weights_json = json.loads(str(row["weights_json"]))
            weights = np.asarray([weights_json.get(feature, 0.0) for feature in FEATURES])
            prediction = predictions @ weights
            losses[:, mask9] = (target - prediction) ** 2
            reconstruction_errors.append(abs(float(losses[:, mask9].mean()) - float(row["test_mse"])))
        max_reconstruction_error = float(max(reconstruction_errors))
        if max_reconstruction_error > 2e-4:
            raise RuntimeError(
                f"Saved game reconstruction mismatch for {dataset}: {max_reconstruction_error}"
            )

        values = -losses.mean(axis=0)
        interaction_point = values @ coefficients
        test_corr_point = pair_values(rank_correlation_matrix(predictions), pairs)

        interaction_table = interactions[interactions["dataset"] == dataset].copy()
        oof_lookup = {
            tuple(sorted((row.feature_i, row.feature_j))): float(row.oof_spearman)
            for row in interaction_table.itertuples()
        }
        oof_corr = np.asarray(
            [oof_lookup[tuple(sorted((FEATURES[left], FEATURES[right])))] for left, right in pairs]
        )

        groups, inverse = np.unique(dataset_metadata["system_id"].astype(str), return_inverse=True)
        n_groups = len(groups)
        group_sizes = np.bincount(inverse)
        group_loss_sums = np.zeros((n_groups, losses.shape[1]), dtype=np.float64)
        np.add.at(group_loss_sums, inverse, losses)
        bootstrap_counts = rng.multinomial(
            n_groups,
            np.full(n_groups, 1.0 / n_groups),
            size=args.n_bootstrap,
        )
        bootstrap_total_items = bootstrap_counts @ group_sizes
        bootstrap_values = -(
            (bootstrap_counts @ group_loss_sums)
            / bootstrap_total_items[:, None]
        )
        bootstrap_interactions = bootstrap_values @ coefficients

        rho_oof_fixed = np.empty(args.n_bootstrap, dtype=float)
        rho_test_joint = np.empty(args.n_bootstrap, dtype=float)
        for replicate in range(args.n_bootstrap):
            repeat_per_item = bootstrap_counts[replicate, inverse]
            indices = np.repeat(np.arange(len(target)), repeat_per_item)
            test_corr = pair_values(rank_correlation_matrix(predictions[indices]), pairs)
            negative_interaction = -bootstrap_interactions[replicate]
            rho_oof_fixed[replicate] = spearmanr(oof_corr, negative_interaction).statistic
            rho_test_joint[replicate] = spearmanr(test_corr, negative_interaction).statistic

        negative_interaction_point = -interaction_point
        rho_oof_point = float(spearmanr(oof_corr, negative_interaction_point).statistic)
        rho_test_point = float(spearmanr(test_corr_point, negative_interaction_point).statistic)
        oof_interval = np.quantile(rho_oof_fixed, [0.025, 0.975])
        test_interval = np.quantile(rho_test_joint, [0.025, 0.975])

        summary_rows.append(
            {
                "dataset": dataset,
                "protocol": "reduced_9_player_no_egemaps_common_alpha10",
                "n_pairs": len(pairs),
                "n_test_systems": n_groups,
                "rho_oof_correlation_vs_negative_interaction": rho_oof_point,
                "rho_oof_ci95_test_system_sampling_interaction_only_low": float(oof_interval[0]),
                "rho_oof_ci95_test_system_sampling_interaction_only_high": float(oof_interval[1]),
                "rho_test_correlation_vs_negative_interaction": rho_test_point,
                "rho_test_joint_system_bootstrap_ci95_low": float(test_interval[0]),
                "rho_test_joint_system_bootstrap_ci95_high": float(test_interval[1]),
                "n_bootstrap": args.n_bootstrap,
                "max_saved_mse_reconstruction_error": max_reconstruction_error,
                "uncertainty_scope": "test-system sampling; fitted experts and gates fixed",
            }
        )

        for pair_index, (left, right) in enumerate(pairs):
            pair_rows.append(
                {
                    "dataset": dataset,
                    "feature_i": FEATURES[left],
                    "feature_j": FEATURES[right],
                    "oof_spearman": float(oof_corr[pair_index]),
                    "test_spearman": float(test_corr_point[pair_index]),
                    "reduced_game_interaction": float(interaction_point[pair_index]),
                    "negative_interaction_redundancy": float(negative_interaction_point[pair_index]),
                }
            )

    pair_frame = pd.DataFrame(pair_rows)
    summary_frame = pd.DataFrame(summary_rows)
    pair_frame.to_csv(args.output_dir / "test_correlation_interaction_pairs.csv", index=False)
    summary_frame.to_csv(args.output_dir / "test_correlation_interaction_bootstrap.csv", index=False)
    audit = {
        "status": "verified",
        "protocol": "reduced_9_player_no_egemaps_common_alpha10",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "scope": "test-system sampling with fitted experts and gates fixed",
        "reason_for_reduced_game": "post-winsorized final eGeMAPS prediction vector absent locally",
        "max_saved_mse_reconstruction_error": float(
            summary_frame["max_saved_mse_reconstruction_error"].max()
        ),
    }
    (args.output_dir / "test_correlation_interaction_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
