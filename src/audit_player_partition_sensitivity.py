#!/usr/bin/env python3
"""Audit Shapley sensitivity to the representation-player partition.

This script performs no model fitting.  It reuses the final common-alpha=10
coalition lattice and evaluates exact reduced/grouped empirical games:

* the original ten singleton players;
* RMVPE-continuous and RMVPE-quantized as one group;
* RMVPE plus eGeMAPS as one broad prosody group;
* games with RMVPE-quantized and/or eGeMAPS removed.

For non-singleton groups it also computes exact Owen values under the stated
partition.  All quantities remain conditional on the saved test-analysis game.
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
BITS = {feature: 1 << index for index, feature in enumerate(FEATURES)}


def subset_indices(n: int, excluded: int):
    others = [index for index in range(n) if index != excluded]
    for size in range(len(others) + 1):
        for subset in combinations(others, size):
            yield subset


def union_mask(partition: list[list[str]], group_indices) -> int:
    mask = 0
    for group_index in group_indices:
        for feature in partition[group_index]:
            mask |= BITS[feature]
    return mask


def exact_group_and_owen(
    values: np.ndarray,
    partition: list[list[str]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return group Shapley values and individual Owen values."""
    n_groups = len(partition)
    group_values: dict[str, float] = {}
    owen_values = {feature: 0.0 for group in partition for feature in group}

    for group_index, group in enumerate(partition):
        group_mask = sum(BITS[feature] for feature in group)
        group_label = "+".join(group)
        group_phi = 0.0

        for group_subset in subset_indices(n_groups, group_index):
            background_mask = union_mask(partition, group_subset)
            background_size = len(group_subset)
            group_weight = 1.0 / (
                n_groups * math.comb(n_groups - 1, background_size)
            )
            group_phi += group_weight * (
                values[background_mask | group_mask] - values[background_mask]
            )

            for feature in group:
                other_members = [member for member in group if member != feature]
                for within_size in range(len(other_members) + 1):
                    within_weight = 1.0 / (
                        len(group) * math.comb(len(group) - 1, within_size)
                    )
                    for within_subset in combinations(other_members, within_size):
                        within_mask = sum(BITS[member] for member in within_subset)
                        before = background_mask | within_mask
                        after = before | BITS[feature]
                        owen_values[feature] += (
                            group_weight
                            * within_weight
                            * (values[after] - values[before])
                        )

        group_values[group_label] = group_phi

    return group_values, owen_values


def make_partitions() -> dict[str, list[list[str]]]:
    singleton = [[feature] for feature in FEATURES]
    no_quant = [[feature] for feature in FEATURES if feature != "rmvpe_quant"]
    no_egemaps = [[feature] for feature in FEATURES if feature != "egemaps"]
    no_quant_egemaps = [
        [feature]
        for feature in FEATURES
        if feature not in {"rmvpe_quant", "egemaps"}
    ]
    rmvpe_group = [
        [feature]
        for feature in FEATURES
        if feature not in {"rmvpe_cont", "rmvpe_quant"}
    ] + [["rmvpe_cont", "rmvpe_quant"]]
    prosody_group = [
        [feature]
        for feature in FEATURES
        if feature not in {"rmvpe_cont", "rmvpe_quant", "egemaps"}
    ] + [["rmvpe_cont", "rmvpe_quant", "egemaps"]]
    return {
        "original_10": singleton,
        "rmvpe_grouped_9": rmvpe_group,
        "prosody_grouped_8": prosody_group,
        "remove_rmvpe_quant_9": no_quant,
        "remove_egemaps_9": no_egemaps,
        "remove_quant_and_egemaps_8": no_quant_egemaps,
    }


def safe_rho(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return float("nan")
    return float(spearmanr(left, right).statistic)


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
        "--output-dir",
        type=Path,
        default=Path("review_evidence"),
    )
    args = parser.parse_args()

    lattice = pd.read_csv(args.lattice)
    importance = pd.read_csv(args.importance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partitions = make_partitions()

    group_rows: list[dict[str, object]] = []
    owen_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for dataset in sorted(lattice["dataset"].unique()):
        dataset_lattice = lattice[lattice["dataset"] == dataset].copy()
        dataset_lattice["mask"] = dataset_lattice["mask"].astype(int)
        by_mask = dataset_lattice.set_index("mask")
        dataset_importance = importance[importance["dataset"] == dataset].set_index(
            "feature"
        )

        empty_estimates = []
        for feature in FEATURES:
            singleton_mse = float(by_mask.loc[BITS[feature], "test_mse"])
            singleton_utility = float(
                dataset_importance.loc[feature, "singleton_mse_utility"]
            )
            empty_estimates.append(singleton_mse + singleton_utility)
        if np.ptp(empty_estimates) > 1e-8:
            raise RuntimeError(
                f"Inconsistent empty-game reconstruction for {dataset}: "
                f"range={np.ptp(empty_estimates)}"
            )

        values = np.full(1024, np.nan, dtype=np.float64)
        values[0] = -float(np.mean(empty_estimates))
        values[dataset_lattice["mask"].to_numpy()] = -dataset_lattice[
            "test_mse"
        ].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete coalition game for {dataset}")

        base_phi = dataset_importance["exact_shapley_mse_utility"].to_dict()

        for scenario, partition in partitions.items():
            group_phi, owen_phi = exact_group_and_owen(values, partition)
            full_mask = union_mask(partition, range(len(partition)))
            saved_weights = json.loads(str(by_mask.loc[full_mask, "weights_json"]))

            aggregate_reference = []
            computed_group = []
            zero_positive = 0
            for group in partition:
                label = "+".join(group)
                gate_mass = float(sum(saved_weights.get(feature, 0.0) for feature in group))
                reference_phi = float(sum(base_phi[feature] for feature in group))
                phi = float(group_phi[label])
                zero_positive_flag = gate_mass < 1e-6 and phi > 0.0
                zero_positive += int(zero_positive_flag)
                aggregate_reference.append(reference_phi)
                computed_group.append(phi)
                group_rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "player": label,
                        "members": ";".join(group),
                        "n_players": len(partition),
                        "full_coalition_mask": full_mask,
                        "gate_mass": gate_mass,
                        "group_shapley_mse_utility": phi,
                        "original_shapley_sum": reference_phi,
                        "zero_gate_positive_shapley": zero_positive_flag,
                    }
                )

            common_owen = []
            common_base = []
            sign_matches = 0
            for feature, phi in owen_phi.items():
                base = float(base_phi[feature])
                sign_same = bool(np.sign(phi) == np.sign(base))
                sign_matches += int(sign_same)
                common_owen.append(float(phi))
                common_base.append(base)
                owen_rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "feature": feature,
                        "owen_mse_utility": float(phi),
                        "original_shapley_mse_utility": base,
                        "sign_same": sign_same,
                    }
                )

            efficiency_target = float(values[full_mask] - values[0])
            efficiency_error = abs(sum(group_phi.values()) - efficiency_target)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "n_players": len(partition),
                    "zero_gate_positive_shapley_count": zero_positive,
                    "group_rank_rho_vs_original_aggregated": safe_rho(
                        aggregate_reference, computed_group
                    ),
                    "owen_rank_rho_vs_original": safe_rho(common_base, common_owen),
                    "owen_sign_agreement": sign_matches / len(common_base),
                    "efficiency_error": efficiency_error,
                    "game_total_mse_utility": efficiency_target,
                }
            )

    group_frame = pd.DataFrame(group_rows)
    owen_frame = pd.DataFrame(owen_rows)
    summary_frame = pd.DataFrame(summary_rows)
    group_frame.to_csv(args.output_dir / "player_partition_group_shapley.csv", index=False)
    owen_frame.to_csv(args.output_dir / "player_partition_owen.csv", index=False)
    summary_frame.to_csv(args.output_dir / "player_partition_summary.csv", index=False)

    audit = {
        "status": "verified",
        "protocol": "common_alpha_10_final_foldwise_egemaps_winsor_0p1pct",
        "scope": "saved held-out empirical game; no model refitting",
        "datasets": sorted(lattice["dataset"].unique().tolist()),
        "scenarios": list(partitions),
        "max_efficiency_error": float(summary_frame["efficiency_error"].max()),
        "outputs": [
            "player_partition_group_shapley.csv",
            "player_partition_owen.csv",
            "player_partition_summary.csv",
        ],
    }
    (args.output_dir / "player_partition_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
