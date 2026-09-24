#!/usr/bin/env python3
"""QAP inference for expert-correlation versus Shapley-interaction matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "paper_reviewer_priority_cpu" / "shapley_interactions_and_expert_correlations.csv"
OUTPUT = HERE.parent / "paper_reviewer_priority_cpu" / "interaction_qap_inference.csv"
FEATURES = [
    "whisper", "contentvec12", "wavlm", "beats", "auditory_erb",
    "speaker", "rmvpe_cont", "rmvpe_quant", "ced", "egemaps",
]
N_PERMUTATIONS = 100_000


def main() -> None:
    frame = pd.read_csv(INPUT)
    rows: list[dict[str, object]] = []
    upper = np.triu_indices(len(FEATURES), 1)
    index = {feature: i for i, feature in enumerate(FEATURES)}
    for dataset, group in frame.groupby("dataset", sort=True):
        correlation = np.zeros((len(FEATURES), len(FEATURES)), dtype=np.float64)
        redundancy = np.zeros_like(correlation)
        for row in group.itertuples(index=False):
            i, j = index[row.feature_i], index[row.feature_j]
            correlation[i, j] = correlation[j, i] = row.oof_spearman
            redundancy[i, j] = redundancy[j, i] = -row.shapley_interaction_mse_utility
        correlation_ranks = rankdata(correlation[upper])
        redundancy_ranks = rankdata(redundancy[upper])
        correlation_centered = correlation_ranks - correlation_ranks.mean()
        redundancy_centered = redundancy_ranks - redundancy_ranks.mean()
        denominator = float(
            np.linalg.norm(correlation_centered) * np.linalg.norm(redundancy_centered)
        )
        observed = float(correlation_centered @ redundancy_centered / denominator)
        redundancy_rank_matrix = np.zeros_like(redundancy)
        redundancy_rank_matrix[upper] = redundancy_ranks
        redundancy_rank_matrix[(upper[1], upper[0])] = redundancy_ranks
        rng = np.random.default_rng(20270910 + sum(map(ord, dataset)))
        exceed_two_sided = 0
        exceed_positive = 0
        for _ in range(N_PERMUTATIONS):
            permutation = rng.permutation(len(FEATURES))
            permuted = redundancy_rank_matrix[np.ix_(permutation, permutation)][upper]
            statistic = float(
                correlation_centered @ (permuted - redundancy_ranks.mean()) / denominator
            )
            exceed_two_sided += abs(statistic) >= abs(observed) - 1e-15
            exceed_positive += statistic >= observed - 1e-15
        rows.append(
            {
                "dataset": dataset,
                "n_features": len(FEATURES),
                "n_pairs": len(group),
                "n_negative_correlations": int(np.sum(correlation[upper] < 0)),
                "minimum_oof_spearman": float(correlation[upper].min()),
                "spearman_correlation_vs_negative_interaction": observed,
                "qap_p_two_sided": (exceed_two_sided + 1) / (N_PERMUTATIONS + 1),
                "qap_p_positive": (exceed_positive + 1) / (N_PERMUTATIONS + 1),
                "n_qap_permutations": N_PERMUTATIONS,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
