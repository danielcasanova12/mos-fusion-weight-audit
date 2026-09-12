#!/usr/bin/env python3
"""Assemble canonical public result files from the archived run directories.

This maintainer script is not needed by artifact users.  It records the exact
mapping used to package the release and refuses to proceed if a source is
missing.  No private audio, ratings, embeddings, or absolute paths are copied.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
RESULTS = ROOT / "results"
REVIEW = WORKSPACE / "paper" / "icassp2027" / "review_evidence"
CPU = WORKSPACE / "paper" / "paper_reviewer_priority_cpu"
FINAL = WORKSPACE / "paper" / "submission_final_cpu_review"

COPIES = {
    REVIEW / "attribution_40cells_alpha10.csv": RESULTS / "attribution_40cells.csv",
    FINAL / "full_gate_shapley_40cells.csv": RESULTS / "test_shapley_system_bootstrap.csv",
    CPU / "system_level_contrasts.csv": RESULTS / "fixed_coalition_contrasts.csv",
    CPU / "shapley_interactions_and_expert_correlations.csv": RESULTS / "interaction_values.csv",
    CPU / "rashomon_weight_ranges.csv": RESULTS / "rashomon_intervals.csv",
    CPU / "rashomon_kkt_audit.csv": RESULTS / "rashomon_kkt_audit.csv",
    REVIEW / "train_shapley_50reps.csv": RESULTS / "training_refits.csv",
    CPU / "full_train_bootstrap_weight_summary.csv": RESULTS / "training_gate_refit_summary.csv",
    REVIEW / "vsys_bootstrap_intervals.csv": RESULTS / "bootstrap_intervals.csv",
    REVIEW / "system_disjoint_selection" / "outer_fold_selection.csv": RESULTS / "nested_selections.csv",
    REVIEW / "system_disjoint_selection" / "selection_metrics.csv": RESULTS / "nested_selection_metrics.csv",
    REVIEW / "test_correlation_interaction_bootstrap.csv": RESULTS / "interaction_bootstrap.csv",
}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for source, target in COPIES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, target)

    attribution = pd.read_csv(REVIEW / "attribution_40cells_alpha10.csv")
    refits = pd.read_csv(REVIEW / "train_shapley_50reps.csv")
    grouped = refits.groupby(["dataset", "feature"])["shapley_mse_utility"]
    refit_stats = grouped.agg(
        refit50_shapley_n="size",
        refit50_shapley_median="median",
    ).reset_index()
    quantiles = grouped.quantile([0.025, 0.975]).unstack().reset_index()
    quantiles.columns = ["dataset", "feature", "refit50_shapley_p025", "refit50_shapley_p975"]
    positive = grouped.apply(lambda values: int((values > 0).sum())).rename(
        "refit50_shapley_positive_count"
    ).reset_index()
    all_cells = attribution.merge(refit_stats, on=["dataset", "feature"], validate="one_to_one")
    all_cells = all_cells.merge(quantiles, on=["dataset", "feature"], validate="one_to_one")
    all_cells = all_cells.merge(positive, on=["dataset", "feature"], validate="one_to_one")
    all_cells["refit50_shapley_positive_frequency"] = (
        all_cells["refit50_shapley_positive_count"] / all_cells["refit50_shapley_n"]
    )
    all_cells.to_csv(RESULTS / "all_40_cells.csv", index=False)
    all_cells.loc[all_cells["gate_zero_positive_shapley"].astype(bool)].to_csv(
        RESULTS / "discordant_cases.csv", index=False
    )
    all_cells[[
        "dataset", "feature", "protocol", "exact_shapley_mse_utility",
        "test_shapley_ci95_low", "test_shapley_ci95_high",
        "test_shapley_bootstrap_n", "refit50_shapley_median",
        "refit50_shapley_p025", "refit50_shapley_p975",
        "refit50_shapley_positive_frequency",
    ]].to_csv(RESULTS / "shapley_values.csv", index=False)

    null_sources = {
        "permuted_redundancy": REVIEW / "redundancy_null.csv",
        "covariance_matched_gaussian": REVIEW / "redundancy_matched_null.csv",
        "residual_controls": REVIEW / "redundancy_residual_controls.csv",
        "full_pipeline": REVIEW / "null_pipeline_counts.csv",
    }
    tables = []
    for experiment, source in null_sources.items():
        frame = pd.read_csv(source)
        frame.insert(0, "experiment", experiment)
        tables.append(frame)
    pd.concat(tables, ignore_index=True, sort=False).to_csv(
        RESULTS / "null_experiments.csv", index=False
    )
    print(f"assembled {len(COPIES) + 4} canonical result files")


if __name__ == "__main__":
    main()
