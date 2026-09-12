#!/usr/bin/env python3
"""Validate the public ICASSP 2027 artifact without private data."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


SCHEMAS = {
    "all_40_cells.csv": {"dataset", "feature", "gate_weight", "exact_shapley_mse_utility", "test_shapley_ci95_low", "test_shapley_ci95_high", "refit50_shapley_n", "refit50_shapley_positive_frequency"},
    "attribution_40cells.csv": {"dataset", "feature", "gate_weight", "singleton_mse_utility", "leave_one_out_mse_utility", "exact_shapley_mse_utility", "train_shapley_positive_frequency_gt_0"},
    "discordant_cases.csv": {"dataset", "feature", "gate_weight", "exact_shapley_mse_utility", "test_shapley_ci95_low", "test_shapley_ci95_high", "refit50_shapley_positive_frequency"},
    "fixed_coalition_contrasts.csv": {"dataset", "comparison", "ci95_low", "ci95_high", "p_holm_16"},
    "shapley_values.csv": {"dataset", "feature", "exact_shapley_mse_utility", "test_shapley_ci95_low", "test_shapley_ci95_high", "refit50_shapley_positive_frequency"},
    "interaction_values.csv": {"dataset", "feature_i", "feature_j", "oof_pearson", "shapley_interaction_mse_utility"},
    "rashomon_intervals.csv": {"dataset", "feature", "relative_mse_tolerance", "weight_min", "weight_max", "range_width"},
    "training_refits.csv": {"dataset", "bootstrap", "feature", "shapley_mse_utility"},
    "bootstrap_intervals.csv": {"dataset", "feature", "phi_utt", "utt_lo", "utt_hi", "phi_sys", "sys_lo", "sys_hi"},
    "nested_selections.csv": {"dataset", "outer_fold", "method", "k", "mask", "features", "inner_mse", "outer_utterance_mse"},
    "nested_selection_metrics.csv": {"dataset", "method", "k", "utterance_mse", "equal_system_mse", "system_srcc"},
    "null_experiments.csv": {"experiment", "dataset"},
    "coalition_values.csv": {"dataset", "protocol", "mask"},
}


def read_checked(name: str) -> pd.DataFrame:
    path = RESULTS / name
    require(path.is_file(), f"missing required result file: {name}")
    frame = pd.read_csv(path)
    missing = SCHEMAS.get(name, set()).difference(frame.columns)
    require(not missing, f"{name} lacks columns: {sorted(missing)}")
    return frame


def validate_rows() -> None:
    attribution = read_checked("attribution_40cells.csv")
    all_cells = read_checked("all_40_cells.csv")
    cases = read_checked("discordant_cases.csv")
    lattice = read_checked("coalition_values.csv")
    summary = pd.read_csv(RESULTS / "nested_selection_summary.csv")
    bootstrap = pd.read_csv(RESULTS / "nested_selection_bootstrap.csv")
    folds = read_checked("nested_selections.csv")
    metrics = read_checked("nested_selection_metrics.csv")
    shapley = read_checked("shapley_values.csv")
    interactions = read_checked("interaction_values.csv")
    rashomon = read_checked("rashomon_intervals.csv")
    refits = read_checked("training_refits.csv")
    intervals = read_checked("bootstrap_intervals.csv")
    contrasts = read_checked("fixed_coalition_contrasts.csv")
    nulls = read_checked("null_experiments.csv")
    test_summary = pd.read_csv(RESULTS / "test_shapley_system_bootstrap.csv")

    require(len(attribution) == 40, "expected 4 corpora x 10 attribution cells")
    require(len(all_cells) == 40, "complete test attribution table must contain 40 cells")
    require(int(attribution.gate_zero_positive_shapley.sum()) == 16,
            "expected 16 zero-gate/positive-Shapley cells")
    require(len(cases) == 16, "public discordant-case table must contain 16 rows")
    keys = ["dataset", "feature"]
    require(not attribution.duplicated(keys).any(), "attribution cell keys must be unique")
    require(set(map(tuple, attribution[keys].to_numpy())) == set(map(tuple, all_cells[keys].to_numpy())),
            "the two 40-cell tables identify different corpus/feature cells")
    expected_cases = attribution.loc[attribution.gate_zero_positive_shapley.astype(bool), keys]
    require(set(map(tuple, expected_cases.to_numpy())) == set(map(tuple, cases[keys].to_numpy())),
            "discordant cases do not equal the flagged attribution cells")
    require((cases.gate_weight.astype(float) < 1e-6).all(), "discordant gate weights must be below 1e-6")
    require((cases.exact_shapley_mse_utility.astype(float) > 0).all(), "discordant Shapley values must be positive")
    require((all_cells.refit50_shapley_n.astype(int) == 50).all(),
            "every complete attribution cell must summarize exactly 50 refits")
    require(all_cells.refit50_shapley_positive_frequency.astype(float).between(0, 1).all(),
            "50-refit positive frequencies must lie in [0,1]")
    joined = all_cells.merge(test_summary, on=keys, validate="one_to_one", suffixes=("", "_test"))
    for left, right in [
        ("gate_weight", "gate_weight_test"),
        ("exact_shapley_mse_utility", "test_shapley_mse_reduction"),
        ("test_shapley_ci95_low", "ci95_low"),
        ("test_shapley_ci95_high", "ci95_high"),
    ]:
        require(np.allclose(joined[left], joined[right], rtol=0, atol=1e-12),
                f"cross-file test attribution mismatch: {left}")

    group_sizes = lattice.groupby(["dataset", "protocol"]).size()
    require(len(group_sizes) == 16 and (group_sizes == 1023).all(),
            "every corpus/protocol lattice must contain 1,023 coalitions")
    for _, group in lattice.groupby(["dataset", "protocol"]):
        require(set(group["mask"].astype(int)) == set(range(1, 1024)),
                "coalition masks must be exactly 1..1023")

    datasets = {"brspeech", "bvcc", "singmos", "tmhintqi"}
    require(set(attribution.dataset) == datasets, "expected exactly four named corpora")
    require(len(shapley) == 40 and len(intervals) == 40, "Shapley and interval tables must have 40 cells")
    require(len(interactions) == 180, "expected 4 corpora x C(10,2) interaction rows")
    require((interactions.groupby("dataset").size() == 45).all(), "each corpus needs 45 expert pairs")
    require(len(rashomon) == 120, "expected 4 corpora x 10 experts x 3 tolerances")
    require(set(np.round(rashomon.relative_mse_tolerance.astype(float), 6)) == {0.001, 0.005, 0.01},
            "unexpected Rashomon tolerances")
    require(rashomon.success_min.astype(bool).all() and rashomon.success_max.astype(bool).all(),
            "at least one Rashomon coordinate optimization failed")
    require((rashomon.weight_min <= rashomon.weight_max + 1e-10).all(),
            "invalid Rashomon interval ordering")
    require(len(refits) == 2000, "expected 4 corpora x 50 refits x 10 experts")
    require((refits.groupby(["dataset", "feature"]).size() == 50).all(),
            "every corpus/expert cell must contain 50 training refits")
    positive_frequency = refits.groupby(keys).shapley_mse_utility.apply(lambda x: float((x > 0).mean()))
    published_frequency = all_cells.set_index(keys).refit50_shapley_positive_frequency
    require(np.allclose(positive_frequency.sort_index(), published_frequency.sort_index(), atol=1e-12),
            "published 50-refit positive frequencies do not match raw refits")
    require(len(contrasts) == 16, "expected 16 fixed-coalition contrasts")
    require(set(nulls.experiment) == {"permuted_redundancy", "covariance_matched_gaussian", "residual_controls", "full_pipeline"},
            "null experiment families are incomplete")
    require(nulls.groupby("experiment").size().to_dict() == {
        "permuted_redundancy": 4,
        "covariance_matched_gaussian": 4,
        "residual_controls": 4,
        "full_pipeline": 8,
    }, "unexpected row counts in null families")

    expected = {
        "gate": (1.083379, 2.375, 1),
        "singleton": (1.078571, 2.250, 10),
        "shapley": (1.079163, 2.0625, 1023),
        "inner_search": (1.077147, 2.000, 1023),
    }
    require(set(summary.method) == set(expected), "unexpected selection methods")
    for method, (ratio, rank, cost) in expected.items():
        row = summary.loc[summary.method == method].iloc[0]
        require(np.isclose(row.mean_mse_ratio_to_full, ratio, atol=5e-6),
                f"unexpected mean ratio for {method}")
        require(np.isclose(row.mean_rank, rank, atol=5e-4),
                f"unexpected mean rank for {method}")
        require(int(row.nonempty_coalitions_scored) == cost,
                f"unexpected coalition count for {method}")

    gate = bootstrap[(bootstrap.method == "gate") &
                     (bootstrap.reference == "inner_search")].iloc[0]
    require(np.isclose(gate.mean_normalized_mse_delta, 0.0062320709, atol=1e-9),
            "gate contrast changed")
    require(np.isclose(gate.p_holm_all_five, 0.208, atol=1e-12),
            "Holm-adjusted gate p-value changed")
    require(len(folds) == 800, "expected 4 corpora x 5 folds x 4 rules x 10 budgets")
    require(set(folds.method) == set(expected), "fold table has unexpected methods")
    require(set(folds.k.astype(int)) == set(range(1, 11)), "budgets must be 1..10")
    require(not folds.duplicated(["dataset", "outer_fold", "method", "k"]).any(),
            "nested fold selections must be unique")
    require(len(metrics) == 160, "expected 4 corpora x 4 methods x 10 budgets")
    require(not metrics.duplicated(["dataset", "method", "k"]).any(),
            "nested metric rows must be unique")


def validate_configuration() -> None:
    experiments = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    datasets = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text(encoding="utf-8"))
    representations = yaml.safe_load((ROOT / "configs" / "representations.yaml").read_text(encoding="utf-8"))
    require(experiments["primary_protocol"]["nonempty_coalitions_per_corpus"] == 1023,
            "configuration must freeze 1,023 coalitions")
    require(len(datasets["datasets"]) == 4, "configuration must define four corpora")
    require(len(representations["representations"]) == 10, "configuration must define ten players")
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    require(len(citation["authors"]) == 7, "CITATION.cff must list all seven manuscript authors")
    require("doi" not in citation, "do not publish a placeholder DOI")


def validate_public_build() -> None:
    """Regenerate human-readable tables and exercise all modular stages."""
    subprocess.run([sys.executable, "-m", "compileall", "-q", "src", "scripts"],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_public_tables.py")],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_pipeline.py")],
                   cwd=ROOT, check=True)
    for driver in ["run_missing_experiments.py", "run_final_icasp_controls.py", "run_system_disjoint_selection.py"]:
        subprocess.run([sys.executable, str(ROOT / "src" / driver), "--help"],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def validate_manifest() -> None:
    path = ROOT / "metadata" / "artifact_manifest.json"
    if not path.exists():
        print("manifest not generated yet; content checks passed")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sha_lines = (ROOT / "metadata" / "file_hashes.sha256").read_text(encoding="utf-8").splitlines()
    sha_map = {}
    for line in sha_lines:
        digest, rel = line.split("  ", 1)
        sha_map[rel] = digest
    require(set(sha_map) == set(manifest["files"]), "JSON and SHA-256 manifests list different files")
    for rel, expected in manifest["files"].items():
        target = ROOT / rel
        require(target.is_file(), f"missing manifest file: {rel}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        require(digest == expected["sha256"], f"hash mismatch: {rel}")
        require(digest == sha_map[rel], f"text manifest mismatch: {rel}")


if __name__ == "__main__":
    validate_rows()
    validate_configuration()
    validate_manifest()
    validate_public_build()
    print("OK: hashes, schemas, four corpora, 40 cells, 1,023-coalition lattices, 50 refits, nested folds, nulls, tables, and modular smoke test are valid")
