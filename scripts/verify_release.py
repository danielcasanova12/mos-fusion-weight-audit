#!/usr/bin/env python3
"""Validate the public ICASSP 2027 artifact without private data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_rows() -> None:
    attribution = pd.read_csv(RESULTS / "attribution_40cells.csv")
    cases = pd.read_csv(RESULTS / "zero_gate_positive_shapley_cases.csv")
    lattice = pd.read_csv(RESULTS / "coalition_values.csv")
    summary = pd.read_csv(RESULTS / "nested_selection_summary.csv")
    bootstrap = pd.read_csv(RESULTS / "nested_selection_bootstrap.csv")
    folds = pd.read_csv(RESULTS / "nested_selection_folds.csv")

    require(len(attribution) == 40, "expected 4 corpora x 10 attribution cells")
    require(int(attribution.gate_zero_positive_shapley.sum()) == 16,
            "expected 16 zero-gate/positive-Shapley cells")
    require(len(cases) == 16, "public discordant-case table must contain 16 rows")

    group_sizes = lattice.groupby(["dataset", "protocol"]).size()
    require(len(group_sizes) == 16 and (group_sizes == 1023).all(),
            "every corpus/protocol lattice must contain 1,023 coalitions")
    for _, group in lattice.groupby(["dataset", "protocol"]):
        require(set(group["mask"].astype(int)) == set(range(1, 1024)),
                "coalition masks must be exactly 1..1023")

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


def validate_manifest() -> None:
    path = ROOT / "metadata" / "artifact_manifest.json"
    if not path.exists():
        print("manifest not generated yet; content checks passed")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        target = ROOT / rel
        require(target.is_file(), f"missing manifest file: {rel}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        require(digest == expected["sha256"], f"hash mismatch: {rel}")


if __name__ == "__main__":
    validate_rows()
    validate_manifest()
    print("OK: public artifact schemas, headline values, and hashes are valid")

