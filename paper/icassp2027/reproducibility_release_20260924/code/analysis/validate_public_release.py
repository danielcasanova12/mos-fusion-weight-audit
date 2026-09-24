#!/usr/bin/env python3
"""Validate the corrected public ICASSP artifact without private data.

The check is deliberately limited to cached, release-level evidence.  It does
not claim to re-extract audio features or refit the neural encoders.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def close(a: float, b: float, tol: float = 1e-12) -> None:
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{a!r} != {b!r} (tol={tol})")


def main() -> None:
    sel = ROOT / "results" / "selection" / "equal_system"
    sens = ROOT / "results" / "selection" / "sensitivity"
    attr = ROOT / "results" / "attribution"
    macro = pd.read_csv(sel / "macro_method_bootstrap.csv")
    expected = {
        "gate": -0.015056024906991356,
        "singleton": -0.0071584692922981585,
        "shapley": -0.021451644672244856,
    }
    for method, value in expected.items():
        row = macro[(macro.method == method) & (macro.reference == "inner_search")]
        if len(row) != 1:
            raise AssertionError(f"missing final selector row: {method}")
        close(row.iloc[0].mean_normalized_mse_delta, value)

    sensitivity = pd.read_csv(sens / "macro_sensitivity.csv")
    row = sensitivity[(sensitivity.scenario == "exclude_brspeechmos") &
                      (sensitivity.candidate == "shapley") &
                      (sensitivity.reference == "inner_search")]
    if len(row) != 1:
        raise AssertionError("missing BRSpeechMOS-excluded Shapley row")
    close(row.iloc[0].point_delta_pct, 0.629978258331526, tol=1e-9)

    table = pd.read_csv(attr / "attribution_40cells_alpha10.csv")
    reps = pd.read_csv(attr / "train_shapley_50reps.csv")
    if len(table) != 40 or len(reps) != 2000:
        raise AssertionError("unexpected attribution table size")
    counts = reps.groupby(["dataset", "feature"]).size()
    if not counts.eq(50).all():
        raise AssertionError("every attribution cell must have 50 refits")
    for key, group in reps.groupby(["dataset", "feature"], sort=False):
        row = table[(table.dataset == key[0]) & (table.feature == key[1])]
        if len(row) != 1:
            raise AssertionError(f"missing attribution cell: {key}")
        row = row.iloc[0]
        values = group.shapley_mse_utility.to_numpy(float)
        close(row.train_shapley_n, 50, tol=0)
        close(row.train_shapley_median, np.median(values))
        close(row.train_shapley_p025, np.quantile(values, 0.025))
        close(row.train_shapley_p975, np.quantile(values, 0.975))
        close(row.train_shapley_positive_count, (values > 0).sum(), tol=0)
        close(row.train_shapley_positive_frequency_gt_0, (values > 0).mean())

    discordant = table[table.gate_zero_positive_shapley]
    if len(discordant) != 16:
        raise AssertionError(f"expected 16 discordant cells, found {len(discordant)}")
    if int((discordant.train_shapley_positive_frequency_gt_0 == 1).sum()) != 8:
        raise AssertionError("unexpected 50-refit stability count")

    required = [
        ROOT / "manuscript" / "submitted_manuscript.pdf",
        ROOT / "configs" / "experiments.yaml",
        ROOT / "code" / "extraction" / "extract_explainability_embeddings.py",
        sel / "run_system_disjoint_selection_equal_system.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"missing release files: {missing}")

    result = {
        "status": "verified",
        "scope": "cached release evidence only; no audio or neural encoder refitting",
        "selector": {
            "objective": "equal_system_MSE_inner_and_outer",
            "outer_folds": 20,
            "bootstrap_replicates": 10000,
            "macro_delta_relative_to_exhaustive": expected,
            "exclude_brspeechmos_shapley_delta_pct": 0.629978258331526,
        },
        "attribution": {
            "cells": 40,
            "training_refits_per_cell": 50,
            "discordant_zero_gate_positive_shapley": 16,
            "discordant_cells_positive_in_all_50_refits": 8,
        },
        "sha256": {
            "submitted_manuscript.pdf": sha256(ROOT / "manuscript" / "submitted_manuscript.pdf"),
            "macro_method_bootstrap.csv": sha256(sel / "macro_method_bootstrap.csv"),
            "train_shapley_50reps.csv": sha256(attr / "train_shapley_50reps.csv"),
        },
    }
    out = ROOT / "results" / "audits" / "public_release_validation.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
