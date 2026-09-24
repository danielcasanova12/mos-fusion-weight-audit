#!/usr/bin/env python3
"""Recompute selection sensitivities from the completed nested run.

This is a CPU-only reanalysis: it does not refit encoders or ridge experts.
It consumes the saved outer-system errors from the equal-system nested run and
repeats the paired system bootstrap for (i) excluding BRSpeechMOS, (ii)
excluding k=1, and (iii) leave-one-corpus-out variants.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ["brspeech", "bvcc", "singmos", "tmhintqi"]
METHODS = ["gate", "singleton", "shapley"]
ALL_METHODS = METHODS + ["inner_search"]
PRIMARY_BUDGETS = [1, 2, 3, 5]
SEED = 20260917  # primary run seed + 1, matching its macro bootstrap
N_BOOTSTRAP = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("review_evidence/system_disjoint_selection_equal_system"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("review_evidence/selection_sensitivity"),
    )
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    return parser.parse_args()


def verify_protocol(source: Path, errors: pd.DataFrame) -> dict[str, object]:
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    completion = json.loads((source / "completion.json").read_text(encoding="utf-8"))
    audit = json.loads((source / "audit.json").read_text(encoding="utf-8"))
    checks: dict[str, object] = {
        "completion_status": completion.get("status") == "complete",
        "outer_folds": completion.get("outer_folds") == 20,
        "inner_selection_metric": protocol.get("inner_selection_metric")
        == "equal-system MSE",
        "gate_calibration_metric": protocol.get("gate_calibration_metric")
        == "equal-system MSE",
        "outer_evaluation_metric": protocol.get("outer_evaluation_metric")
        == "equal-system MSE",
        "utterance_mse_diagnostic_only": protocol.get("utterance_mse_role")
        == "diagnostic only; never used for selection or gate fitting",
        "complete_item_coverage": audit.get("complete_item_coverage_every_method_budget")
        is True,
        "one_outer_fold_per_system": audit.get("one_outer_fold_per_system") is True,
        "expected_selection_rows": audit.get("selection_row_count")
        == audit.get("expected_selection_row_count")
        == 800,
        "errors_file_absent": not (source / "errors.jsonl").exists(),
    }
    checks["all_protocol_checks_pass"] = all(bool(value) for value in checks.values())

    required = {"dataset", "method", "k", "system_id", "system_mse"}
    missing = sorted(required.difference(errors.columns))
    if missing:
        raise ValueError(f"Missing columns in outer_system_errors.csv: {missing}")
    return {"checks": checks, "protocol": protocol, "completion": completion, "audit": audit}


def build_arrays(errors: pd.DataFrame) -> dict[str, dict[str, object]]:
    arrays: dict[str, dict[str, object]] = {}
    for dataset in DATASETS:
        subset = errors[errors.dataset == dataset]
        full = (
            subset[(subset.method == "inner_search") & (subset.k == 10)]
            .set_index("system_id")["system_mse"]
            .sort_index()
        )
        if full.empty:
            raise ValueError(f"No full-model reference for {dataset}")
        delta: dict[tuple[int, str], np.ndarray] = {}
        for k in PRIMARY_BUDGETS:
            pivot = subset[subset.k == k].pivot(
                index="system_id", columns="method", values="system_mse"
            ).loc[full.index]
            missing_methods = sorted(set(ALL_METHODS).difference(pivot.columns))
            if missing_methods:
                raise ValueError(f"{dataset}, k={k}: missing {missing_methods}")
            for method in METHODS:
                delta[(k, method)] = (
                    pivot[method].to_numpy(dtype=float)
                    - pivot["inner_search"].to_numpy(dtype=float)
                )
        arrays[dataset] = {"systems": full.index.tolist(), "full": full.to_numpy(dtype=float), "delta": delta}
    return arrays


def holm_adjust(raw: list[float]) -> list[float]:
    order = np.argsort(np.asarray(raw, dtype=float))
    adjusted = np.ones(len(raw), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(raw) - rank) * raw[index]))
        adjusted[index] = running
    return adjusted.tolist()


def run_scenario(
    name: str,
    datasets: list[str],
    budgets: list[int],
    arrays: dict[str, dict[str, object]],
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    point: dict[str, list[float]] = {method: [] for method in METHODS}
    cells: list[dict[str, object]] = []
    for dataset in datasets:
        cached = arrays[dataset]
        full = np.asarray(cached["full"], dtype=float)
        full_point = float(full.mean())
        delta = cached["delta"]
        for k in budgets:
            for method in METHODS:
                value = float(np.asarray(delta[(k, method)]).mean() / full_point)
                point[method].append(value)
                cells.append(
                    {
                        "scenario": name,
                        "dataset": dataset,
                        "k": k,
                        "method": method,
                        "reference": "inner_search",
                        "delta_pct": 100.0 * value,
                    }
                )

    rng = np.random.default_rng(seed)
    samples = {method: np.empty(bootstrap, dtype=float) for method in METHODS}
    for replicate in range(bootstrap):
        macro = {method: [] for method in METHODS}
        for dataset in datasets:
            cached = arrays[dataset]
            full = np.asarray(cached["full"], dtype=float)
            sampled = rng.integers(0, len(full), size=len(full))
            sampled_full = max(float(full[sampled].mean()), 1e-12)
            for k in budgets:
                for method in METHODS:
                    sampled_delta = np.asarray(cached["delta"][(k, method)])[sampled]
                    macro[method].append(float(sampled_delta.mean() / sampled_full))
        for method in METHODS:
            samples[method][replicate] = float(np.mean(macro[method]))

    contrast_defs = [
        ("gate", "inner_search"),
        ("shapley", "inner_search"),
        ("singleton", "inner_search"),
        ("gate", "singleton"),
        ("shapley", "singleton"),
    ]
    rows: list[dict[str, object]] = []
    raw_p: list[float] = []
    contrast_samples: list[np.ndarray] = []
    for candidate, reference in contrast_defs:
        if reference == "inner_search":
            values = samples[candidate]
            point_value = float(np.mean(point[candidate]))
        else:
            values = samples[candidate] - samples[reference]
            point_value = float(np.mean(point[candidate]) - np.mean(point[reference]))
        probability_negative = float(np.mean(values < 0.0))
        p_value = min(1.0, 2.0 * min(probability_negative, 1.0 - probability_negative))
        raw_p.append(p_value)
        contrast_samples.append(values)
        low, high = np.quantile(values, (0.025, 0.975))
        rows.append(
            {
                "scenario": name,
                "candidate": candidate,
                "reference": reference,
                "point_delta_pct": 100.0 * point_value,
                "ci95_low_pct": 100.0 * float(low),
                "ci95_high_pct": 100.0 * float(high),
                "p_two_sided_bootstrap": p_value,
                "n_corpora": len(datasets),
                "n_budgets": len(budgets),
                "n_bootstrap": bootstrap,
                "datasets": "+".join(datasets),
                "budgets": "+".join(map(str, budgets)),
            }
        )
    for row, adjusted in zip(rows, holm_adjust(raw_p)):
        row["p_holm_all_five"] = adjusted
    return rows, cells


def main() -> None:
    args = parse_args()
    source = args.input
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    errors = pd.read_csv(source / "outer_system_errors.csv")
    audit = verify_protocol(source, errors)
    arrays = build_arrays(errors)

    scenarios = [
        ("all_primary", DATASETS, PRIMARY_BUDGETS),
        ("exclude_brspeechmos", DATASETS[1:], PRIMARY_BUDGETS),
        ("exclude_k1", DATASETS, [2, 3, 5]),
        ("exclude_brspeechmos_k_ge_2", DATASETS[1:], [2, 3, 5]),
    ]
    for dataset in DATASETS:
        scenarios.append((f"leave_out_{dataset}", [d for d in DATASETS if d != dataset], PRIMARY_BUDGETS))

    all_rows: list[dict[str, object]] = []
    all_cells: list[dict[str, object]] = []
    for index, (name, datasets, budgets) in enumerate(scenarios):
        rows, cells = run_scenario(name, datasets, budgets, arrays, args.bootstrap, SEED + index)
        all_rows.extend(rows)
        all_cells.extend(cells)

    macro = pd.DataFrame(all_rows)
    cells = pd.DataFrame(all_cells)
    macro.to_csv(output / "macro_sensitivity.csv", index=False)
    cells.to_csv(output / "cell_sensitivity.csv", index=False)

    parity = macro[macro.scenario == "all_primary"].merge(
        pd.read_csv(source / "macro_method_bootstrap.csv"),
        left_on=["candidate", "reference"],
        right_on=["method", "reference"],
        how="left",
        suffixes=("_recomputed", "_published"),
    )
    parity["point_difference"] = parity.point_delta_pct / 100.0 - parity.mean_normalized_mse_delta
    parity.to_csv(output / "primary_parity.csv", index=False)
    audit["sensitivity"] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": str(source),
        "bootstrap": args.bootstrap,
        "seed_first_scenario": SEED,
        "scenarios": [name for name, _, _ in scenarios],
        "max_primary_point_difference": float(np.max(np.abs(parity.point_difference))),
        "note": "Reanalysis of saved outer-system errors; no model or encoder was refit.",
    }
    (output / "protocol_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    report = [
        "# Selection sensitivity reanalysis",
        "",
        "CPU-only reanalysis of the completed equal-system nested run. No encoder, ridge expert, or gate was refit.",
        "",
        f"- Bootstrap replicates per scenario: {args.bootstrap}",
        f"- Protocol checks passed: {audit['checks']['all_protocol_checks_pass']}",
        f"- Primary parity max point difference: {audit['sensitivity']['max_primary_point_difference']:.3e}",
        "",
        "The primary parity row must match the published macro run; the other scenarios are new sensitivity analyses.",
        "",
        "## Scenarios",
        "",
        "- `exclude_brspeechmos`: all primary budgets on BVCC, SingMOS, and TMHINT-QI.",
        "- `exclude_k1`: budgets 2, 3, and 5 on all four corpora.",
        "- `exclude_brspeechmos_k_ge_2`: both restrictions together.",
        "- `leave_out_*`: one leave-one-corpus-out analysis per corpus.",
    ]
    (output / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "output": str(output),
        "protocol_checks_passed": audit["checks"]["all_protocol_checks_pass"],
        "primary_parity_max_point_difference": audit["sensitivity"]["max_primary_point_difference"],
        "rows": len(macro),
    }, indent=2))


if __name__ == "__main__":
    main()
