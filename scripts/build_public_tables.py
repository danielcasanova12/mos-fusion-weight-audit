#!/usr/bin/env python3
"""Regenerate compact public tables from committed CSV outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "supplement"
OUT.mkdir(exist_ok=True)


def main() -> None:
    selection = pd.read_csv(RESULTS / "nested_selection_summary.csv")
    bootstrap = pd.read_csv(RESULTS / "nested_selection_bootstrap.csv")
    cases = pd.read_csv(RESULTS / "zero_gate_positive_shapley_cases.csv")
    players = pd.read_csv(RESULTS / "player_partition_summary.csv")

    order = ["gate", "singleton", "shapley", "inner_search"]
    labels = {
        "gate": "Gate weight",
        "singleton": "Singleton MSE",
        "shapley": "Exact Shapley",
        "inner_search": "Inner search",
    }
    selection = selection.set_index("method").loc[order].reset_index()

    latex = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Rule & Coalitions & MSE/full & Rank & Best \\",
        r"\midrule",
    ]
    for row in selection.itertuples():
        latex.append(
            f"{labels[row.method]} & {int(row.nonempty_coalitions_scored):,} & "
            f"{row.mean_mse_ratio_to_full:.4f} & {row.mean_rank:.2f} & "
            f"{int(row.wins)}/{int(row.comparisons)} " + r"\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "table_nested_selection.tex").write_text("\n".join(latex) + "\n",
                                                     encoding="utf-8")

    md = [
        "# Generated public tables",
        "",
        "## Nested system-disjoint selection",
        "",
        "| Rule | Coalition scores | Mean MSE/full | Mean rank | Best/tied |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selection.itertuples():
        md.append(
            f"| {labels[row.method]} | {int(row.nonempty_coalitions_scored):,} | "
            f"{row.mean_mse_ratio_to_full:.4f} | {row.mean_rank:.3f} | "
            f"{int(row.wins)}/{int(row.comparisons)} |"
        )
    md.extend([
        "",
        "## Macro bootstrap contrasts",
        "",
        bootstrap.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Zero-gate / positive-Shapley cases",
        "",
        cases.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Player-partition sensitivity",
        "",
        players.to_markdown(index=False, floatfmt=".6f"),
        "",
    ])
    (OUT / "generated_tables.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote supplement/table_nested_selection.tex and supplement/generated_tables.md")


if __name__ == "__main__":
    main()

