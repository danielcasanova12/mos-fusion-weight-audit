#!/usr/bin/env python3
"""Build a short, claim-organized PDF supplement from committed CSVs."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "supplement" / "supplementary_material.pdf"


def page(pdf: PdfPages, title: str, paragraphs: list[str], table=None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.07, 0.84, 0.87])
    ax.axis("off")
    ax.text(0, 1.0, title, fontsize=17, fontweight="bold", va="top")
    y = 0.94
    for paragraph in paragraphs:
        wrapped = textwrap.fill(paragraph, width=105)
        ax.text(0, y, wrapped, fontsize=9.5, va="top", linespacing=1.35)
        y -= 0.034 * (wrapped.count("\n") + 1) + 0.025
    if table is not None:
        columns, rows = table
        height = min(0.46, 0.035 * (len(rows) + 2))
        tab_ax = fig.add_axes([0.08, max(0.08, y - height), 0.84, height])
        tab_ax.axis("off")
        artist = tab_ax.table(cellText=rows, colLabels=columns,
                              loc="upper left", cellLoc="left")
        artist.auto_set_font_size(False)
        artist.set_fontsize(7.5)
        artist.scale(1, 1.25)
        for (row, _), cell in artist.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#E8EEF7")
            cell.set_edgecolor("#9AA6B2")
    fig.text(0.08, 0.025, "MOS Fusion Weight Audit — ICASSP 2027 reproducibility artifact",
             fontsize=7, color="#56616B")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cases = pd.read_csv(RESULTS / "zero_gate_positive_shapley_cases.csv")
    players = pd.read_csv(RESULTS / "player_partition_summary.csv")
    selection = pd.read_csv(RESULTS / "nested_selection_summary.csv")
    boot = pd.read_csv(RESULTS / "nested_selection_bootstrap.csv")
    rashomon = pd.read_csv(RESULTS / "rashomon_weight_ranges.csv")

    with PdfPages(OUTPUT) as pdf:
        page(pdf, "Supplement A — Scope and protocol", [
            "This supplement is organized by the claims in the paper. It contains public aggregate results only. Audio, listener-level ratings, cached embeddings, and credentials are not redistributed.",
            "Ten frozen representation experts are fused by a nonnegative simplex gate. The complete ten-player game contains 1,023 non-empty coalitions per corpus. Gate weight describes allocation in one fitted coalition; exact loss-based Shapley describes average held-out MSE utility across coalition contexts.",
            "The primary audit spans BRSpeechMOS, BVCC, SingMOS, and TMHINT-QI. Uncertainty is clustered by system. BRSpeechMOS has only six systems and is treated as exploratory.",
            "The public integrity command is: python scripts/verify_release.py. Public tables and this PDF are regenerated with the two build scripts in scripts/.",
        ])

        case_rows = []
        for row in cases.itertuples():
            case_rows.append([
                str(row.dataset), str(row.feature), f"{row.gate_weight:.2g}",
                f"{row.leave_one_out_mse_utility:.4f}",
                f"{row.exact_shapley_mse_utility:.4f}",
            ])
        page(pdf, "Supplement B — Allocation is not contextual utility", [
            "Sixteen corpus/representation cells have gate weight below 1e-6 and positive exact Shapley utility. Eleven have descriptive test-system bootstrap intervals above zero and eight remain positive in every one of 50 training refits.",
            "Across 100 system-bootstrap refits, the four corpora produce 51/4/17/14 distinct gate supports while their MSE standard deviations are .278/.008/.021/.012. Broad support variation with stable predictive error is evidence of allocation non-identifiability, not evidence that the encoders themselves change meaning.",
        ], (["Corpus", "Representation", "Gate", "LOO", "Shapley"], case_rows))

        tol = rashomon[rashomon.relative_mse_tolerance == 0.005]
        widest = tol.sort_values("range_width", ascending=False).head(8)
        width_rows = [[str(r.dataset), str(r.feature), f"{r.optimal_weight:.3f}",
                       f"{r.weight_min:.3f}", f"{r.weight_max:.3f}"]
                      for r in widest.itertuples()]
        scenarios = (players.groupby("scenario")["zero_gate_positive_shapley_count"]
                     .sum().sort_values(ascending=False))
        page(pdf, "Supplement C — Rashomon and player granularity", [
            "Coordinatewise Rashomon envelopes search gate vectors whose OOF MSE is within a fixed relative tolerance of the optimum. Wide intervals show that one coefficient vector is an incomplete description of utility.",
            "The discordant-case count depends on the definition of a player. Merging the two RMVPE views reduces 16 cases to 13; grouping RMVPE with eGeMAPS gives nine; reduced games yield 7–13. Every corpus retains at least one mismatch, while grouped/Owen rankings remain stable.",
            "Scenario totals in the public partition file: " + ", ".join(f"{k}={int(v)}" for k, v in scenarios.items()) + ".",
        ], (["Corpus", "Feature", "Optimum", "Min", "Max"], width_rows))

        order = ["gate", "singleton", "shapley", "inner_search"]
        labels = {"gate": "Gate", "singleton": "Singleton", "shapley": "Shapley",
                  "inner_search": "Exhaustive"}
        sel = selection.set_index("method").loc[order].reset_index()
        sel_rows = [[labels[r.method], f"{int(r.nonempty_coalitions_scored):,}",
                     f"{r.mean_mse_ratio_to_full:.4f}", f"{r.mean_rank:.3f}",
                     f"{int(r.wins)}/{int(r.comparisons)}"] for r in sel.itertuples()]
        gate = boot[(boot.method == "gate") & (boot.reference == "inner_search")].iloc[0]
        page(pdf, "Supplement D — System-disjoint subset selection", [
            "Five outer grouped folds hold complete systems out of preprocessing, expert fitting, importance scoring, subset choice, and gate calibration. Four inner grouped folds provide the selection-stage predictions. All methods use the same cardinality budgets k in {1,2,3,5} for the primary comparison.",
            "Exhaustive inner search has the best point mean. No significant difference is detected against singleton or Shapley. Singleton uses 10 coalition scores instead of 1,023. Gate ranking is directionally worse by 0.62% of full-model MSE; its raw interval excludes zero, but the Holm-adjusted p-value across five displayed contrasts is " + f"{gate.p_holm_all_five:.3f}" + ", so the paper does not make a confirmatory superiority claim.",
            "Interpretation: gate describes the fitted allocation; Shapley audits contextual utility and substitutability; selection performance must be validated separately. In this benchmark singleton is the attractive low-cost operating point, not a universally optimal rule.",
        ], (["Rule", "Scores", "MSE/full", "Rank", "Best/tied"], sel_rows))

    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

