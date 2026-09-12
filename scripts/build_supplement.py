#!/usr/bin/env python3
"""Build the complete, claim-organized reproducibility supplement."""

from __future__ import annotations

import datetime as dt
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "supplement" / "supplementary_material.pdf"
TITLE = "Do MOS Fusion Weights Select Useful Embeddings? A System-Disjoint Attribution Audit"
AUTHORS = ("Daniel Casanova, Alef Iury Ferreira, Edresson Casanova, Lucas Gris, "
           "Pedro Lustosa Rege Botelho, Fernanda Silva, Anderson da Silva Soares")
VERSION = "v1.0-icassp2027-submission"


def number(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(x) < 0.5 * 10 ** (-digits):
        if x == 0:
            return "0"
        return f"{x:.2e}"
    if abs(x) >= 1000:
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def canvas(title: str, subtitle: str | None = None):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0.055, 0.055, 0.89, 0.90])
    ax.axis("off")
    ax.text(0, 1, title, fontsize=15, fontweight="bold", va="top", color="#172A3A")
    if subtitle:
        ax.text(0, 0.955, textwrap.fill(subtitle, 112), fontsize=8.5, va="top",
                color="#465864", linespacing=1.3)
    fig.text(0.055, 0.022, f"{VERSION}  •  MOS Fusion Weight Audit",
             fontsize=6.7, color="#667681")
    return fig, ax


def add_table(fig, frame: pd.DataFrame, columns: list[str], labels: list[str],
              *, top: float = 0.84, bottom: float = 0.07, fontsize: float = 6.3,
              digits: int = 4) -> None:
    values = [[number(row[col], digits) for col in columns] for _, row in frame.iterrows()]
    tax = fig.add_axes([0.055, bottom, 0.89, top - bottom])
    tax.axis("off")
    table = tax.table(cellText=values, colLabels=labels, loc="upper left", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.08)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#B7C1C8")
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_text_props(fontweight="bold", color="#172A3A")
            cell.set_facecolor("#DFEAF2")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F8FA")


def save_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    fig, ax = canvas(title)
    y = 0.92
    for paragraph in paragraphs:
        wrapped = textwrap.fill(paragraph, 105)
        ax.text(0, y, wrapped, fontsize=9.2, va="top", linespacing=1.4, color="#263843")
        y -= 0.034 * (wrapped.count("\n") + 1) + 0.034
    pdf.savefig(fig)
    plt.close(fig)


def save_table_pages(pdf: PdfPages, title: str, subtitle: str, frame: pd.DataFrame,
                     columns: list[str], labels: list[str], rows_per_page: int,
                     fontsize: float = 6.3, digits: int = 4) -> None:
    pages = int(np.ceil(len(frame) / rows_per_page))
    for page_index, start in enumerate(range(0, len(frame), rows_per_page), 1):
        chunk = frame.iloc[start:start + rows_per_page]
        fig, _ = canvas(f"{title} ({page_index}/{pages})", subtitle)
        add_table(fig, chunk, columns, labels, fontsize=fontsize, digits=digits)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    cells = pd.read_csv(RESULTS / "all_40_cells.csv")
    cases = pd.read_csv(RESULTS / "discordant_cases.csv")
    rashomon = pd.read_csv(RESULTS / "rashomon_intervals.csv")
    nested = pd.read_csv(RESULTS / "nested_selection_metrics.csv")
    nulls = pd.read_csv(RESULTS / "null_experiments.csv")
    interactions = pd.read_csv(RESULTS / "interaction_bootstrap.csv")

    OUTPUT.parent.mkdir(exist_ok=True)
    fixed_date = dt.datetime(2026, 9, 12, tzinfo=dt.timezone.utc)
    metadata = {
        "Title": f"Supplement — {TITLE}", "Author": AUTHORS,
        "Subject": "ICASSP 2027 reproducibility artifact",
        "Keywords": "MOS, Shapley, fusion, system-disjoint selection",
        "Creator": "scripts/build_supplement.py", "Producer": "Matplotlib",
        "CreationDate": fixed_date, "ModDate": fixed_date,
    }
    with PdfPages(OUTPUT, metadata=metadata) as pdf:
        save_text_page(pdf, "Reproducibility supplement", [
            TITLE, AUTHORS,
            f"Artifact version: {VERSION}. The release tag resolves the immutable Git commit. Zenodo DOI: pending release publication; no placeholder DOI is asserted.",
            "This document accompanies the machine-readable CSVs and executable analysis modules. It reports all 40 attribution cells, all 16 discordant cells with uncertainty and refit frequency, all 120 Rashomon coordinate intervals, system-disjoint selection by corpus and budget, and the null controls. Values are copied from the canonical committed CSVs; scripts/build_public_tables.py produces platform-stable text equivalents.",
            "Data and pretrained checkpoints are not redistributed. See metadata/dataset_splits.md and metadata/checkpoints.md for sources, versions, licenses, and preprocessing instructions. The public artifact begins from already extracted representations.",
            "Integrity command: python scripts/verify_release.py. It checks SHA-256 hashes, schemas, cross-file keys, four corpora, 40 attribution cells, all 1,023 masks in every released lattice, 50 training refits per cell, Rashomon solver success, nested folds, null families, table generation, and a synthetic end-to-end smoke test.",
        ])

        save_table_pages(pdf, "Table S1 — All attribution cells",
            "Gate is the full-coalition allocation. Singleton, LOO, and Shapley are held-out MSE utilities; positive means lower MSE. Train +freq is the fraction of training-system refits with positive Shapley. Test CI is a system bootstrap with fitted models fixed.",
            cells.sort_values(["dataset", "feature"]),
            ["dataset", "feature", "gate_weight", "singleton_mse_utility", "leave_one_out_mse_utility", "exact_shapley_mse_utility", "test_shapley_ci95_low", "test_shapley_ci95_high", "refit50_shapley_positive_frequency"],
            ["Corpus", "Expert", "Gate", "Singleton", "LOO", "Shapley", "CI low", "CI high", "Train +freq"],
            rows_per_page=20, fontsize=5.8)

        save_table_pages(pdf, "Table S2 — Zero-gate / positive-Shapley cells",
            "The full set of 16 discordant cells. Gate zero uses the predeclared threshold <1e-6. Frozen zero freq is the frequency of a zero gate across frozen-expert system bootstraps; train +freq is the frequency of positive Shapley across refits.",
            cases.sort_values(["dataset", "feature"]),
            ["dataset", "feature", "gate_weight", "leave_one_out_mse_utility", "exact_shapley_mse_utility", "test_shapley_ci95_low", "test_shapley_ci95_high", "frozen_expert_gate_probability_zero_lt_1e_6", "refit50_shapley_positive_frequency"],
            ["Corpus", "Expert", "Gate", "LOO", "Shapley", "CI low", "CI high", "Frozen 0 freq", "Train +freq"],
            rows_per_page=16, fontsize=5.8)

        save_table_pages(pdf, "Table S3 — Complete Rashomon intervals",
            "Coordinatewise minimum and maximum gate weights among mixtures whose OOF MSE is within the stated relative tolerance of the optimum. All solver-success flags are checked by verify_release.py; KKT diagnostics are in results/rashomon_kkt_audit.csv.",
            rashomon.sort_values(["relative_mse_tolerance", "dataset", "feature"]),
            ["dataset", "feature", "relative_mse_tolerance", "optimal_weight", "weight_min", "weight_max", "range_width", "success_min", "success_max"],
            ["Corpus", "Expert", "Tol.", "Opt.", "Min", "Max", "Width", "Min OK", "Max OK"],
            rows_per_page=30, fontsize=5.7)

        pivot = nested.pivot_table(index=["dataset", "k"], columns="method", values="equal_system_mse", aggfunc="first").reset_index()
        pivot.columns.name = None
        save_table_pages(pdf, "Table S4 — Nested selection by corpus and budget",
            "Equal-system MSE in outer system-disjoint folds. Every rule uses the same cardinality k. Exhaustive is inner-validation search, never oracle selection on the outer test systems. The full 160-row metric table also reports utterance MSE and system SRCC.",
            pivot.sort_values(["dataset", "k"]),
            ["dataset", "k", "gate", "singleton", "shapley", "inner_search"],
            ["Corpus", "k", "Gate", "Singleton", "Shapley", "Exhaustive"],
            rows_per_page=20, fontsize=6.4)

        null_columns = {
            "permuted_redundancy": ["experiment", "dataset", "true_rho_corr_vs_negI", "null_mean", "null_sd", "null_p025", "null_p975", "emp_p_null_ge_true", "n_null"],
            "covariance_matched_gaussian": ["experiment", "dataset", "n_pairs", "feasible", "rho_obs_corr_vs_negI", "rho_matched_null"],
            "residual_controls": ["experiment", "dataset", "beta_corr", "beta_mean", "beta_min"],
            "full_pipeline": ["experiment", "dataset", "variant", "observed_nzero", "observed_count", "null_nzero_mean", "null_nzero_max", "null_count_mean", "null_count_max", "null_conv_rate", "n_null"],
        }
        for experiment, columns in null_columns.items():
            frame = nulls.loc[nulls.experiment == experiment].dropna(axis=1, how="all")
            save_table_pages(pdf, f"Table S5 — Null: {experiment.replace('_', ' ')}",
                "Complete numeric outcomes for this null family. Protocol notes remain verbatim in results/null_experiments.csv. These controls diagnose how much of the correlation–interaction pattern follows from regression geometry rather than speech-specific structure.",
                frame, columns, [col.replace("_", " ") for col in columns],
                rows_per_page=max(1, len(frame)), fontsize=5.2)

        save_table_pages(pdf, "Table S6 — Interaction-correlation uncertainty",
            "Correlation of expert-prediction similarity with negative Shapley interaction. The test correlation interval resamples test systems while experts and fitted gates remain fixed; this is descriptive rather than full-refit uncertainty.",
            interactions.sort_values("dataset"),
            ["dataset", "n_test_systems", "rho_oof_correlation_vs_negative_interaction", "rho_oof_ci95_test_system_sampling_interaction_only_low", "rho_oof_ci95_test_system_sampling_interaction_only_high", "rho_test_correlation_vs_negative_interaction", "rho_test_joint_system_bootstrap_ci95_low", "rho_test_joint_system_bootstrap_ci95_high", "n_bootstrap"],
            ["Corpus", "Systems", "OOF rho", "OOF low", "OOF high", "Test rho", "Test low", "Test high", "B"],
            rows_per_page=4, fontsize=6.0)

        save_text_page(pdf, "Artifact map and interpretation boundaries", [
            "Allocation: results/all_40_cells.csv and results/discordant_cases.csv contain point estimates, system intervals, and refit frequencies. Contextual utility: results/shapley_values.csv and results/interaction_values.csv contain exact coalition-game quantities. Near-optimal allocation: results/rashomon_intervals.csv and results/rashomon_kkt_audit.csv contain every coordinate extreme and its solver diagnostics.",
            "Selection: results/nested_selections.csv contains every outer-fold selected subset and fitted weight vector; results/nested_selection_metrics.csv contains every corpus × rule × budget outcome; results/nested_selection_bootstrap.csv contains macro contrasts and Holm corrections. Exhaustive means best inner-validation subset, not best outer-test subset.",
            "Nulls: results/null_experiments.csv records four control families, and results/interaction_bootstrap.csv records uncertainty for the observed redundancy relation. The matched Gaussian control reproduces much of the association, so the artifact supports a geometric explanation and does not claim a uniquely speech-specific law.",
            "Scope: Shapley quantifies contextual utility under the declared player partition; it is not a universal selection rule. Singleton ranking was the attractive low-cost operating point in this benchmark, but no equivalence or universal encoder ranking is claimed. BRSpeechMOS system-level inference is exploratory because it has six systems.",
            "Release identity: the version is embedded above and metadata/artifact_manifest.json binds tracked files to SHA-256 digests. The Git tag binds that manifest to a commit. The DOI field must be updated only after Zenodo archives the final GitHub Release.",
        ])
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
