#!/usr/bin/env python3
"""Audit and export the 40-cell attribution evidence for common ridge alpha=10.

Run from any directory with a Python environment containing numpy and pandas:
    python paper/icassp2027/audit_attribution_evidence.py

This performs no fitting. It checks the saved final-protocol coalition lattice,
the explicitly labelled common-alpha-10 results, and the sources that produced
them before joining uncertainty summaries. Selected-alpha and older raw-eGeMAPS
results are never substituted. Missing optional uncertainty inputs remain NA.
The training Shapley percentiles use 50 refitted training-system replicates on
the fixed observed test set; they are descriptive stability ranges, not precise
95% confidence intervals or joint training-and-test uncertainty estimates.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
KEYS = ["dataset", "feature"]
DATASETS = ["brspeech", "bvcc", "singmos", "tmhintqi"]
FEATURES = ["whisper", "contentvec12", "wavlm", "beats", "auditory_erb",
            "speaker", "rmvpe_cont", "rmvpe_quant", "ced", "egemaps"]
PROTOCOL = "common_alpha_10_final_foldwise_egemaps_winsor_0p1pct"
POLICY = "egemaps_train_fold_winsor_0p1pct_each_tail"
THRESHOLDS = [0.0, 1e-8, 1e-6, 1e-4, 1e-3]
TOLERANCE = 1e-6
FULL = 2 ** len(FEATURES) - 1
LABELS = {"brspeech": "BRSpeechMOS", "bvcc": "BVCC", "singmos": "SingMOS",
          "tmhintqi": "TMHINT-QI", "whisper": "Whisper", "contentvec12": "ContentVec",
          "wavlm": "WavLM", "beats": "BEATs", "auditory_erb": "Auditory ERB",
          "speaker": "Speaker", "rmvpe_cont": "RMVPE-cont", "rmvpe_quant": "RMVPE-quant",
          "ced": "CED", "egemaps": "eGeMAPS"}


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise ValueError(explanation)


def close(a, b, label: str, atol: float = 1e-12) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label)
    error = float(np.max(np.abs(a - b)))
    require(error <= atol, f"{label}: maximum absolute difference {error:g} exceeds {atol:g}")
    return error


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    require(not frame.duplicated(KEYS).any(), "Duplicate corpus-feature rows")
    index = pd.MultiIndex.from_product([DATASETS, FEATURES], names=KEYS)
    require(set(frame.set_index(KEYS).index) == set(index), "Expected exactly 40 corpus-feature cells")
    return frame.set_index(KEYS).reindex(index).reset_index()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_shapley(losses: dict[int, float]) -> np.ndarray:
    values = np.zeros(len(FEATURES), dtype=float)
    n = len(FEATURES)
    for i in range(n):
        bit = 1 << i
        for mask in range(FULL + 1):
            if mask & bit:
                continue
            size = mask.bit_count()
            coefficient = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            values[i] += coefficient * (losses[mask] - losses[mask | bit])
    return values


def source_evidence(path: Path, functions: list[str]) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    result = {}
    for name in functions:
        require(name in definitions, f"Missing source function {name} in {path}")
        node = definitions[name]
        result[name] = {"start_line": node.lineno, "end_line": node.end_lineno,
                        "source": ast.get_source_segment(source, node)}
    return result


def verify_protocol(priority: Path, final: Path, table: pd.DataFrame, sources: dict) -> dict:
    manifest = json.loads((priority / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((final / "primary_protocol.json").read_text(encoding="utf-8"))
    require(manifest["features"] == FEATURES and manifest["datasets"] == DATASETS,
            "Manifest player/corpus order does not match this audit")
    require(Path(manifest["primary_bundles"]).name == "primary_oof", "Unrecognized primary expert bundle")
    require(Path(manifest["final_lattice"]).name == "converged_oof_lattice.csv", "Unrecognized final lattice")
    require(manifest["full_train_bootstrap_alpha"] == 10, "Training bootstrap uses another alpha")
    require(protocol["egemaps_policy"] == POLICY and protocol["n_folds"] == 5
            and protocol["cross_fitting"] == "GroupKFold by system_id", "Final preprocessing/cross-fit mismatch")
    runner = sources["run_reviewer_priority_cpu.py"]
    require('math.isclose(alpha, 10.0)' in runner["task_alpha_bundles"]["source"]
            and 'load_bundle(args.primary, dataset)' in runner["task_alpha_bundles"]["source"],
            "Cannot establish reuse of primary experts by the alpha10 protocol")
    refit = runner["bootstrap_refit_bundle"]["source"]
    require('(0.001, 0.999)' in refit and 'GroupKFold(n_splits=min(5,' in refit
            and 'args.bootstrap_alpha' in refit and 'test_arrays' in refit,
            "Training bootstrap preprocessing/cross-fitting could not be verified")
    require('load_primary_bundle(out, dataset)' in sources["run_cpu_submission_finalization.py"]["cmd_explain"]["source"],
            "Test uncertainty does not establish final primary expert provenance")
    require('prediction_from_lattice(bundle, lattice' in sources["run_cpu_submission_finalization.py"]["cmd_explain"]["source"],
            "Test uncertainty does not establish saved coalition-gate provenance")
    lattice = pd.read_csv(final / "converged_oof_lattice.csv")
    alpha_lattice = pd.read_csv(priority / "alpha_coalition_lattices.csv")
    alpha_lattice = alpha_lattice[alpha_lattice.protocol == "common_alpha_10"]
    require(lattice.preprocessing.eq(POLICY).all(), "Final lattice preprocessing mismatch")
    require(not lattice.duplicated(["dataset", "mask"]).any(), "Duplicate lattice cells")
    require(not alpha_lattice.duplicated(["dataset", "mask"]).any(), "Duplicate alpha10 lattice cells")
    merged = lattice.merge(alpha_lattice, on=["dataset", "mask"], suffixes=("", "_alpha10"), validate="one_to_one")
    require(len(merged) == 4 * FULL, "Missing final/alpha10 lattice cells")
    checks = {"4092_alpha10_vs_final_coalition_losses_max_abs_error":
              close(merged.test_mse, merged.test_mse_alpha10, "Final versus alpha10 coalition losses")}
    alpha = pd.read_csv(priority / "alpha_shapley_and_gate.csv")
    alpha = ordered(alpha[alpha.protocol == "common_alpha_10"])
    checks["40_alpha10_gate_weights_max_abs_error"] = close(table.gate_weight, alpha.full_gate_weight, "Gate alpha10 identity")
    checks["40_alpha10_shapley_values_max_abs_error"] = close(table.exact_shapley_mse_utility, alpha.shapley_mse_utility, "Shapley alpha10 identity")
    for dataset in DATASETS:
        part = table[table.dataset == dataset]
        game = lattice[lattice.dataset == dataset].set_index("mask")
        require(set(game.index) == set(range(1, FULL + 1)), f"Incomplete lattice: {dataset}")
        weights = json.loads(game.loc[FULL, "weights_json"])
        checks[f"{dataset}_saved_gate_max_abs_error"] = close(part.gate_weight, [weights[f] for f in FEATURES], "Saved gate identity")
        losses = game.test_mse.to_dict()
        # Every singleton reports loss(empty)-loss(singleton), so all ten
        # independently recover the same train-mean empty-coalition loss.
        empties = part.singleton_mse_utility.to_numpy() + np.array([losses[1 << i] for i in range(len(FEATURES))])
        close(empties, np.repeat(empties[0], len(FEATURES)), "Empty-coalition baseline identity")
        losses[0] = float(empties[0])
        checks[f"{dataset}_recomputed_shapley_max_abs_error"] = close(part.exact_shapley_mse_utility, exact_shapley(losses), "Exact Shapley recomputation")
        checks[f"{dataset}_recomputed_loo_max_abs_error"] = close(part.leave_one_out_mse_utility,
            [losses[FULL ^ (1 << i)] - losses[FULL] for i in range(len(FEATURES))], "LOO recomputation")
    return {"status": "verified", "protocol": PROTOCOL, "checks": checks,
            "evidence_scope": "Saved numeric outputs, manifests, and producing source; raw experts were not refitted by this audit.",
            "selected_alpha_results_used": False, "older_preprocessing_results_used": False}


def bootstrap_summary(frame: pd.DataFrame, value: str, prefix: str, expected_n: int,
                      positive: bool = False) -> pd.DataFrame:
    require(not frame.duplicated(KEYS + ["bootstrap"]).any(), f"Duplicate {prefix} replicate cells")
    require(np.isfinite(frame[value]).all(), f"Non-finite {prefix} values")
    require(set(frame.bootstrap.unique()) == set(range(expected_n)), f"Unexpected {prefix} replicate indices")
    rows = []
    for (dataset, feature), part in frame.groupby(KEYS, sort=False):
        values = part[value].to_numpy(float)
        require(len(values) == expected_n, f"Incomplete {prefix} replicates for {dataset}/{feature}")
        rows.append({"dataset": dataset, "feature": feature,
                     f"{prefix}_n": len(values), f"{prefix}_median": float(np.median(values)),
                     f"{prefix}_p025": float(np.quantile(values, .025)),
                     f"{prefix}_p975": float(np.quantile(values, .975)),
                     f"{prefix}_{'positive_count' if positive else 'zero_count_lt_1e-6'}": int(np.sum(values > 0) if positive else np.sum(values < TOLERANCE)),
                     f"{prefix}_{'positive_frequency_gt_0' if positive else 'probability_zero_lt_1e-6'}": float(np.mean(values > 0) if positive else np.mean(values < TOLERANCE))})
    return ordered(pd.DataFrame(rows))


def gate_zero(values: pd.Series, threshold: float) -> pd.Series:
    # At exactly zero, report exact zeros; a strict <0 test is uninformative
    # for nonnegative simplex coefficients. At positive t retain the runner's <t.
    return values.eq(0) if threshold == 0 else values.lt(threshold)


def threshold_sweep(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ["ALL"] + DATASETS:
        part = table if dataset == "ALL" else table[table.dataset == dataset]
        positive = part.exact_shapley_mse_utility.gt(0)
        for threshold in THRESHOLDS:
            zero = gate_zero(part.gate_weight, threshold)
            signed = part.leave_one_out_mse_utility.le(threshold)
            absolute = part.leave_one_out_mse_utility.abs().le(threshold)
            rows.append({"dataset": dataset, "protocol": PROTOCOL, "threshold": threshold,
                         "gate_rule": "w == 0" if threshold == 0 else "w < threshold",
                         "n_cells": len(part), "gate_zero_count": int(zero.sum()),
                         "gate_zero_positive_shapley_count": int((zero & positive).sum()),
                         "loo_signed_le_threshold_count": int(signed.sum()),
                         "loo_signed_le_threshold_positive_shapley_count": int((signed & positive).sum()),
                         "loo_absolute_le_threshold_count": int(absolute.sum()),
                         "loo_absolute_le_threshold_positive_shapley_count": int((absolute & positive).sum())})
    return pd.DataFrame(rows)


def number(value, digits: int = 5) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def latex_number(value) -> str:
    if pd.isna(value):
        return "NA"
    if value != 0 and abs(value) < 1e-4:
        return f"{value:.2e}"
    return f"{value:.5f}"


def interval(row, prefix: str) -> str:
    return f"[{number(row[f'{prefix}_p025'])}, {number(row[f'{prefix}_p975'])}]"


def write_latex(table: pd.DataFrame, output: Path) -> None:
    lines = [r"% Generated by audit_attribution_evidence.py; requires longtable and booktabs.",
             r"\begingroup", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}{llrrrrrlr}",
             r"\caption{All 40 corpus--representation attribution cells under common ridge $\alpha=10$, with fold-wise eGeMAPS winsorization. Training stability and conditional test uncertainty are reported separately.}\label{tab:attribution-evidence}\\",
             r"\toprule",
             r"Corpus & Representation & Gate $w$ & $\Pr_{\rm train}(w<10^{-6})$ & LOO $\Delta$MSE & Shapley $\phi$ & Train median $\phi$ & Train [2.5,97.5]\% & Train $\Pr(\phi>0)$ \\",
             r"\midrule", r"\endfirsthead", r"\toprule",
             r"Corpus & Representation & Gate $w$ & $\Pr_{\rm train}(w<10^{-6})$ & LOO $\Delta$MSE & Shapley $\phi$ & Train median $\phi$ & Train [2.5,97.5]\% & Train $\Pr(\phi>0)$ \\",
             r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, row in table.iterrows():
        values = [LABELS[row.dataset], LABELS[row.feature], latex_number(row.gate_weight),
                  number(row.train_gate_probability_zero_lt_1e_6, 2),
                  latex_number(row.leave_one_out_mse_utility), latex_number(row.exact_shapley_mse_utility),
                  number(row.train_shapley_median), interval(row, "train_shapley"),
                  number(row.train_shapley_positive_frequency_gt_0, 3)]
        lines.append(" & ".join(values) + r" \\")
    lines += [r"\end{longtable}",
              r"\noindent Gate probabilities use 100 training-system bootstrap refits per corpus. Shapley medians, percentile ranges, and positive frequencies use all 50 complete training-system bootstrap games per corpus, conditional on the fixed observed test set. These are descriptive stability summaries, not precise 95\% confidence intervals or joint training/test uncertainty. Positive frequency has resolution 1/50. LOO is signed test MSE without the representation minus full-model test MSE. Negative LOO means removal improves test MSE; it is not zero utility.",
              r"\begin{longtable}{lllr}",
              r"\caption{Conditional test-system uncertainty for the same 40 common-$\alpha=10$ Shapley values. Fixed fitted experts and coalition gates; 10,000 resamples of test systems.}\label{tab:attribution-test-uncertainty}\\",
              r"\toprule", r"Corpus & Representation & Test-system 95\% interval & Train-only $\Pr(w<10^{-6})$, frozen experts \\",
              r"\midrule", r"\endfirsthead", r"\toprule",
              r"Corpus & Representation & Test-system 95\% interval & Train-only $\Pr(w<10^{-6})$, frozen experts \\",
              r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, row in table.iterrows():
        values = [LABELS[row.dataset], LABELS[row.feature],
                  f"[{number(row.test_shapley_ci95_low)}, {number(row.test_shapley_ci95_high)}]",
                  number(row.frozen_expert_gate_probability_zero_lt_1e_6, 3)]
        lines.append(" & ".join(values) + r" \\")
    lines += [r"\end{longtable}",
              r"\noindent The final column uses 200 resamples of training systems with the experts held fixed. Test intervals resample systems but retain utterance-weighted means and are conditional on all fitted models. The two uncertainty sources are not combined. NA denotes unavailable or unverified inputs. All columns belong to the common-$\alpha=10$ protocol; no selected-$\alpha$ uncertainty is implied.",
              r"\endgroup", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority", type=Path, default=PAPER / "paper_reviewer_priority_cpu")
    parser.add_argument("--final", type=Path, default=PAPER / "submission_final_cpu_review")
    parser.add_argument("--output", type=Path, default=HERE / "review_evidence")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_functions = {
        "run_reviewer_priority_cpu.py": ["task_importance_comparison", "task_alpha_bundles", "bootstrap_refit_bundle", "task_full_train_bootstrap", "task_conditional_gate_bootstrap", "coalition_game"],
        "run_cpu_submission_finalization.py": ["cmd_prepare_primary", "cmd_explain", "bootstrap_cluster_mean", "prediction_from_lattice"],
        "run_final_icasp_controls.py": ["robust_oof", "transformed_arrays", "fit_simplex"],
    }
    sources = {name: source_evidence(PAPER / name, names) for name, names in source_functions.items()}
    table = ordered(pd.read_csv(args.priority / "importance_methods.csv"))
    audit = verify_protocol(args.priority, args.final, table, sources)
    audit["missing_optional_inputs"] = []
    used_inputs = [args.priority / name for name in ["manifest.json", "importance_methods.csv", "alpha_coalition_lattices.csv", "alpha_shapley_and_gate.csv"]]
    used_inputs += [args.final / "primary_protocol.json", args.final / "converged_oof_lattice.csv"]
    table = table[KEYS + ["gate_weight", "singleton_mse_utility", "leave_one_out_mse_utility", "exact_shapley_mse_utility"]]
    table.insert(2, "protocol", PROTOCOL)
    for filename, value, prefix, expected, positive in [
        ("full_train_bootstrap_weights.csv", "weight", "train_gate", 100, False),
        ("train_shapley_50reps.csv", "shapley_mse_utility", "train_shapley", 50, True),
        ("conditional_gate_bootstrap_replicates.csv", "weight", "frozen_expert_gate", 200, False),
    ]:
        public_path = args.output / filename
        path = public_path if public_path.exists() else args.priority / filename
        if path.exists():
            frame = pd.read_csv(path)
            if "spec" in frame:
                frame = frame[frame.spec == "full"]
            if "alpha" in frame:
                require(frame.alpha.eq(10).all(), f"Wrong alpha in {filename}")
            summary = bootstrap_summary(frame, value, prefix, expected, positive)
            table = table.merge(summary, on=KEYS, how="left", validate="one_to_one")
            used_inputs.append(path)
        else:
            audit["missing_optional_inputs"].append(filename)
            suffixes = ["n", "median", "p025", "p975"] + (["positive_count", "positive_frequency_gt_0"] if positive else ["zero_count_lt_1e-6", "probability_zero_lt_1e-6"])
            for suffix in suffixes:
                table[f"{prefix}_{suffix}"] = np.nan
    # Use identifier-friendly column names in the exported table.
    table = table.rename(columns=lambda x: x.replace("1e-6", "1e_6"))
    path = args.final / "oof_shapley_test_cluster_bootstrap.csv"
    if path.exists():
        test = ordered(pd.read_csv(path))
        try:
            audit["checks"]["40_test_ci_point_values_max_abs_error"] = close(table.exact_shapley_mse_utility,
                test.test_shapley_mse_reduction, "Test-CI point identity")
            require(test.bootstrap_unit.eq("system_id").all() and test.n_bootstrap.eq(10000).all(), "Unexpected test resampling protocol")
        except ValueError as error:
            audit["missing_optional_inputs"].append(f"Test CI withheld: {error}")
            test = None
        if test is not None:
            table["test_shapley_ci95_low"] = test.ci95_low
            table["test_shapley_ci95_high"] = test.ci95_high
            table["test_shapley_bootstrap_n"] = test.n_bootstrap
            used_inputs.append(path)
    else:
        audit["missing_optional_inputs"].append(path.name)
    for column in ["test_shapley_ci95_low", "test_shapley_ci95_high", "test_shapley_bootstrap_n"]:
        if column not in table:
            table[column] = np.nan
    table["train_shapley_uncertainty_scope"] = "training_system_refit_fixed_observed_test_descriptive_50_replicates"
    table["test_shapley_uncertainty_scope"] = "test_system_resampling_fixed_fitted_models_utterance_weighted"
    positive = table.exact_shapley_mse_utility.gt(0)
    table["gate_zero_lt_1e_6"] = table.gate_weight.lt(TOLERANCE)
    table["gate_zero_positive_shapley"] = table.gate_zero_lt_1e_6 & positive
    table["loo_signed_le_1e_6_positive_shapley"] = table.leave_one_out_mse_utility.le(TOLERANCE) & positive
    table["loo_absolute_le_1e_6_positive_shapley"] = table.leave_one_out_mse_utility.abs().le(TOLERANCE) & positive
    table.to_csv(args.output / "attribution_40cells_alpha10.csv", index=False, na_rep="NA")
    sweep = threshold_sweep(table)
    sweep.to_csv(args.output / "attribution_threshold_sweep.csv", index=False)
    membership = table[KEYS + ["gate_weight", "leave_one_out_mse_utility", "exact_shapley_mse_utility", "gate_zero_lt_1e_6", "gate_zero_positive_shapley", "loo_signed_le_1e_6_positive_shapley", "loo_absolute_le_1e_6_positive_shapley"]]
    membership.to_csv(args.output / "attribution_case_membership_40cells.csv", index=False)
    case_lists = {}
    for label, column in [("loo_signed_18_cases", "loo_signed_le_1e_6_positive_shapley"), ("zero_gate_positive_shapley_cases", "gate_zero_positive_shapley"), ("loo_absolute_zero_positive_shapley_cases", "loo_absolute_le_1e_6_positive_shapley")]:
        subset = membership[membership[column]]
        subset.to_csv(args.output / f"{label}.csv", index=False)
        case_lists[label] = [f"{row.dataset}/{row.feature}" for row in subset.itertuples()]
    saved_18_path = args.priority / "loo_zero_shapley_positive.csv"
    if saved_18_path.exists():
        saved = pd.read_csv(saved_18_path)
        require(set(map(tuple, saved[KEYS].to_numpy())) == set(map(tuple, membership[membership.loo_signed_le_1e_6_positive_shapley][KEYS].to_numpy())), "Saved 18-case membership differs")
        used_inputs.append(saved_18_path)
    audit["case_lists"] = case_lists
    audit["sources"] = {name: {fn: {k: v for k, v in detail.items() if k != "source"} for fn, detail in functions.items()} for name, functions in sources.items()}
    audit["input_sha256"] = {str(path.relative_to(PAPER)) if path.is_relative_to(PAPER) else path.name: sha256(path) for path in used_inputs + [PAPER / name for name in sources]}
    audit["uncertainty"] = {"training_gate": "100 system bootstrap refits, including experts and gate", "training_shapley": "All 50 system bootstrap refits, full 1023-coalition games, fixed observed test set; descriptive percentile ranges", "conditional_gate": "200 training-system resamples, gate refits with frozen expert predictions", "test_shapley": "10000 test-system resamples, fixed fitted models, utterance-weighted bootstrap mean; same-protocol point values verified"}
    write_latex(table, args.output / "attribution_supplement_alpha10.tex")
    (args.output / "attribution_supplement_standalone.tex").write_text("\n".join([
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,landscape,margin=15mm]{geometry}",
        r"\usepackage{booktabs,longtable}",
        r"\begin{document}",
        r"\input{attribution_supplement_alpha10.tex}",
        r"\end{document}", "",
    ]), encoding="utf-8")
    zero_cases = table[table.gate_zero_positive_shapley]
    audit["zero_gate_positive_shapley_stability"] = {
        "n_cases": len(zero_cases),
        "training_percentile_low_above_zero": int(zero_cases.train_shapley_p025.gt(0).sum()),
        "conditional_test_ci_low_above_zero": int(zero_cases.test_shapley_ci95_low.gt(0).sum()),
        "training_positive_frequency_below_one": zero_cases.loc[
            zero_cases.train_shapley_positive_frequency_gt_0.lt(1),
            KEYS + ["train_shapley_positive_count", "train_shapley_n", "train_shapley_positive_frequency_gt_0"],
        ].to_dict("records"),
    }
    (args.output / "attribution_provenance_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    notes = ["# Attribution evidence audit", "", "Run `python paper/icassp2027/audit_attribution_evidence.py` with numpy and pandas.", "",
             "The 40-cell table uses only common ridge alpha=10, grouped OOF predictions, and fold-wise 0.1% winsorization of both eGeMAPS tails. All 4,092 coalition losses, 40 gates, and 40 Shapley point estimates agree with the explicitly labelled alpha10 game. Shapley and LOO were independently recomputed from the final lattice. The same-protocol test interval point estimates agree within 1e-12. Selected-alpha and older preprocessing results are excluded.", "",
             "Training gate probabilities use 100 system-bootstrap refits per corpus. Training Shapley summaries use all 50 full-system refits and complete coalition games per corpus, conditional on the fixed observed test set. The percentile ranges are descriptive stability ranges, not precise 95% confidence intervals. Positive frequency means the observed fraction of those 50 replicates with Shapley > 0; its resolution is 1/50. It is not a posterior probability or a joint training/test confidence statement.", "",
             "The separate test intervals use 10,000 test-system resamples with the experts and coalition gates fixed; system clusters are sampled, then utterance-weighted means are formed. The separate frozen-expert gate probabilities use 200 training-system resamples. None of these uncertainty sources is combined.", "",
             f"Among the {len(zero_cases)} point zero-gate/positive-Shapley cells, {int(zero_cases.train_shapley_p025.gt(0).sum())} have training percentile lower bounds above zero and {int(zero_cases.test_shapley_ci95_low.gt(0).sum())} have conditional test interval lower bounds above zero. These are descriptive, unadjusted per-cell summaries. Positive point utility is not uniformly stable under retraining: all frequencies are exposed in the table.", "",
             "## The 18-case wording correction", "",
             "The producer calls LOO <= 1e-6 and Shapley > 0 effectively zero LOO. This is a signed threshold, so it also accepts substantially negative utility. Use 'nonpositive or numerically negligible LOO utility' for the 18-case set. Absolute LOO <= 1e-6 yields 16 cases with positive Shapley, exactly the 16 zero-gate/positive-Shapley cases. There are 17 zero gates in total; BRSpeechMOS RMVPE-quant has negative Shapley and is excluded from the positive-utility set.", "",
             "The two additional signed-LOO cases are BVCC Whisper (LOO -0.002392055566558887, gate 0.24948356301409777) and TMHINT-QI Speaker (LOO -1.1852257730815552e-6, gate 0.01175879950483767). Their gates are active; removal improves test MSE.", "",
             "## Threshold convention", "",
             "Gate zero means w < t for positive t, and exact w == 0 at t=0. Both signed LOO <= t and absolute LOO <= t are exported. Every positive-Shapley count uses phi > 0. MSE thresholds are not gate-weight units. Counts are provided overall and by corpus at 0, 1e-8, 1e-6, 1e-4, and 1e-3.", ""]
    for label, cases in case_lists.items():
        notes += [f"## {label} ({len(cases)})", ""] + [f"- {case}" for case in cases] + [""]
    notes += ["## Files", "", "`attribution_40cells_alpha10.csv` contains all numeric summaries, replicate counts, and scope labels. `attribution_supplement_alpha10.tex` contains two longtables (requires booktabs and longtable). `attribution_supplement_standalone.tex` supplies an A4 landscape wrapper; run a LaTeX compiler on it from this evidence directory. Do not insert a longtable directly in the two-column paper body. `attribution_threshold_sweep.csv` and the case-membership/list CSVs make each threshold claim inspectable. `attribution_provenance_audit.json` records source line references, input SHA-256 hashes, and numerical identity checks. NA is reserved for missing or unverified uncertainty evidence.", ""]
    (args.output / "ATTRIBUTION_EVIDENCE_README.md").write_text("\n".join(notes), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "rows": len(table), "case_counts": {k: len(v) for k, v in case_lists.items()}, "missing_optional_inputs": audit["missing_optional_inputs"], "output": str(args.output)}, indent=2))
    print(sweep[sweep.dataset == "ALL"].to_string(index=False))


if __name__ == "__main__":
    main()
