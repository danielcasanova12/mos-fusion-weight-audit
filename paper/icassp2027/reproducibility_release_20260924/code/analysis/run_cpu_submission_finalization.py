#!/usr/bin/env python3
"""CPU-only finalization of the ICASSP MOS explainability results.

This runner deliberately writes to a new output directory.  It uses a
train-fold-only 0.1%/99.9% winsorization rule for eGeMAPS, a deterministic
simplex gate, paired energy-matched reverb controls, an expanded concatenated
ridge grid, and system-cluster uncertainty for MSE and SRCC.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_final_icasp_controls as fc  # noqa: E402
import run_icasp_followups_v2 as v2  # noqa: E402
import run_missing_experiments as base  # noqa: E402


FEATURES = base.FEATURES
DATASETS = base.DATASETS
PRIMARY_BOUNDS = (0.001, 0.999)
PRIMARY_POLICY = "egemaps_train_fold_winsor_0p1pct_each_tail"
FIXED_SPECS = {
    "basic": fc.MASKS["basic"],
    "core": fc.MASKS["core"],
    "stable_ssl": fc.MASKS["stable_ssl"],
    "full": fc.MASKS["full"],
    "basic_plus_ced": fc.MASKS["basic_plus_ced"],
    "core_plus_ced": fc.MASKS["core_plus_ced"],
    "core_plus_beats": fc.MASKS["core_plus_beats"],
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.root).expanduser().resolve()
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    return root, out


def primary_bundle_path(out: Path, dataset: str) -> Path:
    return out / "primary_oof" / f"{dataset}.npz"


def load_primary_bundle(out: Path, dataset: str) -> dict[str, np.ndarray]:
    path = primary_bundle_path(out, dataset)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def transform_egemaps(train: dict[str, np.ndarray], evaluation: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    transformed_train, transformed_eval, _ = fc.transformed_arrays(
        train, evaluation, "egemaps", PRIMARY_BOUNDS
    )
    return transformed_train, transformed_eval


def resolve_audio(root: Path, dataset: str, relative: str) -> Path:
    sys.path.insert(0, str(root))
    import extract_explainability_embeddings as extractor

    candidate = Path(relative)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    candidate = Path(extractor.DATASETS[dataset]["root"]) / relative
    if candidate.exists():
        return candidate
    raise FileNotFoundError(candidate)


def cmd_prepare_primary(args: argparse.Namespace) -> None:
    import opensmile

    root, out = paths(args)
    device = base.device_arg("cpu")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    audit_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    (out / "primary_oof").mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets.split(","):
        train_e = np.asarray(base.load_x(root, args.cache, dataset, "train", "egemaps"), dtype=np.float64)
        mu = train_e.mean(axis=0)
        sd = train_e.std(axis=0)
        sd[sd < 1e-8] = 1.0
        for split in ("val", "test"):
            values = np.asarray(base.load_x(root, args.cache, dataset, split, "egemaps"), dtype=np.float64)
            z = (values - mu) / sd
            item, dim = np.unravel_index(np.abs(z).argmax(), z.shape)
            meta = base.load_metadata(root, dataset, split)
            row = meta.iloc[int(item)]
            audio = resolve_audio(root, dataset, str(row.filepath))
            extracted = smile.process_file(str(audio))
            names = [str(x) for x in extracted.columns]
            fresh = np.asarray(extracted.values.squeeze(), dtype=np.float64)
            cached = values[int(item)]
            audit_rows.append({
                "dataset": dataset,
                "split": split,
                "item_index": int(item),
                "filepath": str(row.filepath),
                "dimension": int(dim),
                "descriptor": names[int(dim)] if int(dim) < len(names) else f"dim_{dim}",
                "cached_value": float(cached[int(dim)]),
                "reextracted_value": float(fresh[int(dim)]),
                "absolute_difference": float(abs(cached[int(dim)] - fresh[int(dim)])),
                "max_vector_absolute_difference": float(np.max(np.abs(cached - fresh))),
                "max_abs_train_standardized_value": float(np.max(np.abs(z))),
            })
            lo, hi = np.quantile(train_e, PRIMARY_BOUNDS, axis=0)
            clipped = np.clip(values, lo, hi)
            diagnostic_rows.append({
                "dataset": dataset,
                "split": split,
                "worst_item": int(item),
                "worst_dimension": int(dim),
                "max_abs_z_raw": float(np.max(np.abs(z))),
                "max_abs_z_after_primary_winsor": float(np.max(np.abs((clipped - mu) / sd))),
                "n_values_clipped": int(np.sum((values < lo) | (values > hi))),
            })

        bundle = fc.robust_oof(
            root, args.cache, dataset, args.alpha, device, PRIMARY_BOUNDS, args.folds
        )
        np.savez(primary_bundle_path(out, dataset), **bundle)
        log(f"primary robust OOF complete: {dataset}")

    pd.DataFrame(audit_rows).to_csv(out / "egemaps_reextraction_audit.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(out / "egemaps_primary_diagnostics.csv", index=False)
    write_json(out / "primary_protocol.json", {
        "device": "cpu",
        "cross_fitting": "GroupKFold by system_id",
        "n_folds": args.folds,
        "gate": "deterministic converged simplex (SLSQP, ftol=1e-12)",
        "egemaps_policy": PRIMARY_POLICY,
        "selection": "validation only",
        "test_use": "final estimates and paired uncertainty only",
    })


def mask_prediction(bundle: dict[str, np.ndarray], mask: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    idx = [FEATURES.index(feature) for feature in fc.names(mask)]
    weights = fc.fit_simplex(bundle["oof"][:, idx], bundle["y_train"])
    return bundle[split][:, idx] @ weights, weights


def cmd_lattice(args: argparse.Namespace) -> None:
    _, out = paths(args)
    rows: list[dict[str, object]] = []
    item_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    fixed_masks = set(fc.MASKS.values())

    for dataset in args.datasets.split(","):
        bundle = load_primary_bundle(out, dataset)
        dataset_rows: list[dict[str, object]] = []
        for mask in range(1, 1 << len(FEATURES)):
            feature_names = fc.names(mask)
            idx = [FEATURES.index(feature) for feature in feature_names]
            weights = fc.fit_simplex(bundle["oof"][:, idx], bundle["y_train"])
            val_pred = bundle["val"][:, idx] @ weights
            test_pred = bundle["test"][:, idx] @ weights
            row = {
                "dataset": dataset,
                "mask": mask,
                "features": "+".join(feature_names),
                "n_features": int(mask.bit_count()),
                "gate": "converged_simplex",
                "preprocessing": PRIMARY_POLICY,
                "weights_json": json.dumps(dict(zip(feature_names, weights.tolist()))),
                **{f"val_{key}": value for key, value in fc.metric_dict(bundle["y_val"], val_pred).items()},
                **{f"test_{key}": value for key, value in fc.metric_dict(bundle["y_test"], test_pred).items()},
            }
            rows.append(row)
            dataset_rows.append(row)
            if mask in fixed_masks:
                spec = next(name for name, value in fc.MASKS.items() if value == mask)
                item_rows.extend({
                    "dataset": dataset,
                    "item_index": int(i),
                    "mos": float(bundle["y_test"][i]),
                    "spec": spec,
                    "mask": mask,
                    "prediction": float(test_pred[i]),
                } for i in range(len(test_pred)))
            if mask % 128 == 0 or mask == 1023:
                log(f"primary lattice {dataset}: {mask}/1023")
        frame = pd.DataFrame(dataset_rows)
        best = frame.loc[frame.val_spearman.idxmax()]
        selected_rows.append({
            "dataset": dataset,
            "selection": "validation_best",
            "mask": int(best["mask"]),
            "features": best["features"],
            "n_features": int(best["n_features"]),
            "val_spearman": float(best["val_spearman"]),
            "test_spearman": float(best["test_spearman"]),
            "test_mse": float(best["test_mse"]),
        })

    pd.DataFrame(rows).to_csv(out / "converged_oof_lattice.csv", index=False)
    pd.DataFrame(item_rows).to_csv(out / "converged_item_predictions.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out / "converged_selected_summary.csv", index=False)


def bootstrap_cluster_mean(values: np.ndarray, groups: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    sums = np.asarray([values[index].sum() for index in members], dtype=np.float64)
    sizes = np.asarray([len(index) for index in members], dtype=np.float64)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)), size=n)
    boot = (counts @ sums) / np.maximum(counts @ sizes, 1.0)
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def prediction_from_lattice(bundle: dict[str, np.ndarray], lattice: pd.DataFrame, mask: int, split: str) -> np.ndarray:
    if mask == 0:
        return np.full(len(bundle[f"y_{split}"]), float(np.mean(bundle["y_train"])), dtype=np.float64)
    row = lattice[lattice["mask"] == mask].iloc[0]
    weights = np.asarray(list(json.loads(row.weights_json).values()), dtype=np.float64)
    idx = [FEATURES.index(feature) for feature in fc.names(mask)]
    return bundle[split][:, idx] @ weights


def cmd_explain(args: argparse.Namespace) -> None:
    root, out = paths(args)
    lattice_all = pd.read_csv(out / "converged_oof_lattice.csv")
    shapley_rows: list[dict[str, object]] = []
    displacement_rows: list[dict[str, object]] = []
    factorial = [math.factorial(i) for i in range(len(FEATURES) + 1)]
    n_features = len(FEATURES)

    for dataset in args.datasets.split(","):
        bundle = load_primary_bundle(out, dataset)
        lattice = lattice_all[lattice_all.dataset == dataset]
        predictions = {0: prediction_from_lattice(bundle, lattice, 0, "test")}
        for mask in range(1, 1 << n_features):
            predictions[mask] = prediction_from_lattice(bundle, lattice, mask, "test")
        y = bundle["y_test"].astype(np.float64)
        groups = base.load_metadata(root, dataset, "test").system_id.astype(str).to_numpy()
        errors = {mask: (y - prediction) ** 2 for mask, prediction in predictions.items()}
        for feature_index, feature in enumerate(FEATURES):
            contribution = np.zeros(len(y), dtype=np.float64)
            bit = 1 << feature_index
            for mask in range(0, 1 << n_features):
                if mask & bit:
                    continue
                size = mask.bit_count()
                weight = factorial[size] * factorial[n_features - size - 1] / factorial[n_features]
                contribution += weight * (errors[mask] - errors[mask | bit])
            estimate, low, high = bootstrap_cluster_mean(
                contribution, groups, args.n_bootstrap, args.seed + feature_index
            )
            shapley_rows.append({
                "dataset": dataset,
                "feature": feature,
                "test_shapley_mse_reduction": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_unit": "system_id",
                "n_bootstrap": args.n_bootstrap,
            })

        core = fc.MASKS["core"]
        core_prediction, core_weights = mask_prediction(bundle, core, "test")
        del core_prediction
        core_map = dict(zip(fc.names(core), core_weights.tolist()))
        for added in ("whisper", "wavlm", "beats", "ced"):
            child = core | base.BITS[added]
            _, child_weights = mask_prediction(bundle, child, "test")
            child_map = dict(zip(fc.names(child), child_weights.tolist()))
            for feature in fc.names(child):
                old = float(core_map.get(feature, 0.0))
                new = float(child_map.get(feature, 0.0))
                displacement_rows.append({
                    "dataset": dataset,
                    "baseline": "core",
                    "added_feature": added,
                    "feature": feature,
                    "weight_before": old,
                    "weight_after": new,
                    "weight_delta": new - old,
                })
        log(f"Shapley and displacement complete: {dataset}")

    pd.DataFrame(shapley_rows).to_csv(out / "oof_shapley_test_cluster_bootstrap.csv", index=False)
    pd.DataFrame(displacement_rows).to_csv(out / "oof_core_displacement.csv", index=False)


def bootstrap_spearman(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    unique = np.unique(groups.astype(str))
    members = {group: np.flatnonzero(groups.astype(str) == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(n, dtype=np.float64)
    for iteration in range(n):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([members[group] for group in chosen])
        values[iteration] = spearmanr(y[index], pred[index]).statistic
    return float(spearmanr(y, pred).statistic), float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * float(p_values[position])))
        adjusted[position] = running
    return adjusted


def cmd_bootstrap(args: argparse.Namespace) -> None:
    root, out = paths(args)
    item = pd.read_csv(out / "converged_item_predictions.csv")
    fixed_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    comparisons = [
        ("basic", "full"),
        ("basic", "basic_plus_ced"),
        ("core", "core_plus_ced"),
        ("core", "core_plus_beats"),
    ]
    for dataset_index, dataset in enumerate(args.datasets.split(",")):
        q = item[item.dataset == dataset]
        y = q[q.spec == "basic"].sort_values("item_index").mos.to_numpy(float)
        groups = base.load_metadata(root, dataset, "test").system_id.astype(str).to_numpy()
        predictions = {
            spec: q[q.spec == spec].sort_values("item_index").prediction.to_numpy(float)
            for spec in q.spec.unique()
        }
        for spec in ("basic", "core", "stable_ssl", "full"):
            pred = predictions[spec]
            squared_error = (y - pred) ** 2
            mse, mse_low, mse_high = bootstrap_cluster_mean(
                squared_error, groups, args.n_bootstrap, args.seed + dataset_index
            )
            srcc, srcc_low, srcc_high = bootstrap_spearman(
                y, pred, groups, args.n_bootstrap, args.seed + 100 + dataset_index
            )
            fixed_rows.append({
                "dataset": dataset,
                "spec": spec,
                "n_items": len(y),
                "n_systems": len(np.unique(groups)),
                "mse": mse,
                "mse_ci95_low": mse_low,
                "mse_ci95_high": mse_high,
                "srcc": srcc,
                "srcc_ci95_low": srcc_low,
                "srcc_ci95_high": srcc_high,
                "n_bootstrap": args.n_bootstrap,
            })
        unique, inverse = np.unique(groups.astype(str), return_inverse=True)
        members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
        rng = np.random.default_rng(args.seed + 200 + dataset_index)
        counts = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)), size=args.n_bootstrap)
        sizes = np.asarray([len(index) for index in members])
        denominator = np.maximum(counts @ sizes, 1)
        for baseline, child in comparisons:
            delta = (y - predictions[baseline]) ** 2 - (y - predictions[child]) ** 2
            sums = np.asarray([delta[index].sum() for index in members])
            boot = (counts @ sums) / denominator
            p_value = min(1.0, 2.0 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))
            comparison_rows.append({
                "dataset": dataset,
                "comparison": f"{baseline}_vs_{child}",
                "estimate_mse_improvement": float(delta.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "bootstrap_p_two_sided": p_value,
                "n_groups": len(unique),
            })
        log(f"MSE/SRCC bootstrap complete: {dataset}")
    comparisons_frame = pd.DataFrame(comparison_rows)
    comparisons_frame["holm_p"] = holm_adjust(comparisons_frame.bootstrap_p_two_sided.to_numpy(float))
    comparisons_frame["family"] = "16_pre_specified_dataset_by_contrast_MSE_tests"
    pd.DataFrame(fixed_rows).to_csv(out / "fixed_model_cluster_bootstrap.csv", index=False)
    comparisons_frame.to_csv(out / "confirmatory_tests_holm.csv", index=False)


def make_paired_controls(root: Path, out: Path, datasets: str, n: int, seed: int) -> Path:
    import torch
    import torchaudio

    manifest_path = out / "controls" / "manifest.csv"
    old = pd.read_csv(root / "results/paper_followups_v2/perturbation/manifest.csv")
    chosen = old.drop_duplicates(["dataset", "source_index"]).groupby("dataset", sort=True).head(n)
    sys.path.insert(0, str(root))
    import extract_explainability_embeddings as extractor

    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    audio_root = out / "controls" / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    for dataset_index, dataset in enumerate(datasets.split(",")):
        subset = chosen[chosen.dataset == dataset].head(n)
        for row in subset.itertuples():
            original = extractor._load_mono_16k(row.source_filepath)
            reverbed = fc.make_energy_reverb(
                original, 16000, seed + dataset_index * 100000 + int(row.source_index)
            )
            common_scale = min(1.0, 0.98 / max(float(original.abs().max()), float(reverbed.abs().max()), 1e-9))
            paired = {
                "identity": original * common_scale,
                "gain_minus6db": original * common_scale * (10.0 ** (-6.0 / 20.0)),
                "reverb_energy": reverbed * common_scale,
            }
            identity_rms = float(paired["identity"].square().mean().sqrt())
            for control, waveform in paired.items():
                destination = audio_root / dataset / control / f"item_{int(row.source_index)}.wav"
                destination.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(destination), waveform.unsqueeze(0), 16000, encoding="PCM_F", bits_per_sample=32)
                rms = float(waveform.square().mean().sqrt())
                diagnostics.append({
                    "dataset": dataset,
                    "source_index": int(row.source_index),
                    "control": control,
                    "common_scale": common_scale,
                    "peak": float(waveform.abs().max()),
                    "rms": rms,
                    "rms_delta_vs_paired_identity_db": 20.0 * math.log10(max(rms, 1e-12) / max(identity_rms, 1e-12)),
                })
                rows.append({
                    "key": f"{dataset}|{int(row.source_index)}|{control}",
                    "filepath": str(destination.relative_to(out / "controls")),
                    "dataset": dataset,
                    "source_index": int(row.source_index),
                    "source_filepath": row.source_filepath,
                    "system_id": row.system_id,
                    "condition": getattr(row, "condition", "unknown"),
                    "mos": float(row.mos),
                    "control": control,
                    "sample_rate": 16000,
                })
        log(f"paired controls prepared: {dataset}")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    pd.DataFrame(diagnostics).to_csv(out / "control_audio_diagnostics.csv", index=False)
    return manifest_path


def cmd_controls_prepare(args: argparse.Namespace) -> None:
    root, out = paths(args)
    make_paired_controls(root, out, args.datasets, args.items_per_dataset, args.seed)


def cmd_controls_extract(args: argparse.Namespace) -> None:
    root, out = paths(args)
    extractor, pseudo = fc.configure_control_extractor(root, out)
    report = extractor.extract_one(
        pseudo,
        "test",
        args.feature,
        SimpleNamespace(device="cpu", whisper_model="whisper-large-v3", deep_audit_after=False),
    )
    write_json(out / "controls" / f"extract_{args.feature}.json", report)
    if report.get("status") == "error":
        raise RuntimeError(report.get("error", "control extraction failed"))


def load_control_arrays(root: Path, out: Path, manifest: pd.DataFrame, feature: str) -> np.ndarray:
    sys.path.insert(0, str(root))
    from run_new_explainability_ridge import pool_tensor

    paths_ = [
        out / "controls" / "embeddings" / "paper_controls" / "test" / fc.DIR_NAMES[feature] / Path(value).with_suffix(".pt")
        for value in manifest.filepath.astype(str)
    ]
    missing = [str(path) for path in paths_ if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    return np.stack([pool_tensor(str(path), feature) for path in paths_]).astype(np.float32)


def cmd_controls_analyze(args: argparse.Namespace) -> None:
    root, out = paths(args)
    manifest = pd.read_csv(out / "controls" / "manifest.csv")
    device = base.device_arg("cpu")
    embedding_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for dataset in args.datasets.split(","):
        subset = manifest[manifest.dataset == dataset].copy().reset_index(drop=True)
        by_feature = {feature: load_control_arrays(root, out, subset, feature) for feature in FEATURES}
        source_values = sorted(subset.source_index.unique().tolist())
        indices = {
            control: np.asarray([
                int(subset.index[(subset.source_index == source) & (subset.control == control)][0])
                for source in source_values
            ])
            for control in ("identity", "gain_minus6db", "reverb_energy")
        }
        for feature in FEATURES:
            identity = by_feature[feature][indices["identity"]]
            for control in ("gain_minus6db", "reverb_energy"):
                changed = by_feature[feature][indices[control]]
                cosine = 1.0 - np.sum(identity * changed, axis=1) / (
                    np.linalg.norm(identity, axis=1) * np.linalg.norm(changed, axis=1) + 1e-9
                )
                relative = np.linalg.norm(changed - identity, axis=1) / (np.linalg.norm(identity, axis=1) + 1e-9)
                embedding_rows.extend({
                    "dataset": dataset,
                    "source_index": int(source),
                    "control": control,
                    "reference": "paired_identity",
                    "feature": feature,
                    "cosine_distance": float(cosine[position]),
                    "relative_l2": float(relative[position]),
                } for position, source in enumerate(source_values))

        train = fc.load_arrays(root, args.cache, dataset, "train")
        bundle = load_primary_bundle(out, dataset)
        for control in ("gain_minus6db", "reverb_energy"):
            identity_arrays = {feature: by_feature[feature][indices["identity"]] for feature in FEATURES}
            control_arrays = {feature: by_feature[feature][indices[control]] for feature in FEATURES}
            train_i, identity_primary = transform_egemaps(train, identity_arrays)
            train_c, control_primary = transform_egemaps(train, control_arrays)
            p_identity = v2.solve_np(root, train_i, identity_primary, bundle["y_train"], device, args.alpha)
            p_control = v2.solve_np(root, train_c, control_primary, bundle["y_train"], device, args.alpha)
            for spec in ("basic", "core", "full"):
                mask = fc.MASKS[spec]
                idx = [FEATURES.index(feature) for feature in fc.names(mask)]
                weights = fc.fit_simplex(bundle["oof"][:, idx], bundle["y_train"])
                identity_prediction = p_identity[:, idx] @ weights
                control_prediction = p_control[:, idx] @ weights
                prediction_rows.extend({
                    "dataset": dataset,
                    "source_index": int(source),
                    "control": control,
                    "reference": "paired_identity",
                    "spec": spec,
                    "prediction_identity": float(identity_prediction[position]),
                    "prediction_control": float(control_prediction[position]),
                    "prediction_delta": float(control_prediction[position] - identity_prediction[position]),
                } for position, source in enumerate(source_values))
        log(f"paired control analysis complete: {dataset}")
    embedding = pd.DataFrame(embedding_rows)
    prediction = pd.DataFrame(prediction_rows)
    embedding.to_csv(out / "control_embedding_consistency.csv", index=False)
    prediction.to_csv(out / "control_prediction_sensitivity.csv", index=False)
    embedding.groupby(["dataset", "control", "feature"], as_index=False)[["cosine_distance", "relative_l2"]].mean().to_csv(
        out / "control_embedding_summary.csv", index=False
    )
    prediction.groupby(["dataset", "control", "spec"], as_index=False).prediction_delta.agg(["mean", "median", "std", "count"]).reset_index().to_csv(
        out / "control_prediction_summary.csv", index=False
    )


def fold_concat(arrays: dict[str, np.ndarray], mask: int, fit_index: np.ndarray, hold_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    hold_parts: list[np.ndarray] = []
    for feature in fc.names(mask):
        train_part = np.asarray(arrays[feature][fit_index], dtype=np.float64)
        hold_part = np.asarray(arrays[feature][hold_index], dtype=np.float64)
        if feature == "egemaps":
            lo, hi = np.quantile(train_part, PRIMARY_BOUNDS, axis=0)
            train_part = np.clip(train_part, lo, hi)
            hold_part = np.clip(hold_part, lo, hi)
        train_parts.append(train_part)
        hold_parts.append(hold_part)
    train_matrix = np.concatenate(train_parts, axis=1)
    hold_matrix = np.concatenate(hold_parts, axis=1)
    mean, sd = train_matrix.mean(axis=0), train_matrix.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (train_matrix - mean) / sd, (hold_matrix - mean) / sd


def full_concat(train: dict[str, np.ndarray], evaluation: dict[str, np.ndarray], mask: int) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    eval_parts: list[np.ndarray] = []
    for feature in fc.names(mask):
        train_part = np.asarray(train[feature], dtype=np.float64)
        eval_part = np.asarray(evaluation[feature], dtype=np.float64)
        if feature == "egemaps":
            lo, hi = np.quantile(train_part, PRIMARY_BOUNDS, axis=0)
            train_part = np.clip(train_part, lo, hi)
            eval_part = np.clip(eval_part, lo, hi)
        train_parts.append(train_part)
        eval_parts.append(eval_part)
    train_matrix = np.concatenate(train_parts, axis=1)
    eval_matrix = np.concatenate(eval_parts, axis=1)
    mean, sd = train_matrix.mean(axis=0), train_matrix.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (train_matrix - mean) / sd, (eval_matrix - mean) / sd


def cmd_concat(args: argparse.Namespace) -> None:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    root, out = paths(args)
    alphas = [float(value) for value in args.alphas.split(",")]
    rows: list[dict[str, object]] = []
    for dataset in args.datasets.split(","):
        arrays = {split: fc.load_arrays(root, args.cache, dataset, split) for split in ("train", "val", "test")}
        targets = {split: base.load_y(root, args.cache, dataset, split) for split in ("train", "val", "test")}
        groups = base.load_metadata(root, dataset, "train").system_id.astype(str).to_numpy()
        split = list(GroupKFold(n_splits=min(args.folds, len(np.unique(groups)))).split(np.zeros(len(groups)), targets["train"], groups))
        for spec, mask in fc.MASKS.items():
            scores = {alpha: [] for alpha in alphas}
            spears = {alpha: [] for alpha in alphas}
            for fit_index, hold_index in split:
                fit_x, hold_x = fold_concat(arrays["train"], mask, fit_index, hold_index)
                for alpha in alphas:
                    model = Ridge(alpha=alpha, solver="lsqr", fit_intercept=True).fit(fit_x, targets["train"][fit_index])
                    metric = fc.metric_dict(targets["train"][hold_index], model.predict(hold_x))
                    scores[alpha].append(metric["mse"])
                    spears[alpha].append(metric["spearman"])
            chosen = min(alphas, key=lambda alpha: float(np.mean(scores[alpha])))
            train_val, val_x = full_concat(arrays["train"], arrays["val"], mask)
            train_test, test_x = full_concat(arrays["train"], arrays["test"], mask)
            val_model = Ridge(alpha=chosen, solver="lsqr", fit_intercept=True).fit(train_val, targets["train"])
            test_model = Ridge(alpha=chosen, solver="lsqr", fit_intercept=True).fit(train_test, targets["train"])
            val_metric = fc.metric_dict(targets["val"], val_model.predict(val_x))
            test_metric = fc.metric_dict(targets["test"], test_model.predict(test_x))
            rows.append({
                "dataset": dataset,
                "spec": spec,
                "mask": mask,
                "n_dimensions": int(train_test.shape[1]),
                "selected_alpha": chosen,
                "alpha_at_grid_boundary": chosen in (min(alphas), max(alphas)),
                "alpha_grid": json.dumps(alphas),
                "group_cv_mse": float(np.mean(scores[chosen])),
                "group_cv_spearman": float(np.mean(spears[chosen])),
                **{f"val_{key}": value for key, value in val_metric.items()},
                **{f"test_{key}": value for key, value in test_metric.items()},
            })
            log(f"concat {dataset}/{spec}: alpha={chosen}, boundary={chosen in (min(alphas), max(alphas))}")
    pd.DataFrame(rows).to_csv(out / "concat_baseline_groupcv.csv", index=False)


def cmd_report(args: argparse.Namespace) -> None:
    _, out = paths(args)
    files = [
        "primary_protocol.json",
        "egemaps_reextraction_audit.csv",
        "egemaps_primary_diagnostics.csv",
        "converged_selected_summary.csv",
        "fixed_model_cluster_bootstrap.csv",
        "confirmatory_tests_holm.csv",
        "oof_shapley_test_cluster_bootstrap.csv",
        "oof_core_displacement.csv",
        "control_audio_diagnostics.csv",
        "control_embedding_summary.csv",
        "control_prediction_summary.csv",
        "concat_baseline_groupcv.csv",
    ]
    sections = ["# CPU-only ICASSP submission finalization", "", f"Primary preprocessing: `{PRIMARY_POLICY}`.", ""]
    for filename in files:
        path = out / filename
        sections.append(f"## {filename}")
        sections.append("")
        if not path.exists():
            sections.append("MISSING")
        elif path.suffix == ".json":
            sections.append("```json")
            sections.append(path.read_text(encoding="utf-8"))
            sections.append("```")
        else:
            frame = pd.read_csv(path)
            sections.append(f"Rows: {len(frame)}")
            sections.append("")
            sections.append(frame.head(100).to_markdown(index=False))
        sections.append("")
    (out / "submission_final_report.md").write_text("\n".join(sections), encoding="utf-8")
    write_json(out / "completion.json", {"status": "complete", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="results/paper_submission_final_cpu")
    parser.add_argument("--cache", default="results/explainability_ridge_full/cache")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--alpha", type=float, default=10.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-primary"); add_common(p); p.add_argument("--folds", type=int, default=5); p.set_defaults(func=cmd_prepare_primary)
    p = sub.add_parser("lattice"); add_common(p); p.set_defaults(func=cmd_lattice)
    p = sub.add_parser("explain"); add_common(p); p.add_argument("--n-bootstrap", type=int, default=10000); p.add_argument("--seed", type=int, default=20260909); p.set_defaults(func=cmd_explain)
    p = sub.add_parser("bootstrap"); add_common(p); p.add_argument("--n-bootstrap", type=int, default=10000); p.add_argument("--seed", type=int, default=20260909); p.set_defaults(func=cmd_bootstrap)
    p = sub.add_parser("controls-prepare"); add_common(p); p.add_argument("--items-per-dataset", type=int, default=32); p.add_argument("--seed", type=int, default=20260909); p.set_defaults(func=cmd_controls_prepare)
    p = sub.add_parser("controls-extract"); add_common(p); p.add_argument("--feature", required=True, choices=FEATURES); p.set_defaults(func=cmd_controls_extract)
    p = sub.add_parser("controls-analyze"); add_common(p); p.set_defaults(func=cmd_controls_analyze)
    p = sub.add_parser("concat"); add_common(p); p.add_argument("--folds", type=int, default=5); p.add_argument("--alphas", default="0.1,1,10,100,1000,10000,100000,1000000,10000000"); p.set_defaults(func=cmd_concat)
    p = sub.add_parser("report"); add_common(p); p.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
