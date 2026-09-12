#!/usr/bin/env python3
"""Second-stage robustness and mechanism experiments for the ICASSP MOS paper.

The first queue produced the complete coalition table and its immediate
follow-ups.  This runner addresses the remaining reviewer-facing controls:
out-of-fold stacking, system-level uncertainty, group-disjoint evaluation,
representation redundancy, true multi-source leave-one-dataset-out transfer,
gate/data stability, real extraction timing, and valid waveform perturbations.

Every subcommand writes independent artifacts below
``results/paper_followups_v2`` and is safe to resume through the shell queue.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.stats import kendalltau, pearsonr, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_missing_experiments as base  # noqa: E402


FEATURES = base.FEATURES
DATASETS = base.DATASETS
SEEDS = base.SEEDS
BITS = base.BITS
FIXED_MASKS = base.FIXED_MASKS


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def root_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def output_root(args: argparse.Namespace) -> Path:
    out = root_path(args.root) / args.output
    out.mkdir(parents=True, exist_ok=True)
    return out


def append_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    if isinstance(rows, list):
        if not rows:
            return
        frame = pd.DataFrame(rows)
    else:
        frame = rows
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def metric_dict(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    result = base.metric_dict(y, pred)
    result["kendall"] = float(kendalltau(y, pred).statistic) if len(y) > 1 else float("nan")
    return result


def split_indices(groups: np.ndarray, n_splits: int, seed: int = 42):
    from sklearn.model_selection import GroupKFold, KFold

    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    if len(unique) >= 2 and not (len(unique) == 1 and unique[0] == "unknown"):
        folds = min(int(n_splits), len(unique))
        if folds >= 2:
            return list(GroupKFold(n_splits=folds).split(np.zeros(len(groups)), groups=groups))
    folds = min(int(n_splits), len(groups))
    if folds < 2:
        raise ValueError("At least two items are required for out-of-fold prediction")
    return list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(groups))))


def arrays_for(root: Path, cache: str, dataset: str, split: str) -> dict[str, np.ndarray]:
    return {feature: base.load_x(root, cache, dataset, split, feature) for feature in FEATURES}


def solve_np(root: Path, train: dict[str, np.ndarray], evaluate: dict[str, np.ndarray],
             y_train: np.ndarray, device, alpha: float) -> np.ndarray:
    _, solve_experts = base.m6_functions(root)
    result = solve_experts(train, evaluate, y_train, device, alpha)
    return result.detach().cpu().numpy().astype(np.float32, copy=False)


def fit_gate_np(root: Path, predictions: np.ndarray, y: np.ndarray, seeds: list[int],
                device, gate_steps: int, batch_size: int) -> np.ndarray:
    import torch

    fit_gate, _ = base.m6_functions(root)
    tensor = torch.as_tensor(predictions, dtype=torch.float32, device=device)
    weights = fit_gate(tensor, y, seeds, gate_steps, batch_size, device)
    return weights.detach().cpu().numpy().astype(np.float32, copy=False)


def oof_bundle_path(out: Path, dataset: str, alpha: float, folds: int) -> Path:
    token = str(alpha).replace(".", "p")
    return out / "oof_cache" / f"{dataset}_alpha{token}_folds{folds}.npz"


def build_oof_bundle(root: Path, out: Path, cache: str, dataset: str, alpha: float,
                     folds: int, device, force: bool = False) -> dict[str, np.ndarray]:
    path = oof_bundle_path(out, dataset, alpha, folds)
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as saved:
            return {key: saved[key] for key in saved.files}

    train = arrays_for(root, cache, dataset, "train")
    val = arrays_for(root, cache, dataset, "val")
    test = arrays_for(root, cache, dataset, "test")
    y_train = base.load_y(root, cache, dataset, "train")
    y_val = base.load_y(root, cache, dataset, "val")
    y_test = base.load_y(root, cache, dataset, "test")
    metadata = base.load_metadata(root, dataset, "train")
    groups = metadata["system_id"].astype(str).to_numpy()
    oof = np.full((len(y_train), len(FEATURES)), np.nan, dtype=np.float32)

    for fold, (fit_idx, hold_idx) in enumerate(split_indices(groups, folds)):
        fold_train = {f: np.asarray(train[f][fit_idx], dtype=np.float32) for f in FEATURES}
        fold_hold = {f: np.asarray(train[f][hold_idx], dtype=np.float32) for f in FEATURES}
        oof[hold_idx] = solve_np(root, fold_train, fold_hold, y_train[fit_idx], device, alpha)
        log(f"OOF experts {dataset} alpha={alpha} fold={fold + 1}")
    if not np.isfinite(oof).all():
        raise RuntimeError(f"Non-finite or missing OOF predictions for {dataset}")

    p_val = solve_np(root, train, val, y_train, device, alpha)
    p_test = solve_np(root, train, test, y_train, device, alpha)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        oof=oof,
        val=p_val,
        test=p_test,
        y_train=np.asarray(y_train, dtype=np.float32),
        y_val=np.asarray(y_val, dtype=np.float32),
        y_test=np.asarray(y_test, dtype=np.float32),
        groups=groups.astype("U"),
    )
    return {
        "oof": oof, "val": p_val, "test": p_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "groups": groups.astype("U"),
    }


def representative_specs() -> dict[str, int]:
    specs = {k: FIXED_MASKS[k] for k in ["basic", "core", "speech_ssl", "audio_ssl", "stable_ssl", "full"]}
    for feature in ["whisper", "contentvec12", "wavlm", "beats", "ced"]:
        specs[f"basic_plus_{feature}"] = FIXED_MASKS["basic"] | BITS[feature]
    for feature in ["whisper", "wavlm", "beats", "ced"]:
        specs[f"core_plus_{feature}"] = FIXED_MASKS["core"] | BITS[feature]
    return specs


def cmd_oof_sweep(args: argparse.Namespace) -> None:
    root = root_path(args.root)
    out = output_root(args)
    result = out / "oof_m6_results.csv"
    state_path = out / "oof_m6_state.json"
    done = set(json.loads(state_path.read_text(encoding="utf-8"))) if state_path.exists() else set()
    device = base.device_arg(args.device)
    masks = list(range(1, 1 << len(FEATURES)))
    if args.max_masks is not None:
        masks = masks[: max(1, int(args.max_masks))]
    for dataset in args.datasets.split(","):
        bundle = build_oof_bundle(root, out, args.cache, dataset, args.alpha, args.folds, device)
        for mask in masks:
            key = f"{dataset}|{mask}"
            if key in done:
                continue
            idx = [FEATURES.index(f) for f in base.names(mask)]
            weights = fit_gate_np(root, bundle["oof"][:, idx], bundle["y_train"], SEEDS,
                                  device, args.gate_steps, args.batch_size)
            rows = []
            for si, seed in enumerate(SEEDS):
                weights_json = json.dumps(dict(zip(base.names(mask), weights[si].astype(float).tolist())))
                for split in ["val", "test"]:
                    pred = bundle[split][:, idx] @ weights[si]
                    rows.append({
                        "dataset": dataset, "mask": mask, "features": "+".join(base.names(mask)),
                        "seed": seed, "split": split, "stacking": "group_oof",
                        "weights_json": weights_json, **metric_dict(bundle[f"y_{split}"], pred),
                    })
            append_csv(result, rows)
            done.add(key)
            write_json(state_path, sorted(done))
            if mask % 32 == 0 or mask == masks[-1]:
                log(f"OOF sweep {dataset} mask={mask}/{masks[-1]}")

    frame = pd.read_csv(result)
    rows = []
    for (dataset, seed), group in frame.groupby(["dataset", "seed"]):
        val = group[group["split"] == "val"]
        selected = val.loc[val["spearman"].idxmax()]
        test = group[(group["split"] == "test") & (group["mask"] == int(selected["mask"]))].iloc[0]
        rows.append({"dataset": dataset, "seed": seed, "selection": "validation_best",
                     "mask": int(selected["mask"]), "features": selected["features"],
                     "val_spearman": selected["spearman"], "test_spearman": test["spearman"],
                     "test_mse": test["mse"]})
        for spec in ["basic", "full", "stable_ssl"]:
            mask = FIXED_MASKS[spec]
            vq = val[val["mask"] == mask]
            tq = group[(group["split"] == "test") & (group["mask"] == mask)]
            if vq.empty or tq.empty:
                continue
            v = vq.iloc[0]
            t = tq.iloc[0]
            rows.append({"dataset": dataset, "seed": seed, "selection": spec, "mask": mask,
                         "features": v["features"], "val_spearman": v["spearman"],
                         "test_spearman": t["spearman"], "test_mse": t["mse"]})
    pd.DataFrame(rows).to_csv(out / "oof_selected_summary.csv", index=False)
    log(f"OOF sweep complete output={result}")


def cmd_alpha_robustness(args: argparse.Namespace) -> None:
    root = root_path(args.root)
    out = output_root(args)
    result = out / "alpha_robustness.csv"
    state_path = out / "alpha_robustness_state.json"
    done = set(json.loads(state_path.read_text(encoding="utf-8"))) if state_path.exists() else set()
    device = base.device_arg(args.device)
    alphas = [float(x) for x in args.alphas.split(",")]
    for dataset in args.datasets.split(","):
        for alpha in alphas:
            bundle = build_oof_bundle(root, out, args.cache, dataset, alpha, args.folds, device)
            for spec, mask in representative_specs().items():
                key = f"{dataset}|{alpha}|{spec}"
                if key in done:
                    continue
                idx = [FEATURES.index(f) for f in base.names(mask)]
                weights = fit_gate_np(root, bundle["oof"][:, idx], bundle["y_train"], SEEDS,
                                      device, args.gate_steps, args.batch_size)
                rows = []
                for si, seed in enumerate(SEEDS):
                    for split in ["val", "test"]:
                        pred = bundle[split][:, idx] @ weights[si]
                        rows.append({"dataset": dataset, "alpha": alpha, "spec": spec,
                                     "mask": mask, "seed": seed, "split": split,
                                     "weights_json": json.dumps(dict(zip(base.names(mask), weights[si].astype(float).tolist()))),
                                     **metric_dict(bundle[f"y_{split}"], pred)})
                append_csv(result, rows)
                done.add(key)
                write_json(state_path, sorted(done))
            log(f"alpha robustness {dataset} alpha={alpha}")
    log(f"alpha robustness complete output={result}")


def bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    if len(unique) < 2:
        return rng.integers(0, len(groups), size=len(groups))
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in sampled])


def summarize_bootstrap(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)), float(np.nanquantile(array, 0.025)), float(np.nanquantile(array, 0.975))


def ensemble_frames(predictions: pd.DataFrame, dataset: str) -> dict[str, pd.DataFrame]:
    d = predictions[predictions["dataset"] == dataset].copy()
    outputs: dict[str, pd.DataFrame] = {}
    fixed = ["basic", "core", "speech_ssl", "audio_ssl", "stable_ssl", "full"]
    keys = ["item_index", "mos", "system_id", "condition"]
    for spec in fixed:
        q = d[d["spec"] == spec]
        if q.empty:
            continue
        outputs[spec] = q.groupby(keys, as_index=False, dropna=False)["prediction"].mean()
    selected = d[d["spec"] == ("selected_" + d["seed"].astype(str))]
    if not selected.empty:
        outputs["validation_selected"] = selected.groupby(keys, as_index=False, dropna=False)["prediction"].mean()
    return outputs


def cmd_system_metrics(args: argparse.Namespace) -> None:
    root = root_path(args.root)
    out = output_root(args)
    source = root / args.predictions
    usecols = ["dataset", "spec", "seed", "item_index", "mos", "prediction", "system_id", "condition"]
    predictions = pd.read_csv(source, usecols=usecols)
    rows = []
    rng = np.random.default_rng(args.seed)
    metrics = ["mse", "mae", "pearson", "spearman", "kendall"]
    for dataset in args.datasets.split(","):
        for spec, frame in ensemble_frames(predictions, dataset).items():
            y = frame["mos"].to_numpy(float)
            pred = frame["prediction"].to_numpy(float)
            groups = frame["system_id"].fillna("unknown").astype(str).to_numpy()
            item_point = metric_dict(y, pred)
            aggregated = frame.groupby("system_id", as_index=False)[["mos", "prediction"]].mean()
            system_point = metric_dict(aggregated["mos"].to_numpy(float), aggregated["prediction"].to_numpy(float))
            y_within = y - pd.Series(y).groupby(groups).transform("mean").to_numpy()
            p_within = pred - pd.Series(pred).groupby(groups).transform("mean").to_numpy()
            within_point = metric_dict(y_within, p_within)
            boot = {(level, metric): [] for level in ["item", "system", "within_system"] for metric in metrics}
            unique_groups = np.unique(groups)
            group_members = {group: np.flatnonzero(groups == group) for group in unique_groups}
            for _ in range(args.n_bootstrap):
                if len(unique_groups) >= 2:
                    sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                    pieces = [group_members[group] for group in sampled_groups]
                    idx = np.concatenate(pieces)
                else:
                    sampled_groups = unique_groups
                    idx = rng.integers(0, len(groups), size=len(groups))
                for metric, value in metric_dict(y[idx], pred[idx]).items():
                    boot[("item", metric)].append(value)
                system_y = np.asarray([y[group_members[group]].mean() for group in sampled_groups])
                system_p = np.asarray([pred[group_members[group]].mean() for group in sampled_groups])
                for metric, value in metric_dict(system_y, system_p).items():
                    boot[("system", metric)].append(value)
                by_parts, bp_parts = [], []
                for occurrence, group in enumerate(sampled_groups):
                    members = group_members[group]
                    by_parts.append(y[members] - y[members].mean())
                    bp_parts.append(pred[members] - pred[members].mean())
                by = np.concatenate(by_parts) if by_parts else y[idx] - y[idx].mean()
                bp = np.concatenate(bp_parts) if bp_parts else pred[idx] - pred[idx].mean()
                for metric, value in metric_dict(by, bp).items():
                    boot[("within_system", metric)].append(value)
            points = {"item": item_point, "system": system_point, "within_system": within_point}
            for level in points:
                for metric in metrics:
                    _, low, high = summarize_bootstrap(boot[(level, metric)])
                    rows.append({"dataset": dataset, "spec": spec, "level": level, "metric": metric,
                                 "estimate": points[level][metric], "ci95_low": low, "ci95_high": high,
                                 "n_items": len(frame), "n_systems": len(np.unique(groups)),
                                 "ensemble_seeds": len(SEEDS), "n_bootstrap": args.n_bootstrap})
            log(f"system metrics {dataset}/{spec}")
    pd.DataFrame(rows).to_csv(out / "system_level_bootstrap.csv", index=False)
    log("system metrics complete")


def concat_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {f: np.concatenate([part[f] for part in parts], axis=0) for f in FEATURES}


def nested_group_predictions(root: Path, arrays: dict[str, np.ndarray], y: np.ndarray,
                             groups: np.ndarray, outer_train: np.ndarray, outer_test: np.ndarray,
                             device, alpha: float, inner_folds: int) -> tuple[np.ndarray, np.ndarray]:
    inner_groups = groups[outer_train]
    inner_oof = np.full((len(outer_train), len(FEATURES)), np.nan, dtype=np.float32)
    for fit_local, hold_local in split_indices(inner_groups, inner_folds):
        fit_idx = outer_train[fit_local]
        hold_idx = outer_train[hold_local]
        train = {f: np.asarray(arrays[f][fit_idx], dtype=np.float32) for f in FEATURES}
        hold = {f: np.asarray(arrays[f][hold_idx], dtype=np.float32) for f in FEATURES}
        inner_oof[hold_local] = solve_np(root, train, hold, y[fit_idx], device, alpha)
    train = {f: np.asarray(arrays[f][outer_train], dtype=np.float32) for f in FEATURES}
    test = {f: np.asarray(arrays[f][outer_test], dtype=np.float32) for f in FEATURES}
    outer_pred = solve_np(root, train, test, y[outer_train], device, alpha)
    return inner_oof, outer_pred


def cmd_group_ood(args: argparse.Namespace) -> None:
    root = root_path(args.root)
    out = output_root(args)
    result = out / "group_ood_predictions.csv"
    state_path = out / "group_ood_state.json"
    done = set(json.loads(state_path.read_text(encoding="utf-8"))) if state_path.exists() else set()
    device = base.device_arg(args.device)
    specs = {k: FIXED_MASKS[k] for k in ["basic", "core", "speech_ssl", "audio_ssl", "stable_ssl", "full"]}
    for dataset in args.datasets.split(","):
        train = arrays_for(root, args.cache, dataset, "train")
        val = arrays_for(root, args.cache, dataset, "val")
        arrays = {f: np.concatenate([train[f], val[f]], axis=0) for f in FEATURES}
        y = np.concatenate([base.load_y(root, args.cache, dataset, "train"), base.load_y(root, args.cache, dataset, "val")])
        meta = pd.concat([base.load_metadata(root, dataset, "train"), base.load_metadata(root, dataset, "val")], ignore_index=True)
        meta["global_index"] = np.arange(len(meta))
        for group_type in ["system_id", "condition"]:
            groups = meta[group_type].fillna("unknown").astype(str).to_numpy()
            unique = np.unique(groups)
            if len(unique) < 2 or (len(unique) == 1 and unique[0] == "unknown"):
                log(f"group OOD skip {dataset}/{group_type}: no usable groups")
                continue
            outer_splits = split_indices(groups, args.outer_folds)
            for fold, (outer_train, outer_test) in enumerate(outer_splits):
                expected = {f"{dataset}|{group_type}|{fold}|{spec}" for spec in specs}
                if expected.issubset(done):
                    continue
                inner_oof, outer_experts = nested_group_predictions(
                    root, arrays, y, groups, outer_train, outer_test,
                    device, args.alpha, args.inner_folds,
                )
                for spec, mask in specs.items():
                    key = f"{dataset}|{group_type}|{fold}|{spec}"
                    if key in done:
                        continue
                    idx = [FEATURES.index(f) for f in base.names(mask)]
                    weights = fit_gate_np(root, inner_oof[:, idx], y[outer_train], SEEDS,
                                          device, args.gate_steps, args.batch_size)
                    block = meta.iloc[outer_test][["global_index", "filepath", "system_id", "condition"]].copy()
                    block["dataset"] = dataset
                    block["group_type"] = group_type
                    block["outer_fold"] = fold
                    block["spec"] = spec
                    block["mask"] = mask
                    block["mos"] = y[outer_test]
                    for si, seed in enumerate(SEEDS):
                        frame = block.copy()
                        frame["seed"] = seed
                        frame["prediction"] = outer_experts[:, idx] @ weights[si]
                        frame["weights_json"] = json.dumps(dict(zip(base.names(mask), weights[si].astype(float).tolist())))
                        append_csv(result, frame)
                    done.add(key)
                    write_json(state_path, sorted(done))
                log(f"group OOD {dataset}/{group_type} fold={fold + 1}/{len(outer_splits)}")

    frame = pd.read_csv(result)
    ensemble = frame.groupby(
        ["dataset", "group_type", "spec", "global_index", "mos", "system_id", "condition"],
        as_index=False, dropna=False,
    )["prediction"].mean()
    rows = []
    for (dataset, group_type, spec), group in ensemble.groupby(["dataset", "group_type", "spec"]):
        metrics = metric_dict(group["mos"].to_numpy(float), group["prediction"].to_numpy(float))
        rows.append({"dataset": dataset, "group_type": group_type, "spec": spec,
                     "n_items": len(group), "n_groups": group[group_type].nunique(), **metrics})
    pd.DataFrame(rows).to_csv(out / "group_ood_summary.csv", index=False)
    log("group OOD complete")


def random_project(x: np.ndarray, dimension: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = x / np.where(std < 1e-6, 1.0, std)
    if x.shape[1] <= dimension:
        return x
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / math.sqrt(dimension), size=(x.shape[1], dimension)).astype(np.float32)
    return x @ projection


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denominator = float(np.linalg.norm(xx) * np.linalg.norm(yy))
    return float(np.linalg.norm(cross) ** 2 / denominator) if denominator > 0 else float("nan")


def cmd_redundancy(args: argparse.Namespace) -> None:
    root = root_path(args.root)
    out = output_root(args)
    device = base.device_arg(args.device)
    rows, displacement = [], []
    ssl_features = ["whisper", "wavlm", "beats", "ced"]
    core_features = ["contentvec12", "speaker", "rmvpe_cont", "rmvpe_quant"]
    m6 = base.load_m6(root, args.m6_csv)
    for di, dataset in enumerate(args.datasets.split(",")):
        arrays = arrays_for(root, args.cache, dataset, "train")
        n = min(args.max_items, len(next(iter(arrays.values()))))
        rng = np.random.default_rng(args.seed + di)
        chosen = np.sort(rng.choice(len(next(iter(arrays.values()))), size=n, replace=False))
        projected = {f: random_project(arrays[f][chosen], args.projection_dim,
                                       args.seed + di * 100 + FEATURES.index(f)) for f in FEATURES}
        bundle = build_oof_bundle(root, out, args.cache, dataset, args.alpha, args.folds, device)
        pred_corr = np.corrcoef(bundle["oof"], rowvar=False)
        for i, feature_a in enumerate(FEATURES):
            for j in range(i + 1, len(FEATURES)):
                feature_b = FEATURES[j]
                rows.append({"dataset": dataset, "feature_a": feature_a, "feature_b": feature_b,
                             "n_items": n, "projection_dim": args.projection_dim,
                             "linear_cka_approx": linear_cka(projected[feature_a], projected[feature_b]),
                             "oof_expert_prediction_correlation": float(pred_corr[i, j])})

        q = m6[(m6["dataset"] == dataset) & (m6["regime"] == f"within_{dataset}") & (m6["split"] == "val")]
        for ssl in ssl_features:
            child_mask = FIXED_MASKS["core"] | BITS[ssl]
            core = q[q["mask"] == FIXED_MASKS["core"]]
            child = q[q["mask"] == child_mask]
            if core.empty or child.empty:
                continue
            core_weights = []
            child_weights = []
            for row in core.itertuples():
                core_weights.append(dict(zip(base.names(FIXED_MASKS["core"]), json.loads(row.weights))))
            for row in child.itertuples():
                child_weights.append(dict(zip(base.names(child_mask), json.loads(row.weights))))
            for old in core_features:
                deltas = [cw.get(old, 0.0) - bw.get(old, 0.0) for bw, cw in zip(core_weights, child_weights)]
                a, b = sorted([ssl, old], key=FEATURES.index)
                pair = next(r for r in rows if r["dataset"] == dataset and r["feature_a"] == a and r["feature_b"] == b)
                displacement.append({"dataset": dataset, "added_ssl": ssl, "baseline_feature": old,
                                     "mean_gate_displacement": float(np.mean(deltas)),
                                     "linear_cka_approx": pair["linear_cka_approx"],
                                     "oof_expert_prediction_correlation": pair["oof_expert_prediction_correlation"]})
        log(f"redundancy {dataset}")
    pd.DataFrame(rows).to_csv(out / "representation_redundancy.csv", index=False)
    disp = pd.DataFrame(displacement)
    disp.to_csv(out / "cka_gate_displacement.csv", index=False)
    if len(disp) >= 3:
        summary = {
            "cka_vs_absolute_displacement_spearman": float(spearmanr(disp["linear_cka_approx"], disp["mean_gate_displacement"].abs()).statistic),
            "prediction_corr_vs_absolute_displacement_spearman": float(spearmanr(disp["oof_expert_prediction_correlation"].abs(), disp["mean_gate_displacement"].abs()).statistic),
            "n_pairs": int(len(disp)),
        }
        write_json(out / "redundancy_summary.json", summary)
    log("redundancy complete")


def cmd_true_lodo(args: argparse.Namespace) -> None:
    root = root_path(args.root)
    out = output_root(args)
    result = out / "true_lodo_results.csv"
    state_path = out / "true_lodo_state.json"
    done = set(json.loads(state_path.read_text(encoding="utf-8"))) if state_path.exists() else set()
    device = base.device_arg(args.device)
    specs = {k: FIXED_MASKS[k] for k in ["basic", "core", "speech_ssl", "audio_ssl", "stable_ssl", "full"]}
    datasets = args.datasets.split(",")
    for target in datasets:
        sources = [d for d in datasets if d != target]
        source_arrays = {d: arrays_for(root, args.cache, d, "train") for d in sources}
        source_y = {d: base.load_y(root, args.cache, d, "train") for d in sources}
        offsets: dict[str, slice] = {}
        start = 0
        for source in sources:
            offsets[source] = slice(start, start + len(source_y[source]))
            start += len(source_y[source])
        domain_oof = np.full((start, len(FEATURES)), np.nan, dtype=np.float32)
        y_oof = np.concatenate([source_y[d] for d in sources])
        for held_source in sources:
            fit_sources = [d for d in sources if d != held_source]
            fit_arrays = concat_parts([source_arrays[d] for d in fit_sources])
            fit_y = np.concatenate([source_y[d] for d in fit_sources])
            domain_oof[offsets[held_source]] = solve_np(
                root, fit_arrays, source_arrays[held_source], fit_y, device, args.alpha,
            )
            log(f"true LODO target={target} source-OOF={held_source}")
        final_train = concat_parts([source_arrays[d] for d in sources])
        final_y = np.concatenate([source_y[d] for d in sources])
        target_arrays = arrays_for(root, args.cache, target, "test")
        target_y = base.load_y(root, args.cache, target, "test")
        target_experts = solve_np(root, final_train, target_arrays, final_y, device, args.alpha)
        for spec, mask in specs.items():
            key = f"{target}|{spec}"
            if key in done:
                continue
            idx = [FEATURES.index(f) for f in base.names(mask)]
            weights = fit_gate_np(root, domain_oof[:, idx], y_oof, SEEDS,
                                  device, args.gate_steps, args.batch_size)
            rows = []
            for si, seed in enumerate(SEEDS):
                pred = target_experts[:, idx] @ weights[si]
                rows.append({"target": target, "sources": "+".join(sources), "spec": spec,
                             "mask": mask, "seed": seed, "gate_training": "leave_one_source_domain_out",
                             "weights_json": json.dumps(dict(zip(base.names(mask), weights[si].astype(float).tolist()))),
                             **metric_dict(target_y, pred)})
            append_csv(result, rows)
            done.add(key)
            write_json(state_path, sorted(done))
        log(f"true LODO target={target} complete")
    log(f"true LODO complete output={result}")


def simplex_weights(predictions: np.ndarray, y: np.ndarray, entropy_lambda: float = 0.0) -> np.ndarray:
    k = predictions.shape[1]
    initial = np.full(k, 1.0 / k)

    def objective(w):
        mse = np.mean((y - predictions @ w) ** 2)
        if entropy_lambda <= 0:
            return float(mse)
        safe = np.clip(w, 1e-12, 1.0)
        kl_uniform = np.sum(safe * np.log(safe * k))
        return float(mse + entropy_lambda * kl_uniform)

    result = minimize(objective, initial, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                      constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                      options={"maxiter": 500, "ftol": 1e-10})
    if not result.success:
        raise RuntimeError(f"Simplex optimizer failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def cmd_gate_robustness(args: argparse.Namespace) -> None:
    from sklearn.linear_model import RidgeCV

    root = root_path(args.root)
    out = output_root(args)
    alternatives, boot_rows = [], []
    specs = {k: FIXED_MASKS[k] for k in ["basic", "core", "speech_ssl", "audio_ssl", "stable_ssl", "full"]}
    entropy_grid = [float(x) for x in args.entropy_lambdas.split(",")]
    rng = np.random.default_rng(args.seed)
    for dataset in args.datasets.split(","):
        bundle = build_oof_bundle(root, out, args.cache, dataset, args.alpha, args.folds,
                                  base.device_arg(args.device))
        for spec, mask in specs.items():
            idx = [FEATURES.index(f) for f in base.names(mask)]
            p, pv, pt = bundle["oof"][:, idx], bundle["val"][:, idx], bundle["test"][:, idx]
            y, yv, yt = bundle["y_train"], bundle["y_val"], bundle["y_test"]
            candidates: dict[str, tuple[np.ndarray, float]] = {}
            candidates["equal"] = (np.full(len(idx), 1.0 / len(idx)), 0.0)
            candidates["simplex"] = (simplex_weights(p, y), 0.0)
            nnls_w, _ = nnls(p, y)
            candidates["nnls"] = (nnls_w, 0.0)
            ridge = RidgeCV(alphas=np.logspace(-4, 4, 17), fit_intercept=True).fit(p, y)
            candidates["ridge_stacking"] = (ridge.coef_.astype(float), float(ridge.intercept_))
            entropy_trials = []
            for lam in entropy_grid:
                w = simplex_weights(p, y, lam)
                entropy_trials.append((metric_dict(yv, pv @ w)["mse"], lam, w))
            _, chosen_lambda, chosen_weights = min(entropy_trials, key=lambda item: item[0])
            candidates["entropy_simplex"] = (chosen_weights, 0.0)
            for mode, (weights, intercept) in candidates.items():
                mv = metric_dict(yv, pv @ weights + intercept)
                mt = metric_dict(yt, pt @ weights + intercept)
                alternatives.append({"dataset": dataset, "spec": spec, "mode": mode,
                                     "selected_entropy_lambda": chosen_lambda if mode == "entropy_simplex" else np.nan,
                                     "intercept": intercept,
                                     "weights_json": json.dumps(dict(zip(base.names(mask), weights.astype(float).tolist()))),
                                     **{f"val_{k}": v for k, v in mv.items()},
                                     **{f"test_{k}": v for k, v in mt.items()}})

            groups = bundle["groups"].astype(str)
            for bootstrap in range(args.n_gate_bootstrap):
                sample = bootstrap_indices(groups, rng)
                weights = simplex_weights(p[sample], y[sample])
                mt = metric_dict(yt, pt @ weights)
                boot_rows.append({"dataset": dataset, "spec": spec, "bootstrap": bootstrap,
                                  "weights_json": json.dumps(dict(zip(base.names(mask), weights.astype(float).tolist()))),
                                  **{f"test_{k}": v for k, v in mt.items()}})
            log(f"gate robustness {dataset}/{spec}")
    pd.DataFrame(alternatives).to_csv(out / "gate_alternatives_oof.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(out / "gate_data_bootstrap.csv", index=False)
    log("gate robustness complete")


def dataset_audio_records(root: Path, dataset: str, split: str = "test") -> tuple[pd.DataFrame, list[Path]]:
    sys.path.insert(0, str(root))
    import extract_explainability_embeddings as extractor

    _, frame, _, absolute = extractor.load_records(dataset, split)
    return frame, absolute


def stratified_indices(frame: pd.DataFrame, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    group_col = "system_id" if "system_id" in frame and frame["system_id"].nunique() > 1 else None
    chosen: list[int] = []
    if group_col:
        groups = list(frame.groupby(group_col, sort=True).groups.values())
        rng.shuffle(groups)
        for group in groups:
            chosen.append(int(rng.choice(np.asarray(list(group), dtype=int))))
            if len(chosen) >= n:
                break
    remaining = np.setdiff1d(np.arange(len(frame)), np.asarray(chosen, dtype=int))
    if len(chosen) < n and len(remaining):
        extra = rng.choice(remaining, size=min(n - len(chosen), len(remaining)), replace=False)
        chosen.extend(int(x) for x in extra)
    return sorted(chosen[:n])


def configure_pseudo_extractor(root: Path, pseudo_name: str, pseudo_root: Path,
                               csv_name: str, embeddings_root: Path):
    sys.path.insert(0, str(root))
    import extract_explainability_embeddings as extractor

    extractor.DATASETS[pseudo_name] = {
        "root": pseudo_root,
        "csv": csv_name,
        "path_column": "filepath",
        "output_name": pseudo_name,
    }
    extractor.EMBEDDINGS_ROOT = embeddings_root
    return extractor


def cmd_latency(args: argparse.Namespace) -> None:
    import torch

    root = root_path(args.root)
    out = output_root(args)
    benchmark = out / "latency"
    input_root = benchmark / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    records = []
    for di, dataset in enumerate(args.datasets.split(",")):
        frame, paths = dataset_audio_records(root, dataset)
        for index in stratified_indices(frame, args.items_per_dataset, args.seed + di):
            destination = input_root / f"{dataset}_{index}{paths[index].suffix.lower()}"
            if not destination.exists():
                destination.symlink_to(paths[index])
            records.append({"filepath": destination.name, "mos": float(frame.iloc[index].get("mos", np.nan)),
                            "source_dataset": dataset, "source_index": index})
    pd.DataFrame(records).to_csv(input_root / "test.csv", index=False)
    pseudo = "paper_latency"
    extractor = configure_pseudo_extractor(root, pseudo, input_root, "{split}.csv", benchmark / "embeddings")
    result = out / "extraction_latency.csv"
    if result.exists():
        previous = pd.read_csv(result)
        done = set(previous.loc[previous["status"].astype(str).eq("ok"), "feature"].astype(str))
    else:
        done = set()
    features = args.features.split(",")
    for feature in features:
        if feature in done:
            continue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        status, error = "ok", ""
        try:
            report = extractor.extract_one(pseudo, "test", feature, SimpleNamespace(
                device=args.device, whisper_model="whisper-large-v3", deep_audit_after=False,
            ))
            if report.get("status") == "error":
                status = "error"
                error = str(report.get("error", "extractor returned an error"))
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        audio_seconds = 0.0
        for record in records:
            try:
                info = __import__("torchaudio").info(str(input_root / record["filepath"]))
                audio_seconds += float(info.num_frames) / float(info.sample_rate)
            except Exception:
                pass
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        append_csv(result, [{"feature": feature, "status": status, "error": error,
                             "n_items": len(records), "audio_seconds": audio_seconds,
                             "wall_seconds": elapsed, "seconds_per_item": elapsed / max(len(records), 1),
                             "real_time_factor": elapsed / max(audio_seconds, 1e-9),
                             "peak_gpu_memory_bytes": peak, "includes_model_load": True}])
        log(f"latency {feature} status={status} seconds={elapsed:.2f}")
    log("latency benchmark complete")


def make_reverb(waveform, sample_rate: int, seed: int):
    import torch
    import torch.nn.functional as functional

    rng = torch.Generator().manual_seed(seed)
    length = int(0.4 * sample_rate)
    times = torch.arange(length, dtype=torch.float32) / sample_rate
    tail = torch.randn(length, generator=rng) * torch.exp(-6.9 * times / 0.4)
    impulse = 0.12 * tail
    impulse[0] += 1.0
    impulse /= impulse.abs().sum().clamp_min(1e-6)
    result = functional.conv1d(waveform[None, None], impulse.flip(0)[None, None], padding=length - 1).squeeze()[: waveform.numel()]
    return result


def safe_peak(waveform):
    peak = float(waveform.abs().max())
    return waveform * (0.98 / peak) if peak > 0.98 else waveform


def cmd_perturb_prepare(args: argparse.Namespace) -> None:
    import torch
    import torchaudio

    root = root_path(args.root)
    out = output_root(args)
    perturb_root = out / "perturbation"
    audio_root = perturb_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_path = perturb_root / "manifest.csv"
    error_path = perturb_root / "generation_errors.csv"
    completed = set()
    existing_rows: list[dict] = []
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        completed = set(old["key"].astype(str))
        existing_rows = old.to_dict("records")
    new_rows, errors = [], []
    perturbations = ["noise_20db", "lowpass_4khz", "reverb_rt60_0p4", "pitch_up_2st", "tempo_0p9", "opus_24k"]
    for di, dataset in enumerate(args.datasets.split(",")):
        frame, paths = dataset_audio_records(root, dataset)
        for index in stratified_indices(frame, args.items_per_dataset, args.seed + di):
            waveform = None
            try:
                sys.path.insert(0, str(root))
                import extract_explainability_embeddings as extractor
                waveform = extractor._load_mono_16k(paths[index])
            except Exception as exc:
                errors.append({"dataset": dataset, "source_index": index, "perturbation": "load",
                               "error": f"{type(exc).__name__}: {exc}"})
                continue
            for pi, perturbation in enumerate(perturbations):
                key = f"{dataset}|{index}|{perturbation}"
                if key in completed:
                    continue
                destination = audio_root / dataset / perturbation / f"item_{index}.wav"
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if perturbation == "noise_20db":
                        generator = torch.Generator().manual_seed(args.seed + di * 100000 + index)
                        noise = torch.randn(waveform.shape, generator=generator)
                        signal_rms = waveform.pow(2).mean().sqrt().clamp_min(1e-8)
                        noise_rms = noise.pow(2).mean().sqrt().clamp_min(1e-8)
                        changed = waveform + noise * (signal_rms / noise_rms) * (10.0 ** (-20.0 / 20.0))
                    elif perturbation == "lowpass_4khz":
                        changed = torchaudio.functional.lowpass_biquad(waveform, 16000, 4000.0)
                    elif perturbation == "reverb_rt60_0p4":
                        changed = make_reverb(waveform, 16000, args.seed + index)
                    elif perturbation == "pitch_up_2st":
                        changed = torchaudio.functional.pitch_shift(waveform, 16000, 2.0)
                    elif perturbation == "tempo_0p9":
                        source = destination.with_suffix(".source.wav")
                        torchaudio.save(str(source), waveform.unsqueeze(0), 16000)
                        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(source),
                                        "-filter:a", "atempo=0.9", str(destination)], check=True)
                        source.unlink(missing_ok=True)
                        changed = None
                    elif perturbation == "opus_24k":
                        source = destination.with_suffix(".source.wav")
                        encoded = destination.with_suffix(".opus")
                        torchaudio.save(str(source), waveform.unsqueeze(0), 16000)
                        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(source),
                                        "-c:a", "libopus", "-b:a", "24k", str(encoded)], check=True)
                        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(encoded),
                                        "-ar", "16000", "-ac", "1", str(destination)], check=True)
                        source.unlink(missing_ok=True)
                        encoded.unlink(missing_ok=True)
                        changed = None
                    else:
                        raise ValueError(perturbation)
                    if changed is not None:
                        torchaudio.save(str(destination), safe_peak(changed).unsqueeze(0), 16000)
                    rel = destination.relative_to(perturb_root)
                    new_rows.append({"key": key, "filepath": str(rel), "dataset": dataset,
                                     "source_index": index, "source_filepath": str(paths[index]),
                                     "system_id": str(frame.iloc[index].get("system_id", "unknown")),
                                     "condition": str(frame.iloc[index].get("condition", "unknown")),
                                     "mos": float(frame.iloc[index].get("mos", np.nan)),
                                     "perturbation": perturbation, "sample_rate": 16000})
                    completed.add(key)
                    if len(new_rows) % 25 == 0:
                        pd.DataFrame(existing_rows + new_rows).to_csv(manifest_path, index=False)
                except Exception as exc:
                    errors.append({"dataset": dataset, "source_index": index, "perturbation": perturbation,
                                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        log(f"perturbation generation {dataset}")
    pd.DataFrame(existing_rows + new_rows).drop_duplicates("key").to_csv(manifest_path, index=False)
    if errors:
        append_csv(error_path, errors)
    log(f"perturbation manifest rows={len(pd.read_csv(manifest_path))} errors={len(errors)}")


def cmd_perturb_extract(args: argparse.Namespace) -> None:
    root = root_path(args.root)
    out = output_root(args)
    perturb_root = out / "perturbation"
    manifest = perturb_root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    pseudo = "paper_perturb"
    extractor = configure_pseudo_extractor(root, pseudo, perturb_root, "manifest.csv", perturb_root / "embeddings")
    report = extractor.extract_one(pseudo, "test", args.feature, SimpleNamespace(
        device=args.device, whisper_model="whisper-large-v3", deep_audit_after=False,
    ))
    write_json(perturb_root / f"extract_{args.feature}.json", report)
    if report.get("status") == "error":
        raise RuntimeError(report.get("error", f"Extraction failed for {args.feature}"))
    log(f"perturb extraction complete feature={args.feature}")


def cmd_perturb_analyze(args: argparse.Namespace) -> None:
    import torch

    root = root_path(args.root)
    out = output_root(args)
    perturb_root = out / "perturbation"
    manifest = pd.read_csv(perturb_root / "manifest.csv")
    sys.path.insert(0, str(root))
    from run_new_explainability_ridge import pool_tensor

    device = base.device_arg(args.device)
    result_rows, summary_rows = [], []
    directory_names = {
        "whisper": "whisper", "contentvec12": "contentvec", "wavlm": "wavlm_final",
        "beats": "beats", "auditory_erb": "auditory_erb", "speaker": "speaker",
        "rmvpe_cont": "f0_rmvpe", "rmvpe_quant": "f0_rmvpe_quant", "ced": "ced", "egemaps": "egemaps",
    }
    for dataset in args.datasets.split(","):
        q = manifest[manifest["dataset"] == dataset].copy().reset_index(drop=True)
        if q.empty:
            continue
        train = arrays_for(root, args.cache, dataset, "train")
        y_train = base.load_y(root, args.cache, dataset, "train")
        original_test = arrays_for(root, args.cache, dataset, "test")
        original_idx = q["source_index"].astype(int).to_numpy()
        pert_arrays: dict[str, np.ndarray] = {}
        available = []
        for feature in FEATURES:
            paths = [perturb_root / "embeddings" / "paper_perturb" / "test" / directory_names[feature] /
                     Path(path).with_suffix(".pt") for path in q["filepath"].astype(str)]
            if not all(path.exists() for path in paths):
                log(f"perturb analyze missing feature={feature} dataset={dataset}")
                continue
            pert_arrays[feature] = np.stack([pool_tensor(str(path), feature) for path in paths]).astype(np.float32)
            available.append(feature)
        if not available:
            continue
        eval_arrays = {feature: pert_arrays.get(feature, np.asarray(original_test[feature][original_idx], dtype=np.float32)) for feature in FEATURES}
        original_arrays = {feature: np.asarray(original_test[feature][original_idx], dtype=np.float32) for feature in FEATURES}
        pert_experts = solve_np(root, train, eval_arrays, y_train, device, args.alpha)
        original_experts = solve_np(root, train, original_arrays, y_train, device, args.alpha)
        for fi, feature in enumerate(FEATURES):
            if feature not in available:
                continue
            a = original_arrays[feature]
            b = pert_arrays[feature]
            cosine = 1.0 - np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
            for row_index, row in q.iterrows():
                result_rows.append({"dataset": dataset, "source_index": int(row["source_index"]),
                                    "system_id": row["system_id"], "perturbation": row["perturbation"],
                                    "entity_type": "expert", "entity": feature,
                                    "cosine_distance": float(cosine[row_index]),
                                    "prediction_original": float(original_experts[row_index, fi]),
                                    "prediction_perturbed": float(pert_experts[row_index, fi]),
                                    "prediction_delta": float(pert_experts[row_index, fi] - original_experts[row_index, fi])})

        bundle = build_oof_bundle(root, out, args.cache, dataset, args.alpha, args.folds, device)
        for spec in ["basic", "stable_ssl", "full"]:
            mask = FIXED_MASKS[spec]
            names = base.names(mask)
            idx = [FEATURES.index(f) for f in names]
            weights = simplex_weights(bundle["oof"][:, idx], bundle["y_train"])
            po = original_experts[:, idx] @ weights
            pp = pert_experts[:, idx] @ weights
            for row_index, row in q.iterrows():
                result_rows.append({"dataset": dataset, "source_index": int(row["source_index"]),
                                    "system_id": row["system_id"], "perturbation": row["perturbation"],
                                    "entity_type": "model", "entity": spec, "cosine_distance": np.nan,
                                    "prediction_original": float(po[row_index]),
                                    "prediction_perturbed": float(pp[row_index]),
                                    "prediction_delta": float(pp[row_index] - po[row_index])})
        log(f"perturb analysis {dataset}")
    result = pd.DataFrame(result_rows)
    result.to_csv(perturb_root / "perturbation_sensitivity.csv", index=False)
    if not result.empty:
        summary = result.groupby(["dataset", "perturbation", "entity_type", "entity"], as_index=False).agg(
            n=("prediction_delta", "size"), mean_prediction_delta=("prediction_delta", "mean"),
            median_prediction_delta=("prediction_delta", "median"),
            mean_cosine_distance=("cosine_distance", "mean"),
        )
        summary.to_csv(perturb_root / "perturbation_summary.csv", index=False)
    log("perturbation analysis complete")


def cmd_report(args: argparse.Namespace) -> None:
    out = output_root(args)
    sections = ["# ICASSP robustness queue report\n",
                "Generated evidence index. Statistical interpretation belongs in the manuscript.\n"]
    files = [
        "oof_selected_summary.csv", "alpha_robustness.csv", "system_level_bootstrap.csv",
        "group_ood_summary.csv", "representation_redundancy.csv", "cka_gate_displacement.csv",
        "true_lodo_results.csv", "gate_alternatives_oof.csv", "gate_data_bootstrap.csv",
        "extraction_latency.csv", "perturbation/perturbation_summary.csv",
    ]
    for relative in files:
        path = out / relative
        sections.append(f"\n## {relative}\n")
        if not path.exists():
            sections.append("Missing or failed. See queue_errors.tsv.\n")
            continue
        frame = pd.read_csv(path)
        sections.append(f"Rows: {len(frame)}.\n\n")
        sections.append(frame.head(80).to_markdown(index=False) + "\n")
    (out / "followup_v2_report.md").write_text("\n".join(sections), encoding="utf-8")
    log("v2 report complete")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="results/paper_followups_v2")
    parser.add_argument("--cache", default="results/explainability_ridge_full/cache")
    parser.add_argument("--datasets", default=",".join(DATASETS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("oof-sweep"); add_common(p)
    p.add_argument("--device", default="cuda"); p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--folds", type=int, default=5); p.add_argument("--gate-steps", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=4096); p.add_argument("--max-masks", type=int, default=None)
    p.set_defaults(func=cmd_oof_sweep)

    p = sub.add_parser("alpha-robustness"); add_common(p)
    p.add_argument("--device", default="cuda"); p.add_argument("--alphas", default="0.1,1,10,100,1000")
    p.add_argument("--folds", type=int, default=5); p.add_argument("--gate-steps", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=4096); p.set_defaults(func=cmd_alpha_robustness)

    p = sub.add_parser("system-metrics"); add_common(p)
    p.add_argument("--predictions", default="results/paper_missing/item_predictions.csv")
    p.add_argument("--n-bootstrap", type=int, default=2000); p.add_argument("--seed", type=int, default=20260908)
    p.set_defaults(func=cmd_system_metrics)

    p = sub.add_parser("group-ood"); add_common(p)
    p.add_argument("--device", default="cuda"); p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--outer-folds", type=int, default=5); p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--gate-steps", type=int, default=25); p.add_argument("--batch-size", type=int, default=4096)
    p.set_defaults(func=cmd_group_ood)

    p = sub.add_parser("redundancy"); add_common(p)
    p.add_argument("--device", default="cuda"); p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--folds", type=int, default=5); p.add_argument("--max-items", type=int, default=1024)
    p.add_argument("--projection-dim", type=int, default=192); p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--m6-csv", default="results/explainability_m6_gpu_sequential/m6_results.csv")
    p.set_defaults(func=cmd_redundancy)

    p = sub.add_parser("true-lodo"); add_common(p)
    p.add_argument("--device", default="cuda"); p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--gate-steps", type=int, default=25); p.add_argument("--batch-size", type=int, default=4096)
    p.set_defaults(func=cmd_true_lodo)

    p = sub.add_parser("gate-robustness"); add_common(p)
    p.add_argument("--device", default="cuda"); p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--folds", type=int, default=5); p.add_argument("--entropy-lambdas", default="0,0.0001,0.001,0.01,0.1")
    p.add_argument("--n-gate-bootstrap", type=int, default=200); p.add_argument("--seed", type=int, default=20260908)
    p.set_defaults(func=cmd_gate_robustness)

    p = sub.add_parser("latency"); add_common(p)
    p.add_argument("--device", default="cuda"); p.add_argument("--items-per-dataset", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--features", default=",".join(["egemaps", "auditory_erb", "rmvpe_cont", "rmvpe_quant", "speaker", "contentvec12", "wavlm", "beats", "ced", "whisper"]))
    p.set_defaults(func=cmd_latency)

    p = sub.add_parser("perturb-prepare"); add_common(p)
    p.add_argument("--items-per-dataset", type=int, default=32); p.add_argument("--seed", type=int, default=20260908)
    p.set_defaults(func=cmd_perturb_prepare)

    p = sub.add_parser("perturb-extract"); add_common(p)
    p.add_argument("--device", default="cuda"); p.add_argument("--feature", required=True, choices=FEATURES)
    p.set_defaults(func=cmd_perturb_extract)

    p = sub.add_parser("perturb-analyze"); add_common(p)
    p.add_argument("--device", default="cuda"); p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--folds", type=int, default=5); p.set_defaults(func=cmd_perturb_analyze)

    p = sub.add_parser("report"); add_common(p); p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
