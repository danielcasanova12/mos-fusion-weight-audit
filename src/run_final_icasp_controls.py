#!/usr/bin/env python3
"""Final, protocol-consistent controls for the MOS explainability paper.

The runner consumes the already extracted caches and v2 OOF bundles. It writes
new artifacts below ``results/paper_final_controls`` and never overwrites the
earlier exploratory results. All selections in this runner use train/OOF or
validation only; the official test split is used for the final estimates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_missing_experiments as base  # noqa: E402
import run_icasp_followups_v2 as v2  # noqa: E402

FEATURES = base.FEATURES
DATASETS = base.DATASETS
SEEDS = base.SEEDS
BITS = base.BITS
MASKS = {
    "basic": base.FIXED_MASKS["basic"],
    "core": base.FIXED_MASKS["core"],
    "speech_ssl": base.FIXED_MASKS["speech_ssl"],
    "audio_ssl": base.FIXED_MASKS["audio_ssl"],
    "stable_ssl": base.FIXED_MASKS["stable_ssl"],
    "full": base.FIXED_MASKS["full"],
}
for _f in ["whisper", "contentvec12", "wavlm", "beats", "ced"]:
    MASKS["basic_plus_" + _f] = MASKS["basic"] | BITS[_f]
for _f in ["whisper", "wavlm", "beats", "ced"]:
    MASKS["core_plus_" + _f] = MASKS["core"] | BITS[_f]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def root_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def out_path(args: argparse.Namespace) -> Path:
    path = root_path(args.root) / args.output
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def metric_dict(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "mse": float(np.mean((y - pred) ** 2)),
        "mae": float(np.mean(np.abs(y - pred))),
        "pearson": float(pearsonr(y, pred)[0]) if len(y) > 1 else float("nan"),
        "spearman": float(spearmanr(y, pred).statistic) if len(y) > 1 else float("nan"),
    }


def names(mask: int) -> list[str]:
    return [feature for i, feature in enumerate(FEATURES) if mask & (1 << i)]


def load_bundle(root: Path, out: Path, dataset: str, alpha: float = 10.0) -> dict[str, np.ndarray]:
    token = str(alpha).replace(".", "p")
    path = out / ".." / "paper_followups_v2" / "oof_cache" / f"{dataset}_alpha{token}_folds5.npz"
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def load_arrays(root: Path, cache: str, dataset: str, split: str) -> dict[str, np.ndarray]:
    return {f: base.load_x(root, cache, dataset, split, f) for f in FEATURES}


def fit_simplex(predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Deterministic convex gate fit; the objective is convex in the simplex."""
    predictions = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k = predictions.shape[1]
    if k == 1:
        return np.ones(1, dtype=np.float64)
    initial = np.full(k, 1.0 / k, dtype=np.float64)

    def objective(weights: np.ndarray) -> float:
        return float(np.mean((y - predictions @ weights) ** 2))

    def gradient(weights: np.ndarray) -> np.ndarray:
        return (-2.0 / len(y)) * predictions.T @ (y - predictions @ weights)

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
                     "jac": lambda w: np.ones(k, dtype=np.float64)},
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    if not result.success:
        # A projected least-squares fallback is preferable to silently dropping
        # a coalition. It starts from the same uniform simplex point.
        weights = initial.copy()
        step = 0.2 / max(float(np.linalg.norm(predictions, ord=2) ** 2), 1.0)
        for _ in range(4000):
            weights -= step * gradient(weights)
            u = np.sort(weights)[::-1]
            cssv = np.cumsum(u) - 1.0
            rho = np.nonzero(u - cssv / (np.arange(k) + 1) > 0)[0][-1]
            theta = cssv[rho] / (rho + 1.0)
            weights = np.maximum(weights - theta, 0.0)
        result.x = weights
    weights = np.asarray(result.x, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    return weights / max(float(weights.sum()), 1e-12)


def fit_mask(bundle: dict[str, np.ndarray], mask: int) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    idx = [FEATURES.index(f) for f in names(mask)]
    weights = fit_simplex(bundle["oof"][:, idx], bundle["y_train"])
    val = metric_dict(bundle["y_val"], bundle["val"][:, idx] @ weights)
    test = metric_dict(bundle["y_test"], bundle["test"][:, idx] @ weights)
    return weights, val, test


def save_state(out: Path, name: str, payload: object) -> None:
    write_json(out / f"{name}_state.json", payload)


def cmd_converged_lattice(args: argparse.Namespace) -> None:
    root, out = root_path(args.root), out_path(args)
    rows, item_rows, selected_rows = [], [], []
    for dataset in args.datasets.split(","):
        bundle = load_bundle(root, out, dataset, args.alpha)
        yv, yt = bundle["y_val"], bundle["y_test"]
        for mask in range(1, 1 << len(FEATURES)):
            weights, val, test = fit_mask(bundle, mask)
            rows.append({"dataset": dataset, "mask": mask, "features": "+".join(names(mask)),
                         "n_features": int(mask.bit_count()), "gate": "converged_simplex",
                         "weights_json": json.dumps(dict(zip(names(mask), weights.tolist()))),
                         **{f"val_{k}": v for k, v in val.items()},
                         **{f"test_{k}": v for k, v in test.items()}})
            if mask in set(MASKS.values()):
                p = bundle["test"][:, [FEATURES.index(f) for f in names(mask)]] @ weights
                item_rows.extend({"dataset": dataset, "item_index": i, "mos": float(yt[i]),
                                  "spec": next(k for k, v in MASKS.items() if v == mask),
                                  "mask": mask, "prediction": float(p[i])}
                                 for i in range(len(yt)))
            if mask % 128 == 0 or mask == 1023:
                log(f"converged lattice {dataset} mask={mask}/1023")
        frame = pd.DataFrame([r for r in rows if r["dataset"] == dataset])
        best = frame.loc[frame["val_spearman"].idxmax()]
        selected_rows.append({"dataset": dataset, "selection": "validation_best",
                              "mask": int(best["mask"]), "features": best["features"],
                              "n_features": int(best["n_features"]),
                              "val_spearman": float(best["val_spearman"]),
                              "test_spearman": float(best["test_spearman"]),
                              "test_mse": float(best["test_mse"])})
    pd.DataFrame(rows).to_csv(out / "converged_oof_lattice.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out / "converged_selected_summary.csv", index=False)
    pd.DataFrame(item_rows).to_csv(out / "converged_item_predictions.csv", index=False)
    save_state(out, "01_converged_lattice", {"rows": len(rows), "items": len(item_rows), "status": "complete"})
    log("converged lattice complete")


def random_oof_bundle(root: Path, cache: str, dataset: str, alpha: float, device, folds: int, seed: int) -> dict[str, np.ndarray]:
    from sklearn.model_selection import KFold
    train = load_arrays(root, cache, dataset, "train")
    val = load_arrays(root, cache, dataset, "val")
    test = load_arrays(root, cache, dataset, "test")
    y = base.load_y(root, cache, dataset, "train")
    yv, yt = base.load_y(root, cache, dataset, "val"), base.load_y(root, cache, dataset, "test")
    oof = np.full((len(y), len(FEATURES)), np.nan, dtype=np.float32)
    split = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fi, (fit_idx, hold_idx) in enumerate(split.split(np.zeros(len(y)))):
        fit = {f: np.asarray(train[f][fit_idx], dtype=np.float32) for f in FEATURES}
        hold = {f: np.asarray(train[f][hold_idx], dtype=np.float32) for f in FEATURES}
        oof[hold_idx] = v2.solve_np(root, fit, hold, y[fit_idx], device, alpha)
        log(f"random OOF {dataset} fold={fi + 1}/{folds}")
    pval = v2.solve_np(root, train, val, y, device, alpha)
    ptest = v2.solve_np(root, train, test, y, device, alpha)
    return {"oof": oof, "val": pval, "test": ptest, "y_train": y, "y_val": yv, "y_test": yt}


def cmd_protocol(args: argparse.Namespace) -> None:
    root, out = root_path(args.root), out_path(args)
    device = base.device_arg(args.device)
    rows = []
    specs = {k: MASKS[k] for k in ["basic", "core", "full", "stable_ssl",
                                   "basic_plus_whisper", "basic_plus_contentvec12",
                                   "basic_plus_wavlm", "basic_plus_beats", "basic_plus_ced",
                                   "core_plus_whisper", "core_plus_wavlm", "core_plus_beats", "core_plus_ced"]}
    for dataset in args.datasets.split(","):
        grouped = load_bundle(root, out, dataset, args.alpha)
        random = random_oof_bundle(root, args.cache, dataset, args.alpha, device, args.folds, args.seed)
        np.savez(out / f"random_oof_{dataset}.npz", **random)
        for protocol, bundle in [("group_system", grouped), ("random_item", random)]:
            for spec, mask in specs.items():
                weights, val, test = fit_mask(bundle, mask)
                rows.append({"dataset": dataset, "protocol": protocol, "spec": spec,
                             "mask": mask, "weights_json": json.dumps(dict(zip(names(mask), weights.tolist()))),
                             **{f"val_{k}": v for k, v in val.items()},
                             **{f"test_{k}": v for k, v in test.items()}})
        log(f"protocol comparison complete {dataset}")
    pd.DataFrame(rows).to_csv(out / "oof_protocol_compare.csv", index=False)
    save_state(out, "02_oof_protocol", {"rows": len(rows), "status": "complete"})


def transformed_arrays(train: dict[str, np.ndarray], eval_arrays: dict[str, np.ndarray],
                       feature: str, bounds: tuple[float, float] | None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    if bounds is None:
        return train, eval_arrays, {"bounds": None}
    lo, hi = np.quantile(train[feature].astype(np.float64), bounds, axis=0)
    a, b = dict(train), dict(eval_arrays)
    a[feature] = np.clip(a[feature], lo, hi).astype(np.float32)
    b[feature] = np.clip(b[feature], lo, hi).astype(np.float32)
    return a, b, {"bounds": [float(bounds[0]), float(bounds[1])], "n_clipped_train": int(np.sum((train[feature] < lo) | (train[feature] > hi)))}


def robust_oof(root: Path, cache: str, dataset: str, alpha: float, device, bounds: tuple[float, float] | None,
               folds: int) -> dict[str, np.ndarray]:
    from sklearn.model_selection import GroupKFold
    train = load_arrays(root, cache, dataset, "train")
    val = load_arrays(root, cache, dataset, "val")
    test = load_arrays(root, cache, dataset, "test")
    y = base.load_y(root, cache, dataset, "train")
    result = {"oof": np.full((len(y), len(FEATURES)), np.nan, dtype=np.float32),
              "y_train": y, "y_val": base.load_y(root, cache, dataset, "val"),
              "y_test": base.load_y(root, cache, dataset, "test")}
    meta = base.load_metadata(root, dataset, "train")
    groups = meta["system_id"].astype(str).to_numpy()
    for fi, (fit_idx, hold_idx) in enumerate(GroupKFold(n_splits=min(folds, len(np.unique(groups)))).split(np.zeros(len(y)), groups=groups)):
        fold_train, fold_hold, _ = transformed_arrays(
            {f: np.asarray(train[f][fit_idx], dtype=np.float32) for f in FEATURES},
            {f: np.asarray(train[f][hold_idx], dtype=np.float32) for f in FEATURES},
            "egemaps", bounds,
        )
        result["oof"][hold_idx] = v2.solve_np(root, fold_train, fold_hold, y[fit_idx], device, alpha)
        log(f"robust OOF {dataset} bounds={bounds} fold={fi + 1}")
    full_train, full_val, _ = transformed_arrays(train, val, "egemaps", bounds)
    full_train2, full_test, _ = transformed_arrays(train, test, "egemaps", bounds)
    result["val"] = v2.solve_np(root, full_train, full_val, y, device, alpha)
    result["test"] = v2.solve_np(root, full_train2, full_test, y, device, alpha)
    return result


def cmd_egemaps(args: argparse.Namespace) -> None:
    root, out = root_path(args.root), out_path(args)
    device = base.device_arg(args.device)
    configs = {"raw": None, "winsor_0p1": (0.001, 0.999), "winsor_0p5": (0.005, 0.995), "winsor_1p0": (0.01, 0.99)}
    rows, diagnostics = [], []
    for dataset in args.datasets.split(","):
        raw = load_bundle(root, out, dataset, args.alpha)
        train_e = np.asarray(load_arrays(root, args.cache, dataset, "train")["egemaps"], dtype=np.float64)
        val_e = np.asarray(load_arrays(root, args.cache, dataset, "val")["egemaps"], dtype=np.float64)
        mu, sd = train_e.mean(0), train_e.std(0); sd[sd < 1e-8] = 1.0
        z = (val_e - mu) / sd
        worst = int(np.unravel_index(np.abs(z).argmax(), z.shape)[0])
        diagnostics.append({"dataset": dataset, "worst_val_item": worst,
                            "max_abs_train_standardized_value": float(np.abs(z).max()),
                            "worst_item_max_abs_z": float(np.abs(z[worst]).max()),
                            "worst_item_euclidean_z": float(np.linalg.norm(z[worst]))})
        for config, bounds in configs.items():
            bundle = raw if bounds is None else robust_oof(root, args.cache, dataset, args.alpha, device, bounds, args.folds)
            for spec, mask in {"basic": MASKS["basic"], "core": MASKS["core"], "full": MASKS["full"],
                               "core_plus_ced": MASKS["core_plus_ced"]}.items():
                weights, val, test = fit_mask(bundle, mask)
                rows.append({"dataset": dataset, "config": config, "spec": spec,
                             "mask": mask, "weights_json": json.dumps(dict(zip(names(mask), weights.tolist()))),
                             **{f"val_{k}": v for k, v in val.items()},
                             **{f"test_{k}": v for k, v in test.items()}})
        log(f"eGeMAPS sensitivity {dataset}")
    pd.DataFrame(rows).to_csv(out / "egemaps_robust_sensitivity.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "egemaps_outlier_diagnostics.csv", index=False)
    save_state(out, "03_egemaps", {"rows": len(rows), "status": "complete"})


def make_energy_reverb(waveform, sample_rate: int, seed: int):
    import torch
    import torch.nn.functional as functional
    generator = torch.Generator().manual_seed(seed)
    length = int(0.4 * sample_rate)
    times = torch.arange(length, dtype=torch.float32) / sample_rate
    impulse = torch.randn(length, generator=generator) * torch.exp(-6.9 * times / 0.4)
    impulse = 0.12 * impulse
    impulse[0] += 1.0
    impulse /= impulse.abs().sum().clamp_min(1e-6)
    result = functional.conv1d(waveform[None, None], impulse.flip(0)[None, None], padding=length - 1).squeeze()[:waveform.numel()]
    original_rms = waveform.square().mean().sqrt().clamp_min(1e-8)
    result_rms = result.square().mean().sqrt().clamp_min(1e-8)
    return result * (original_rms / result_rms)


def safe_peak(waveform):
    peak = float(waveform.abs().max())
    return waveform * (0.98 / peak) if peak > 0.98 else waveform


CONTROL_NAMES = {"identity": "identity", "gain_minus6db": "gain_-6db", "reverb_energy": "reverb_energy"}
DIR_NAMES = {"whisper": "whisper", "contentvec12": "contentvec", "wavlm": "wavlm_final", "beats": "beats",
             "auditory_erb": "auditory_erb", "speaker": "speaker", "rmvpe_cont": "f0_rmvpe",
             "rmvpe_quant": "f0_rmvpe_quant", "ced": "ced", "egemaps": "egemaps"}


def control_manifest(root: Path, out: Path, datasets: str, n: int, seed: int) -> Path:
    import torchaudio
    manifest_path = out / "controls" / "manifest.csv"
    if manifest_path.exists() and len(pd.read_csv(manifest_path)) == n * 3 * len(datasets.split(",")):
        return manifest_path
    old = pd.read_csv(root / "results/paper_followups_v2/perturbation/manifest.csv")
    chosen = old.drop_duplicates(["dataset", "source_index"]).groupby("dataset", sort=True).head(n)
    rows = []
    audio_root = out / "controls" / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    import extract_explainability_embeddings as extractor
    for di, dataset in enumerate(datasets.split(",")):
        q = chosen[chosen.dataset == dataset].head(n)
        for row in q.itertuples():
            original = extractor._load_mono_16k(row.source_filepath)
            for ci, control in enumerate(CONTROL_NAMES):
                destination = audio_root / dataset / control / f"item_{int(row.source_index)}.wav"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if control == "identity":
                    changed = original
                elif control == "gain_minus6db":
                    changed = original * (10.0 ** (-6.0 / 20.0))
                elif control == "reverb_energy":
                    changed = make_energy_reverb(original, 16000, seed + di * 100000 + int(row.source_index))
                else:
                    raise ValueError(control)
                torchaudio.save(str(destination), safe_peak(changed).unsqueeze(0), 16000)
                rows.append({"key": f"{dataset}|{int(row.source_index)}|{control}", "filepath": str(destination.relative_to(out / "controls")),
                             "dataset": dataset, "source_index": int(row.source_index), "source_filepath": row.source_filepath,
                             "system_id": row.system_id, "condition": getattr(row, "condition", "unknown"),
                             "mos": float(row.mos), "control": control, "sample_rate": 16000})
        log(f"control audio prepared {dataset}")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def configure_control_extractor(root: Path, out: Path):
    sys.path.insert(0, str(root))
    import extract_explainability_embeddings as extractor
    pseudo = "paper_controls"
    controls = out / "controls"
    extractor.DATASETS[pseudo] = {"root": controls, "csv": "manifest.csv", "path_column": "filepath", "output_name": pseudo}
    extractor.EMBEDDINGS_ROOT = controls / "embeddings"
    return extractor, pseudo


def cmd_controls_prepare(args: argparse.Namespace) -> None:
    root, out = root_path(args.root), out_path(args)
    path = control_manifest(root, out, args.datasets, args.items_per_dataset, args.seed)
    log(f"control manifest ready {path}")
    save_state(out, "04_controls_prepare", {"manifest": str(path), "rows": len(pd.read_csv(path)), "status": "complete"})


def cmd_controls_extract(args: argparse.Namespace) -> None:
    root, out = root_path(args.root), out_path(args)
    manifest = out / "controls" / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    extractor, pseudo = configure_control_extractor(root, out)
    report = extractor.extract_one(pseudo, "test", args.feature, SimpleNamespace(device=args.device, whisper_model="whisper-large-v3", deep_audit_after=False))
    write_json(out / "controls" / f"extract_{args.feature}.json", report)
    if report.get("status") == "error":
        raise RuntimeError(report.get("error", "control extraction failed"))
    log(f"control extraction complete feature={args.feature}")


def cmd_controls_analyze(args: argparse.Namespace) -> None:
    root, out = root_path(args.root), out_path(args)
    manifest = pd.read_csv(out / "controls" / "manifest.csv")
    sys.path.insert(0, str(root))
    from run_new_explainability_ridge import pool_tensor
    device = base.device_arg(args.device)
    embedding_rows, prediction_rows = [], []
    for dataset in args.datasets.split(","):
        q = manifest[manifest.dataset == dataset].copy().reset_index(drop=True)
        original_test = load_arrays(root, args.cache, dataset, "test")
        train = load_arrays(root, args.cache, dataset, "train")
        y_train = base.load_y(root, args.cache, dataset, "train")
        bundle = load_bundle(root, out, dataset, args.alpha)
        for feature in FEATURES:
            paths = [out / "controls" / Path(x) for x in q.filepath.astype(str)]
            tensor_paths = [out / "controls" / "embeddings" / "paper_controls" / "test" / DIR_NAMES[feature] / Path(p).with_suffix(".pt") for p in q.filepath.astype(str)]
            if not all(p.exists() for p in tensor_paths):
                raise FileNotFoundError(f"missing control tensor for {dataset}/{feature}")
            controls = np.stack([pool_tensor(str(p), feature) for p in tensor_paths]).astype(np.float32)
            for control in q.control.unique():
                indices = np.flatnonzero(q.control.to_numpy() == control)
                source_idx = q.iloc[indices].source_index.astype(int).to_numpy()
                original = np.asarray(original_test[feature][source_idx], dtype=np.float32)
                changed = controls[indices]
                cosine = 1.0 - np.sum(original * changed, axis=1) / (np.linalg.norm(original, axis=1) * np.linalg.norm(changed, axis=1) + 1e-9)
                rel = np.linalg.norm(changed - original, axis=1) / (np.linalg.norm(original, axis=1) + 1e-9)
                for i, value in enumerate(indices):
                    embedding_rows.append({"dataset": dataset, "source_index": int(source_idx[i]), "control": control,
                                           "feature": feature, "cosine_distance": float(cosine[i]), "relative_l2": float(rel[i])})
        # Model sensitivity uses the converged OOF gate for the full and basic specs.
        for control in q.control.unique():
            indices = np.flatnonzero(q.control.to_numpy() == control)
            source_idx = q.iloc[indices].source_index.astype(int).to_numpy()
            eval_arrays = {}
            original_arrays = {}
            for feature in FEATURES:
                paths = [out / "controls" / "embeddings" / "paper_controls" / "test" / DIR_NAMES[feature] / Path(p).with_suffix(".pt") for p in q.iloc[indices].filepath.astype(str)]
                eval_arrays[feature] = np.stack([pool_tensor(str(p), feature) for p in paths]).astype(np.float32)
                original_arrays[feature] = np.asarray(load_arrays(root, args.cache, dataset, "test")[feature][source_idx], dtype=np.float32)
            p_original = v2.solve_np(root, train, original_arrays, y_train, device, args.alpha)
            p_control = v2.solve_np(root, train, eval_arrays, y_train, device, args.alpha)
            for spec in ["basic", "full", "core"]:
                mask = MASKS[spec]
                idx = [FEATURES.index(f) for f in names(mask)]
                weights = fit_simplex(bundle["oof"][:, idx], bundle["y_train"])
                delta = (p_control[:, idx] @ weights) - (p_original[:, idx] @ weights)
                for i, value in enumerate(indices):
                    prediction_rows.append({"dataset": dataset, "source_index": int(source_idx[i]), "control": control,
                                            "spec": spec, "prediction_delta": float(delta[i]),
                                            "prediction_original": float(p_original[i, idx] @ weights),
                                            "prediction_control": float(p_control[i, idx] @ weights)})
        log(f"control analysis {dataset}")
    embedding = pd.DataFrame(embedding_rows)
    prediction = pd.DataFrame(prediction_rows)
    embedding.to_csv(out / "control_embedding_consistency.csv", index=False)
    prediction.to_csv(out / "control_prediction_sensitivity.csv", index=False)
    embedding.groupby(["dataset", "control", "feature"], as_index=False)[["cosine_distance", "relative_l2"]].mean().to_csv(out / "control_embedding_summary.csv", index=False)
    prediction.groupby(["dataset", "control", "spec"], as_index=False)["prediction_delta"].agg(["mean", "median", "std", "count"]).reset_index().to_csv(out / "control_prediction_summary.csv", index=False)
    save_state(out, "04_controls_analyze", {"embedding_rows": len(embedding), "prediction_rows": len(prediction), "status": "complete"})


def _encode_labels(train: pd.Series, val: pd.Series, test: pd.Series):
    all_values = pd.concat([train, val, test]).fillna("unknown").astype(str)
    categories = {x: i for i, x in enumerate(sorted(all_values.unique()))}
    return tuple(s.fillna("unknown").astype(str).map(categories).to_numpy() for s in [train, val, test]), categories


def cmd_concepts(args: argparse.Namespace) -> None:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score
    root, out = root_path(args.root), out_path(args)
    rows, residual_rows = [], []
    for dataset in args.datasets.split(","):
        meta = {s: base.load_metadata(root, dataset, s) for s in ["train", "val", "test"]}
        arrays = {s: load_arrays(root, args.cache, dataset, s) for s in ["train", "val", "test"]}
        for feature in FEATURES:
            mu = arrays["train"][feature].mean(0); sd = arrays["train"][feature].std(0); sd[sd < 1e-8] = 1.0
            X = {s: (arrays[s][feature] - mu) / sd for s in arrays}
            for task, column in [("system_proxy", "system_id"), ("condition", "condition")]:
                if column not in meta["train"] or meta["train"][column].nunique() < 2:
                    continue
                (tr, va, te), categories = _encode_labels(meta["train"][column], meta["val"][column], meta["test"][column])
                clf = LogisticRegression(max_iter=1000, C=args.logistic_c, solver="lbfgs", multi_class="auto")
                clf.fit(X["train"], tr)
                pv, pt = clf.predict(X["val"]), clf.predict(X["test"])
                rows.append({"dataset": dataset, "feature": feature, "task": task, "label_column": column,
                             "n_classes": len(categories), "val_balanced_accuracy": balanced_accuracy_score(va, pv),
                             "test_balanced_accuracy": balanced_accuracy_score(te, pt),
                             "test_macro_f1": f1_score(te, pt, average="macro", zero_division=0)})
            if feature != "rmvpe_cont":
                target = arrays["train"]["rmvpe_cont"]
                target_v = arrays["val"]["rmvpe_cont"]
                target_t = arrays["test"]["rmvpe_cont"]
                # Five interpretable pitch targets: mean, spread, extrema, voiced fraction.
                for j, target_name in enumerate(["f0_mean", "f0_std", "f0_min", "f0_max", "voiced_fraction"]):
                    model = Ridge(alpha=args.ridge_alpha).fit(X["train"], target[:, j])
                    rows.append({"dataset": dataset, "feature": feature, "task": "pitch_probe", "label_column": target_name,
                                 "val_r2": r2_score(target_v[:, j], model.predict(X["val"])),
                                 "test_r2": r2_score(target_t[:, j], model.predict(X["test"])),
                                 "test_spearman": spearmanr(target_t[:, j], model.predict(X["test"])).statistic})
        bundle = load_bundle(root, out, dataset, args.alpha)
        core_idx = [FEATURES.index(f) for f in names(MASKS["core"])]
        core_w = fit_simplex(bundle["oof"][:, core_idx], bundle["y_train"])
        core_oof = bundle["oof"][:, core_idx] @ core_w
        core_test = bundle["test"][:, core_idx] @ core_w
        for feature in FEATURES:
            if MASKS["core"] & BITS[feature]:
                continue
            j = FEATURES.index(feature)
            candidate_oof = bundle["oof"][:, j]
            candidate_test = bundle["test"][:, j]
            residual = bundle["y_train"] - core_oof
            candidate_residual = candidate_oof - np.mean(candidate_oof)
            cross = float(np.mean(residual * candidate_residual))
            energy = float(np.mean(candidate_residual ** 2))
            chosen_lambda = float(np.clip(cross / max(energy, 1e-12), -1.0, 1.0))
            adjusted = core_test + chosen_lambda * (candidate_test - np.mean(candidate_oof))
            rows_core = metric_dict(bundle["y_test"], core_test)
            rows_adj = metric_dict(bundle["y_test"], adjusted)
            residual_rows.append({"dataset": dataset, "added_feature": feature, "core_mse": rows_core["mse"],
                                  "residual_cross_term": cross, "candidate_difference_energy": energy,
                                  "oof_chosen_residual_lambda": chosen_lambda,
                                  "test_mse_after_oof_residual_adjustment": rows_adj["mse"],
                                  "test_mse_gain": rows_core["mse"] - rows_adj["mse"],
                                  "test_spearman_delta": rows_adj["spearman"] - rows_core["spearman"]})
        log(f"concept probes {dataset}")
    pd.DataFrame(rows).to_csv(out / "concept_probe_results.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(out / "residual_complementarity.csv", index=False)
    save_state(out, "05_concepts", {"rows": len(rows), "residual_rows": len(residual_rows), "status": "complete"})


def concat_matrix(arrays: dict[str, np.ndarray], mask: int) -> np.ndarray:
    return np.concatenate([np.asarray(arrays[f], dtype=np.float32) for f in names(mask)], axis=1)


def cmd_concat(args: argparse.Namespace) -> None:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    root, out = root_path(args.root), out_path(args)
    alphas = [float(x) for x in args.alphas.split(",")]
    rows = []
    for dataset in args.datasets.split(","):
        arrays = {s: load_arrays(root, args.cache, dataset, s) for s in ["train", "val", "test"]}
        y = {s: base.load_y(root, args.cache, dataset, s) for s in ["train", "val", "test"]}
        meta = base.load_metadata(root, dataset, "train")
        groups = meta.system_id.astype(str).to_numpy()
        for spec, mask in MASKS.items():
            train_raw = concat_matrix(arrays["train"], mask).astype(np.float64)
            mean, sd = train_raw.mean(0), train_raw.std(0); sd[sd < 1e-8] = 1.0
            X = {s: (concat_matrix(arrays[s], mask).astype(np.float64) - mean) / sd for s in arrays}
            split = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
            cv_rows = []
            for alpha in alphas:
                fold_mse, fold_spear = [], []
                for fit_idx, hold_idx in split.split(X["train"], y["train"], groups):
                    model = Ridge(alpha=alpha, solver="lsqr", fit_intercept=True).fit(X["train"][fit_idx], y["train"][fit_idx])
                    pred = model.predict(X["train"][hold_idx])
                    met = metric_dict(y["train"][hold_idx], pred)
                    fold_mse.append(met["mse"]); fold_spear.append(met["spearman"])
                cv_rows.append((alpha, float(np.mean(fold_mse)), float(np.mean(fold_spear))))
            chosen_alpha, cv_mse, cv_spear = min(cv_rows, key=lambda v: v[1])
            model = Ridge(alpha=chosen_alpha, solver="lsqr", fit_intercept=True).fit(X["train"], y["train"])
            val, test = metric_dict(y["val"], model.predict(X["val"])), metric_dict(y["test"], model.predict(X["test"]))
            rows.append({"dataset": dataset, "spec": spec, "mask": mask, "n_dimensions": int(train_raw.shape[1]),
                         "selected_alpha": chosen_alpha, "group_cv_mse": cv_mse, "group_cv_spearman": cv_spear,
                         **{f"val_{k}": v for k, v in val.items()}, **{f"test_{k}": v for k, v in test.items()}})
            log(f"concat baseline {dataset}/{spec} alpha={chosen_alpha}")
    pd.DataFrame(rows).to_csv(out / "concat_baseline_groupcv.csv", index=False)
    save_state(out, "06_concat", {"rows": len(rows), "status": "complete"})


def cluster_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray, n: int, seed: int):
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    delta = (y - a) ** 2 - (y - b) ** 2
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)), size=n)
    sizes = np.asarray([len(x) for x in members])
    denom = counts @ sizes
    sums = np.asarray([delta[x].sum() for x in members])
    values = (counts @ sums) / np.maximum(denom, 1)
    return float(delta.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), values


def cmd_statistics(args: argparse.Namespace) -> None:
    root, out = root_path(args.root), out_path(args)
    item = pd.read_csv(out / "converged_item_predictions.csv")
    rows = []
    primary = []
    for dataset in args.datasets.split(","):
        q = item[item.dataset == dataset]
        y = q[q.spec == "basic"].sort_values("item_index").mos.to_numpy(float)
        groups = base.load_metadata(root, dataset, "test").system_id.astype(str).to_numpy()
        predictions = {spec: q[q.spec == spec].sort_values("item_index").prediction.to_numpy(float) for spec in q.spec.unique()}
        comparisons = [("basic", "full"), ("basic", "basic_plus_ced"), ("core", "core_plus_ced"), ("core", "core_plus_beats")]
        for baseline, child in comparisons:
            mean, lo, hi, boot = cluster_bootstrap(y, predictions[baseline], predictions[child], groups, args.n_bootstrap, args.seed)
            p = 2.0 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
            primary.append({"dataset": dataset, "comparison": baseline + "_vs_" + child, "estimate_mse_improvement": mean,
                            "ci95_low": lo, "ci95_high": hi, "bootstrap_p_two_sided": min(p, 1.0), "n_groups": len(np.unique(groups))})
    primary_frame = pd.DataFrame(primary)
    order = primary_frame.bootstrap_p_two_sided.argsort().to_numpy()
    m = len(primary_frame)
    adjusted = np.empty(m)
    running = 0.0
    for rank, pos in enumerate(order):
        value = min(1.0, (m - rank) * float(primary_frame.iloc[pos].bootstrap_p_two_sided))
        running = max(running, value)
        adjusted[pos] = running
    primary_frame["holm_p"] = adjusted
    primary_frame["family"] = "pre_specified_MSE_comparisons"
    primary_frame.to_csv(out / "confirmatory_tests_holm.csv", index=False)
    # Compactness is reported as an equivalence sensitivity, not as a claim of superiority.
    selected = pd.read_csv(out / "converged_selected_summary.csv")
    compact_rows = []
    for row in selected.itertuples():
        q = item[item.dataset == row.dataset]
        y = q[q.spec == "full"].sort_values("item_index").mos.to_numpy(float)
        full = q[q.spec == "full"].sort_values("item_index").prediction.to_numpy(float)
        compact = q[q.spec == "full"].sort_values("item_index").prediction.to_numpy(float)
        # Selection mask is not one of the fixed item rows; use lattice test values
        lattice = pd.read_csv(out / "converged_oof_lattice.csv")
        cand = lattice[(lattice.dataset == row.dataset) & (lattice["mask"] == int(row.mask))].iloc[0]
        # Reconstruct selected predictions from the stored test bundle and weights.
        bundle = load_bundle(root, out, row.dataset, args.alpha)
        w = np.array(list(json.loads(cand.weights_json).values()), dtype=float)
        idx = [FEATURES.index(f) for f in names(int(row.mask))]
        compact = bundle["test"][:, idx] @ w
        groups = base.load_metadata(root, row.dataset, "test").system_id.astype(str).to_numpy()
        mean, lo, hi, _ = cluster_bootstrap(y, full, compact, groups, args.n_bootstrap, args.seed)
        compact_rows.append({"dataset": row.dataset, "selected_mask": int(row.mask), "selected_features": row.features,
                             "relative_mse_difference_pct": 100 * mean / max(float(np.mean((y - full) ** 2)), 1e-12),
                             "ci95_low_relative_pct": 100 * lo / max(float(np.mean((y - full) ** 2)), 1e-12),
                             "ci95_high_relative_pct": 100 * hi / max(float(np.mean((y - full) ** 2)), 1e-12),
                             "equivalence_margin_pct": args.equivalence_margin_pct})
    pd.DataFrame(compact_rows).to_csv(out / "compact_equivalence_sensitivity.csv", index=False)
    latency = pd.read_csv(root / "results/paper_followups_v2/extraction_latency.csv")
    perf = pd.read_csv(out / "converged_oof_lattice.csv")
    frontier = []
    total_cost = float(latency.wall_seconds.sum())
    for row in perf[perf["mask"].isin(set(MASKS.values()))].itertuples():
        n = row.n_features
        cost = float(latency[latency.feature.isin(names(int(row.mask)))].wall_seconds.sum())
        frontier.append({"dataset": row.dataset, "spec": next(k for k, v in MASKS.items() if v == row.mask),
                         "test_spearman": row.test_spearman, "test_mse": row.test_mse,
                         "n_features": n, "estimated_cold_wall_seconds": cost,
                         "estimated_fraction_of_full_cost": cost / max(total_cost, 1e-9)})
    frontier = pd.DataFrame(frontier)
    frontier["pareto_efficient"] = True
    for i, r in frontier.iterrows():
        dominated = frontier[(frontier.dataset == r.dataset) & (frontier.estimated_cold_wall_seconds <= r.estimated_cold_wall_seconds) & (frontier.test_spearman >= r.test_spearman)]
        if len(dominated) > 1:
            frontier.loc[i, "pareto_efficient"] = False
    frontier.to_csv(out / "efficiency_frontier.csv", index=False)
    plan = {
        "primary_question": "Does representation utility depend on the coalition and the cross-fitting protocol?",
        "primary_comparisons": ["basic_vs_full", "basic_vs_basic_plus_ced", "core_vs_core_plus_ced", "core_vs_core_plus_beats"],
        "bootstrap": {"unit": "system cluster", "n": args.n_bootstrap, "seed": args.seed},
        "multiple_comparisons": "Holm correction over the four pre-specified MSE comparisons",
        "compact_equivalence": {"margin_relative_mse_percent": args.equivalence_margin_pct, "selection": "validation only"},
        "test_use": "only final estimate and paired uncertainty; no test-based selection",
        "interpretation": "predictive conditional evidence, not a causal effect or proof of human perceptual mechanism",
    }
    write_json(out / "statistical_analysis_plan.json", plan)
    save_state(out, "07_statistics", {"primary_rows": len(primary_frame), "status": "complete"})


def cmd_report(args: argparse.Namespace) -> None:
    root, out = root_path(args.root), out_path(args)
    sections = ["# Final ICASSP controls report\n", "All primary comparisons are defined in statistical_analysis_plan.json.\n"]
    files = ["converged_selected_summary.csv", "oof_protocol_compare.csv", "egemaps_robust_sensitivity.csv",
             "egemaps_outlier_diagnostics.csv", "control_embedding_summary.csv", "control_prediction_summary.csv",
             "concept_probe_results.csv", "residual_complementarity.csv", "concat_baseline_groupcv.csv",
             "confirmatory_tests_holm.csv", "compact_equivalence_sensitivity.csv", "efficiency_frontier.csv"]
    for relative in files:
        path = out / relative
        sections.append(f"\n## {relative}\n")
        if not path.exists():
            sections.append("Missing; inspect queue_errors.tsv.\n")
            continue
        frame = pd.read_csv(path)
        sections.append(f"Rows: {len(frame)}.\n\n")
        sections.append(frame.head(80).to_markdown(index=False) + "\n")
    sections.append("\n## Reproducibility\n\n")
    sections.append(f"Generated at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}.\n")
    (out / "final_controls_report.md").write_text("\n".join(sections), encoding="utf-8")
    save_state(out, "08_report", {"status": "complete"})
    log("final controls report complete")


def add_common(parser):
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="results/paper_final_controls")
    parser.add_argument("--cache", default="results/explainability_ridge_full/cache")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--alpha", type=float, default=10.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("converged-lattice"); add_common(p); p.set_defaults(func=cmd_converged_lattice)
    p = sub.add_parser("protocol"); add_common(p); p.add_argument("--device", default="cuda"); p.add_argument("--folds", type=int, default=5); p.add_argument("--seed", type=int, default=20260909); p.set_defaults(func=cmd_protocol)
    p = sub.add_parser("egemaps"); add_common(p); p.add_argument("--device", default="cuda"); p.add_argument("--folds", type=int, default=5); p.set_defaults(func=cmd_egemaps)
    p = sub.add_parser("controls-prepare"); add_common(p); p.add_argument("--items-per-dataset", type=int, default=32); p.add_argument("--seed", type=int, default=20260909); p.set_defaults(func=cmd_controls_prepare)
    p = sub.add_parser("controls-extract"); add_common(p); p.add_argument("--device", default="cuda"); p.add_argument("--feature", required=True, choices=FEATURES); p.set_defaults(func=cmd_controls_extract)
    p = sub.add_parser("controls-analyze"); add_common(p); p.add_argument("--device", default="cuda"); p.set_defaults(func=cmd_controls_analyze)
    p = sub.add_parser("concepts"); add_common(p); p.add_argument("--ridge-alpha", type=float, default=10.0); p.add_argument("--logistic-c", type=float, default=1.0); p.set_defaults(func=cmd_concepts)
    p = sub.add_parser("concat"); add_common(p); p.add_argument("--alphas", default="0.1,1,10,100,1000"); p.add_argument("--folds", type=int, default=5); p.set_defaults(func=cmd_concat)
    p = sub.add_parser("statistics"); add_common(p); p.add_argument("--n-bootstrap", type=int, default=10000); p.add_argument("--seed", type=int, default=20260909); p.add_argument("--equivalence-margin-pct", type=float, default=2.0); p.set_defaults(func=cmd_statistics)
    p = sub.add_parser("report"); add_common(p); p.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
