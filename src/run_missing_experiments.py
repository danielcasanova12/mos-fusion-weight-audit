#!/usr/bin/env python3
"""Reproducible follow-up experiments for the MOS coalition paper.

The main M6 sweep is already complete.  This runner performs the analyses that
turn the sweep into a defensible explanatory paper: item-level predictions and
paired bootstrap, exact Shapley values from the existing coalition table,
leave-one-dataset-out transfer, shortcut probes, permutation importance, gate
stability, and a cache-cost audit.

Run from the project root, for example::

    python paper/run_missing_experiments.py shapley --root .
    python paper/run_missing_experiments.py predictions --device cuda --root .

Every command writes a status file and is safe to rerun after interruption.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


FEATURES = [
    "whisper", "contentvec12", "wavlm", "beats", "auditory_erb",
    "speaker", "rmvpe_cont", "rmvpe_quant", "ced", "egemaps",
]
DATASETS = ["brspeech", "bvcc", "singmos", "tmhintqi"]
SEEDS = [42, 123, 456, 789, 1011]

BITS = {name: 1 << i for i, name in enumerate(FEATURES)}
FIXED_MASKS = {
    "basic": sum(BITS[x] for x in ["auditory_erb", "speaker", "rmvpe_cont", "rmvpe_quant", "egemaps"]),
    "core": sum(BITS[x] for x in ["contentvec12", "speaker", "rmvpe_cont", "rmvpe_quant"]),
    "speech_ssl": sum(BITS[x] for x in ["whisper", "contentvec12", "wavlm"]),
    "audio_ssl": sum(BITS[x] for x in ["beats", "ced"]),
    "stable_ssl": sum(BITS[x] for x in ["wavlm", "beats", "ced"]),
    "full": (1 << len(FEATURES)) - 1,
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def root_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def cache_path(root: Path, cache: str, dataset: str, split: str, feature: str) -> Path:
    path = root / cache / f"{dataset}_{split}_{feature}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_x(root: Path, cache: str, dataset: str, split: str, feature: str) -> np.ndarray:
    return np.asarray(np.load(cache_path(root, cache, dataset, split, feature), mmap_mode="r"), dtype=np.float32)


def load_y(root: Path, cache: str, dataset: str, split: str) -> np.ndarray:
    return load_x(root, cache, dataset, split, "y")


def metadata_path(root: Path, dataset: str, split: str) -> Path:
    path = root / "embeddings" / dataset / split / "metadata_explainability.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_metadata(root: Path, dataset: str, split: str) -> pd.DataFrame:
    frame = pd.read_csv(metadata_path(root, dataset, split))
    frame.insert(0, "item_index", np.arange(len(frame), dtype=np.int64))
    for col in ("system_id", "condition"):
        if col not in frame:
            frame[col] = "unknown"
        frame[col] = frame[col].fillna("unknown").astype(str)
    return frame


def names(mask: int) -> list[str]:
    return [f for i, f in enumerate(FEATURES) if mask & (1 << i)]


def mask_for(items: Iterable[str]) -> int:
    return sum(BITS[x] for x in items)


def metric_dict(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return {
        "mse": float(np.mean((y - pred) ** 2)),
        "mae": float(np.mean(np.abs(y - pred))),
        "pearson": float(pearsonr(y, pred)[0]) if len(y) > 1 else float("nan"),
        "spearman": float(spearmanr(y, pred).statistic) if len(y) > 1 else float("nan"),
    }


def append_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_m6(root: Path, m6_csv: str) -> pd.DataFrame:
    path = root / m6_csv
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def selected_masks(m6: pd.DataFrame) -> dict[tuple[str, int], int]:
    q = m6[(m6["split"] == "val") & m6["regime"].astype(str).str.startswith("within_")].copy()
    out: dict[tuple[str, int], int] = {}
    for (dataset, seed), group in q.groupby(["dataset", "seed"], sort=True):
        row = group.loc[group["spearman"].astype(float).idxmax()]
        out[(str(dataset), int(seed))] = int(row["mask"])
    return out


def device_arg(value: str):
    import torch
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    return torch.device(value)


def m6_functions(root: Path):
    sys.path.insert(0, str(root))
    from run_new_explainability_m6_gpu import fit_gate, solve_experts
    return fit_gate, solve_experts


def expert_predictions(root: Path, cache: str, dataset: str, device, alpha: float = 10.0):
    """Fit the frozen Ridge experts once and return train/val/test matrices."""
    import torch
    _, solve_experts = m6_functions(root)
    train = {f: load_x(root, cache, dataset, "train", f) for f in FEATURES}
    val = {f: load_x(root, cache, dataset, "val", f) for f in FEATURES}
    test = {f: load_x(root, cache, dataset, "test", f) for f in FEATURES}
    y_train = load_y(root, cache, dataset, "train")
    pred_train = solve_experts(train, train, y_train, device, alpha)
    pred_val = solve_experts(train, val, y_train, device, alpha)
    pred_test = solve_experts(train, test, y_train, device, alpha)
    return y_train, pred_train, pred_val, pred_test


def spec_masks(root: Path, m6_csv: str, dataset: str) -> dict[str, int]:
    m6 = load_m6(root, m6_csv)
    selection = selected_masks(m6)
    specs = dict(FIXED_MASKS)
    for feature in FEATURES:
        specs[f"basic_plus_{feature}"] = FIXED_MASKS["basic"] | BITS[feature]
        specs[f"core_plus_{feature}"] = FIXED_MASKS["core"] | BITS[feature]
    for seed in SEEDS:
        if (dataset, seed) in selection:
            specs[f"selected_{seed}"] = selection[(dataset, seed)]
    return specs


def cmd_predictions(args: argparse.Namespace) -> None:
    import torch
    root = root_path(args.root)
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    result = out / "item_predictions.csv"
    state_path = out / "predictions_state.json"
    done = set(json.loads(state_path.read_text()) if state_path.exists() else [])
    device = device_arg(args.device)
    for dataset in args.datasets.split(","):
        y_train, p_train, _, p_test = expert_predictions(root, args.cache, dataset, device, args.alpha)
        y_test = load_y(root, args.cache, dataset, "test")
        meta = load_metadata(root, dataset, "test")
        specs = spec_masks(root, args.m6_csv, dataset)
        for spec, mask in specs.items():
            for si, seed in enumerate(SEEDS):
                key = f"{dataset}|{spec}|{seed}"
                if key in done:
                    continue
                idx = [FEATURES.index(f) for f in names(mask)]
                fit_gate, _ = m6_functions(root)
                weights = fit_gate(p_train[:, idx], y_train, [seed], args.gate_steps, args.batch_size, device)[0]
                pred = (p_test[:, idx] * weights[None, :]).sum(dim=1).detach().cpu().numpy()
                frame = meta[["item_index", "filepath", "condition", "system_id"]].copy()
                frame["dataset"] = dataset
                frame["spec"] = spec
                frame["mask"] = mask
                frame["features"] = "+".join(names(mask))
                frame["seed"] = seed
                frame["mos"] = y_test
                frame["prediction"] = pred
                frame["weights_json"] = json.dumps({f: float(weights[j].detach().cpu()) for j, f in enumerate(names(mask))})
                for j, feature in enumerate(FEATURES):
                    frame[f"expert_{feature}"] = p_test[:, j].detach().cpu().numpy()
                    frame[f"weight_{feature}"] = 0.0
                for j, feature in enumerate(names(mask)):
                    frame[f"weight_{feature}"] = float(weights[j].detach().cpu())
                append_frame(result, frame)
                done.add(key)
                state_path.write_text(json.dumps(sorted(done), indent=2))
                log(f"predictions {dataset} {spec} seed={seed} rows={len(frame)}")
        del p_train, p_test
        if device.type == "cuda":
            torch.cuda.empty_cache()


def bootstrap_indices(groups: np.ndarray | None, n: int, rng: np.random.Generator) -> np.ndarray:
    if groups is None or len(np.unique(groups)) < 2:
        return rng.integers(0, n, size=n)
    unique = np.unique(groups)
    selected = rng.choice(unique, size=len(unique), replace=True)
    pieces = [np.flatnonzero(groups == g) for g in selected]
    return np.concatenate(pieces) if pieces else rng.integers(0, n, size=n)


def bootstrap_compare(y: np.ndarray, base: np.ndarray, child: np.ndarray,
                      groups: np.ndarray | None, n_boot: int, seed: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    records: dict[str, list[float]] = {"mse_improvement": [], "mae_improvement": [],
                                       "pearson_delta": [], "spearman_delta": []}
    for _ in range(n_boot):
        idx = bootstrap_indices(groups, len(y), rng)
        mb = metric_dict(y[idx], base[idx]); mc = metric_dict(y[idx], child[idx])
        records["mse_improvement"].append(mb["mse"] - mc["mse"])
        records["mae_improvement"].append(mb["mae"] - mc["mae"])
        records["pearson_delta"].append(mc["pearson"] - mb["pearson"])
        records["spearman_delta"].append(mc["spearman"] - mb["spearman"])
    rows = []
    for metric, vals in records.items():
        arr = np.asarray(vals, dtype=float)
        rows.append({"metric": metric, "estimate": float(arr.mean()),
                     "ci95_low": float(np.quantile(arr, .025)),
                     "ci95_high": float(np.quantile(arr, .975)),
                     "prob_positive": float(np.mean(arr > 0))})
    return rows


def cmd_bootstrap(args: argparse.Namespace) -> None:
    root = root_path(args.root)
    pred_path = root / args.predictions
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    df = pd.read_csv(pred_path)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    selection = selected_masks(load_m6(root, args.m6_csv))
    comparisons = [("basic", "full"), ("basic", "basic_plus_whisper"),
                   ("basic", "basic_plus_contentvec12"), ("basic", "basic_plus_wavlm"),
                   ("basic", "basic_plus_beats"), ("basic", "basic_plus_ced")]
    rows = []
    for dataset in args.datasets.split(","):
        d = df[df["dataset"] == dataset]
        for seed in SEEDS:
            selected_mask = selection.get((dataset, seed))
            if selected_mask is not None:
                dsel = d[(d["seed"] == seed) & (d["mask"] == selected_mask)]
                db = d[(d["seed"] == seed) & (d["mask"] == FIXED_MASKS["basic"])]
                if not dsel.empty and not db.empty:
                    merged = db.merge(dsel, on=["item_index", "seed"], suffixes=("_base", "_child"))
                    group_col = "system_id_base" if "system_id_base" in merged else None
                    for rec in bootstrap_compare(merged["mos_base"].to_numpy(), merged["prediction_base"].to_numpy(), merged["prediction_child"].to_numpy(), merged[group_col].to_numpy() if group_col else None, args.n_bootstrap, seed + 5000):
                        rows.append({"dataset": dataset, "seed": seed, "comparison": f"basic->selected_mask_{selected_mask}", **rec})
            for base_spec, child_spec in comparisons:
                db = d[(d["seed"] == seed) & (d["spec"] == base_spec)]
                dc = d[(d["seed"] == seed) & (d["spec"] == child_spec)]
                if db.empty or dc.empty:
                    continue
                merged = db.merge(dc, on=["item_index", "seed"], suffixes=("_base", "_child"))
                group_col = "system_id_base" if "system_id_base" in merged else None
                for rec in bootstrap_compare(merged["mos_base"].to_numpy(), merged["prediction_base"].to_numpy(), merged["prediction_child"].to_numpy(), merged[group_col].to_numpy() if group_col else None, args.n_bootstrap, seed + 7000):
                    rows.append({"dataset": dataset, "seed": seed, "comparison": f"{base_spec}->{child_spec}", **rec})
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"bootstrap rows={len(rows)} output={out}")


def cmd_shapley(args: argparse.Namespace) -> None:
    root = root_path(args.root)
    m6 = load_m6(root, args.m6_csv)
    rows, interactions, baselines = [], [], []
    n = len(FEATURES)
    factorial = [math.factorial(k) for k in range(n + 1)]
    for dataset in args.datasets.split(","):
        y_train = load_y(root, args.cache, dataset, "train")
        y_val = load_y(root, args.cache, dataset, "val")
        empty_mse = float(np.mean((y_val - np.mean(y_train)) ** 2))
        baselines.append({"dataset": dataset, "empty_validation_mse": empty_mse, "train_mean": float(np.mean(y_train))})
        q = m6[(m6["dataset"] == dataset) & (m6["regime"] == f"within_{dataset}") & (m6["split"] == "val")]
        for seed in SEEDS:
            g = q[q["seed"] == seed]
            mse = {int(r.mask): float(r.mse) for r in g.itertuples()}
            if len(mse) < (1 << n) - 1:
                log(f"warning: incomplete M6 table for {dataset} seed={seed}; found {len(mse)} masks")
            mse[0] = empty_mse
            for i, feature in enumerate(FEATURES):
                bit = 1 << i
                value = 0.0
                for mask in range(1 << n):
                    if mask & bit:
                        continue
                    s = mask.bit_count()
                    w = factorial[s] * factorial[n - s - 1] / factorial[n]
                    if mask not in mse or (mask | bit) not in mse:
                        continue
                    value += w * (mse[mask] - mse[mask | bit])
                rows.append({"dataset": dataset, "seed": seed, "feature": feature,
                             "shapley_mse_reduction": value})
            for i in range(n):
                for j in range(i + 1, n):
                    total = []
                    bi, bj = 1 << i, 1 << j
                    for mask in range(1 << n):
                        if mask & (bi | bj):
                            continue
                        needed = [mask, mask | bi, mask | bj, mask | bi | bj]
                        if not all(k in mse for k in needed):
                            continue
                        total.append(mse[mask] - mse[mask | bi] - mse[mask | bj] + mse[mask | bi | bj])
                    interactions.append({"dataset": dataset, "seed": seed, "feature_a": FEATURES[i],
                                         "feature_b": FEATURES[j], "mean_pair_interaction_mse": float(np.mean(total)) if total else float("nan")})
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(interactions).to_csv(out.with_name("pair_interaction_validation.csv"), index=False)
    pd.DataFrame(baselines).to_csv(out.with_name("empty_baselines.csv"), index=False)
    log(f"shapley rows={len(rows)} output={out}")


def cmd_lodo(args: argparse.Namespace) -> None:
    import torch
    root = root_path(args.root)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        old = pd.read_csv(out, usecols=["source", "target", "spec", "seed"])
        done = {(r.source, r.target, r.spec, int(r.seed)) for r in old.itertuples()}
    device = device_arg(args.device)
    fit_gate, solve_experts = m6_functions(root)
    specs = {k: FIXED_MASKS[k] for k in ["basic", "core", "stable_ssl", "speech_ssl", "audio_ssl", "full"]}
    for source in args.datasets.split(","):
        train = {f: load_x(root, args.cache, source, "train", f) for f in FEATURES}
        val = {f: load_x(root, args.cache, source, "val", f) for f in FEATURES}
        y_train = load_y(root, args.cache, source, "train")
        y_val = load_y(root, args.cache, source, "val")
        p_train = solve_experts(train, train, y_train, device, args.alpha)
        p_val = solve_experts(train, val, y_train, device, args.alpha)
        for target in args.datasets.split(","):
            if target == source:
                continue
            test = {f: load_x(root, args.cache, target, "test", f) for f in FEATURES}
            y_test = load_y(root, args.cache, target, "test")
            p_test = solve_experts(train, test, y_train, device, args.alpha)
            for spec, mask in specs.items():
                idx = [FEATURES.index(f) for f in names(mask)]
                weights = fit_gate(p_train[:, idx], y_train, SEEDS, args.gate_steps, args.batch_size, device)
                for si, seed in enumerate(SEEDS):
                    key = (source, target, spec, seed)
                    if key in done:
                        continue
                    mv = metric_dict(y_val, (p_val[:, idx] * weights[si][None, :]).sum(dim=1).detach().cpu().numpy())
                    mt = metric_dict(y_test, (p_test[:, idx] * weights[si][None, :]).sum(dim=1).detach().cpu().numpy())
                    row = {"source": source, "target": target, "spec": spec, "mask": mask, "seed": seed,
                           "features": "+".join(names(mask)), "source_val_spearman": mv["spearman"],
                           "target_test_spearman": mt["spearman"], "target_test_pearson": mt["pearson"],
                           "target_test_mse": mt["mse"], "target_test_mae": mt["mae"],
                           "weights_json": json.dumps({f: float(weights[si][j].detach().cpu()) for j, f in enumerate(names(mask))})}
                    append_frame(out, pd.DataFrame([row])); done.add(key)
            del p_test
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del p_train, p_val
    log(f"lodo output={out}")


def encode_labels(train: pd.Series, val: pd.Series, test: pd.Series):
    all_values = pd.concat([train, val, test]).fillna("unknown").astype(str)
    categories = {x: i for i, x in enumerate(sorted(all_values.unique()))}
    return (train.fillna("unknown").astype(str).map(categories).to_numpy(),
            val.fillna("unknown").astype(str).map(categories).to_numpy(),
            test.fillna("unknown").astype(str).map(categories).to_numpy(), categories)


def cmd_probes(args: argparse.Namespace) -> None:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    root = root_path(args.root)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets.split(","):
        meta = {s: load_metadata(root, dataset, s) for s in ["train", "val", "test"]}
        y = {s: load_y(root, args.cache, dataset, s) for s in ["train", "val", "test"]}
        for feature in FEATURES:
            arrays = {s: load_x(root, args.cache, dataset, s, feature) for s in ["train", "val", "test"]}
            mean = arrays["train"].mean(axis=0); std = arrays["train"].std(axis=0); std[std < 1e-6] = 1.0
            X = {s: (arrays[s] - mean) / std for s in arrays}
            mos = Ridge(alpha=args.ridge_alpha).fit(X["train"], y["train"])
            pv = mos.predict(X["val"]); pt = mos.predict(X["test"])
            mv, mt = metric_dict(y["val"], pv), metric_dict(y["test"], pt)
            rows.append({"dataset": dataset, "feature": feature, "task": "mos", "val_spearman": mv["spearman"], "test_spearman": mt["spearman"], "test_pearson": mt["pearson"], "test_mse": mt["mse"], "test_mae": mt["mae"]})
            for task in ["system_id", "condition"]:
                tr, va, te, categories = encode_labels(meta["train"][task], meta["val"][task], meta["test"][task])
                if len(categories) < 2:
                    continue
                clf = LogisticRegression(max_iter=1000, C=args.logistic_c, solver="lbfgs", multi_class="auto")
                clf.fit(X["train"], tr)
                qv, qt = clf.predict(X["val"]), clf.predict(X["test"])
                rows.append({"dataset": dataset, "feature": feature, "task": task,
                             "n_classes": len(categories), "val_accuracy": accuracy_score(va, qv),
                             "test_accuracy": accuracy_score(te, qt), "test_balanced_accuracy": balanced_accuracy_score(te, qt),
                             "test_macro_f1": f1_score(te, qt, average="macro", zero_division=0)})
            log(f"probes {dataset}/{feature}")
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"probes output={out}")


def cmd_baselines(args: argparse.Namespace) -> None:
    """Matched raw-feature Ridge baselines for basic/core/full coalitions."""
    from sklearn.linear_model import Ridge
    root = root_path(args.root)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    specs = {"basic": FIXED_MASKS["basic"], "core": FIXED_MASKS["core"], "full": FIXED_MASKS["full"]}
    for dataset in args.datasets.split(","):
        y = {s: load_y(root, args.cache, dataset, s) for s in ["train", "val", "test"]}
        for spec, mask in specs.items():
            fs = names(mask)
            arrays = {s: [load_x(root, args.cache, dataset, s, f) for f in fs] for s in ["train", "val", "test"]}
            mean = [x.mean(axis=0) for x in arrays["train"]]
            std = [x.std(axis=0) for x in arrays["train"]]
            std = [np.where(s < 1e-6, 1.0, s) for s in std]
            X = {s: np.concatenate([(x - m) / z for x, m, z in zip(arrays[s], mean, std)], axis=1) for s in arrays}
            model = Ridge(alpha=args.ridge_alpha, solver="lsqr")
            model.fit(X["train"], y["train"])
            mv = metric_dict(y["val"], model.predict(X["val"]))
            mt = metric_dict(y["test"], model.predict(X["test"]))
            rows.append({"dataset": dataset, "spec": spec, "mask": mask, "features": "+".join(fs),
                         "dimension": X["train"].shape[1], "val_spearman": mv["spearman"],
                         "test_spearman": mt["spearman"], "test_pearson": mt["pearson"],
                         "test_mse": mt["mse"], "test_mae": mt["mae"]})
            log(f"matched baseline {dataset}/{spec} dim={X['train'].shape[1]}")
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"matched baselines output={out}")


def cmd_permutation(args: argparse.Namespace) -> None:
    root = root_path(args.root)
    df = pd.read_csv(root / args.predictions)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (dataset, spec, seed), g in df.groupby(["dataset", "spec", "seed"], sort=True):
        if spec not in FIXED_MASKS and not str(spec).startswith(("selected_", "basic_plus_", "core_plus_")):
            continue
        g = g.sort_values("item_index")
        mask = int(g["mask"].iloc[0]); y = g["mos"].to_numpy(float); base = g["prediction"].to_numpy(float)
        bm = metric_dict(y, base)
        for feature in names(mask):
            rng = np.random.default_rng(int(seed) + BITS[feature] + mask * 17)
            shuffled = g[f"expert_{feature}"].to_numpy(float)[rng.permutation(len(g))]
            pred = base - g[f"weight_{feature}"].to_numpy(float) * g[f"expert_{feature}"].to_numpy(float) + g[f"weight_{feature}"].to_numpy(float) * shuffled
            pm = metric_dict(y, pred)
            rows.append({"dataset": dataset, "spec": spec, "seed": seed, "mask": mask, "feature": feature,
                         "base_mse": bm["mse"], "permuted_mse": pm["mse"], "mse_drop": pm["mse"] - bm["mse"],
                         "base_spearman": bm["spearman"], "permuted_spearman": pm["spearman"],
                         "spearman_drop": bm["spearman"] - pm["spearman"]})
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"permutation rows={len(rows)} output={out}")


def cmd_stability(args: argparse.Namespace) -> None:
    import torch
    root = root_path(args.root)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    device = device_arg(args.device)
    fit_gate, solve_experts = m6_functions(root)
    modes = ["softmax", "equal", "linear"]
    rows = []
    seeds = list(range(args.n_stability_seeds))
    for dataset in args.datasets.split(","):
        y_train, p_train, p_val, p_test = expert_predictions(root, args.cache, dataset, device, args.alpha)
        y_val = load_y(root, args.cache, dataset, "val"); y_test = load_y(root, args.cache, dataset, "test")
        for spec in ["basic", "core", "stable_ssl", "speech_ssl", "audio_ssl", "full"]:
            mask = FIXED_MASKS[spec]; idx = [FEATURES.index(f) for f in names(mask)]
            if "softmax" in modes:
                weights = fit_gate(p_train[:, idx], y_train, seeds, args.gate_steps, args.batch_size, device)
                for si, seed in enumerate(seeds):
                    pv = (p_val[:, idx] * weights[si][None, :]).sum(dim=1).detach().cpu().numpy()
                    pt = (p_test[:, idx] * weights[si][None, :]).sum(dim=1).detach().cpu().numpy()
                    mv, mt = metric_dict(y_val, pv), metric_dict(y_test, pt)
                    rows.append({"dataset": dataset, "spec": spec, "mode": "softmax", "seed": seed,
                                 "val_spearman": mv["spearman"], "test_spearman": mt["spearman"],
                                 "test_mse": mt["mse"], "weights_json": json.dumps({f: float(weights[si][j].detach().cpu()) for j, f in enumerate(names(mask))})})
            if "equal" in modes:
                w = np.full(len(idx), 1.0 / len(idx), dtype=np.float32)
                pv = (p_val[:, idx].detach().cpu().numpy() * w).sum(axis=1); pt = (p_test[:, idx].detach().cpu().numpy() * w).sum(axis=1)
                mv, mt = metric_dict(y_val, pv), metric_dict(y_test, pt)
                rows.append({"dataset": dataset, "spec": spec, "mode": "equal", "seed": -1,
                             "val_spearman": mv["spearman"], "test_spearman": mt["spearman"], "test_mse": mt["mse"], "weights_json": json.dumps(dict(zip(names(mask), w.tolist())))})
            if "linear" in modes:
                p = p_train[:, idx].detach().cpu().numpy(); pv0 = p_val[:, idx].detach().cpu().numpy(); pt0 = p_test[:, idx].detach().cpu().numpy()
                design = np.column_stack([p, np.ones(len(p))]); coef = np.linalg.lstsq(design, y_train, rcond=None)[0]
                pv = np.column_stack([pv0, np.ones(len(pv0))]) @ coef; pt = np.column_stack([pt0, np.ones(len(pt0))]) @ coef
                mv, mt = metric_dict(y_val, pv), metric_dict(y_test, pt)
                rows.append({"dataset": dataset, "spec": spec, "mode": "linear", "seed": -1,
                             "val_spearman": mv["spearman"], "test_spearman": mt["spearman"], "test_mse": mt["mse"],
                             "weights_json": json.dumps(dict(zip(names(mask), coef[:-1].tolist())))})
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"stability rows={len(rows)} output={out}")


def cmd_cost(args: argparse.Namespace) -> None:
    root = root_path(args.root)
    rows = []
    for dataset in args.datasets.split(","):
        for split in ["train", "val", "test"]:
            for feature in FEATURES:
                path = cache_path(root, args.cache, dataset, split, feature)
                start = time.perf_counter(); array = np.load(path, mmap_mode="r"); _ = float(np.asarray(array[: min(len(array), 16)]).mean()); elapsed = time.perf_counter() - start
                rows.append({"dataset": dataset, "split": split, "feature": feature, "n_items": len(array), "dimension": int(array.shape[1]) if array.ndim > 1 else 1, "cache_bytes": path.stat().st_size, "sample_load_seconds": elapsed})
    out = root / args.output; out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False)
    log(f"cost rows={len(rows)} output={out}")


def cmd_report(args: argparse.Namespace) -> None:
    """Build a compact Markdown evidence report from whatever queue stages exist."""
    root = root_path(args.root)
    folder = root / args.folder
    out = root / args.output
    sections = ["# Follow-up evidence report\n", "This file is generated from the queued experiments. It is an evidence index, not a replacement for the paper.\n"]
    files = {
        "Bootstrap": folder / "bootstrap_summary.csv",
        "Shapley": folder / "shapley_validation.csv",
        "Pair interactions": folder / "pair_interaction_validation.csv",
        "LODO": folder / "lodo_results.csv",
        "Shortcut probes": folder / "probe_results.csv",
        "Matched baselines": folder / "matched_baselines.csv",
        "Permutation": folder / "permutation_results.csv",
        "Gate stability": folder / "gate_stability.csv",
        "Cost": folder / "cost_audit.csv",
    }
    for title, path in files.items():
        sections.append(f"## {title}\n")
        if not path.exists():
            sections.append("Not run yet.\n")
            continue
        frame = pd.read_csv(path)
        sections.append(f"Rows: {len(frame)}. Source: `{path.relative_to(root)}`.\n")
        if title == "Bootstrap" and not frame.empty:
            cols = [c for c in ["dataset", "comparison", "metric", "estimate", "ci95_low", "ci95_high", "prob_positive"] if c in frame]
            sections.append(frame[cols].head(40).to_markdown(index=False) + "\n")
        elif title == "Shapley" and not frame.empty:
            summary = frame.groupby("feature", as_index=False)["shapley_mse_reduction"].mean().sort_values("shapley_mse_reduction", ascending=False)
            sections.append(summary.to_markdown(index=False) + "\n")
        elif title == "LODO" and not frame.empty:
            summary = frame.groupby(["source", "target", "spec"], as_index=False)["target_test_spearman"].mean()
            sections.append(summary.to_markdown(index=False) + "\n")
        elif title == "Shortcut probes" and not frame.empty:
            cols = [c for c in ["dataset", "feature", "task", "test_spearman", "test_accuracy", "test_balanced_accuracy", "test_macro_f1"] if c in frame]
            sections.append(frame[cols].head(80).to_markdown(index=False) + "\n")
        elif title == "Matched baselines" and not frame.empty:
            cols = [c for c in ["dataset", "spec", "dimension", "val_spearman", "test_spearman", "test_mse"] if c in frame]
            sections.append(frame[cols].to_markdown(index=False) + "\n")
        elif title == "Permutation" and not frame.empty:
            summary = frame.groupby(["dataset", "feature"], as_index=False)[["mse_drop", "spearman_drop"]].mean().sort_values("spearman_drop", ascending=False)
            sections.append(summary.to_markdown(index=False) + "\n")
        elif title == "Gate stability" and not frame.empty:
            cols = [c for c in ["dataset", "spec", "mode", "test_spearman", "test_mse"] if c in frame]
            sections.append(frame[cols].head(80).to_markdown(index=False) + "\n")
        elif title == "Cost" and not frame.empty:
            summary = frame.groupby("feature", as_index=False).agg(cache_gb=("cache_bytes", lambda x: float(np.sum(x)) / 1e9), dimension=("dimension", "first"))
            sections.append(summary.to_markdown(index=False) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")
    log(f"report output={out}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    def common(q):
        q.add_argument("--root", default="."); q.add_argument("--cache", default="results/explainability_ridge_full/cache")
        q.add_argument("--datasets", default=",".join(DATASETS))
    q = sub.add_parser("predictions"); common(q); q.add_argument("--m6-csv", default="results/explainability_m6_gpu_sequential/m6_results.csv"); q.add_argument("--output", default="results/paper_missing"); q.add_argument("--device", default="auto"); q.add_argument("--alpha", type=float, default=10.0); q.add_argument("--gate-steps", type=int, default=25); q.add_argument("--batch-size", type=int, default=4096); q.set_defaults(func=cmd_predictions)
    q = sub.add_parser("bootstrap"); common(q); q.add_argument("--m6-csv", default="results/explainability_m6_gpu_sequential/m6_results.csv"); q.add_argument("--predictions", default="results/paper_missing/item_predictions.csv"); q.add_argument("--output", default="results/paper_missing/bootstrap_summary.csv"); q.add_argument("--n-bootstrap", type=int, default=2000); q.set_defaults(func=cmd_bootstrap)
    q = sub.add_parser("shapley"); common(q); q.add_argument("--m6-csv", default="results/explainability_m6_gpu_sequential/m6_results.csv"); q.add_argument("--output", default="results/paper_missing/shapley_validation.csv"); q.set_defaults(func=cmd_shapley)
    q = sub.add_parser("lodo"); common(q); q.add_argument("--output", default="results/paper_missing/lodo_results.csv"); q.add_argument("--device", default="auto"); q.add_argument("--alpha", type=float, default=10.0); q.add_argument("--gate-steps", type=int, default=25); q.add_argument("--batch-size", type=int, default=4096); q.set_defaults(func=cmd_lodo)
    q = sub.add_parser("probes"); common(q); q.add_argument("--output", default="results/paper_missing/probe_results.csv"); q.add_argument("--ridge-alpha", type=float, default=10.0); q.add_argument("--logistic-c", type=float, default=1.0); q.set_defaults(func=cmd_probes)
    q = sub.add_parser("baselines"); common(q); q.add_argument("--output", default="results/paper_missing/matched_baselines.csv"); q.add_argument("--ridge-alpha", type=float, default=10.0); q.set_defaults(func=cmd_baselines)
    q = sub.add_parser("permutation"); common(q); q.add_argument("--predictions", default="results/paper_missing/item_predictions.csv"); q.add_argument("--output", default="results/paper_missing/permutation_results.csv"); q.set_defaults(func=cmd_permutation)
    q = sub.add_parser("stability"); common(q); q.add_argument("--output", default="results/paper_missing/gate_stability.csv"); q.add_argument("--device", default="auto"); q.add_argument("--alpha", type=float, default=10.0); q.add_argument("--gate-steps", type=int, default=25); q.add_argument("--batch-size", type=int, default=4096); q.add_argument("--n-stability-seeds", type=int, default=20); q.set_defaults(func=cmd_stability)
    q = sub.add_parser("cost"); common(q); q.add_argument("--output", default="results/paper_missing/cost_audit.csv"); q.set_defaults(func=cmd_cost)
    q = sub.add_parser("report"); common(q); q.add_argument("--folder", default="results/paper_missing"); q.add_argument("--output", default="results/paper_missing/followup_report.md"); q.set_defaults(func=cmd_report)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
