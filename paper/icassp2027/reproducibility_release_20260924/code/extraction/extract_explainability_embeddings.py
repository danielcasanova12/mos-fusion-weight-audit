#!/usr/bin/env python3
"""Resumable multi-dataset extraction for the MOS explainability study.

The script extracts one file per utterance and mirrors the audio path below
``embeddings/<dataset>/<split>/<feature>``. Existing valid files are preserved.
It also writes ``metadata_explainability.csv`` and JSON audit reports.

Core power-set features:
  whisper, contentvec12, wavlm, beats, auditory_erb, speaker,
  rmvpe_cont, rmvpe_quant, ced, egemaps

Controlled encoder-swap ablations (not extra power-set families):
  hubert, wav2vec2
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


# The alphaai host currently exposes an incompatible cuDNN runtime. Keep the
# workaround opt-in so compatible machines still use cuDNN normally.
if os.environ.get("MOS_DISABLE_CUDNN") == "1":
    torch.backends.cudnn.enabled = False


PROJECT = Path(__file__).resolve().parent
EMBEDDINGS_ROOT = PROJECT / "embeddings"
REPORT_ROOT = PROJECT / "reports" / "explainability_extraction"

DATASETS = {
    "brspeech": {
        "root": PROJECT / "datasets/Datasets_mos/Datasets_mos/BRSPEECH_MOS_DATASET_v2",
        "csv": "{split}.csv",
        "path_column": "filepath",
        "output_name": "brspeech",
    },
    "bvcc": {
        "root": PROJECT / "datasets/Datasets_mos/bvcc/main/DATA",
        "csv": "sets/{split}.csv",
        "path_column": "filepath",
        "output_name": "bvcc",
    },
    "singmos": {
        "root": PROJECT / "datasets/Datasets_mos/Datasets_mos/singmos/DATA",
        "csv": "sets/{split}.csv",
        "path_column": "filepath",
        "output_name": "singmos",
    },
    "tmhintqi": {
        "root": PROJECT / "datasets/Datasets_mos/Datasets_mos/TMHINTQI",
        "csv": "{split}.csv",
        "path_column": "filepath",
        "output_name": "tmhintqi",
    },
}

CORE_FEATURES = [
    "whisper",
    "contentvec12",
    "wavlm",
    "beats",
    "auditory_erb",
    "speaker",
    "rmvpe_cont",
    "rmvpe_quant",
    "ced",
    "egemaps",
]
SWAP_FEATURES = ["hubert", "wav2vec2"]
ALL_FEATURES = CORE_FEATURES + SWAP_FEATURES

OUTPUT_DIR_NAMES = {
    "whisper": "whisper",
    "contentvec12": "contentvec",
    "wavlm": "wavlm_final",
    "beats": "beats",
    "auditory_erb": "auditory_erb",
    "speaker": "speaker",
    "rmvpe_cont": "f0_rmvpe",
    "rmvpe_quant": "f0_rmvpe_quant",
    "ced": "ced",
    "egemaps": "egemaps",
    "hubert": "hubert_final",
    "wav2vec2": "wav2vec2_final",
}

EXPECTED = {
    "whisper": {"ndim": 2, "last_dim": 1280, "dtype": "float"},
    "contentvec12": {"ndim": 2, "last_dim": 768, "dtype": "float"},
    "wavlm": {"ndim": 2, "last_dim": 768, "dtype": "float"},
    "beats": {"ndim": 2, "last_dim": 768, "dtype": "float"},
    "auditory_erb": {"ndim": 2, "last_dim": 64, "dtype": "float"},
    "speaker": {"ndim": 1, "last_dim": 192, "dtype": "float"},
    "rmvpe_cont": {"ndim": 1, "dtype": "float"},
    "rmvpe_quant": {"ndim": 1, "dtype": "int", "min": 0, "max": 255},
    "ced": {"ndim": 2, "dtype": "float"},
    "egemaps": {"ndim": 1, "last_dim": 88, "dtype": "float"},
    "hubert": {"ndim": 2, "last_dim": 768, "dtype": "float"},
    "wav2vec2": {"ndim": 2, "last_dim": 768, "dtype": "float"},
}

MODEL_MANIFEST = {
    "whisper": {"checkpoint": "openai/whisper-large-v3", "hidden": "last", "output": "[T,1280] fp16"},
    "contentvec12": {"checkpoint": "lengyue233/content-vec-best", "hidden": 12, "output": "[T,768]"},
    "wavlm": {"checkpoint": "microsoft/wavlm-base-plus", "hidden": 12, "output": "[T,768]"},
    "beats": {"checkpoint": "BEATs_iter3_plus_AS2M", "hidden": "final", "output": "[T,768] fp16"},
    "auditory_erb": {"frontend": "64-band triangular ERB filterbank", "n_fft": 512, "hop": 160, "output": "[T,64] fp16"},
    "speaker": {"checkpoint": "speechbrain/spkrec-ecapa-voxceleb", "output": "[192] L2-normalized"},
    "rmvpe_cont": {"checkpoint": "weights/rmvpe.pt", "output": "F0 Hz [T], unvoiced=0"},
    "rmvpe_quant": {"source": "rmvpe_cont", "scale": "mel", "bins": 256, "output": "indices [T], bin0=unvoiced"},
    "ced": {"checkpoint": "mispeech/ced-small", "hidden": "encoder final", "output": "[T,D] fp16"},
    "egemaps": {"feature_set": "eGeMAPSv02", "level": "Functionals", "output": "[88]"},
    "hubert": {"checkpoint": "facebook/hubert-base-ls960", "hidden": 12, "output": "[T,768]"},
    "wav2vec2": {"checkpoint": "facebook/wav2vec2-base", "hidden": 12, "output": "[T,768]"},
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_torch_save(value: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_records(dataset: str, split: str) -> tuple[dict, pd.DataFrame, list[Path], list[Path]]:
    spec = DATASETS[dataset]
    root = Path(spec["root"])
    csv_path = root / str(spec["csv"]).format(split=split)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    column = str(spec["path_column"])
    if column not in frame.columns:
        raise KeyError(f"Column {column!r} absent from {csv_path}")
    relative = [Path(str(value)) for value in frame[column].tolist()]
    absolute = [path if path.is_absolute() else root / path for path in relative]
    missing_audio = [str(path) for path in absolute if not path.exists()]
    if missing_audio:
        raise RuntimeError(f"{dataset}/{split}: {len(missing_audio)} missing audio files; first={missing_audio[:3]}")
    return spec, frame, relative, absolute


def output_directory(dataset: str, split: str, feature: str) -> Path:
    name = str(DATASETS[dataset]["output_name"])
    return EMBEDDINGS_ROOT / name / split / OUTPUT_DIR_NAMES[feature]


def expected_paths(dataset: str, split: str, feature: str, relative: Iterable[Path]) -> list[Path]:
    out = output_directory(dataset, split, feature)
    return [out / path.with_suffix(".pt") for path in relative]


def tensor_error(tensor: object, feature: str) -> str | None:
    if not isinstance(tensor, torch.Tensor):
        return "not_tensor"
    expected = EXPECTED[feature]
    if tensor.ndim != expected["ndim"]:
        return f"ndim={tensor.ndim} expected={expected['ndim']}"
    if tensor.numel() == 0:
        return "empty"
    if "last_dim" in expected and tensor.shape[-1] != expected["last_dim"]:
        return f"last_dim={tensor.shape[-1]} expected={expected['last_dim']}"
    if expected["dtype"] == "float" and not tensor.dtype.is_floating_point:
        return f"dtype={tensor.dtype} expected=float"
    if expected["dtype"] == "int" and tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return f"dtype={tensor.dtype} expected=int"
    if tensor.dtype.is_floating_point and not torch.isfinite(tensor).all():
        return "non_finite"
    if "min" in expected and int(tensor.min()) < expected["min"]:
        return f"min={int(tensor.min())}"
    if "max" in expected and int(tensor.max()) > expected["max"]:
        return f"max={int(tensor.max())}"
    return None


def audit_feature(dataset: str, split: str, feature: str, deep: bool) -> dict:
    _, _, relative, _ = load_records(dataset, split)
    paths = expected_paths(dataset, split, feature, relative)
    missing: list[str] = []
    corrupt: list[dict] = []
    shapes: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    for path in tqdm(paths, desc=f"audit {dataset}/{split}/{feature}", disable=not deep):
        if not path.exists():
            missing.append(str(path))
            continue
        if not deep:
            continue
        try:
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            error = tensor_error(tensor, feature)
            if error:
                corrupt.append({"path": str(path), "error": error})
            else:
                shapes[str(tuple(tensor.shape))] += 1
                dtypes[str(tensor.dtype)] += 1
        except Exception as exc:
            corrupt.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return {
        "dataset": dataset,
        "split": split,
        "feature": feature,
        "expected": len(paths),
        "present": len(paths) - len(missing),
        "missing": len(missing),
        "corrupt_or_invalid": len(corrupt),
        "complete": not missing and not corrupt,
        "missing_examples": missing[:10],
        "invalid_examples": corrupt[:10],
        "shape_counts": dict(shapes),
        "dtype_counts": dict(dtypes),
        "deep_audit": deep,
    }


def write_metadata(dataset: str, split: str) -> Path:
    spec, frame, relative, _ = load_records(dataset, split)
    for feature in ALL_FEATURES:
        paths = expected_paths(dataset, split, feature, relative)
        frame[f"{feature}_path"] = [str(path) for path in paths]
        frame[f"{feature}_available"] = [path.exists() for path in paths]
    destination = EMBEDDINGS_ROOT / str(spec["output_name"]) / split / "metadata_explainability.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _delegate_existing(dataset: str, split: str, feature: str, whisper_model: str) -> None:
    spec, _, _, files = load_records(dataset, split)
    input_root = str(spec["root"])
    out = str(output_directory(dataset, split, feature))
    filelist = [str(path) for path in files]

    if feature in ("wavlm", "hubert", "wav2vec2"):
        _materialize_final_layer_from_legacy(dataset, split, feature)
        if audit_feature(dataset, split, feature, deep=False)["complete"]:
            return

    if feature == "whisper":
        from extract_embs.extract_whisper_embeddings import extract_whisper_embeddings
        extract_whisper_embeddings(filelist, input_root, out, whisper_model, layers=[-1], save_dtype=torch.float16, crop_padding=True)
    elif feature == "contentvec12":
        from extract_embs.extract_contentvec import extract_contentvec_embeddings
        extract_contentvec_embeddings(filelist, input_root, out, "contentvec-best", layer=12, pool=False, validate_existing=True)
    elif feature == "wavlm":
        from extract_embs.extract_wavlm_embeddings import extract_wavlm_embeddings
        extract_wavlm_embeddings(filelist, input_root, out, "wavlm-base-plus", layer=12, pool=False)
    elif feature == "hubert":
        from extract_embs.extract_hubert_embeddings import extract_hubert_embeddings
        extract_hubert_embeddings(filelist, input_root, out, "hubert-base", layer=12, pool=False)
    elif feature == "wav2vec2":
        from extract_embs.extract_wav2vec2_embeddings import extract_wav2vec2_embeddings
        extract_wav2vec2_embeddings(filelist, input_root, out, "wav2vec2-base", layer=12, pool=False)
    elif feature == "speaker":
        from extract_embs.extract_speaker_embeddings import extract_speaker_embeddings
        extract_speaker_embeddings(filelist, input_root, out, "ecapa-tdnn", aggregate="mean", normalize=True)
    elif feature == "rmvpe_cont":
        from extract_embs.extract_f0_rmvpe import extract_f0_rmvpe_embeddings
        extract_f0_rmvpe_embeddings(filelist, input_root, out)
    elif feature == "rmvpe_quant":
        from extract_embs.extract_f0_rmvpe import extract_f0_rmvpe_quant_embeddings
        source = str(output_directory(dataset, split, "rmvpe_cont"))
        extract_f0_rmvpe_quant_embeddings(filelist, input_root, out, source_f0_dir=source)
    else:
        raise ValueError(f"No delegated extractor for {feature}")


def _materialize_final_layer_from_legacy(dataset: str, split: str, feature: str) -> None:
    """Reuse legacy [layers,T,D] files without overwriting them.

    Older experiments stored all 13 hidden states in ``wavlm/``, ``hubert/``
    and ``wav2vec2/``. The explainability study fixes the final representation
    and writes it to a distinct ``*_final`` directory.
    """
    _, _, relative, _ = load_records(dataset, split)
    legacy_root = EMBEDDINGS_ROOT / str(DATASETS[dataset]["output_name"]) / split / feature
    destinations = expected_paths(dataset, split, feature, relative)
    for rel_path, destination in tqdm(
        list(zip(relative, destinations)),
        desc=f"materialize {feature}_final {dataset}/{split}",
    ):
        if destination.exists():
            continue
        source = legacy_root / rel_path.with_suffix(".pt")
        if not source.exists():
            continue
        try:
            tensor = torch.load(source, map_location="cpu", weights_only=True)
            if tensor.ndim == 3 and tensor.shape[0] >= 2:
                tensor = tensor[-1]
            if tensor_error(tensor, feature) is None:
                atomic_torch_save(tensor, destination)
        except Exception:
            continue


def _load_mono_16k(path: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    if waveform.numel() == 0:
        raise ValueError("empty waveform")
    return waveform.float()


def extract_egemaps(dataset: str, split: str) -> None:
    import opensmile

    spec, _, relative, files = load_records(dataset, split)
    out_paths = expected_paths(dataset, split, "egemaps", relative)
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    errors = []
    for audio, output in tqdm(list(zip(files, out_paths)), desc=f"eGeMAPS {dataset}/{split}"):
        if output.exists():
            continue
        try:
            features = smile.process_file(str(audio))
            atomic_torch_save(torch.tensor(features.values.squeeze(), dtype=torch.float32), output)
        except Exception as exc:
            errors.append((str(audio), str(exc)))
    if errors:
        raise RuntimeError(f"eGeMAPS errors={len(errors)} first={errors[:3]}")


def _hz_to_erb(frequency: torch.Tensor) -> torch.Tensor:
    return 21.4 * torch.log10(1.0 + 0.00437 * frequency)


def _erb_to_hz(erb: torch.Tensor) -> torch.Tensor:
    return (torch.pow(10.0, erb / 21.4) - 1.0) / 0.00437


def erb_filterbank(sample_rate: int = 16000, n_fft: int = 512, bands: int = 64, f_min: float = 50.0) -> torch.Tensor:
    bins = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    erb_points = torch.linspace(_hz_to_erb(torch.tensor(f_min)), _hz_to_erb(torch.tensor(sample_rate / 2.0)), bands + 2)
    hz_points = _erb_to_hz(erb_points)
    bank = torch.zeros(bands, bins.numel())
    for index in range(bands):
        left, center, right = hz_points[index : index + 3]
        rising = (bins - left) / max(float(center - left), 1e-8)
        falling = (right - bins) / max(float(right - center), 1e-8)
        bank[index] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return bank


def extract_auditory_erb(dataset: str, split: str) -> None:
    _, _, relative, files = load_records(dataset, split)
    out_paths = expected_paths(dataset, split, "auditory_erb", relative)
    window = torch.hann_window(400)
    bank = erb_filterbank()
    errors = []
    for audio, output in tqdm(list(zip(files, out_paths)), desc=f"ERB {dataset}/{split}"):
        if output.exists():
            continue
        try:
            waveform = _load_mono_16k(audio)
            spectrum = torch.stft(waveform, n_fft=512, hop_length=160, win_length=400, window=window, return_complex=True)
            power = spectrum.abs().pow(2)
            representation = torch.log1p(bank @ power).transpose(0, 1).contiguous()
            atomic_torch_save(representation.to(torch.float16), output)
        except Exception as exc:
            errors.append((str(audio), str(exc)))
    if errors:
        raise RuntimeError(f"ERB errors={len(errors)} first={errors[:3]}")


def extract_ced(dataset: str, split: str, device: torch.device) -> None:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    _, _, relative, files = load_records(dataset, split)
    out_paths = expected_paths(dataset, split, "ced", relative)
    processor = AutoFeatureExtractor.from_pretrained("mispeech/ced-small", trust_remote_code=True)
    model = AutoModelForAudioClassification.from_pretrained("mispeech/ced-small", trust_remote_code=True).to(device).eval()
    errors = []
    for audio, output in tqdm(list(zip(files, out_paths)), desc=f"CED {dataset}/{split}"):
        if output.exists():
            continue
        try:
            waveform = _load_mono_16k(audio)
            inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
            input_values = inputs["input_values"].to(device)
            with torch.inference_mode():
                hidden = model.encoder(input_values).logits.squeeze(0)
            atomic_torch_save(hidden.to(torch.float16).cpu(), output)
        except Exception as exc:
            errors.append((str(audio), f"{type(exc).__name__}: {exc}"))
            clear_memory()
    del model, processor
    clear_memory()
    if errors:
        raise RuntimeError(f"CED errors={len(errors)} first={errors[:3]}")


def ensure_beats_assets() -> tuple[Path, Path]:
    repo = PROJECT / "third_party_beats_repo"
    beats_source = repo / "beats"
    if not (beats_source / "BEATs.py").exists():
        if repo.exists():
            raise RuntimeError(f"Incomplete BEATs checkout: {repo}")
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "https://github.com/microsoft/unilm.git", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "beats"], check=True)
    from huggingface_hub import hf_hub_download
    checkpoint = Path(hf_hub_download(repo_id="Bencr/beats-checkpoints", filename="BEATs_iter3_plus_AS2M.pt", repo_type="dataset"))
    return beats_source, checkpoint


def extract_beats(dataset: str, split: str, device: torch.device) -> None:
    beats_source, checkpoint_path = ensure_beats_assets()
    sys.path.insert(0, str(beats_source))
    from BEATs import BEATs, BEATsConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BEATs(BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()

    _, _, relative, files = load_records(dataset, split)
    out_paths = expected_paths(dataset, split, "beats", relative)
    errors = []
    for audio, output in tqdm(list(zip(files, out_paths)), desc=f"BEATs {dataset}/{split}"):
        if output.exists():
            continue
        try:
            waveform = _load_mono_16k(audio).unsqueeze(0).to(device)
            padding_mask = torch.zeros_like(waveform, dtype=torch.bool)
            with torch.inference_mode():
                hidden, _ = model.extract_features(waveform, padding_mask=padding_mask)
            atomic_torch_save(hidden.squeeze(0).to(torch.float16).cpu(), output)
        except Exception as exc:
            errors.append((str(audio), f"{type(exc).__name__}: {exc}"))
            clear_memory()
    del model, checkpoint
    clear_memory()
    if errors:
        raise RuntimeError(f"BEATs errors={len(errors)} first={errors[:3]}")


CUSTOM_EXTRACTORS: dict[str, Callable] = {
    "egemaps": extract_egemaps,
    "auditory_erb": extract_auditory_erb,
    "ced": extract_ced,
    "beats": extract_beats,
}


def extract_one(dataset: str, split: str, feature: str, args: argparse.Namespace) -> dict:
    started = now()
    report = {"dataset": dataset, "split": split, "feature": feature, "started_at": started, "status": "running"}
    try:
        before = audit_feature(dataset, split, feature, deep=False)
        if before["complete"]:
            report["status"] = "already_complete"
        elif feature in CUSTOM_EXTRACTORS:
            extractor = CUSTOM_EXTRACTORS[feature]
            if feature in ("ced", "beats"):
                extractor(dataset, split, get_device(args.device))
            else:
                extractor(dataset, split)
            report["status"] = "extracted"
        else:
            _delegate_existing(dataset, split, feature, args.whisper_model)
            report["status"] = "extracted"
        after = audit_feature(dataset, split, feature, deep=args.deep_audit_after)
        report["audit"] = after
        if not after["complete"]:
            raise RuntimeError(f"post-extraction audit incomplete: {after}")
        report["metadata"] = str(write_metadata(dataset, split))
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    report["finished_at"] = now()
    destination = REPORT_ROOT / dataset / split / f"{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    atomic_json_save(report, destination)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "error":
        raise RuntimeError(report["error"])
    return report


def run_audit(args: argparse.Namespace) -> int:
    results = []
    for dataset in args.datasets:
        for split in args.splits:
            for feature in args.features:
                result = audit_feature(dataset, split, feature, deep=args.deep_audit)
                results.append(result)
                print(
                    f"{dataset:9s} {split:5s} {feature:14s} "
                    f"present={result['present']:5d}/{result['expected']:5d} "
                    f"missing={result['missing']:5d} invalid={result['corrupt_or_invalid']:5d}"
                )
            write_metadata(dataset, split)
    report = {"created_at": now(), "deep_audit": args.deep_audit, "results": results}
    destination = REPORT_ROOT / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    atomic_json_save(report, destination)
    print(f"Audit report: {destination}")
    return 1 if any(not result["complete"] for result in results) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--features", nargs="+", choices=ALL_FEATURES, default=ALL_FEATURES)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--whisper-model", default="whisper-large-v3")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--deep-audit", action="store_true")
    parser.add_argument("--deep-audit-after", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json_save(MODEL_MANIFEST, REPORT_ROOT / "model_manifest.json")
    if args.audit_only:
        return run_audit(args)
    for dataset in args.datasets:
        for split in args.splits:
            for feature in args.features:
                extract_one(dataset, split, feature, args)
                clear_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
