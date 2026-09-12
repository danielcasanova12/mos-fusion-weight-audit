#!/usr/bin/env python3
"""Write deterministic SHA-256 metadata for public artifact files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "metadata" / "artifact_manifest.json"


def main() -> None:
    files = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts or path == OUTPUT:
            continue
        rel = path.relative_to(ROOT).as_posix()
        files[rel] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    payload = {
        "artifact_version": "v1.0-icassp2027-submission",
        "hash_algorithm": "sha256",
        "files": files,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} with {len(files)} files")


if __name__ == "__main__":
    main()
