#!/usr/bin/env python3
"""Write deterministic SHA-256 metadata for public artifact files."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "metadata" / "artifact_manifest.json"
SHA_OUTPUT = ROOT / "metadata" / "file_hashes.sha256"


def main() -> None:
    files = {}
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    for rel in sorted(tracked):
        path = ROOT / rel
        if not path.is_file() or path in {OUTPUT, SHA_OUTPUT}:
            continue
        normalized = Path(rel).as_posix()
        files[normalized] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    payload = {
        "artifact_version": "v1.0-icassp2027-submission",
        "hash_algorithm": "sha256",
        "files": files,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [f"{entry['sha256']}  {rel}" for rel, entry in files.items()]
    SHA_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote JSON and SHA-256 manifests for {len(files)} tracked files")


if __name__ == "__main__":
    main()
