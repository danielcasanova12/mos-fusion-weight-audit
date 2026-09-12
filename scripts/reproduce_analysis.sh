#!/usr/bin/env bash
set -euo pipefail
python scripts/verify_release.py
python scripts/build_public_tables.py
python scripts/build_supplement.py
python scripts/build_manifest.py
