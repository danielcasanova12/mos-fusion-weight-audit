#!/usr/bin/env bash
set -euo pipefail
python scripts/build_public_tables.py
python scripts/verify_release.py
