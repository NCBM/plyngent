#!/usr/bin/env bash
# Thin wrapper — all path logic lives in wine_pdm.py / wine_env.py.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$root/scripts/wine_pdm.py" "$@"
