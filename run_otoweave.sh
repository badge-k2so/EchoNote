#!/usr/bin/env bash
# OtoWeave launcher for macOS (Apple Silicon).
#
# Usage:
#   ./run_otoweave.sh                 # normal launch
#   ./run_otoweave.sh --demo          # seed the demo lesson
#   ./run_otoweave.sh --data-root DIR # use a custom lesson storage folder
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -x ".venv/bin/python" ]; then
  echo "[エラー] .venv が見つかりません。先に ./setup_mac.sh を実行してください。" >&2
  exit 1
fi

# Same idea as the Windows distribution's OtoWeaveを起動.bat: keep the
# HuggingFace cache local to this folder and never let libraries reach out
# to the network at runtime.
export HF_HOME="$ROOT/hf-cache"
export HF_HUB_OFFLINE=1

exec .venv/bin/python -m otoweave_app.main "$@"
