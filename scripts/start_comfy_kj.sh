#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLCONFIGDIR="$PWD/.local/matplotlib"
exec .venv-comfy/bin/python vendor/ComfyUI/main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --offline
