#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON_BIN:-python3.12}"
command -v "$python_bin" >/dev/null || { echo 'Install Python 3.12 first (brew install python@3.12).'; exit 1; }
mkdir -p vendor
clone_pinned() {
  local url="$1" target="$2" revision="$3"
  if [ ! -d "$target/.git" ]; then git clone "$url" "$target"; git -C "$target" checkout "$revision"; fi
  [ "$(git -C "$target" rev-parse HEAD)" = "$revision" ] || { echo "Unexpected revision in $target; preserve your changes and restore $revision."; exit 1; }
}
clone_pinned https://github.com/barnent1/musetalk-mac vendor/musetalk-mac 7fd019315127d7f31e2e7f9853547ddceabbeb6e
clone_pinned https://github.com/TMElyralab/MuseTalk vendor/MuseTalk 0a89dec45a0192b824e3cf4daf96c239440c5ed8
"$python_bin" -m venv .venv
.venv/bin/pip install -r backend/requirements.lock.txt
.venv/bin/python scripts/patch_models.py
PATH="$PWD/.venv/bin:$PATH" bash vendor/musetalk-mac/download_weights_mac.sh
.venv/bin/python backend/worker.py doctor
