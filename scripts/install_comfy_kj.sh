#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor
clone_pinned() {
 local url="$1" target="$2" revision="$3"
 if [ ! -d "$target/.git" ]; then git clone "$url" "$target"; git -C "$target" checkout "$revision"; fi
 [ "$(git -C "$target" rev-parse HEAD)" = "$revision" ] || { echo "Unexpected source revision: $target"; exit 1; }
}
clone_pinned https://github.com/kijai/ComfyUI-LivePortraitKJ vendor/ComfyUI-LivePortraitKJ 4d9dc6205b793ffd0fb319816136d9b8c0dbfdff
clone_pinned https://github.com/comfyanonymous/ComfyUI vendor/ComfyUI 4ac1c173afd42f1141dcebace42d8091998d3b8c
[ -x .venv/bin/python ] || { echo 'Install the existing local model runtime first.'; exit 1; }
python3.12 -m venv .venv-comfy
.venv-comfy/bin/python - <<'PY'
import sysconfig
from pathlib import Path
Path(sysconfig.get_paths()['purelib'],'avatar_shared.pth').write_text(str(Path.cwd()/'.venv/lib/python3.12/site-packages')+'\n')
PY
.venv-comfy/bin/pip install -r research/comfy-runtime-requirements.txt
.venv/bin/python scripts/prepare_kj_models.py
mkdir -p vendor/ComfyUI/custom_nodes
ln -sfn "$PWD/vendor/ComfyUI-LivePortraitKJ" vendor/ComfyUI/custom_nodes/LivePortraitKJ
ln -sfn "$PWD/backend/comfy_io_nodes.py" vendor/ComfyUI/custom_nodes/avatar_studio_io.py
