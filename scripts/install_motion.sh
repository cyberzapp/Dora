#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d vendor/LivePortrait/.git ]; then
 git clone https://github.com/KlingAIResearch/LivePortrait vendor/LivePortrait
 git -C vendor/LivePortrait checkout 9b294b3d0536135442ea73cb01e6cb3ca7029dd3
fi
[ "$(git -C vendor/LivePortrait rev-parse HEAD)" = "9b294b3d0536135442ea73cb01e6cb3ca7029dd3" ]
.venv/bin/pip install 'numpy==1.26.4' 'tifffile==2024.8.30' 'tyro==0.8.5' 'onnxruntime==1.20.1' 'pykalman==0.9.7' 'scikit-image==0.24.0'
# Only human animation models. Our adapter uses MediaPipe, never InsightFace weights.
.venv/bin/huggingface-cli download KwaiVGI/LivePortrait --local-dir vendor/LivePortrait/pretrained_weights --include 'liveportrait/*'
echo "Motion models installed. Use backend/motion.py with the MediaPipe detector."
