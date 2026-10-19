#!/usr/bin/env bash
# Create the training / serving environment for PointmapVLA.
#
# Requirements: conda (or mamba), an NVIDIA driver supporting CUDA 12.8, and
# internet access. Compiled kernels (flash-attn, causal-conv1d) ship prebuilt
# wheels for torch 2.7 + CUDA 12 + Python 3.10; if no wheel matches your
# platform they are built from source, which needs nvcc 12.8 and ~30 minutes.
#
# Usage: scripts/install_train_env.sh [env-name]
set -euo pipefail

ENV_NAME="${1:-pointmapvla}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda create -y -n "$ENV_NAME" python=3.10
PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"

"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -r "$ROOT/requirements-pointmapvla.txt"

# Compiled kernels. The prebuilt wheels below match torch 2.7 + CUDA 12 +
# Python 3.10 + the cxx11 ABI of the PyPI torch wheels, and need no CUDA
# toolkit. On another platform, set CUDA_HOME to a CUDA 12.8 toolkit and run
#   pip install --no-build-isolation flash_attn==2.8.3.post1 causal-conv1d==1.6.2.post1
# instead (their setup.py downloads a matching wheel when one exists, else builds).
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
CC_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
"$PY" -m pip install "$FA_URL" "$CC_URL"

"$PY" -m pip install -e "$ROOT"

"$PY" - <<'EOF'
import torch, transformers, flash_attn, fla
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("flash_attn", flash_attn.__version__)
print("fla", fla.__version__)
EOF
echo "Environment '$ENV_NAME' is ready. Activate with: conda activate $ENV_NAME"
