#!/usr/bin/env bash
# Create the RoboCasa kitchen-24 simulator environment used to evaluate
# PointmapVLA. The simulator runs in its own conda environment because it pins
# numpy 1.23 and an older robosuite, which are incompatible with the training
# environment. The policy server runs in the training environment and talks to
# the simulator over a websocket.
#
# Installs the exact revisions the paper results were produced with:
#   robosuite        DAVIAN-Robotics/robosuite        tag pointmapvla-v1 (upstream 232ce7d)
#   robocasa-kitchen DAVIAN-Robotics/robocasa-kitchen tag pointmapvla-v1 (robocasa 0.2.0 fork)
# plus ~5 GB of kitchen assets downloaded by robocasa's own script.
#
# Usage: scripts/install_sim_robocasa.sh [env-name] [checkout-dir]
set -euo pipefail

ENV_NAME="${1:-robocasa_kitchen}"
SIM_DIR="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party}"
TAG="pointmapvla-v1"
mkdir -p "$SIM_DIR"

conda create -y -c conda-forge -n "$ENV_NAME" python=3.10
# EGL/GL runtime libraries for headless MuJoCo rendering.
conda install -y -n "$ENV_NAME" -c conda-forge "libegl=1.7.0" "libgl=1.7.0" "libglvnd=1.7.0" "libglx=1.7.0"
PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"

[[ -d "$SIM_DIR/robosuite" ]]        || git clone --branch "$TAG" --depth 1 https://github.com/DAVIAN-Robotics/robosuite.git "$SIM_DIR/robosuite"
[[ -d "$SIM_DIR/robocasa-kitchen" ]] || git clone --branch "$TAG" --depth 1 https://github.com/DAVIAN-Robotics/robocasa-kitchen.git "$SIM_DIR/robocasa-kitchen"

"$PY" -m pip install --upgrade pip
# Two steps on purpose: robosuite declares mujoco>=3.3.0 while robocasa 0.2.0 pins
# mujoco==3.2.6, so installing them together fails to resolve. robocasa's pin wins
# (that is the version both were run with); pip only warns about robosuite's range.
"$PY" -m pip install -e "$SIM_DIR/robosuite"
"$PY" -m pip install -e "$SIM_DIR/robocasa-kitchen" "mujoco==3.2.6" "gymnasium==0.29.1" websockets msgpack av opencv-python tyro typing_extensions
# numpy must be installed last: robocasa 0.2.0 requires 1.23 and newer numpy breaks its numba dependency.
"$PY" -m pip install "numpy==1.23.3"

(
  cd "$SIM_DIR/robocasa-kitchen"
  echo y | "$PY" robocasa/scripts/download_kitchen_assets.py
  "$PY" robocasa/scripts/setup_macros.py
)

"$PY" -c "import robocasa, robosuite, mujoco, numpy; print('mujoco', mujoco.__version__, 'numpy', numpy.__version__)"
echo "Environment '$ENV_NAME' is ready. Activate with: conda activate $ENV_NAME"
