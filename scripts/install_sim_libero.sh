#!/usr/bin/env bash
# Create the LIBERO (and optionally LIBERO-Plus) simulator environment used to
# evaluate PointmapVLA. The simulator runs in its own conda environment; the
# policy server runs in the training environment and talks to it over a
# websocket.
#
# Installs the exact revisions the paper results were produced with:
#   LIBERO       DAVIAN-Robotics/LIBERO       tag pointmapvla-v1 (upstream 8f1084e + torch>=2.6 load fix)
#   LIBERO-plus  DAVIAN-Robotics/LIBERO-plus  tag pointmapvla-v1 (upstream 4976dc3 + torch>=2.6 load fix)
#
# Usage: scripts/install_sim_libero.sh [--plus] [env-name] [checkout-dir]
#   without --plus: env 'libero' with LIBERO
#   with    --plus: env 'libero_plus' with LIBERO-plus (its own fork of LIBERO)
set -euo pipefail

PLUS=0
if [[ "${1:-}" == "--plus" ]]; then PLUS=1; shift; fi
if [[ $PLUS -eq 1 ]]; then DEFAULT_ENV=libero_plus; else DEFAULT_ENV=libero; fi
ENV_NAME="${1:-$DEFAULT_ENV}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_DIR="${2:-$ROOT/third_party}"
TAG="pointmapvla-v1"
mkdir -p "$SIM_DIR"

conda create -y -n "$ENV_NAME" python=3.10
# EGL/GL runtime libraries for headless MuJoCo rendering.
conda install -y -n "$ENV_NAME" -c conda-forge "libegl=1.7.0" "libgl=1.7.0" "libglvnd=1.7.0" "libglx=1.7.0"
PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
# egl_probe (pulled in by robosuite 1.4) builds with CMake. Use the conda-forge
# binary: the PyPI `cmake` shim is not importable from pip's isolated build env.
conda install -y -n "$ENV_NAME" -c conda-forge cmake
export PATH="$(dirname "$PY"):$PATH"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install "mujoco==3.2.3" tyro matplotlib mediapy websockets msgpack imageio "opencv-python-headless==4.6.0.66"

if [[ $PLUS -eq 1 ]]; then
  REPO_DIR="$SIM_DIR/LIBERO-plus"
  [[ -d "$REPO_DIR" ]] || git clone --branch "$TAG" --depth 1 https://github.com/DAVIAN-Robotics/LIBERO-plus.git "$REPO_DIR"
  "$PY" -m pip install -e "$REPO_DIR"
  "$PY" -m pip install -r "$ROOT/examples/simBenchmarks/LIBERO-plus/eval_files/libero_plus_requirements.txt" scikit-image huggingface_hub
  # LIBERO-Plus perturbations use Wand, which needs the ImageMagick runtime.
  conda install -y -n "$ENV_NAME" -c conda-forge imagemagick
  # LIBERO-Plus keeps its scenes, textures and extra objects (6.4 GB) outside git, in
  # the upstream Hugging Face dataset Sylvest/LIBERO-plus (assets.zip). The archive
  # nests everything under a long machine-specific prefix, so extract the assets/
  # subtree only. Skipped when the directory is already populated.
  ASSETS="$REPO_DIR/libero/libero/assets"
  if [[ ! -d "$ASSETS/scenes" ]]; then
    "$PY" -m pip install -q "huggingface_hub>=0.30"
    "$PY" - "$REPO_DIR/.assets_download" "$ASSETS" <<'PYEOF'
import sys, zipfile, pathlib
from huggingface_hub import hf_hub_download
download_dir, target = sys.argv[1], pathlib.Path(sys.argv[2])
zip_path = hf_hub_download("Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset", local_dir=download_dir)
with zipfile.ZipFile(zip_path) as z:
    names = z.namelist()
    key = "/assets/"
    prefixes = {n[: n.index(key) + len(key)] for n in names if key in n}
    assert len(prefixes) == 1, f"unexpected archive layout: {sorted(prefixes)[:3]}"
    prefix = prefixes.pop()
    members = [n for n in names if n.startswith(prefix) and not n.endswith("/")]
    for i, n in enumerate(members):
        out = target / n[len(prefix):]
        out.parent.mkdir(parents=True, exist_ok=True)
        with z.open(n) as src, open(out, "wb") as dst:
            dst.write(src.read())
    print(f"extracted {len(members)} asset files to {target}")
PYEOF
    rm -rf "$REPO_DIR/.assets_download"
  fi
  CONFIG_DIR="$REPO_DIR/.libero_plus"
else
  REPO_DIR="$SIM_DIR/LIBERO"
  [[ -d "$REPO_DIR" ]] || git clone --branch "$TAG" --depth 1 https://github.com/DAVIAN-Robotics/LIBERO.git "$REPO_DIR"
  "$PY" -m pip install -e "$REPO_DIR"
  # LIBERO's setup.py declares no dependencies; these are what its env code imports.
  "$PY" -m pip install "robosuite==1.4.0" "bddl==1.0.1" "robomimic==0.2.0" "gym==0.25.2" "future==0.18.2" "easydict==1.9" huggingface_hub
  CONFIG_DIR="$REPO_DIR/.libero"
fi
# LIBERO's top-level `libero/` directory has no __init__.py, so the editable
# install's import finder cannot locate it; put the repository root on sys.path.
"$PY" - "$REPO_DIR" <<'PYEOF'
import pathlib, site, sys
pathlib.Path(site.getsitepackages()[0], "zz_libero_root.pth").write_text(sys.argv[1] + "\n")
PYEOF
# robosuite 1.4 pulls in opencv-python, whose current release needs numpy>=2.
# Keep only the headless 4.6 build the benchmarks were run with.
"$PY" -m pip uninstall -y opencv-python
"$PY" -m pip install --force-reinstall --no-deps "opencv-python-headless==4.6.0.66"
# robosuite 1.4 logs to a hard-coded /tmp/robosuite.log, which fails on shared
# machines where another user owns that file. Disable file logging through
# robosuite's own macros_private.py override (this also silences its warning
# that the override is missing).
"$PY" - <<'PYEOF'
import importlib.util, pathlib, re
pkg = pathlib.Path(importlib.util.find_spec("robosuite").origin).parent
src = (pkg / "macros.py").read_text()
(pkg / "macros_private.py").write_text(re.sub(r"^FILE_LOGGING_LEVEL = .*$", "FILE_LOGGING_LEVEL = None", src, flags=re.M))
PYEOF
# numpy must be installed last so nothing above upgrades it.
"$PY" -m pip install "numpy==1.24.4"

# LIBERO asks interactively for its asset paths the first time it is imported
# unless a config file exists. Write it so evaluation scripts never block on
# input(); scripts/env_libero.sh exports the matching LIBERO_CONFIG_PATH.
mkdir -p "$CONFIG_DIR"
PKG="$REPO_DIR/libero/libero"
cat > "$CONFIG_DIR/config.yaml" <<EOF
benchmark_root: $PKG
bddl_files: $PKG/bddl_files
init_states: $PKG/init_files
datasets: $REPO_DIR/datasets
assets: $PKG/assets
EOF

# Import what the evaluation client imports, with the renderer settings scripts/env_libero.sh uses.
LIBERO_CONFIG_PATH="$CONFIG_DIR" LIBERO_HOME="$REPO_DIR" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LD_LIBRARY_PATH="$(dirname "$PY")/../lib" \
  "$PY" -c "from libero.libero.envs import OffScreenRenderEnv; import robosuite, mujoco, numpy; print('mujoco', mujoco.__version__, 'robosuite', robosuite.__version__, 'numpy', numpy.__version__)"
echo "Environment '$ENV_NAME' is ready. Before evaluating, run: source scripts/env_libero.sh $([[ $PLUS -eq 1 ]] && echo libero_plus || echo libero) $SIM_DIR"
