# Environment variables for the LIBERO / LIBERO-Plus simulator clients.
#
#   source scripts/env_libero.sh [libero|libero_plus] [checkout-dir]
#
# Source this AFTER activating the simulator conda env (it reads CONDA_PREFIX).
# checkout-dir is where scripts/install_sim_libero.sh cloned the simulators
# (default: <repo>/third_party). LIBERO reads its config from LIBERO_CONFIG_PATH,
# which the installer writes inside the checkout; LIBERO-Plus additionally needs
# LIBERO_HOME (its source checkout, for task_classification.json). Override any
# variable by exporting it before sourcing.
_BENCH="${1:-${BENCH:-libero}}"
_SIM_DIR="${2:-${SIM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party}}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export MAGICK_HOME="${MAGICK_HOME:-$CONDA_PREFIX}"   # LIBERO-Plus image corruptions use ImageMagick
fi

if [[ "$_BENCH" == "libero_plus" ]]; then
  export LIBERO_HOME="${LIBERO_HOME:-$_SIM_DIR/LIBERO-plus}"
  export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$_SIM_DIR/LIBERO-plus/.libero_plus}"
else
  export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$_SIM_DIR/LIBERO/.libero}"
fi
unset _BENCH _SIM_DIR
