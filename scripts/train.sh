#!/bin/bash
# Fine-tune PointmapVLA on a single multi-GPU machine (no scheduler).
#
#   scripts/train.sh examples/simBenchmarks/Robocasa_kitchen/train_files/pointmapvla_robocasa_kitchen.yaml
#   scripts/train.sh examples/simBenchmarks/LIBERO/train_files/pointmapvla_libero_80gb.yaml \
#       --trainer.max_train_steps 50          # any extra args are OmegaConf dotlist overrides
#
# Environment knobs (optional):
#   RUN_ID      run directory name under results/runs (default: the yaml's run_id)
#   DS_CONFIG   accelerate/DeepSpeed config (default: deepspeed_zero2.yaml; *_80gb yamls
#               pick deepspeed_zero2_accum4.yaml automatically)
#   NGPU        number of GPUs (default 4; see docs/training.md for why 4 is the minimum)
#
# Run it again with the same yaml/RUN_ID to resume from the newest checkpoint.
set -euo pipefail

CONFIG="${1:?usage: scripts/train.sh <training yaml> [dotlist overrides...]}"
shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [[ -z "${DS_CONFIG:-}" ]]; then
  case "$CONFIG" in
    *_80gb.yaml) DS_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_accum4.yaml ;;
    *)           DS_CONFIG=starVLA/config/deepseeds/deepspeed_zero2.yaml ;;
  esac
fi
NGPU="${NGPU:-4}"
RUN_ID="${RUN_ID:-$(sed -n 's/^run_id:[[:space:]]*//p' "$CONFIG" | head -1)}"

# DeepSpeed needs a CUDA toolkit location at import time (it checks op
# compatibility), even though no ops are compiled here. Point CUDA_HOME at the
# toolkit if it is not already set.
if [[ -z "${CUDA_HOME:-}" ]]; then
  for c in /usr/local/cuda-12.8 /usr/local/cuda; do
    [[ -d "$c" ]] && export CUDA_HOME="$c" && break
  done
fi
[[ -n "${CUDA_HOME:-}" ]] || echo "[train] warning: CUDA_HOME is not set; DeepSpeed may fail to import" >&2

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-offline}"

# A free rendezvous port, so two runs on one machine do not collide on accelerate's default 29500.
MASTER_PORT="${MASTER_PORT:-$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')}"

echo "[train] config=$CONFIG run_id=$RUN_ID gpus=$NGPU ds_config=$DS_CONFIG port=$MASTER_PORT"
accelerate launch \
  --config_file "$DS_CONFIG" \
  --num_processes "$NGPU" \
  --main_process_port "$MASTER_PORT" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG" \
  --run_id "$RUN_ID" \
  --run_root_dir results/runs \
  "$@"
