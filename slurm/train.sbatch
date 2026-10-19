#!/bin/bash
# Fine-tune PointmapVLA on a Slurm cluster (one node, 4 GPUs).
#
#   CONFIG=examples/simBenchmarks/Robocasa_kitchen/train_files/pointmapvla_robocasa_kitchen.yaml \
#   RUN_ID=pointmapvla_robocasa_kitchen sbatch slurm/train.sbatch
#
# Environment knobs (all optional except CONFIG):
#   RUN_ID      run directory name under results/runs (default: the yaml's run_id)
#   CONDA_ENV   conda environment with the training stack (default: pointmapvla)
#   DS_CONFIG   accelerate/DeepSpeed config; use deepspeed_zero2_accum4.yaml with the *_80gb yamls
#   NGPU        number of GPUs (default 4; see docs/training.md for why 4 is the minimum)
#
# Resubmitting the same CONFIG/RUN_ID resumes from the newest checkpoint in the run
# dir (`trainer.is_resume: true`), so a preempted or time-limited job is simply
# submitted again. Adjust --time to your scheduler: 20k RoboCasa steps take about
# 7.5 h and 30k LIBERO steps about 9 h on 4 B200 GPUs.
#SBATCH --job-name=pointmapvla_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=results/logs/slurm-%x-%j.out
set -euo pipefail

: "${CONFIG:?set CONFIG=<training yaml>}"
# Slurm runs a spool copy of this script, so locate the repository from the submit dir.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO"
mkdir -p results/logs

CONDA_ENV="${CONDA_ENV:-pointmapvla}"
DS_CONFIG="${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
NGPU="${NGPU:-${SLURM_GPUS_ON_NODE:-4}}"
RUN_ID="${RUN_ID:-$(sed -n 's/^run_id:[[:space:]]*//p' "$CONFIG" | head -1)}"

if [[ -d "$CONDA_ENV" ]]; then
  # A path to an environment works without conda on PATH (another user's env, a shared install).
  export PATH="$CONDA_ENV/bin:$PATH"
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export PYTHONNOUSERSITE=1            # never pick up ~/.local packages inside the job
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

echo "[train] config=$CONFIG run_id=$RUN_ID gpus=$NGPU ds_config=$DS_CONFIG env=$CONDA_ENV"
accelerate launch \
  --config_file "$DS_CONFIG" \
  --num_processes "$NGPU" \
  --num_machines 1 --machine_rank 0 \
  --main_process_port "$MASTER_PORT" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG" \
  --run_id "$RUN_ID" \
  --run_root_dir results/runs \
  "$@"
