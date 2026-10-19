#!/bin/bash
#SBATCH --job-name=pointmapvla_robocasa24
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=360G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
#
# RoboCasa Kitchen 24-task evaluation, paper protocol: 24 tasks x 50 episodes x 720
# steps, action chunk 16, env seed 42, REPS repetitions with policy seeds 0..REPS-1.
# One repetition takes about 1 h 15 min on 2 GPUs with 9 server/client pairs each.
#
# Submit from the repository root (the repo is taken from SLURM_SUBMIT_DIR):
#   sbatch --export=ALL,CKPT=DAVIAN-Robotics/PointmapVLA-RoboCasa \
#       examples/simBenchmarks/Robocasa_kitchen/eval_files/eval_kitchen24.sbatch
# Optional exports: OUT, REPS (3), PAIRS (9), K (2), EPS (50), TASKS, SERVER_ENV, SIM_ENV.
# If your worker image lacks a compiler for runtime JIT kernels, add e.g.
#   --container=nvcr.io/nvidia/pytorch:25.04-py3
# Re-submitting the same command resumes: finished <Task>.json files are skipped.
set -euo pipefail

REPO=${REPO:-$SLURM_SUBMIT_DIR}
export CKPT=${CKPT:?set CKPT=<run dir | checkpoint file | HF repo id>}
export GPUS=$(seq -s, 0 $(( ${SLURM_GPUS_ON_NODE:-1} - 1 )))
export PAIRS=${PAIRS:-9} K=${K:-2} REPS=${REPS:-3} EPS=${EPS:-50}
export OUT=${OUT:-$REPO/results/robocasa}
export PYTHONNOUSERSITE=1

bash "$REPO/scripts/eval/run_eval_robocasa.sh"
