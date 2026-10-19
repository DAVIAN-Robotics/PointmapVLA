#!/bin/bash
#SBATCH --job-name=pointmapvla_libero
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=30
#SBATCH --mem=180G
#SBATCH --time=05:00:00
#SBATCH --output=slurm-%x-%j.out
#
# LIBERO (4 suites x 10 tasks x 50 trials, ~40 min) or LIBERO-Plus (10,030 tasks x 1
# trial, ~3 h on 2 GPUs) evaluation with the vectorized runner: one policy server per
# GPU, K=12 env slots, server + env seed SEED. The paper reports seeds 7, 8, 9.
# Each env slot is one simulator process, so K is bounded by the CPU allocation.
#
# Submit from the repository root (the repo is taken from SLURM_SUBMIT_DIR):
#   sbatch --export=ALL,CKPT=DAVIAN-Robotics/PointmapVLA-LIBERO,BENCH=libero,SEED=7 \
#       examples/simBenchmarks/LIBERO/eval_files/eval_libero_vec.sbatch
#   sbatch --gres=gpu:2 --export=ALL,CKPT=...,BENCH=libero_plus,SEED=7 \
#       examples/simBenchmarks/LIBERO/eval_files/eval_libero_vec.sbatch
# Optional exports: OUT, K (12), EPS, SHARD_SIZE, SERVER_ENV, SIM_ENV, LIBERO_HOME, LIBERO_CONFIG_PATH.
# If your worker image lacks a compiler for runtime JIT kernels, add e.g.
#   --container=nvcr.io/nvidia/pytorch:25.04-py3
# Re-submitting the same command resumes: finished shard JSONs are skipped.
set -euo pipefail

REPO=${REPO:-$SLURM_SUBMIT_DIR}
export CKPT=${CKPT:?set CKPT=<run dir | checkpoint file | HF repo id>}
export BENCH=${BENCH:-libero} SEED=${SEED:-7} K=${K:-12}
export GPUS=$(seq -s, 0 $(( ${SLURM_GPUS_ON_NODE:-1} - 1 )))
export OUT=${OUT:-$REPO/results/$BENCH/s$SEED}
export PYTHONNOUSERSITE=1

bash "$REPO/scripts/eval/run_eval_libero.sh"
