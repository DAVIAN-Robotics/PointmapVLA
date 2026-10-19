#!/usr/bin/env bash
# RoboCasa Kitchen: evaluate ONE task on ONE GPU (server + client on this node).
# Thin wrapper over scripts/eval/run_eval_robocasa.sh with single-task defaults.
#
#   CKPT=<run dir | HF repo id> TASK=OpenSingleDoor bash examples/simBenchmarks/Robocasa_kitchen/eval_files/run_eval.sh
#
# Optional: GPU (0), EPS (50), K (2), OUT (results/robocasa_single), SERVER_ENV, SIM_ENV.
# Result: <OUT>/rep0/eval_json/<TASK>.json
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CKPT=${CKPT:?set CKPT=<run dir | checkpoint file | HF repo id>}
export CKPT TASKS="${TASK:-OpenSingleDoor}" GPUS="${GPU:-0}" PAIRS=1 REPS=1
export EPS="${EPS:-50}" K="${K:-2}" OUT="${OUT:-$REPO/results/robocasa_single}"
# A single task cannot pass the 24-task integrity check; print its JSON instead.
bash "$REPO/scripts/eval/run_eval_robocasa.sh" || true
cat "$OUT/rep0/eval_json/${TASK:-OpenSingleDoor}.json"
