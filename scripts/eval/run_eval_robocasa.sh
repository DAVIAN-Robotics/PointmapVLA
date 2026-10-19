#!/usr/bin/env bash
# RoboCasa Kitchen 24-task evaluation on one node (no scheduler needed).
#
# Starts PAIRS policy servers per GPU, each paired with a simulator client that
# works through its share of the (repetition, task) list sequentially, then runs
# the aggregator. Existing result JSONs are skipped, so re-running resumes.
#
# Required:
#   CKPT         run directory, checkpoint file, or Hugging Face repo id
# Optional (defaults = paper protocol):
#   OUT          results root            (results/robocasa) -> <OUT>/rep<r>/eval_json/<Task>.json
#   GPUS         comma-separated GPU ids (CUDA_VISIBLE_DEVICES, else 0)
#   PAIRS        server/client pairs per GPU (2)
#   K            parallel envs per client (2)
#   EPS          episodes per task (50)
#   REPS         repetitions; policy seeds 0..REPS-1 (3)
#   TASKS        space-separated subset of the 24 tasks (all)
#   SEED         environment seed, episode i uses SEED*100000+i (42)
#   N_ACT        action steps executed per query (16)
#   MAX_STEPS    max env steps per episode (720)
#   SERVER_ENV   conda env with the training/serving stack (pointmapvla)
#   SIM_ENV      conda env with RoboCasa (robocasa_kitchen)
#   PORT_BASE    first websocket port (20000)
#
# Example: CKPT=DAVIAN-Robotics/PointmapVLA-RoboCasa GPUS=0,1 scripts/eval/run_eval_robocasa.sh
set -euo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then sed -n '2,25p' "$0"; exit 0; fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT=${CKPT:?set CKPT=<run dir | checkpoint file | HF repo id>}
OUT=${OUT:-$REPO/results/robocasa}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
PAIRS=${PAIRS:-2}; K=${K:-2}; EPS=${EPS:-50}; REPS=${REPS:-3}
SEED=${SEED:-42}; N_ACT=${N_ACT:-16}; MAX_STEPS=${MAX_STEPS:-720}
SERVER_ENV=${SERVER_ENV:-pointmapvla}; SIM_ENV=${SIM_ENV:-robocasa_kitchen}
PORT_BASE=${PORT_BASE:-20000}
# A task subset cannot produce the full 24-task number; let the aggregator print partial rates.
PARTIAL_FLAG=${TASKS+--allow-partial}
TASKS=${TASKS:-"PnPMicrowaveToCounter PnPCounterToMicrowave TurnOffStove CoffeeSetupMug
PnPCabToCounter PnPCounterToCab PnPCounterToSink PnPCounterToStove PnPStoveToCounter
PnPSinkToCounter OpenDoubleDoor TurnOnMicrowave CoffeeServeMug OpenSingleDoor TurnOnStove
TurnSinkSpout OpenDrawer TurnOffSinkFaucet CoffeePressButton CloseDoubleDoor TurnOnSinkFaucet
TurnOffMicrowave CloseSingleDoor CloseDrawer"}

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1
# shellcheck disable=SC1091
source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
LOGD="$OUT/logs"; mkdir -p "$LOGD"
# Work items "rep task", distributed round-robin over all workers.
ITEMS=()
for ((r = 0; r < REPS; r++)); do for t in $TASKS; do ITEMS+=("$r $t"); done; done
NWORKERS=$(( ${#GPU_LIST[@]} * PAIRS ))
echo "[robocasa] ckpt=$CKPT gpus=$GPUS pairs=$PAIRS K=$K eps=$EPS reps=$REPS items=${#ITEMS[@]} out=$OUT"

worker() {
  local gpu=$1 pair=$2 wid=$3
  local port=$(( PORT_BASE + gpu * 16 + pair ))
  local srv_log="$LOGD/server_gpu${gpu}_p${pair}.log"
  local todo=()
  for ((i = wid; i < ${#ITEMS[@]}; i += NWORKERS)); do
    set -- ${ITEMS[$i]}
    [[ -f "$OUT/rep$1/eval_json/$2.json" ]] && continue
    todo+=("${ITEMS[$i]}")
  done
  if [[ ${#todo[@]} -eq 0 ]]; then echo "[w$wid] nothing to do"; return 0; fi

  conda activate "$SERVER_ENV"
  # Many servers in one node: cap the CPU threads of each (loading and inference are GPU-bound).
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    # Truncate first: on a resumed run the readiness grep below could otherwise see the
    # previous run's "listening on" line before the background redirect truncates the file.
    : > "$srv_log"
    python deployment/model_server/server_policy.py --ckpt "$CKPT" --port "$port" --use_bf16 >> "$srv_log" 2>&1 &
  local srv=$!
  for _ in $(seq 1 360); do
    grep -q 'listening on' "$srv_log" && break
    kill -0 $srv 2>/dev/null || { echo "[w$wid] server died"; tail -20 "$srv_log"; return 2; }
    sleep 5
  done
  grep -q 'listening on' "$srv_log" || { echo "[w$wid] server not ready after 30 min"; kill $srv; return 2; }
  conda deactivate; conda activate "$SIM_ENV"
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$gpu
  local rc=0
  for item in "${todo[@]}"; do
    set -- $item
    local rep=$1 task=$2 t0=$SECONDS
    python -m examples.simBenchmarks.Robocasa_kitchen.eval_files.simulation_env \
      --args.port "$port" --args.env-name "$task" --args.n-episodes "$EPS" --args.n-envs "$K" \
      --args.max-episode-steps "$MAX_STEPS" --args.n-action-steps "$N_ACT" --args.seed "$SEED" \
      --args.policy-seed "$rep" --args.result-dir "$OUT/rep$rep/eval_json" \
      > "$LOGD/client_rep${rep}_${task}.log" 2>&1 || { echo "[w$wid] FAIL rep$rep $task (see $LOGD/client_rep${rep}_${task}.log)"; rc=1; continue; }
    echo "[w$wid] rep$rep $task done in $((SECONDS - t0))s"
    kill -0 $srv 2>/dev/null || { echo "[w$wid] server gone"; rc=2; break; }
  done
  kill $srv 2>/dev/null || true
  return $rc
}

pids=(); wid=0
for gpu in "${GPU_LIST[@]}"; do
  for ((p = 0; p < PAIRS; p++)); do
    worker "$gpu" "$p" "$wid" > "$LOGD/worker_${wid}.log" 2>&1 &
    pids+=($!); wid=$((wid + 1)); sleep 3  # stagger checkpoint reads
  done
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
cat "$LOGD"/worker_*.log

conda activate "$SERVER_ENV"
REP_DIRS=(); for ((r = 0; r < REPS; r++)); do REP_DIRS+=("$OUT/rep$r"); done
python examples/simBenchmarks/aggregate_eval_results.py robocasa "${REP_DIRS[@]}" --episodes "$EPS" --out "$OUT/summary.json" ${PARTIAL_FLAG:-} || rc=1
exit $rc
