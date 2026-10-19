#!/usr/bin/env bash
# LIBERO (4 suites x 10 tasks x 50 trials) or LIBERO-Plus (10,030 tasks x 1 trial)
# evaluation on one node: one policy server per GPU, shards distributed over the
# GPUs and run sequentially per GPU with the vectorized runner, then the aggregator.
# Existing shard JSONs are skipped, so re-running resumes.
#
# Required:
#   CKPT         run directory, checkpoint file, or Hugging Face repo id
# Optional (defaults = paper protocol):
#   BENCH        libero | libero_plus (libero)
#   SEED         server + environment seed (7); the paper uses 7, 8, 9
#   OUT          run directory (results/<BENCH>/s<SEED>) -> <OUT>/<BENCH>/<suite>_shard<A>-<B>.json
#   GPUS         comma-separated GPU ids (CUDA_VISIBLE_DEVICES, else 0)
#   K            parallel env slots = policy batch size (12). Changes the noise draws; keep fixed.
#   EPS          trials per task (50 for libero, 1 for libero_plus)
#   SHARD_SIZE   tasks per shard (5 for libero, 200 for libero_plus)
#   SERVER_ENV   conda env with the training/serving stack (pointmapvla)
#   SIM_ENV      conda env with the simulator (libero | libero_plus)
#   PORT_BASE    first websocket port (21000)
#   LIBERO_HOME / LIBERO_CONFIG_PATH   see scripts/env_libero.sh
#
# Example: CKPT=DAVIAN-Robotics/PointmapVLA-LIBERO BENCH=libero_plus SEED=7 GPUS=0,1 scripts/eval/run_eval_libero.sh
set -euo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then sed -n '2,24p' "$0"; exit 0; fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT=${CKPT:?set CKPT=<run dir | checkpoint file | HF repo id>}
BENCH=${BENCH:-libero}; SEED=${SEED:-7}; K=${K:-12}
OUT=${OUT:-$REPO/results/$BENCH/s$SEED}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
SERVER_ENV=${SERVER_ENV:-pointmapvla}; PORT_BASE=${PORT_BASE:-21000}
SUITES=(libero_spatial libero_object libero_goal libero_10)
if [[ "$BENCH" == "libero" ]]; then
  SIM_ENV=${SIM_ENV:-libero}; EPS=${EPS:-50}; SHARD_SIZE=${SHARD_SIZE:-5}
  COUNTS=(10 10 10 10)
elif [[ "$BENCH" == "libero_plus" ]]; then
  SIM_ENV=${SIM_ENV:-libero_plus}; EPS=${EPS:-1}; SHARD_SIZE=${SHARD_SIZE:-200}
  COUNTS=(2402 2518 2591 2519)
else
  echo "BENCH must be libero or libero_plus"; exit 2
fi

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export STARVLA_EVAL_SEED=$SEED
# shellcheck disable=SC1091
source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
RESD="$OUT/$BENCH"; LOGD="$OUT/logs"; mkdir -p "$RESD" "$LOGD"
# Shards "suite start end"; the aggregator checks that they tile every suite.
SHARDS=()
for i in 0 1 2 3; do
  n=${COUNTS[$i]}; s=0
  while (( s < n )); do e=$(( s + SHARD_SIZE )); (( e > n )) && e=$n; SHARDS+=("${SUITES[$i]} $s $e"); s=$e; done
done
NGPU=${#GPU_LIST[@]}
echo "[$BENCH] ckpt=$CKPT seed=$SEED gpus=$GPUS K=$K eps=$EPS shards=${#SHARDS[@]} out=$OUT"

worker() {
  local gpu=$1 wid=$2
  local port=$(( PORT_BASE + gpu ))
  local srv_log="$LOGD/server_gpu${gpu}.log"
  local todo=()
  for ((i = wid; i < ${#SHARDS[@]}; i += NGPU)); do
    set -- ${SHARDS[$i]}
    [[ -s "$RESD/$1_shard$(printf '%05d' "$2")-$(printf '%05d' "$3").json" ]] && continue
    todo+=("${SHARDS[$i]}")
  done
  if [[ ${#todo[@]} -eq 0 ]]; then echo "[w$wid] nothing to do"; return 0; fi

  conda activate "$SERVER_ENV"
  # Truncate first: on a resumed run the readiness grep below could otherwise see the
  # previous run's "listening on" line before the background redirect truncates the file.
  : > "$srv_log"
  CUDA_VISIBLE_DEVICES=$gpu python deployment/model_server/server_policy.py --ckpt "$CKPT" --port "$port" --use_bf16 \
    >> "$srv_log" 2>&1 &
  local srv=$!
  for _ in $(seq 1 360); do
    grep -q 'listening on' "$srv_log" && break
    kill -0 $srv 2>/dev/null || { echo "[w$wid] server died"; tail -20 "$srv_log"; return 2; }
    sleep 5
  done
  grep -q 'listening on' "$srv_log" || { echo "[w$wid] server not ready after 30 min"; kill $srv; return 2; }
  conda deactivate; conda activate "$SIM_ENV"
  # shellcheck disable=SC1091
  source "$REPO/scripts/env_libero.sh" "$BENCH"
  export MUJOCO_EGL_DEVICE_ID=$gpu
  local rc=0
  for item in "${todo[@]}"; do
    set -- $item
    local suite=$1 a=$2 b=$3 t0=$SECONDS
    python examples/simBenchmarks/LIBERO/eval_files/eval_libero_vec.py \
      --bench "$BENCH" --task-suite-name "$suite" --task-id-start "$a" --task-id-end "$b" \
      --num-trials-per-task "$EPS" --port "$port" --log-path "$RESD" --seed "$SEED" --k "$K" \
      >> "$LOGD/client_gpu${gpu}.log" 2>&1 || { echo "[w$wid] FAIL $suite [$a,$b) (see $LOGD/client_gpu${gpu}.log)"; rc=1; }
    echo "[w$wid] $suite [$a,$b) done in $((SECONDS - t0))s"
    kill -0 $srv 2>/dev/null || { echo "[w$wid] server gone"; rc=2; break; }
  done
  kill $srv 2>/dev/null || true
  return $rc
}

pids=(); wid=0
for gpu in "${GPU_LIST[@]}"; do
  worker "$gpu" "$wid" > "$LOGD/worker_${wid}.log" 2>&1 &
  pids+=($!); wid=$((wid + 1)); sleep 3
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
cat "$LOGD"/worker_*.log

conda activate "$SERVER_ENV"
python examples/simBenchmarks/aggregate_eval_results.py "$BENCH" "$OUT" --episodes "$EPS" --out "$OUT/summary.json" || rc=1
exit $rc
