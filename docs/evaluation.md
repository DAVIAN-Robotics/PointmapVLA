# Evaluating PointmapVLA on LIBERO and LIBERO-Plus

This page walks through the LIBERO (4 suites) and LIBERO-Plus evaluations behind the
paper numbers. RoboCasa is covered in
[`examples/simBenchmarks/Robocasa_kitchen/README.md`](../examples/simBenchmarks/Robocasa_kitchen/README.md).
The upstream starVLA pages `examples/simBenchmarks/LIBERO/README.md` and
`examples/simBenchmarks/LIBERO-plus/README.md` describe upstream's own (non-pointmap)
evaluation flow and are kept verbatim; the PointmapVLA flow is this page.

## How evaluation works

Two processes, two conda environments:

* **Policy server** (`deployment/model_server/server_policy.py`, env `pointmapvla`):
  loads a checkpoint, serves normalised action chunks over a websocket, un-normalises them
  with the checkpoint's `dataset_statistics.json` and converts the 6-D rotation to
  axis-angle, so the client always receives environment-space actions. One server per GPU.
* **Simulator client** (`examples/simBenchmarks/LIBERO/eval_files/eval_libero_vec.py`,
  env `libero` or `libero_plus`): runs K environments in parallel as one policy batch,
  renders depth, builds robot-base-frame pointmaps with the same code the training loader
  uses, and writes one JSON per shard of tasks.

The client refuses to run if the server's handshake does not match (pointmap frame, state
layout, action chunk).

## Protocol

| | LIBERO | LIBERO-Plus |
|---|---|---|
| Checkpoint | `DAVIAN-Robotics/PointmapVLA-LIBERO` | same (zero-shot) |
| Episodes | 4 suites x 10 tasks x 50 trials = 2,000 | 10,030 tasks x 1 trial (2402 / 2518 / 2591 / 2519) |
| Action chunk / executed | 8 / 8 | 8 / 8 |
| Flow-matching Euler steps | 4 | 4 |
| Seeds | server + environment seed 7, 8, 9 (one run each) | same |
| Parallel environments K | 12 | 12 |
| Images | agentview + wrist, rotated 180°, 256² -> 224 | same |
| State | 11-D: EE position 3, rotation r6 6, gripper 2 | same |

K is the policy batch size and changes the noise draws; keep it at 12 to reproduce. The
paper reports the mean over the three seeds.

## Setup

```bash
scripts/install_train_env.sh             # env pointmapvla (serving)
scripts/install_sim_libero.sh            # env libero
scripts/install_sim_libero.sh --plus     # env libero_plus; downloads the 6.4 GB LIBERO-Plus asset pack
```

The simulator installers clone the pinned forks into `third_party/` and write LIBERO's
config file so the first import never prompts. In a simulator env, run
`source scripts/env_libero.sh libero` (or `libero_plus`) to export `MUJOCO_GL=egl`,
`LIBERO_CONFIG_PATH` and, for LIBERO-Plus, `LIBERO_HOME` (its checkout, which holds
`task_classification.json` for the per-category breakdown). The driver below sources it
for you.

## One-node driver

```bash
# LIBERO, one seed, one GPU (about 40 min on a B200)
CKPT=DAVIAN-Robotics/PointmapVLA-LIBERO BENCH=libero SEED=7 GPUS=0 scripts/eval/run_eval_libero.sh

# LIBERO-Plus, one seed, two GPUs (about 3 h)
CKPT=DAVIAN-Robotics/PointmapVLA-LIBERO BENCH=libero_plus SEED=7 GPUS=0,1 scripts/eval/run_eval_libero.sh
```

Variables (`scripts/eval/run_eval_libero.sh -h`):

| Variable | Default | Meaning |
|---|---|---|
| `CKPT` | required | run directory, weight file under `checkpoints/`, or Hugging Face repo id |
| `BENCH` | `libero` | `libero` or `libero_plus` |
| `SEED` | 7 | server + environment seed; the paper uses 7, 8, 9 |
| `OUT` | `results/<BENCH>/s<SEED>` | output root |
| `GPUS` | `CUDA_VISIBLE_DEVICES` or `0` | comma-separated GPU ids; one server per GPU |
| `K` | 12 | parallel env slots = policy batch size |
| `EPS` | 50 / 1 | trials per task |
| `SHARD_SIZE` | 5 / 200 | tasks per shard JSON |
| `SERVER_ENV`, `SIM_ENV` | `pointmapvla`, `libero` / `libero_plus` | conda env names |
| `PORT_BASE` | 21000 | first websocket port (port = base + GPU id) |

The driver starts one server per GPU, waits for it to listen, distributes the shards over
the GPUs, runs the client shard by shard, kills the servers and runs the aggregator on
that seed. Finished shard JSONs are skipped, so re-running the same command resumes.

Each env slot is a simulator process: K = 12 wants roughly 24-30 CPU cores per GPU.

### Output layout

```
results/libero/s7/
├── libero/
│   ├── libero_spatial_shard00000-00005.json
│   ├── libero_spatial_shard00005-00010.json
│   ├── libero_object_shard00000-00005.json
│   └── ...                                   8 shards (4 suites x 2)
├── logs/                                     server_gpu<N>.log, client_gpu<N>.log, worker_<N>.log
└── summary.json                              this seed only
```

LIBERO-Plus writes `results/libero_plus/s7/libero_plus/<suite>_shard<start>-<end>.json`
(200 tasks per shard, 52 shards). Each shard JSON holds per-task successes; the per-task
entries also carry the perturbation category for LIBERO-Plus.

## Aggregating over seeds

```bash
conda activate pointmapvla
python examples/simBenchmarks/aggregate_eval_results.py libero \
    results/libero/s7 results/libero/s8 results/libero/s9 --out results/libero/summary.json
python examples/simBenchmarks/aggregate_eval_results.py libero_plus \
    results/libero_plus/s7 results/libero_plus/s8 results/libero_plus/s9 --out results/libero_plus/summary.json
```

The aggregator first checks integrity: every suite must have exactly 10 tasks (LIBERO) or
its full task count (LIBERO-Plus) covered by shards without gaps or overlaps, with the
expected number of trials. It then prints per-suite and overall success rates as
mean ± sd (sample sd) over the seed directories, and for LIBERO-Plus the 7 perturbation
categories (camera, robot, language, light, background, noise, layout). A run that does
not cover the full protocol is reported as INCOMPLETE and no rate is printed unless
`--allow-partial` is given. Each run directory may contain the JSONs directly or in a
`libero/` / `libero_plus/` subdirectory.

## Running server and client by hand

Useful for debugging a single suite or shard.

Terminal 1 (serving env):

```bash
conda activate pointmapvla
python deployment/model_server/server_policy.py --ckpt DAVIAN-Robotics/PointmapVLA-LIBERO --port 21000 --use_bf16 --seed 7
```

`--seed` (or `STARVLA_EVAL_SEED`) seeds the server's RNGs once at start-up. The server
forces SDPA attention (`STARVLA_FORCE_SDPA=1`; flash-attention's variable-length kernel
is unreliable at serving batch sizes) and enables TF32 matmuls, matching the paper runs.

Terminal 2 (simulator env, repository root):

```bash
conda activate libero
source scripts/env_libero.sh libero
export PYTHONPATH=$PWD
python examples/simBenchmarks/LIBERO/eval_files/eval_libero_vec.py \
    --bench libero --task-suite-name libero_spatial --task-id-start 0 --task-id-end 5 \
    --num-trials-per-task 50 --port 21000 --log-path results/libero/s7/libero --seed 7 --k 12
```

For LIBERO-Plus use `conda activate libero_plus`, `source scripts/env_libero.sh libero_plus`,
`--bench libero_plus`, `--num-trials-per-task 1` and task ranges of up to 200.

Check that the simulator side renders depth and ships base-frame pointmaps without any
server:

```bash
python examples/simBenchmarks/LIBERO/eval_files/eval_libero_vec.py \
    --bench libero --task-suite-name libero_spatial --log-path results/selftest --pm-selftest
```

## Slurm

`examples/simBenchmarks/LIBERO/eval_files/eval_libero_vec.sbatch` wraps the driver for a
one-node job (1 GPU for LIBERO, 2 for LIBERO-Plus); edit the `#SBATCH` lines for your
cluster:

```bash
sbatch --export=ALL,CKPT=DAVIAN-Robotics/PointmapVLA-LIBERO,BENCH=libero,SEED=7 \
    examples/simBenchmarks/LIBERO/eval_files/eval_libero_vec.sbatch
sbatch --gres=gpu:2 --export=ALL,CKPT=DAVIAN-Robotics/PointmapVLA-LIBERO,BENCH=libero_plus,SEED=7 \
    examples/simBenchmarks/LIBERO/eval_files/eval_libero_vec.sbatch
```

Resubmitting the same command resumes. If the worker image has no C compiler, add a
container that does (the Qwen3.5 backbone JIT-compiles Triton kernels on first use), e.g.
`--container=nvcr.io/nvidia/pytorch:25.04-py3`.

## Evaluating your own fine-tune

Point `CKPT` at the run directory written by the trainer
(`results/runs/<run_id>`) or at a weight file inside it
(`results/runs/<run_id>/checkpoints/steps_30000_model.safetensors`). The run directory has
the same layout as the released checkpoints, so nothing else changes.
