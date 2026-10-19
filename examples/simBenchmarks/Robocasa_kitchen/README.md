# PointmapVLA on RoboCasa kitchen-24

Fine-tuning and evaluation of PointmapVLA on the 24 atomic RoboCasa kitchen tasks with
the MimicGen-300 (MG-300) demonstrations. This directory holds the data registry
(`train_files/data_registry/data_config.py`), the training configs (`train_files/*.yaml`)
and the evaluation client (`eval_files/`).

## 1. Install

```bash
scripts/install_train_env.sh          # env pointmapvla: training and the policy server
scripts/install_sim_robocasa.sh       # env robocasa_kitchen: the simulator (~5 GB of kitchen assets)
```

The simulator installer clones the pinned forks `DAVIAN-Robotics/robosuite` and
`DAVIAN-Robotics/robocasa-kitchen` (tag `pointmapvla-v1`) into `third_party/`, installs
mujoco 3.2.6 and numpy 1.23.3, downloads the kitchen assets and runs robocasa's
`setup_macros.py`. The simulator and the training stack cannot share an environment, so
evaluation always runs the two as separate processes connected over a websocket.

## 2. Data

```bash
conda activate pointmapvla
python scripts/download_data.py --bench robocasa
```

This downloads two LeRobot v3.0 datasets to where the configs expect them:

| Hugging Face repo | Local path | Content | Size |
|---|---|---|---|
| `DAVIAN-Robotics/robocasa-MG_300-r6` | `playground/Datasets/robocasa_kitchen_mg300_r6/robocasa-MG_300/` | RGB (3 cameras, 128 x 128), 18-D state, 15-D action; 7,200 episodes, 2.08 M frames | ~11 GB |
| `DAVIAN-Robotics/robocasa-MG_300-pointmap` | `playground/Datasets/robocasa-MG_300-pointmap/` | 256 x 256 depth videos (FFV1 `gray16le`, `uint16 = m x 10000`) + per-frame intrinsics/extrinsics for the same episodes | ~120 GB |

Verify a download with `python scripts/verify_data.py playground/Datasets/robocasa-MG_300-pointmap`
(sizes; add `--sha256` to hash). The loader joins the two datasets by episode and builds
robot-base-frame pointmaps on the fly; see the dataset cards for the full feature tables.

State and action layout ("r6" = rotation as the first two rows of the rotation matrix):

* state (18): base position 3 | base quaternion 4 | EE position 3 | EE rotation r6 6 | gripper 2.
  The action head receives the 11-D end-effector anchor (EE position, r6, gripper).
* action (15): EE position 3 | EE rotation r6 6 | gripper 1 | base motion 4 | control mode 1.

## 3. Fine-tune

```bash
conda activate pointmapvla
scripts/train.sh examples/simBenchmarks/Robocasa_kitchen/train_files/pointmapvla_robocasa_kitchen.yaml        # 4 x 32, B200-class GPUs
scripts/train.sh examples/simBenchmarks/Robocasa_kitchen/train_files/pointmapvla_robocasa_kitchen_80gb.yaml   # 4 x 8 x accum 4, 80 GB GPUs
```

Recipe of the released `DAVIAN-Robotics/PointmapVLA-RoboCasa`: initialised from
`DAVIAN-Robotics/PointmapVLA-Pretrained`, 20,000 optimizer steps, global batch 128 on 4
GPUs, action horizon 16, cosine schedule with 1,000 warm-up steps, `enable_thinking: false`
(an explicit empty `<think>` block in the prompt). Loading the pretrained weights remaps
the four action-head tensors whose shape depends on the 15/11 layout. Four GPUs is the
minimum (DeepSpeed ZeRO-2 partition size); see [docs/training.md](../../../docs/training.md).
The run directory `results/runs/pointmapvla_robocasa_kitchen/` has the same layout as the
released checkpoint and can be passed to the evaluation as is.

## 4. Evaluate

Paper protocol: 24 tasks x 50 episodes, max 720 steps, action chunk 16 executed in full,
flow-matching with 4 Euler steps, environment seed 42 (episode i uses `42*100000 + i`),
**three repetitions with policy noise seeds 0, 1, 2**, reported as mean ± sd over the
repetitions. Images are the three 256² renders (left, right, wrist) resized to 224.

```bash
CKPT=DAVIAN-Robotics/PointmapVLA-RoboCasa GPUS=0,1 scripts/eval/run_eval_robocasa.sh
```

`scripts/eval/run_eval_robocasa.sh` starts `PAIRS` policy servers per GPU, each paired
with a client that works through its share of the (repetition, task) list, then runs the
aggregator. Finished `<Task>.json` files are skipped, so re-running resumes. Variables
(`-h` lists them):

| Variable | Default | Meaning |
|---|---|---|
| `CKPT` | required | run directory, weight file under `checkpoints/`, or Hugging Face repo id |
| `OUT` | `results/robocasa` | results root |
| `GPUS` | `CUDA_VISIBLE_DEVICES` or `0` | comma-separated GPU ids |
| `PAIRS` | 2 | server/client pairs per GPU |
| `K` | 2 | parallel environments per client |
| `EPS` | 50 | episodes per task |
| `REPS` | 3 | repetitions; policy seeds `0..REPS-1` |
| `TASKS` | all 24 | space-separated subset (aggregator then runs with `--allow-partial`) |
| `SEED` | 42 | environment seed |
| `N_ACT` | 16 | action steps executed per query |
| `MAX_STEPS` | 720 | max environment steps per episode |
| `SERVER_ENV`, `SIM_ENV` | `pointmapvla`, `robocasa_kitchen` | conda env names |
| `PORT_BASE` | 20000 | first websocket port |

One repetition takes about 1 h 15 min on 2 GPUs with 9 pairs each
(`eval_files/eval_kitchen24.sbatch`, the Slurm version:
`sbatch --export=ALL,CKPT=DAVIAN-Robotics/PointmapVLA-RoboCasa examples/simBenchmarks/Robocasa_kitchen/eval_files/eval_kitchen24.sbatch`).

### Result layout

```
results/robocasa/
├── rep0/eval_json/<Task>.json        24 files per repetition
├── rep1/eval_json/<Task>.json
├── rep2/eval_json/<Task>.json
├── logs/                             server_gpu<N>_p<M>.log, client_rep<r>_<Task>.log, worker_<N>.log
└── summary.json
```

Each `<Task>.json` records `env`, `success_rate`, the per-episode `successes` list and the
`seed` / `policy_seed` / `n_envs` it was produced with.

### Aggregator

```bash
conda activate pointmapvla
python examples/simBenchmarks/aggregate_eval_results.py robocasa \
    results/robocasa/rep0 results/robocasa/rep1 results/robocasa/rep2 \
    --out results/robocasa/summary.json
```

Integrity first: every repetition must contain exactly the 24 tasks with 50 episodes
each, otherwise it is reported as INCOMPLETE and no number is printed (unless
`--allow-partial`). Then a per-task table and the overall success rate as mean ± sd
(sample sd) over repetitions. The driver runs this automatically at the end.

### One task, by hand

Single task on one GPU through the driver:

```bash
CKPT=DAVIAN-Robotics/PointmapVLA-RoboCasa TASK=OpenSingleDoor bash examples/simBenchmarks/Robocasa_kitchen/eval_files/run_eval.sh
```

Or run the two processes yourself. Terminal 1 (serving env):

```bash
conda activate pointmapvla
python deployment/model_server/server_policy.py --ckpt DAVIAN-Robotics/PointmapVLA-RoboCasa --port 20000 --use_bf16
```

Terminal 2 (simulator env, repository root):

```bash
conda activate robocasa_kitchen
export PYTHONPATH=$PWD MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python -m examples.simBenchmarks.Robocasa_kitchen.eval_files.simulation_env \
    --args.port 20000 --args.env-name OpenSingleDoor --args.n-episodes 50 --args.n-envs 2 \
    --args.max-episode-steps 720 --args.n-action-steps 16 --args.seed 42 \
    --args.policy-seed 0 --args.result-dir results/robocasa/rep0/eval_json
```

With a fixed `--args.policy-seed` the same task evaluated twice produces identical action
logs on the same hardware. Changing the number of parallel environments does not change
the evaluated episode set (episodes are seeded individually).

### What a reproduced number looks like

The paper number (78.2) is the mean of three repetitions. One repetition on its own
typically lands within about ±1.3 points of that mean, so compare three-repetition means,
not single runs.
