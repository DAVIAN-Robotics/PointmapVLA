# Reproduction log

This page records what we measured when we re-ran the release artifacts ourselves, so
that users know what a faithful reproduction looks like and how much a single run
scatters. The headline numbers of the paper are in the [README](../README.md#performance).

## Protocol

| | RoboCasa kitchen-24 | LIBERO (4 suites) | LIBERO-Plus |
|---|---|---|---|
| Checkpoint | `PointmapVLA-RoboCasa` | `PointmapVLA-LIBERO` | `PointmapVLA-LIBERO` (zero-shot) |
| Action chunk / executed | 16 / 16 | 8 / 8 | 8 / 8 |
| Flow-matching Euler steps | 4 | 4 | 4 |
| Episodes | 24 tasks x 50, max 720 steps | 4 x 10 tasks x 50 = 2,000 | 10,030 (2402 / 2518 / 2591 / 2519) |
| Seeds | env `42*100000 + episode`; policy noise seeds 0, 1, 2 (one per repetition) | server + env seed 7, 8, 9; K = 12 parallel envs | same |
| Images | 3 views, 256² -> 224 (INTER_AREA) | agentview + wrist, rotated 180°, 256² -> 224 | same |
| State | 18-D (base pose, EE pose as r6, gripper) | 11-D (EE pos 3, r6 6, gripper 2) | same |

Every reported number is a **mean over three independent evaluations**: RoboCasa
repetitions with policy noise seeds 0, 1, 2 (environment seed 42 in all of them), and
LIBERO / LIBERO-Plus runs with seeds 7, 8, 9 (K = 12 parallel environments; K changes the
noise draws, so keep it fixed). A single evaluation scatters around the mean: one RoboCasa
repetition typically lands within about ±1.3 points of the three-repetition mean, one
LIBERO seed within about ±0.2. Compare means with means. Within one repetition, a fixed
policy seed makes the run bit-reproducible on the same hardware and kernels.

The aggregator (`examples/simBenchmarks/aggregate_eval_results.py <robocasa|libero|libero_plus>
<run dir>...`) checks integrity before printing a number: 24 tasks x 50 episodes per
repetition, 10 tasks per suite tiled by shards without gaps, LIBERO-Plus totals
2402/2518/2591/2519 and the 7-axis perturbation categories. A partial run is reported as
INCOMPLETE unless `--allow-partial` is given.

## What we measured

Success rate (%), mean ± s.d. over the three evaluations, all produced with the scripts in
this repository.

| Benchmark | Paper | Released checkpoint re-evaluated | Re-trained from `PointmapVLA-Pretrained` |
|---|---|---|---|
| LIBERO (4 suites, 2,000 episodes) | 99.0 | 99.1 ± 0.1 | 99.0 ± 0.1 |
| LIBERO-Plus (10,030 episodes) | 92.4 | 92.4 ± 0.3 | 92.1 ± 0.4 |
| RoboCasa kitchen-24 (1,200 episodes) | 78.2 | 77.8 ± 0.6 | 77.8 ± 0.5 |

* **Released checkpoint re-evaluated**: `PointmapVLA-RoboCasa` and `PointmapVLA-LIBERO`
  downloaded from Hugging Face and evaluated with `scripts/eval/run_eval_*.sh`.
* **Re-trained**: `PointmapVLA-Pretrained` fine-tuned with the shipped configs
  (`pointmapvla_robocasa_kitchen.yaml`, 20k steps; `pointmapvla_libero.yaml`, 30k steps)
  on 4 B200 GPUs, then evaluated the same way. The training loss curves match the runs
  behind the paper's checkpoints to three decimals at every logged step.

We repeated the re-training once more from a fresh clone of this repository, with
environments built only by the shipped installers and data obtained only through
`scripts/download_data.py`: RoboCasa 77.4 ± 0.6, LIBERO 99.2 ± 0.2, LIBERO-Plus 92.0 ± 0.4.

## Same-recipe variance

Fine-tuning is not bit-deterministic across runs (data-loader ordering and kernel
autotuning differ between machines), so two fine-tunes from the same pretrained weights
land a little apart. In our runs, repeated RoboCasa fine-tunes spanned 76.4–77.8 and
repeated LIBERO-Plus evaluations of repeated fine-tunes spanned 91.5–92.4. A reproduction
within these bands is a successful one.
