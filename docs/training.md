# Fine-tuning PointmapVLA

This page covers fine-tuning the released pretrained checkpoint on RoboCasa kitchen
and on LIBERO with the exact recipes behind the released fine-tuned checkpoints.
Install the training environment first (`scripts/install_train_env.sh`) and download
the datasets (`scripts/download_data.py --bench robocasa libero`).

## Configs

| Config | Steps | GPUs x batch x accum | Launch with |
|---|---|---|---|
| `examples/simBenchmarks/Robocasa_kitchen/train_files/pointmapvla_robocasa_kitchen.yaml` | 20,000 | 4 x 32 x 1 | `starVLA/config/deepseeds/deepspeed_zero2.yaml` |
| `examples/simBenchmarks/Robocasa_kitchen/train_files/pointmapvla_robocasa_kitchen_80gb.yaml` | 20,000 | 4 x 8 x 4 | `starVLA/config/deepseeds/deepspeed_zero2_accum4.yaml` |
| `examples/simBenchmarks/LIBERO/train_files/pointmapvla_libero.yaml` | 30,000 | 4 x 32 x 1 | `starVLA/config/deepseeds/deepspeed_zero2.yaml` |
| `examples/simBenchmarks/LIBERO/train_files/pointmapvla_libero_80gb.yaml` | 30,000 | 4 x 8 x 4 | `starVLA/config/deepseeds/deepspeed_zero2_accum4.yaml` |

All four use the same global batch (128), the same optimizer (AdamW, betas 0.9/0.999,
weight decay 1e-5), per-module learning rates (VLM 1e-5, pointmap encoder 1e-5,
action head and projection layers 1e-4), a cosine schedule to 1e-6 with 5% warmup,
bf16 autocast with TF32 matmuls, and DeepSpeed ZeRO-2 with gradient clipping 1.0.
`trainer.max_train_steps` counts optimizer updates, so the `_80gb` variants run the
same schedule over four times as many micro-batches.

The plain B200 configs (32 samples per GPU) need roughly 150 GB per GPU. The `_80gb`
variants (8 samples per GPU, 4 accumulation steps) fit on 80 GB GPUs (A100/H100).

## Launching

On a plain machine:

```bash
scripts/train.sh examples/simBenchmarks/Robocasa_kitchen/train_files/pointmapvla_robocasa_kitchen.yaml
```

On Slurm (edit the `#SBATCH` lines for your cluster):

```bash
CONFIG=examples/simBenchmarks/LIBERO/train_files/pointmapvla_libero.yaml sbatch slurm/train.sbatch
CONFIG=examples/simBenchmarks/LIBERO/train_files/pointmapvla_libero_80gb.yaml \
  DS_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_accum4.yaml sbatch slurm/train.sbatch
```

Both scripts wrap

```bash
accelerate launch --config_file <deepspeed yaml> --num_processes 4 \
  starVLA/training/train_starvla.py --config_yaml <config> --run_id <name> --run_root_dir results/runs
```

Any further `--section.key value` arguments override the yaml (OmegaConf dotlist), e.g.
`--trainer.max_train_steps 50` for a smoke test.

DeepSpeed needs `CUDA_HOME` to point at a CUDA toolkit when it is imported (it
checks which ops it could compile, even though PointmapVLA compiles none).
`scripts/train.sh` sets it to `/usr/local/cuda-12.8` or `/usr/local/cuda` when
one exists; otherwise export it yourself. The runtime kernels used by the Qwen3.5
backbone are JIT-compiled by Triton on first use, so a C compiler (`gcc`) must be
on the path of the training machine; the Slurm example uses an NVIDIA PyTorch
container for that reason.

The run directory `results/runs/<run_id>/` ends up with the same layout as the released
checkpoints: `config.yaml`, `dataset_statistics.json`, `base_vlm/` and
`checkpoints/steps_<N>_model.safetensors`. Point the evaluation server at the run
directory or at a weight file inside it.

## Why 4 GPUs is the minimum

PointmapVLA has 6.08 B parameters. DeepSpeed ZeRO-2 keeps a flat fp32 partition of the
parameters on each rank, and in the pinned DeepSpeed/PyTorch versions a partition larger
than 2^31 elements silently corrupts the weights after a few steps. With 1 or 2 GPUs the
partition is 6.08 B or 3.04 B elements, above that limit; with 4 GPUs it is 1.52 B,
safely below. Do not run these configs on fewer than 4 GPUs. For 8 GPUs, halve the
per-device batch to keep the global batch at 128:

```bash
NGPU=8 scripts/train.sh <config> --datasets.vla_data.per_device_batch_size 16
```

(`starVLA/config/deepseeds/deepspeed_zero2.yaml` lists `num_processes: 8`; the launch
scripts pass `--num_processes` explicitly, so the yaml value does not matter.)

The gradient-accumulation count lives in the DeepSpeed json
(`ds_config*.yaml: gradient_accumulation_steps`), which is what the engine actually
uses. The trainer checks it against `trainer.gradient_accumulation_steps` in the
training yaml at start-up and refuses to run when they differ, so a config can never
silently train with a different global batch.

## The head-remap log line

The pretrained checkpoint was trained on a 32-D action / 17-D state layout; RoboCasa
uses 15/11 and LIBERO 10/11. When the pretrained weights are loaded, the model remaps
the four action-head projections whose shape depends on these dims (state encoder
input, action encoder input, action decoder output weight and bias): the slots shared
by both layouts (end-effector position, r6 rotation, gripper) are copied, the rest
are freshly initialised. The load prints one line per remapped tensor and the trainer
summarises the load, e.g.

```
loaded 1620 tensors from DAVIAN-Robotics/PointmapVLA-Pretrained (missing 0, unexpected 0)
```

The released checkpoints store the language-model head once (it is tied to the
token embedding); the loader re-ties it on load, so no key is reported missing.
Everything other than the four remapped tensors transfers unchanged. Only the
pretrained load goes through this path; resuming a run requires exact shapes.

## Resume

Set `trainer.is_resume: true` (the default in the shipped configs) and launch into the
same run directory again. The trainer picks the newest `checkpoints/resume_<N>/`
(written every `trainer.full_state_interval` steps: ZeRO optimizer state, per-rank RNG,
LR scheduler and step counter) and continues from step N+1 with the exact optimizer
state. If only a weights file `steps_<N>_model.safetensors` exists, it resumes from
that with a fresh optimizer and the LR schedule advanced to N. Only the newest
`trainer.resume_keep` full-state directories are kept (each is about 70 GB for this
model). A full state written with a different number of GPUs is loaded weights-only.

## Numerical guard

In bf16 DeepSpeed does not check gradients for inf/NaN. The trainer runs the engine's
overflow check before every optimizer step; a non-finite step is skipped (not counted,
LR not advanced) and logged as `[NAN-GUARD] skip #k at step N`. Five consecutive skips
abort the run. The released checkpoints were trained with zero skips.

## Expected wall time

On 4 B200 GPUs with the plain configs: RoboCasa 20k steps about 7 h 15 min, LIBERO 30k
steps about 9 h. Dataset indexing on the first launch adds a few minutes (RoboCasa) and
is cached in the dataset directory. The `_80gb` variants run the same number of optimizer
steps over 4x the micro-batches and are correspondingly slower per step on the same
hardware.
