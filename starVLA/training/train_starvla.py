# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import gc
import json
import os
import re
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.checkpoint_io import load_run_dir_state_dict, resolve_base_vlm, resolve_run_dir
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args


def save_state_dict_file(state_dict, path: str, save_format: str) -> None:
    """Write a state dict as safetensors or a torch pickle, honouring the process umask.

    safetensors creates files with mode 0600, which hides checkpoints from group members on
    shared machines; torch.save follows the umask. Make both behave like the latter.
    """
    if save_format == "safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, path)
    elif save_format == "pt":
        torch.save(state_dict, path)
    else:
        raise ValueError(f"unknown save_format {save_format!r} (expected 'safetensors' or 'pt')")
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(path, 0o666 & ~umask)

# TF32 for the fp32 matmuls that autocast leaves alone (the action head). The
# released checkpoints were trained this way; the eval server enables it too.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# STARVLA_NO_DEEPSPEED=1 runs a plain Accelerator (torch AdamW, no ZeRO) for
# single-process smoke tests that are launched without `accelerate launch`.
deepspeed_plugin = None if os.getenv("STARVLA_NO_DEEPSPEED", "0") == "1" else DeepSpeedPlugin()
# Rank 0 builds the dataset step index alone while the other ranks wait at a
# barrier; on a large mixture that takes longer than the default 30-minute
# NCCL timeout, so give the process group two hours.
accelerator = Accelerator(
    deepspeed_plugin=deepspeed_plugin,
    kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))],
)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def link_base_vlm(cfg, output_dir: Path) -> None:
    """Resolve a relative `framework.qwenvl.base_vlm` (e.g. `./base_vlm`).

    Relative paths are taken relative to a run directory: the released checkpoints
    ship `base_vlm/` (Qwen config, tokenizer and processor, no weights) next to
    their `config.yaml`. Copy it from the pretrained run dir into this run dir so
    the new run is self-contained, and point the live config at the copy. The
    saved `config.yaml` keeps the relative path (see _save_initial_configs).
    """
    base_vlm = str(cfg.framework.qwenvl.base_vlm)
    if resolve_base_vlm(base_vlm, output_dir) != base_vlm:
        pass  # already copied by an earlier launch of this run (resume)
    elif os.path.isabs(base_vlm) or os.path.exists(base_vlm):
        return
    else:
        src = getattr(cfg.trainer, "pretrained_checkpoint", None)
        assert src, f"base_vlm `{base_vlm}` is relative but trainer.pretrained_checkpoint is not set"
        src_dir = Path(resolve_base_vlm(base_vlm, resolve_run_dir(src)))
        assert src_dir.is_dir(), f"base_vlm `{base_vlm}` not found under {resolve_run_dir(src)}"
        if not dist.is_initialized() or dist.get_rank() == 0:
            shutil.copytree(src_dir, output_dir / base_vlm, dirs_exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    cfg.framework.qwenvl.base_vlm = resolve_base_vlm(base_vlm, output_dir)


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    # The dataset step index is a few very large Python lists. Every full garbage
    # collection walks all of them (seconds per pass, at a different moment on each
    # rank, so the all-reduce waits for the slowest). Freeze what is alive now so
    # the collector never scans it again; done before the loader workers fork.
    gc.collect()
    gc.freeze()

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class _ResumeState:
    """Training state that `accelerator.save_state()` does not capture on its own.

    - `completed_steps`: the optimizer-update counter. A NaN-guard skip leaves it
      unchanged while DeepSpeed's own `global_steps` advances, so it is stored
      explicitly and is the source of truth for checkpoint names.
    - `lr_scheduler`: stepped by hand in `_train_step` and never passed to
      `accelerator.prepare`, so accelerate would otherwise drop it from every save.
    - `param_group_names`: restored ZeRO optimizer state is keyed by param-group
      index; a resume with a different `learning_rate` layout would silently put
      Adam moments on the wrong tensors, so refuse instead.
    """

    def __init__(self, trainer):
        self._t = trainer

    def _group_names(self):
        return [g.get("name", f"group{i}") for i, g in enumerate(self._t.optimizer.param_groups)]

    def state_dict(self):
        return {
            "completed_steps": int(self._t.completed_steps),
            "lr_scheduler": self._t.lr_scheduler.state_dict(),
            "param_group_names": self._group_names(),
        }

    def load_state_dict(self, sd):
        assert self._group_names() == list(sd["param_group_names"]), (
            f"param-group layout changed since the checkpoint: saved={sd['param_group_names']} now={self._group_names()}"
        )
        self._t.lr_scheduler.load_state_dict(sd["lr_scheduler"])
        self._t.completed_steps = int(sd["completed_steps"])


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self._nan_skips = 0  # consecutive optimizer steps skipped by the NaN guard

        # Full-state resume: besides the weights-only `steps_<N>` file, also write
        # `checkpoints/resume_<N>/` (ZeRO fp32 master weights + Adam moments per rank,
        # per-rank RNG, _ResumeState) every `full_state_interval` steps and keep the
        # newest `resume_keep` of them.
        self._full_state = bool(getattr(self.config.trainer, "save_full_state", False))
        self._full_state_interval = int(
            getattr(self.config.trainer, "full_state_interval", None) or self.config.trainer.save_interval
        )
        self._resume_keep = int(getattr(self.config.trainer, "resume_keep", 1))
        self._pending_full_state = None  # chosen in _init_checkpointing, loaded after prepare()
        if self._full_state:
            self.accelerator.register_for_checkpointing(_ResumeState(self))

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        # Needs the DeepSpeed-wrapped model and optimizer, so it must follow prepare().
        self._load_full_state()

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases (best-effort; must not block training)."""
        self._wandb_enabled = False
        if os.environ.get("WANDB_MODE") == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            self.accelerator.wait_for_everyone()
            return
        if self.accelerator.is_main_process:
            try:
                wandb.init(
                    name=self.config.run_id,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                )
                self._wandb_enabled = True
            except Exception as exc:
                logger.warning(f"W&B init failed; continuing without W&B: {exc}")
                self._wandb_enabled = False
        # Rendezvous after rank-0 W&B init. Otherwise a slow or failing init on
        # rank 0 lets the other ranks reach the first collective alone and
        # eventually hit an NCCL watchdog timeout.
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints).
        # Original values keep `framework.qwenvl.base_vlm` as written in the yaml
        # (`./base_vlm`, relative to the run dir) rather than the absolute path the
        # trainer resolved it to, so the run dir stays portable.
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=True)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _load_weights(self, spec):
        """Load a weights-only checkpoint into the model (non-strict).

        The model's own `load_state_dict` pre-hook adapts the action-head input and
        output projections when the pretrained action/state dims differ from this
        run's, so no shape handling happens here.
        """
        state_dict = load_run_dir_state_dict(spec)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.accelerator.print(
            f"📦 loaded {len(state_dict) - len(unexpected)} tensors from {spec} "
            f"(missing {len(missing)}, unexpected {len(unexpected)})"
        )
        if missing or unexpected:
            self.accelerator.print(f"   missing: {sorted(missing)[:8]}{' ...' if len(missing) > 8 else ''}")
            self.accelerator.print(f"   unexpected: {sorted(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)

            # Prefer a complete full-state dir that is at least as new as the newest
            # weights file. Only decide here; the load needs the prepared model.
            fs_dir, fs_step = self._latest_full_state()
            if fs_dir is not None and fs_step >= (self.completed_steps if resume_from_checkpoint else -1):
                self._pending_full_state = fs_dir
                self.completed_steps = fs_step
                self.resume_from_checkpoint = fs_dir
                self.accelerator.print(f"[RESUME] full state found: {fs_dir} (step {fs_step})")
                return

            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self._load_weights(self.resume_from_checkpoint)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            self._load_weights(pretrained_checkpoint)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    # Full-state layout under checkpoint_dir, N = completed_steps:
    #   steps_N_model.safetensors   weights only (rank 0), every save_interval
    #   resume_N/                   accelerate.save_state(), every full_state_interval:
    #     pytorch_model/            DeepSpeed module + per-rank ZeRO optimizer shards
    #     random_states_R.pkl       per-rank RNG
    #     custom_checkpoint_0.pkl   _ResumeState
    #     _COMPLETE                 written last; only complete dirs are resumed from
    _FS_RE = re.compile(r"^resume_(\d+)$")

    def _latest_full_state(self):
        """(dir, step) of the newest complete resume_<N>/, or (None, -1)."""
        best = (None, -1)
        for f in os.listdir(self.checkpoint_dir):
            m = self._FS_RE.match(f)
            p = os.path.join(self.checkpoint_dir, f)
            if m and os.path.isfile(os.path.join(p, "_COMPLETE")) and int(m.group(1)) > best[1]:
                best = (p, int(m.group(1)))
        return best

    def _load_full_state(self):
        """Runs on all ranks after accelerator.prepare(). Consumes _pending_full_state."""
        fs_dir = self._pending_full_state
        if fs_dir is None:
            return
        self._pending_full_state = None
        with open(os.path.join(fs_dir, "_COMPLETE")) as f:
            saved_world = int(json.load(f)["world_size"])
        # ZeRO optimizer shards are bound to the world size they were written with.
        # On a different world size, restore weights + RNG + _ResumeState only.
        load_opt = saved_world == self.accelerator.num_processes
        if not load_opt:
            self.accelerator.print(
                f"[RESUME] world size {saved_world} -> {self.accelerator.num_processes}: "
                "optimizer state not loadable, resuming weights-only"
            )
        self.accelerator.load_state(fs_dir, load_optimizer_states=load_opt, load_lr_scheduler_states=False)
        self.accelerator.wait_for_everyone()
        # The LR scheduler came back through _ResumeState; do not replay steps on top.
        self.accelerator.print(
            f"[RESUME] {'full state' if load_opt else 'weights-only'} loaded at step {self.completed_steps}, "
            f"lr={[round(x, 10) for x in self.lr_scheduler.get_last_lr()]}"
        )

    def _save_full_state(self):
        """Runs on all ranks: save_state into resume_<N>.tmp, mark complete, rename, roll."""
        n = self.completed_steps
        final = os.path.join(self.checkpoint_dir, f"resume_{n}")
        tmp = final + ".tmp"
        t0 = time.perf_counter()
        if self.accelerator.is_main_process and os.path.isdir(tmp):
            shutil.rmtree(tmp)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(tmp)  # collective: every rank writes its shard
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            with open(os.path.join(tmp, "_COMPLETE"), "w") as f:
                json.dump({"step": n, "world_size": int(self.accelerator.num_processes)}, f)
            if os.path.isdir(final):
                shutil.rmtree(final)
            os.replace(tmp, final)
            self.accelerator.print(f"[RESUME] full state saved at {final} in {time.perf_counter() - t0:.0f}s")
            # Roll: the new dir is complete before anything older is deleted.
            complete = sorted(
                (int(self._FS_RE.match(f).group(1)), os.path.join(self.checkpoint_dir, f))
                for f in os.listdir(self.checkpoint_dir)
                if self._FS_RE.match(f) and os.path.isfile(os.path.join(self.checkpoint_dir, f, "_COMPLETE"))
            )
            for _, p in complete[: -self._resume_keep]:
                shutil.rmtree(p)
                self.accelerator.print(f"[RESUME] removed {os.path.basename(p)}")
        self.accelerator.wait_for_everyone()

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self._pending_full_state is not None:
            return  # the scheduler state itself is restored by load_state()
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            suffix = "_model.safetensors" if save_format == "safetensors" else "_pytorch_model.pt"
            save_state_dict_file(state_dict, checkpoint_path + suffix, save_format)

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=True)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and rank == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            if getattr(self, "_wandb_enabled", False):
                try:
                    wandb.log(metrics, step=self.completed_steps)
                except Exception as exc:
                    self._wandb_enabled = False
                    logger.warning(f"W&B log failed; disabling W&B: {exc}")
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        # The DeepSpeed engine reads gradient_accumulation_steps from the DeepSpeed
        # json, not from the yaml. If the two disagree the run silently trains a
        # different global batch, so fail loudly instead.
        if hasattr(self.model, "gradient_accumulation_steps"):
            engine_accum = int(self.model.gradient_accumulation_steps())
            cfg_accum = int(self.config.trainer.get("gradient_accumulation_steps", 1) or 1)
            if engine_accum != cfg_accum:
                raise RuntimeError(
                    f"[ACCUM] DeepSpeed engine gradient_accumulation_steps={engine_accum} != "
                    f"yaml trainer.gradient_accumulation_steps={cfg_accum}; launch with the matching "
                    "accelerate/DeepSpeed config (see starVLA/config/deepseeds/)."
                )
            # The DeepSpeed engine owns the accumulation factor; the accelerator object does not know it.
            self.total_batch_size = (
                self.config.datasets.vla_data.per_device_batch_size * self.accelerator.num_processes * engine_accum
            )
            self.accelerator.print(f"[ACCUM] engine accum={engine_accum}, global batch per optimizer step={self.total_batch_size}")

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # `completed_steps` counts optimizer updates, so `max_train_steps` means
            # updates regardless of gradient accumulation. The loop body runs once per
            # micro-batch; everything below is gated on an actual update.
            optimizer_stepped = step_metrics.pop("_optimizer_step", self.accelerator.sync_gradients)
            if optimizer_stepped:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if not optimizer_stepped:
                continue

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self._full_state and self.completed_steps % self._full_state_interval == 0 and self.completed_steps > 0:
                self._save_full_state()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(examples=examples)

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        if hasattr(self.model, "is_gradient_accumulation_boundary"):
            return self._train_step_deepspeed(batch_vla)

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        return {
            "action_dit_loss": action_loss.item(),
        }

    def _train_step_deepspeed(self, batch_vla):
        """Single micro-batch on the DeepSpeed engine.

        `accelerator.accumulate()` is incompatible with ZeRO-2 gradient partitioning
        and zeroes the gradients on every micro-step, so accumulation is left to the
        engine: `backward()` accumulates and `step()` only updates on its own
        boundary. Gradient clipping comes from the DeepSpeed json.
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_dict = self.model.forward(batch_vla)
            action_loss = output_dict["action_loss"]
        self.model.backward(action_loss)

        optimizer_stepped = bool(self.model.is_gradient_accumulation_boundary())
        if optimizer_stepped:
            # NaN guard. DeepSpeed only checks gradients for inf/nan under fp16; in
            # bf16 a single non-finite gradient would go straight into AdamW and turn
            # every parameter with a live gradient into NaN. `check_overflow()` is the
            # engine's own fp16 machinery (partitioned-grad scan + all-reduce, so all
            # ranks agree); when it sets `overflow`, `step()` skips the update.
            opt = self.model.optimizer
            opt.check_overflow()
            if bool(opt.overflow):
                self._nan_skips += 1
                self.accelerator.print(
                    f"[NAN-GUARD] skip #{self._nan_skips} at step {self.completed_steps + 1}: "
                    f"loss={float(action_loss):.6g}"
                )
                assert self._nan_skips < 5, (
                    "[NAN-GUARD] 5 consecutive non-finite steps; the weights or the data are "
                    "likely already corrupt, refusing to continue"
                )
                optimizer_stepped = False  # not counted, LR not advanced
            else:
                self._nan_skips = 0

        self.model.step()  # no-op except on the accumulation boundary
        if optimizer_stepped:
            self.lr_scheduler.step()

        return {"action_dit_loss": action_loss.item(), "_optimizer_step": optimizer_stepped}

    def _finalize_training(self):
        """Training end processing."""
        if torch.cuda.is_available():
            peak_gib = torch.cuda.max_memory_allocated() / 2**30
            logger.info(f"Peak GPU memory allocated on this rank: {peak_gib:.1f} GiB")
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            name = "model.safetensors" if save_format == "safetensors" else "pytorch_model.pt"
            save_state_dict_file(state_dict, os.path.join(final_checkpoint, name), save_format)
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    link_base_vlm(cfg, output_dir)

    # Seed before the model is built so freshly initialised modules (and the
    # remapped action-head slots) start from the same values on every run and on
    # every rank. prepare_training() re-seeds per rank for the training itself.
    if hasattr(cfg, "seed"):
        set_seed(cfg.seed)

    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_tighten.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
