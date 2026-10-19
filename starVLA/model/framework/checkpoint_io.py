"""Locate and load PointmapVLA checkpoints.

A *run directory* is the unit both the trainer and the released model repos use:

    <run_dir>/config.yaml
    <run_dir>/dataset_statistics.json
    <run_dir>/base_vlm/                       Qwen config, tokenizer and processor (no weights)
    <run_dir>/checkpoints/model-*.safetensors + model.safetensors.index.json   (released repos)
    <run_dir>/checkpoints/steps_<N>_model.safetensors                          (trainer output)
    <run_dir>/checkpoints/steps_<N>_pytorch_model.pt                           (trainer, save_format: pt)

The functions below are shared by the trainer, `PointmapVLA.from_pretrained` and the
evaluation server so that every entry point accepts the same checkpoint specs.
"""

import json
import os
from pathlib import Path

import torch


def resolve_run_dir(spec: str) -> Path:
    """Map a checkpoint spec to a local run directory.

    `spec` may be a run directory, a weight file inside `<run_dir>/checkpoints/`, or a
    Hugging Face repo id (downloaded with `snapshot_download`, cached by `HF_HOME`).
    """
    p = Path(spec)
    if p.is_file():
        return p.parents[1]
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(spec))


def resolve_weight_files(spec: str) -> list:
    """Return the weight file(s) a checkpoint spec refers to, in load order.

    A weight file is returned as is. A run directory (or repo id) yields the shards
    listed in `checkpoints/model.safetensors.index.json` when present, otherwise the
    single `checkpoints/*.safetensors` or `*.pt` file; several step files in one
    directory are ambiguous, so pass the file you want in that case.
    """
    p = Path(spec)
    if p.is_file():
        return [p]
    ckpt_dir = resolve_run_dir(spec) / "checkpoints"
    index = ckpt_dir / "model.safetensors.index.json"
    if index.is_file():
        with open(index) as f:
            return sorted({ckpt_dir / v for v in json.load(f)["weight_map"].values()})
    files = sorted(ckpt_dir.glob("*.safetensors")) or sorted(ckpt_dir.glob("*.pt"))
    assert len(files) == 1, f"expected exactly one weight file under {ckpt_dir}, found {files}"
    return files


# Parameters that share one tensor in the model: {tied key: key of the stored tensor}.
# Released shards store such a tensor once (the index lists the omitted keys under
# `metadata.tied_weights`); trainer saves contain both keys. Loading re-inserts the
# omitted key so both forms give the same dict and strict loads accept either.
TIED_WEIGHTS = {
    "qwen_vl_interface.model.lm_head.weight": "qwen_vl_interface.model.model.language_model.embed_tokens.weight",
}


def load_run_dir_state_dict(spec: str) -> dict:
    """Load the weights a checkpoint spec refers to into one CPU state dict, with
    tied weights (see ``TIED_WEIGHTS``) present under every key."""
    from safetensors.torch import load_file

    state_dict = {}
    for f in resolve_weight_files(spec):
        state_dict.update(load_file(str(f)) if f.suffix == ".safetensors" else torch.load(f, map_location="cpu"))
    for tied, source in TIED_WEIGHTS.items():
        if tied not in state_dict and source in state_dict:
            state_dict[tied] = state_dict[source]
    return state_dict


def resolve_base_vlm(base_vlm: str, run_dir) -> str:
    """Resolve `framework.qwenvl.base_vlm` as written in a run dir's `config.yaml`.

    Relative paths (the shipped `./base_vlm`) are taken relative to the run directory;
    absolute paths, paths that exist from the current directory, and Hugging Face ids
    are returned unchanged.
    """
    if os.path.isabs(base_vlm) or os.path.exists(base_vlm):
        return base_vlm
    candidate = Path(run_dir) / base_vlm
    return str(candidate.resolve()) if candidate.is_dir() else base_vlm
