# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Qwen3.5 interface for PointmapVLA.

Wraps ``Qwen3_5ForConditionalGenerationPointmap`` (the stock Qwen3.5 model with a
pointmap fusion seam in its vision tower) behind the same small surface the other
starVLA VLM interfaces expose: prompt building, a forward that returns every
layer's hidden states, and the pointmap staging call.

Two things differ from the upstream Qwen3.5 interface:

* The attention backend follows ``framework.qwenvl.attn_implementation`` (the
  paper fine-tunes train with flash-attention 2). Setting ``STARVLA_FORCE_SDPA=1``
  overrides it to SDPA, which the policy server does because the varlen flash
  kernel is unstable at serving batch sizes.
* ``base_vlm`` may be a directory that holds only the Qwen3.5 config, tokenizer
  and processor (as shipped inside every PointmapVLA checkpoint repository). The
  model is then built as an uninitialised skeleton and every weight must come from
  the PointmapVLA checkpoint. A directory with weights takes the stock
  ``from_pretrained`` path.
"""

# fla (flash-linear-attention) powers Qwen3.5's gated-delta-rule layers with Triton
# kernels and ships no pretuned configs for some GPUs; without them every process
# autotunes for minutes. ``starVLA/config/fla_configs/<GPU name>/`` holds configs we
# generated; point fla at them unless the user already set FLA_CONFIG_DIR. fla reads
# these variables at import time, so this runs before transformers is imported.
import os as _os
import re as _re
from pathlib import Path as _Path


def _prime_fla_configs() -> None:
    if _os.environ.get("FLA_CONFIG_DIR"):
        return
    try:
        import torch as _torch

        if not _torch.cuda.is_available():
            return
        gpu = _torch.cuda.get_device_name(0)
    except Exception:
        return
    tag = _re.sub(r"[^0-9A-Za-z]+", "_", gpu).strip("_") or "unknown_gpu"
    cfg_dir = _Path(__file__).resolve().parents[3] / "config" / "fla_configs" / tag
    if cfg_dir.is_dir() and any(cfg_dir.glob("*.json")):
        _os.environ["FLA_CONFIG_DIR"] = str(cfg_dir)
        _os.environ.setdefault("FLA_CACHE_MODE", "default")


_prime_fla_configs()

from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoProcessor

from starVLA.dataloader.prompt_norm import apply_cot_prompt, origin_tag
from starVLA.model.modules.vlm.qwen3_5.modeling_qwen3_5_pointmap import Qwen3_5ForConditionalGenerationPointmap
from starVLA.model.tools import has_flash_attn
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# The action-token range of the Qwen3.5 base used for PointmapVLA (2,048 extra tokens
# appended to the vocabulary). Read from ``config.json`` when present; the directory
# name convention is kept for bases built before the field existed.
ACTION_TOKEN_RANGE_KEY = "pointmapvla_action_token_range"
_LEGACY_ACTION_TOKEN_RANGE = (248077, 248077 + 2047)

_WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin", "model*.pt")


def _has_weight_files(model_dir: Path) -> bool:
    return any(any(model_dir.glob(p)) for p in _WEIGHT_PATTERNS)


def _materialize_meta_parameters(model: nn.Module) -> None:
    """Replace meta-device parameters with uninitialised CPU storage, keeping tied
    parameters tied. Buffers were never on meta (``init_empty_weights`` leaves them
    on CPU), so rotary tables etc. stay valid."""
    replacements = {}
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None or not param.is_meta:
                continue
            new = replacements.get(id(param))
            if new is None:
                new = nn.Parameter(torch.empty(param.shape, dtype=param.dtype), requires_grad=param.requires_grad)
                replacements[id(param)] = new
            module._parameters[name] = new


class QwenPointmapInterface(nn.Module):
    """Qwen3.5 backbone with pointmap fusion, plus prompt/processor plumbing."""

    def __init__(self, config, **kwargs):
        super().__init__()
        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = str(qwenvl_config.get("base_vlm", "./base_vlm"))
        attn_implementation = str(qwenvl_config.get("attn_implementation", "flash_attention_2"))

        reason = "from config"
        if attn_implementation == "flash_attention_2" and not has_flash_attn():
            attn_implementation, reason = "sdpa", "flash_attn not installed"
        if _os.environ.get("STARVLA_FORCE_SDPA", "0") == "1" and attn_implementation != "sdpa":
            attn_implementation, reason = "sdpa", "STARVLA_FORCE_SDPA=1"
        print(f"[QwenPointmapInterface] attn_implementation = {attn_implementation} ({reason})", flush=True)

        model_dir = Path(model_id)
        self.built_from_config = model_dir.is_dir() and not _has_weight_files(model_dir)
        if self.built_from_config:
            # Config-only base: build the skeleton; the PointmapVLA checkpoint supplies
            # every weight (the loader verifies nothing is left uninitialised).
            hf_config = AutoConfig.from_pretrained(model_id)
            with init_empty_weights():
                model = Qwen3_5ForConditionalGenerationPointmap._from_config(
                    hf_config, dtype=torch.bfloat16, attn_implementation=attn_implementation
                )
            _materialize_meta_parameters(model)
            model.tie_weights()  # the lm_head / embed_tokens tie is not applied on the meta device
            print(f"[QwenPointmapInterface] built Qwen3.5 skeleton from {model_id} (weights come from the checkpoint)", flush=True)
        else:
            model = Qwen3_5ForConditionalGenerationPointmap.from_pretrained(
                model_id, attn_implementation=attn_implementation, dtype=torch.bfloat16
            )
        lm_head, embed = model.get_output_embeddings(), model.get_input_embeddings()
        assert lm_head.weight is embed.weight, "Qwen3.5 base must tie lm_head to embed_tokens"

        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config
        # Other starVLA frameworks read hidden_size from the top-level config.
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        token_range = getattr(self.model.config, ACTION_TOKEN_RANGE_KEY, None)
        if token_range is None and "-Action" in model_id:
            token_range = _LEGACY_ACTION_TOKEN_RANGE
        self.action_token_range = tuple(int(x) for x in token_range) if token_range is not None else None

    @property
    def device(self) -> torch.device:
        return self.model.device

    def forward(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**kwargs)

    def stage_pointmap(self, merged_add: torch.Tensor, merged_rows: Optional[torch.Tensor] = None,
                       merged_total: Optional[int] = None) -> None:
        """Stage pointmap tokens for the next vision-tower forward (see the model class)."""
        self.model.stage_pointmap(merged_add, merged_rows=merged_rows, merged_total=merged_total)

    def build_qwenvl_inputs(
        self,
        images: Sequence[Sequence],
        instructions: Sequence[str],
        enable_thinking: Optional[bool] = None,
        pointmap_fed: Optional[List[bool]] = None,
    ):
        """Render the PointmapVLA prompt and tokenize it with the Qwen3.5 chat template.

        The prompt is ``datasets.vla_data.CoT_prompt`` with the (optionally normalised)
        instruction substituted, followed by ``pointmap_origin_tag`` on rows whose
        pointmap is fed. The training collator renders the same string, so train and
        inference token streams match.

        ``enable_thinking`` is forwarded to the chat template when not None. Training
        sequences are unaffected by it; at inference ``False`` renders the empty think
        block so the generation prompt is an exact prefix of the training sequence.
        """
        assert len(images) == len(instructions), "images and instructions must have the same length"
        vla_cfg = self.config.datasets.vla_data
        if vla_cfg.get("pointmap_origin_tag", None):
            if pointmap_fed is None:
                raise RuntimeError("pointmap_origin_tag is set: pass pointmap_fed (one bool per row)")
            if len(pointmap_fed) != len(instructions):
                raise RuntimeError(f"pointmap_fed has {len(pointmap_fed)} rows, batch has {len(instructions)}")

        messages = []
        for row, (row_images, instruction) in enumerate(zip(images, instructions)):
            content = [{"type": "image", "image": img} for img in row_images]
            prompt = apply_cot_prompt(vla_cfg, instruction)
            if pointmap_fed is not None:
                prompt += origin_tag(vla_cfg, bool(pointmap_fed[row]))
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
        return batch_inputs.to(self.model.device)
