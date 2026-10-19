# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Dataloader collator that runs Qwen-VL preprocessing inside the worker processes.

Why this exists
---------------
``processor.apply_chat_template`` (chat-template formatting + tokenization + image
resize/normalize) is a few hundred milliseconds of CPU work per batch. Done on the main
training process it sits serially in front of the GPU forward; done in the dataloader
workers it overlaps with GPU compute.

Mechanics
---------
The collator builds the batched Qwen-VL inputs on **CPU** (workers have no GPU) and
attaches them to ``batch[0][QWEN_INPUTS_KEY]``. The batch stays a plain ``list[dict]`` so
the inference path (``predict_action``) and any ``for example in examples`` iteration keep
working unchanged. The framework's ``forward`` picks the precomputed inputs up, moves
them to the GPU and skips the rebuild.

The processor is built lazily (once per worker) so the collator object itself pickles
cheaply when torch forks/spawns the workers.
"""

from typing import Optional

from transformers import AutoProcessor

from starVLA.dataloader.prompt_norm import render_prompt

# Key under which the batched, CPU-side Qwen-VL inputs travel on batch[0].
QWEN_INPUTS_KEY = "_qwen_inputs"

# Key under which the batched pointmap tensors travel on batch[0] (same transport
# pattern as QWEN_INPUTS_KEY): {"pointmap": (B,V,h,w,3) fp16, "valid": (B,V,h,w) bool}.
POINTMAP_INPUTS_KEY = "_pointmap_inputs"


class QwenVLBatchCollator:
    """Batches raw samples AND precomputes Qwen-VL model inputs in the worker.

    Args:
        base_vlm: path / HF id of the Qwen-VL checkpoint (for its AutoProcessor).
        cot_prompt: optional template containing ``{instruction}``; when set, each
            instruction is substituted into it (see ``prompt_norm.render_prompt``).
        instruction_normalize: sentence-case + terminal period before templating.
        pack_pointmap: stack the per-sample ``pointmap`` / ``pointmap_valid`` arrays into
            CPU tensors on ``batch[0][POINTMAP_INPUTS_KEY]``.
        enable_thinking: forwarded to ``apply_chat_template`` when not None. ``False``
            renders the closed empty think block (``<think>\\n\\n</think>\\n\\n``) in the
            generation prompt; ``None`` leaves the template default (``<think>\\n``). The
            released checkpoints were trained with a fixed per-benchmark setting, so the
            value must match the training config byte-for-byte.
        pointmap_origin_tag: appended to the prompt of every row whose pointmap has at
            least one valid pixel (e.g. ``' Pointmap_origin: <robot>'``). Tells the
            language model which frame the geometry is expressed in.
    """

    def __init__(
        self,
        base_vlm: str,
        cot_prompt: Optional[str] = None,
        instruction_normalize: bool = False,
        pack_pointmap: bool = False,
        enable_thinking: Optional[bool] = None,
        pointmap_origin_tag: str = "",
    ):
        self.base_vlm = base_vlm
        self.cot_prompt = cot_prompt
        self.instruction_normalize = bool(instruction_normalize)
        self.pack_pointmap = pack_pointmap
        self.enable_thinking = enable_thinking
        self.pointmap_origin_tag = str(pointmap_origin_tag or "")
        if self.pointmap_origin_tag and not pack_pointmap:
            raise ValueError("pointmap_origin_tag needs collate_pointmap: true")
        self._processor = None  # lazily built once per worker; keeps pickling cheap

    @property
    def processor(self):
        if self._processor is None:
            proc = AutoProcessor.from_pretrained(self.base_vlm)
            # Left padding matches the framework's own processor setup.
            proc.tokenizer.padding_side = "left"
            self._processor = proc
        return self._processor

    def __call__(self, batch):
        images = [ex["image"] for ex in batch]          # [B, [PIL]]
        instructions = [ex["lang"] for ex in batch]     # [B, str]

        # pointmap_fed[i]: this row's pointmap reaches the model (>= 1 valid pixel).
        pointmap_fed = None
        if self.pack_pointmap and "pointmap" in batch[0]:
            import numpy as np

            pointmap_fed = [bool(np.asarray(ex["pointmap_valid"]).any()) for ex in batch]
        elif self.pointmap_origin_tag:
            raise KeyError("pointmap_origin_tag set but the batch carries no 'pointmap' (collate_pointmap?)")

        messages = []
        for i, (imgs, instruction) in enumerate(zip(images, instructions)):
            content = [{"type": "image", "image": img} for img in imgs]
            prompt = render_prompt(self.cot_prompt, instruction, self.instruction_normalize)
            if pointmap_fed is not None and pointmap_fed[i]:
                prompt += self.pointmap_origin_tag
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        # CPU tensors only -- no .to(device); forward moves them to the GPU.
        template_kwargs = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        qwen_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
        batch[0][QWEN_INPUTS_KEY] = dict(qwen_inputs)

        # Pointmap transport: stack (V,h,w,3) fp16 + (V,h,w) bool per sample into
        # (B,V,h,w,3) / (B,V,h,w) CPU tensors.
        if pointmap_fed is not None:
            import numpy as np
            import torch

            pm = torch.from_numpy(np.stack([ex["pointmap"] for ex in batch]))
            pv = torch.from_numpy(np.stack([ex["pointmap_valid"] for ex in batch]))
            batch[0][POINTMAP_INPUTS_KEY] = {"pointmap": pm, "valid": pv}

        return batch
