# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""PointmapVLA: Qwen3.5 + pointmap encoder + layer-wise flow-matching action head.

Architecture
------------
* ``qwen_vl_interface`` — Qwen3.5-4B. Multi-view RGB and the instruction go through
  the stock model; every decoder layer's hidden states condition the action head.
* ``pointmap_encoder`` — a second copy of the Qwen3.5 vision tower (initialised from
  the RGB tower) that encodes a robot-base-frame pointmap (metres, one XYZ per RGB
  pixel) with the exact same patchify/merge pipeline. Its merged tokens are added
  onto the RGB merged tokens inside the vision tower, so each visual token carries
  the 3-D position of its own patch.
* ``project_layers`` — one LayerNorm+Linear per decoder layer mapping the VLM hidden
  size to the action-DiT width.
* ``action_model`` — ``PointmapVLAActionHead``: a layer-wise cross-attention DiT whose
  cross blocks alternate between image and text tokens (image first). The DiT
  sequence is one state token (11-D end-effector anchor) plus the action chunk.

Batch contract (training; produced by ``QwenVLBatchCollator``)
    examples[i]["image"]  multi-view images     examples[i]["lang"]   instruction
    examples[i]["action"] (T, action_dim)       examples[i]["state"]  (1, state_width)
    examples[0]["_qwen_inputs"]     processor output for the whole batch (CPU tensors)
    examples[0]["_pointmap_inputs"] {"pointmap": (B, V, H, W, 3) fp16, "valid": (B, V, H, W) bool}

Inference contract (``predict_action``)
    examples[i]["image"] list of HWC uint8 arrays / PIL images, ["lang"], ["state"] (1, state_width)
    float32 normalised, ["pointmap"] (V, H, W, 3) float16 metres in the robot base
    frame, ["pointmap_valid"] (V, H, W) bool. Returns {"normalized_actions": (B, H, action_dim)}.

Checkpoint layout (``from_pretrained`` accepts a run directory, a legacy
``<run>/checkpoints/*.pt`` file, or a Hugging Face repo id)
    <run>/config.yaml  <run>/dataset_statistics.json  <run>/base_vlm/  (config + tokenizer)
    <run>/checkpoints/model-0000X-of-0000N.safetensors + model.safetensors.index.json

Head transplant across embodiments
----------------------------------
The pretrained checkpoint has a 32-D action / 17-D state head; the benchmark
fine-tunes use 15/11 (RoboCasa) or 10/11 (LIBERO). The leading slots of these
layouts mean the same thing (end-effector position 3, rotation 6, gripper 1), so
when a checkpoint with different head dimensions is loaded, ``_adapt_action_head_io``
copies the shared leading slots of the four input/output projections and keeps the
model's fresh initialisation for the rest. This runs automatically inside
``load_state_dict`` and is a no-op when the shapes already match.
"""

import copy
import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.dataloader.qwenvl_collator import POINTMAP_INPUTS_KEY, QWEN_INPUTS_KEY
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.checkpoint_io import TIED_WEIGHTS, load_run_dir_state_dict, resolve_base_vlm, resolve_run_dir
from starVLA.model.framework.share_tools import apply_config_compat, merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.PointmapVLA_ActionHeader import PointmapVLAActionHead
from starVLA.model.modules.vlm.QWen3_5_pointmap import QwenPointmapInterface
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

# State keys (in order) that form the 11-D end-effector anchor fed to the action head.
STATE_ANCHOR_KEYS = (
    "state.end_effector_position_relative",  # 3
    "state.end_effector_rotation_relative",  # 6, first two rows of the rotation matrix
    "state.gripper_qpos",  # 2
)

# Head tensors whose shape depends on the embodiment: (axis that may differ, number
# of leading slots that mean the same thing in every layout). Action layouts share
# [position 3 | rotation 6 | gripper 1]; state layouts share [position 3 | rotation 6].
HEAD_IO_SHARED_SLOTS = {
    "action_model.state_encoder.layer1.weight": (1, 9),
    "action_model.action_encoder.layer1.weight": (1, 10),
    "action_model.action_decoder.layer2.weight": (0, 10),
    "action_model.action_decoder.layer2.bias": (0, 10),
}


@dataclass
class PointmapVLADefaultConfig:
    """Defaults for ``framework.*``; the YAML overrides them."""

    name: str = "PointmapVLA"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./base_vlm",
            "attn_implementation": "flash_attention_2",
            # Forwarded to the Qwen3.5 chat template when set (True/False). None keeps
            # the template default. Must match what the checkpoint was trained with.
            "enable_thinking": None,
            "gradient_checkpointing": False,
        }
    )

    # Coordinate frame of the input pointmaps; a record for the evaluation client.
    pointmap: dict = field(default_factory=lambda: {"frame": "base"})

    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 15,
            "state_dim": 11,
            "action_horizon": 16,
            "repeated_diffusion_steps": 8,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                "action_dit_hidden_dim": 1024,
                "attention_head_dim": 64,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
            },
        }
    )


def _layout_slices(keys: Sequence[str], key_dims) -> dict:
    """{key: slice} for a concatenated state/action vector, in key order."""
    slices, cursor = {}, 0
    for key in keys:
        width = int(key_dims[key])
        slices[key] = slice(cursor, cursor + width)
        cursor += width
    return slices


@FRAMEWORK_REGISTRY.register("PointmapVLA")
class PointmapVLA(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(PointmapVLADefaultConfig, config)
        self.qwen_vl_interface = QwenPointmapInterface(self.config)

        text_cfg = self.qwen_vl_interface.model.config.text_config
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(text_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        if bool(self.config.framework.qwenvl.get("gradient_checkpointing", False)):
            vlm = self.qwen_vl_interface.model
            vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # HF layers only recompute in train mode; from_pretrained leaves the model in eval.
            vlm.train()

        # Pointmap encoder: a copy of the RGB vision tower (weights come from the
        # checkpoint). Activation checkpointing keeps the 4-GPU training footprint.
        self.pointmap_encoder = copy.deepcopy(self.qwen_vl_interface.model.model.visual)
        self.pointmap_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self._pointmap_cache = None

        action_cfg = self.config.framework.action_model
        self.action_dim = int(action_cfg.action_dim)
        self.state_dim = int(action_cfg.state_dim)
        self.action_horizon = int(action_cfg.action_horizon)
        self.action_chunk_size = self.action_horizon
        self._state_anchor_slices = self._resolve_state_anchor_slices()

        dit_hidden = action_cfg.diffusion_model_cfg.get("action_dit_hidden_dim", None)
        self.action_dit_hidden_dim = int(dit_hidden) if dit_hidden is not None else llm_hidden_size
        populate_layerwise_dit_cfg(self.config, dit_hidden_dim=self.action_dit_hidden_dim, num_dit_layers=num_vl_layers)
        self.action_model = PointmapVLAActionHead(self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)

        self.project_layers = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(llm_hidden_size), nn.Linear(llm_hidden_size, self.action_dit_hidden_dim))
                for _ in range(self.num_action_dit_layers)
            ]
        )

        thinking = self.config.framework.qwenvl.get("enable_thinking", None)
        self.enable_thinking = None if thinking is None else bool(thinking)
        self._vision_token_ids = None

        self.register_load_state_dict_pre_hook(self._adapt_action_head_io)
        self.register_load_state_dict_post_hook(self._check_skeleton_loaded)

        # Metadata read by the policy server / evaluation clients.
        self.uses_pointmap = True
        self.pointmap_frame = str(self.config.framework.pointmap.get("frame", "base"))
        self.state_input = "expert"
        self.state_rotation = "r6"
        # state_width: width of the state vector the client sends (set in _resolve_state_anchor_slices).
        pointmap_hw = self.config.datasets.vla_data.get("pointmap_out_hw", None) if "datasets" in self.config else None
        self.pointmap_hw = tuple(int(x) for x in pointmap_hw) if pointmap_hw else None

    # ------------------------------------------------------------------ state anchor
    def _resolve_state_anchor_slices(self) -> Optional[List[slice]]:
        """Slices of the dataset state vector that form the anchor, read from the data
        registry of ``datasets.vla_data.data_mix``. Without a training mixture (the
        pretrained checkpoint) the state vector is used as-is."""
        vla_cfg = self.config.datasets.get("vla_data", None) if "datasets" in self.config else None
        data_mix = vla_cfg.get("data_mix", None) if vla_cfg is not None else None
        if not data_mix:
            logger.info(f"[PointmapVLA] no data_mix: the {self.state_dim}-D state is used as the anchor unchanged")
            self.state_width = self.state_dim
            return None

        from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP

        mixture = DATASET_NAMED_MIXTURES.get(data_mix)
        if not mixture:
            raise KeyError(f"data_mix {data_mix!r} is not in the dataset registry")
        robot_types = {entry[2] for entry in mixture}
        assert len(robot_types) == 1, f"PointmapVLA expects a single robot type per mixture, got {robot_types}"
        data_config = ROBOT_TYPE_CONFIG_MAP[next(iter(robot_types))]

        state_slices = _layout_slices(data_config.state_keys, data_config.state_key_dims)
        anchor = [state_slices[k] for k in STATE_ANCHOR_KEYS]
        widths = [s.stop - s.start for s in anchor]
        assert widths == [3, 6, 2], (
            f"state anchor expects widths [3, 6, 2] for {STATE_ANCHOR_KEYS}, got {widths} — "
            "is the mixture an end-effector r6 dataset?"
        )
        assert sum(widths) == self.state_dim, f"action_model.state_dim {self.state_dim} != anchor width {sum(widths)}"
        self.state_width = sum(int(v) for v in data_config.state_key_dims.values())

        total_action = sum(int(v) for v in data_config.action_key_dims.values())
        assert total_action == self.action_dim, f"action_model.action_dim {self.action_dim} != registry layout {total_action}"
        logger.info(
            f"[PointmapVLA] state anchor from {data_mix}: {dict(zip(STATE_ANCHOR_KEYS, anchor))} "
            f"-> {self.state_dim}-D; action layout {total_action}-D"
        )
        return anchor

    def _state_anchor(self, state: torch.Tensor) -> torch.Tensor:
        """(B, 1, state_width) normalised state -> (B, 1, state_dim) anchor (pure slice)."""
        assert state.dim() == 3 and state.shape[1] == 1, f"expected state (B, 1, D), got {tuple(state.shape)}"
        state = state.float()
        if self._state_anchor_slices is None:
            assert state.shape[2] == self.state_dim, f"state width {state.shape[2]} != state_dim {self.state_dim}"
            return state
        assert state.shape[2] == self.state_width, f"state width {state.shape[2]} != dataset layout {self.state_width}"
        return torch.cat([state[..., s] for s in self._state_anchor_slices], dim=-1)

    # ------------------------------------------------------------------ pointmap
    def _cache_pointmap(self, examples: List[dict]) -> None:
        """Pick the batch pointmap off ``examples`` (collator tensors or per-example arrays)."""
        packed = examples[0].get(POINTMAP_INPUTS_KEY) if isinstance(examples[0], dict) else None
        if packed is not None:
            self._pointmap_cache = (packed["pointmap"], packed["valid"])
            return
        if "pointmap" not in examples[0]:
            raise ValueError(
                "PointmapVLA needs example['pointmap'] and example['pointmap_valid'] "
                f"(or the collator's examples[0][{POINTMAP_INPUTS_KEY!r}])"
            )
        pointmap = torch.from_numpy(np.stack([np.asarray(ex["pointmap"]) for ex in examples]))
        valid = torch.from_numpy(np.stack([np.asarray(ex["pointmap_valid"]) for ex in examples]))
        self._pointmap_cache = (pointmap, valid)

    def _take_pointmap(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._pointmap_cache is not None, "pointmap not cached — call forward/predict_action"
        pointmap, valid = self._pointmap_cache
        self._pointmap_cache = None
        return pointmap.to(device), valid.to(device)

    @staticmethod
    def _pointmap_to_patches(pointmap: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, V, H, W, 3) metres + valid mask -> Qwen2-VL style patches.

        Invalid pixels are zeroed; coordinates are otherwise fed raw. The reshape /
        permute chain reproduces the image processor's patch order (2x2 merge blocks
        in raster order, 2x2 patches inside each block, channel-major (C, T, P, P)
        inside a patch), with the single frame duplicated along the temporal axis.
        Returns (B*V*gh*gw, 3*2*16*16) patches and (B*V, 3) grid_thw.
        """
        x = pointmap.to(torch.float32)
        x = torch.where(valid.unsqueeze(-1), x, torch.zeros((), dtype=x.dtype, device=x.device))
        B, V, H, W, C = x.shape
        assert C == 3, f"pointmap must have 3 channels, got {C}"
        patch, merge = 16, 2
        gh, gw = H // patch, W // patch
        assert gh * patch == H and gw * patch == W and gh % merge == 0 and gw % merge == 0, (H, W)

        x = x.flatten(0, 1).permute(0, 3, 1, 2)  # (N, C, H, W)
        x = x.unsqueeze(1).expand(-1, 2, -1, -1, -1)  # (N, T=2, C, H, W)
        x = x.reshape(B * V, 2, C, gh // merge, merge, patch, gw // merge, merge, patch)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)  # (N, gh/m, gw/m, m, m, C, T, P, P)
        x = x.reshape(B * V * gh * gw, C * 2 * patch * patch)
        grid_thw = torch.tensor([[1, gh, gw]], dtype=torch.long, device=x.device).repeat(B * V, 1)
        return x, grid_thw

    def _pointmap_encoder_forward(self, pointmap: torch.Tensor, valid: torch.Tensor) -> None:
        """Encode the pointmap views and stage the result for the RGB vision tower.

        A view with no valid pixel (missing camera) is skipped entirely: no encoder
        forward and no add, so its RGB tokens stay as in an RGB-only model.
        """
        B, V = pointmap.shape[:2]
        view_has_pixels = valid.flatten(2).any(dim=-1)  # (B, V)
        kept_rows = None
        if not bool(view_has_pixels.all()):
            kept_rows = view_has_pixels.reshape(-1).nonzero(as_tuple=True)[0]
            if kept_rows.numel() == 0:
                return
            pointmap = pointmap.reshape(B * V, 1, *pointmap.shape[2:])[kept_rows]
            valid = valid.reshape(B * V, 1, *valid.shape[2:])[kept_rows]

        if torch.is_grad_enabled():
            # Activation checkpointing only fires in train mode.
            self.pointmap_encoder.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            patches, grid_thw = self._pointmap_to_patches(pointmap, valid)
            patches = patches.to(next(self.pointmap_encoder.parameters()).dtype)
            merged = self.pointmap_encoder(patches, grid_thw=grid_thw).pooler_output  # (rows, llm_hidden)

        if kept_rows is None:
            self.qwen_vl_interface.stage_pointmap(merged)
            return
        per_view, remainder = divmod(int(merged.shape[0]), int(kept_rows.numel()))
        assert remainder == 0, f"merged rows {merged.shape[0]} not divisible by kept views {kept_rows.numel()}"
        rows = (kept_rows.to(merged.device).unsqueeze(1) * per_view + torch.arange(per_view, device=merged.device)).reshape(-1)
        self.qwen_vl_interface.stage_pointmap(merged, merged_rows=rows, merged_total=B * V * per_view)

    # ------------------------------------------------------------------ VLM encode
    def _encode_vl_hidden_states(
        self,
        batch_images: List,
        instructions: List[str],
        precomputed_inputs: Optional[dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Pointmap encoder -> Qwen3.5 -> per-layer projections.

        Returns (layer-wise projected embeddings, attention_mask, input_ids).
        """
        device = self.qwen_vl_interface.device
        pointmap, valid = self._take_pointmap(device)
        self._pointmap_encoder_forward(pointmap, valid)

        if precomputed_inputs is not None:
            qwen_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in precomputed_inputs.items()}
        else:
            pointmap_fed = valid.flatten(1).any(dim=1).tolist()
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                enable_thinking=self.enable_thinking,
                pointmap_fed=pointmap_fed,
            )
        attention_mask = qwen_inputs.get("attention_mask", None)
        input_ids = qwen_inputs.get("input_ids", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            vl_embs_list = list(outputs.hidden_states[-self.num_action_dit_layers:])
            vl_embs_list = [proj(h) for proj, h in zip(self.project_layers, vl_embs_list)]
        return vl_embs_list, attention_mask, input_ids

    def _modality_masks(self, attention_mask: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(text_mask, image_mask): disjoint bool views of the VLM sequence. The
        vision start/end markers count as image tokens."""
        assert attention_mask is not None and input_ids is not None, "modality split needs attention_mask and input_ids"
        assert attention_mask.shape == input_ids.shape, (tuple(attention_mask.shape), tuple(input_ids.shape))
        if self._vision_token_ids is None:
            cfg = self.qwen_vl_interface.model.config
            self._vision_token_ids = [
                int(getattr(cfg, k)) for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
            ]
        valid = attention_mask.to(torch.bool)
        is_image = torch.isin(input_ids, torch.tensor(self._vision_token_ids, device=input_ids.device))
        image, text = valid & is_image, valid & ~is_image
        if not (bool(image.any(dim=1).all()) and bool(text.any(dim=1).all())):
            raise RuntimeError("a batch row has no image or no text tokens — cross-attention would produce NaN")
        return text, image

    # ------------------------------------------------------------------ train / infer
    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        """Flow-matching loss on a collated batch. Returns {"action_loss": scalar}."""
        self._cache_pointmap(examples)
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        if "state" not in examples[0]:
            raise RuntimeError("PointmapVLA needs example['state'] (set datasets.vla_data.include_state: true)")
        state = [example["state"] for example in examples]
        precomputed = examples[0].get(QWEN_INPUTS_KEY) if isinstance(examples[0], dict) else None

        vl_embs_list, attention_mask, input_ids = self._encode_vl_hidden_states(batch_images, instructions, precomputed)
        base_hidden = vl_embs_list[-1]

        # Autocast to float32: the action head's matmuls run in fp32 even when the
        # parameters are bf16. The released checkpoints were trained and evaluated
        # this way, so the loss and the sampled actions carry fp32 precision.
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon:, :]
            repeats = int(self.config.framework.action_model.get("repeated_diffusion_steps", 2))

            # Every sample is denoised at `repeats` noise levels per step; tile the
            # targets, the conditioning and the masks with the same batch order.
            actions_target = actions_target.repeat(repeats, 1, 1)
            vl_embs_list = [h.repeat(repeats, 1, 1) for h in vl_embs_list]
            text_mask, image_mask = self._modality_masks(attention_mask, input_ids)
            masks = tuple(m.repeat(repeats, 1).to(dtype=torch.bool) for m in (text_mask, image_mask))

            state_t = torch.tensor(np.array(state), device=base_hidden.device, dtype=torch.float32)
            anchor = self._state_anchor(state_t).to(base_hidden.dtype).repeat(repeats, 1, 1)

            action_loss = self.action_model(vl_embs_list, actions_target, anchor, encoder_attention_mask=masks)
        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, noise_seed: Optional[int] = None, **kwargs) -> dict:
        """Denoise one action chunk per example.

        noise_seed: seeds the initial noise of the flow sampler for this call only
        (a dedicated generator; the global RNG is untouched). None = unseeded.
        Returns {"normalized_actions": np.ndarray (B, action_horizon, action_dim)}.
        """
        if type(examples) is not list:
            examples = [examples]
        self._cache_pointmap(examples)
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        if "state" not in examples[0]:
            raise RuntimeError("PointmapVLA needs example['state'] at inference")
        state = [example["state"] for example in examples]

        obs_image_size = self.config.datasets.vla_data.get("obs_image_size", None)
        if obs_image_size:
            batch_images = resize_images(batch_images, target_size=tuple(int(x) for x in obs_image_size))

        vl_embs_list, attention_mask, input_ids = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]
        masks = self._modality_masks(attention_mask, input_ids)
        state_t = torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=torch.float32)
        anchor = self._state_anchor(state_t).to(base_hidden.dtype)

        generator = None
        if noise_seed is not None:
            generator = torch.Generator(device=base_hidden.device).manual_seed(int(noise_seed))
        with torch.autocast("cuda", dtype=torch.float32):  # fp32 head math, see forward()
            pred_actions = self.action_model.predict_action(
                vl_embs_list, anchor, encoder_attention_mask=masks, generator=generator
            )
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    # ------------------------------------------------------------------ loading
    def _adapt_action_head_io(self, module, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load_state_dict pre-hook: transplant the shared leading slots of the four
        embodiment-dependent head tensors when the checkpoint's action/state dims
        differ from this model's. Other shape mismatches are left to PyTorch."""
        for key, (axis, shared) in HEAD_IO_SHARED_SLOTS.items():
            full_key = prefix + key
            if full_key not in state_dict:
                continue
            incoming = state_dict[full_key]
            live = self.get_parameter(key)
            if tuple(incoming.shape) == tuple(live.shape):
                continue
            src_shape, dst_shape = tuple(incoming.shape), tuple(live.shape)
            assert len(src_shape) == len(dst_shape) and all(
                a == b for i, (a, b) in enumerate(zip(src_shape, dst_shape)) if i != axis
            ), f"{key}: only axis {axis} may differ between checkpoint {src_shape} and model {dst_shape}"
            assert src_shape[axis] >= shared and dst_shape[axis] >= shared, (
                f"{key}: both layouts must have at least {shared} slots on axis {axis}, got {src_shape} vs {dst_shape}"
            )
            merged = live.detach().clone()
            merged.narrow(axis, 0, shared).copy_(incoming.narrow(axis, 0, shared).to(merged.dtype))
            state_dict[full_key] = merged
            logger.info(
                f"[PointmapVLA] head remap {key}: copied slots [0:{shared}) on axis {axis} "
                f"(checkpoint {src_shape} -> model {dst_shape}); remaining slots keep their fresh init"
            )

    def _check_skeleton_loaded(self, module, incompatible_keys):
        """load_state_dict post-hook: a VLM built from config only has uninitialised
        memory until the checkpoint fills it — refuse to continue if any of it is missing."""
        if not self.qwen_vl_interface.built_from_config:
            return
        # A tied key whose source tensor was loaded is present (same parameter).
        loaded = set(self.state_dict()) - set(incompatible_keys.missing_keys)
        missing = [
            k for k in incompatible_keys.missing_keys
            if k.startswith(("qwen_vl_interface.", "pointmap_encoder.")) and TIED_WEIGHTS.get(k) not in loaded
        ]
        if missing:
            raise RuntimeError(
                f"{len(missing)} VLM / pointmap-encoder tensors missing from the checkpoint while the base VLM "
                f"was built from config only (first: {missing[:3]}). Load a full PointmapVLA checkpoint or point "
                "framework.qwenvl.base_vlm at a directory with weights."
            )

    @classmethod
    def from_pretrained(cls, pretrained_checkpoint: str, **kwargs) -> "PointmapVLA":
        """Build from ``<run>/config.yaml`` and load the weights strictly.

        ``pretrained_checkpoint``: run directory, legacy ``<run>/checkpoints/*.pt`` file,
        or Hugging Face repo id. ``norm_stats`` (dataset_statistics.json) is attached
        for action un-normalisation. The model is returned on CPU.
        """
        run_dir = resolve_run_dir(str(pretrained_checkpoint))
        config = OmegaConf.load(str(run_dir / "config.yaml"))
        apply_config_compat(config)
        if config.get("trainer", None) is not None and "pretrained_checkpoint" in config.trainer:
            config.trainer.pretrained_checkpoint = None
        config.framework.qwenvl.base_vlm = resolve_base_vlm(str(config.framework.qwenvl.base_vlm), run_dir)

        model = cls(config)
        model.load_state_dict(load_run_dir_state_dict(str(pretrained_checkpoint)), strict=True)
        model.qwen_vl_interface.model.tie_weights()
        with open(run_dir / "dataset_statistics.json") as f:
            model.norm_stats = json.load(f)
        return model
