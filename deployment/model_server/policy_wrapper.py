# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``, plus the
    PointmapVLA observation contract (``uses_pointmap``, ``pointmap_frame``,
    ``pointmap_hw``, ``state_input``, ``state_rotation``, ``state_dim``,
    ``state_norm``). Clients build their observations from these fields
    instead of guessing from the checkpoint config.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}`` in environment units
    (r6 rotations are decoded to axis-angle here).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from starVLA.model.framework.checkpoint_io import resolve_run_dir, resolve_weight_files
from starVLA.model.framework.share_tools import read_mode_config
from starVLA.model.framework.VLM4A.PointmapVLA import PointmapVLA

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.tools.rotation_utils import r6_to_axis_angle_unit


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
    ) -> None:
        # `ckpt_path` may be a run directory, a weight file under <run_dir>/checkpoints/,
        # or a Hugging Face repo id; the framework resolves all three itself. The upstream
        # config/stats helpers below want a weight file and look two levels up from it.
        self._run_dir = str(resolve_run_dir(ckpt_path))
        self._ckpt_path = str(resolve_weight_files(ckpt_path)[0])

        logging.info("PolicyServerWrapper: loading framework from %s", self._run_dir)
        framework = PointmapVLA.from_pretrained(ckpt_path)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        # Co-located metadata.
        model_cfg, _ = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg

        # action_chunk_size = future_action_window_size + 1 (matches old client).
        action_model_cfg = model_cfg["framework"]["action_model"]

        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found in model config for {self._ckpt_path}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._norm_stats = _ns
        self._available_unnorm_keys: List[str] = list(_ns.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
            # The clients run in simulator environments without access to the
            # checkpoint, so the state normalization constants travel with the
            # handshake. Clients apply (state - mean) / std before sending.
            state_stats = self._norm_stats[self._default_unnorm_key]["state"]
            base["state_norm"] = {
                "mean": [float(v) for v in state_stats["mean"]],
                "std": [float(v) for v in state_stats["std"]],
            }
        # Observation contract declared by the model itself, never inferred by
        # the client: a pointmap-trained checkpoint evaluated without pointmaps,
        # or with pointmaps in the wrong frame, degrades silently.
        fw = self._framework
        base["uses_pointmap"] = bool(getattr(fw, "uses_pointmap", True))
        base["pointmap_frame"] = str(getattr(fw, "pointmap_frame", "base"))
        base["pointmap_hw"] = [int(v) for v in self._pointmap_hw()]
        base["state_input"] = str(getattr(fw, "state_input", "expert"))
        base["state_rotation"] = str(getattr(fw, "state_rotation", "r6"))
        # Width of the state vector the CLIENT sends (the model slices its anchor from it).
        base["state_dim"] = int(getattr(fw, "state_width", fw.state_dim))
        if "state_norm" in base and len(base["state_norm"]["mean"]) != base["state_dim"]:
            raise RuntimeError(
                f"state statistics width {len(base['state_norm']['mean'])} != model state width {base['state_dim']}")
        return base

    def _pointmap_hw(self):
        """Pointmap resolution the model was trained with (must match RGB token count)."""
        hw = getattr(self._framework, "pointmap_hw", None)
        if hw is None:
            hw = self._model_cfg.get("datasets", {}).get("vla_data", {}).get("pointmap_out_hw")
        return tuple(hw) if hw is not None else (256, 256)

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``noise_seed``, ``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        out = self._framework.predict_action(examples=examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        unnorm = self._r6_actions_to_env(unnorm, proc)
        return {"actions": unnorm}

    @staticmethod
    def _r6_actions_to_env(actions: np.ndarray, proc: PolicyNormProcessor) -> np.ndarray:
        """Decode the 6-D rotation block to axis-angle so the server always
        returns actions in the environment's layout (RoboCasa 12-D, LIBERO 7-D).

        The rotation slice is located from the declared action layout, never
        hardcoded; a layout with a 3-wide rotation passes through unchanged.
        """
        dims = getattr(proc, "_action_key_dims", None) or {}
        rot_key = "action.end_effector_rotation"
        if int(dims.get(rot_key, 3)) != 6:
            return actions
        start = 0
        for k, d in dims.items():
            if k == rot_key:
                break
            start += int(d)
        end = start + 6
        aa = r6_to_axis_angle_unit(np.asarray(actions[..., start:end], dtype=np.float64))
        out = np.concatenate([actions[..., :start], aa, actions[..., end:]], axis=-1)
        return out.astype(actions.dtype)
