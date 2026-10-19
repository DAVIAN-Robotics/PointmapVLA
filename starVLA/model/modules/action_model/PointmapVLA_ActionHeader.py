# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Flow-matching action head used by PointmapVLA.

Same layer-wise cross-attention DiT as ``LayerwiseFlowmatchingActionHead`` with two
changes:

* **Modality-split cross-attention.** The framework passes the VLM attention mask
  as a ``(text_mask, image_mask)`` pair. Cross-attention blocks alternate between
  the two views of the same VLM sequence, image first: blocks 0, 4, 8, ... attend to
  image tokens, blocks 2, 6, 10, ... to text tokens (odd blocks are self-attention).
* **No learned future tokens.** The DiT sequence is one state token followed by the
  action-chunk tokens. The state token encodes the 11-D end-effector anchor
  (position 3, rotation as the first two rows of the rotation matrix 6, gripper 2).

``predict_action`` accepts a ``torch.Generator`` so the initial noise can be seeded
per query, which makes evaluation episodes reproducible.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    MLP,
    ActionEncoder,
    DiTConfig,
    LayerwiseFlowmatchingActionHead,
)


class ModalitySplitDiT(DiT):
    """Layer-wise DiT whose cross-attention blocks alternate image / text keys."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not (self.config.interleave_self_attention and self.config.use_canonical_forward):
            raise ValueError(
                "ModalitySplitDiT needs interleave_self_attention=True and use_canonical_forward=True "
                "(cross-attention on even blocks, self-attention on odd blocks)"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states,
        timestep: Optional[torch.LongTensor] = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask=None,
        return_pre_output: bool = False,
    ):
        """``encoder_attention_mask`` is ``(text_mask, image_mask)``; a single mask (or
        None) is applied to every cross block unchanged."""
        temb = self.timestep_encoder(timestep)

        hidden_states = hidden_states.contiguous()
        is_layerwise_encoder = isinstance(encoder_hidden_states, (list, tuple))
        encoder_hidden_states = (
            [state.contiguous() for state in encoder_hidden_states]
            if is_layerwise_encoder
            else encoder_hidden_states.contiguous()
        )

        split = isinstance(encoder_attention_mask, (tuple, list))
        if split:
            text_mask, image_mask = encoder_attention_mask

        all_hidden_states = [hidden_states]
        cross_rank = 0
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                block_encoder_hidden_states = encoder_hidden_states[idx] if is_layerwise_encoder else encoder_hidden_states
                block_mask = encoder_attention_mask
                if split:
                    # Image first: even cross blocks see image tokens, odd ones text tokens.
                    block_mask = image_mask if cross_rank % 2 == 0 else text_mask
                cross_rank += 1
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=block_encoder_hidden_states,
                    encoder_attention_mask=block_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        if return_pre_output:
            if return_all_hidden_states:
                return hidden_states, all_hidden_states
            return hidden_states

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)


class PointmapVLAActionHead(LayerwiseFlowmatchingActionHead):
    """Layer-wise flow-matching head: [state token, action tokens] -> velocity."""

    def __init__(self, global_config, **kwargs):
        nn.Module.__init__(self)
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg

        for k, v in DiTConfig.items():
            if diffusion_model_cfg.get(k, None) is None:
                diffusion_model_cfg[k] = v
        # Framework-side hints that are not DiT constructor arguments.
        dit_kwargs = {k: v for k, v in diffusion_model_cfg.items() if k not in {"action_dit_hidden_dim"}}

        self.input_embedding_dim = dit_kwargs["input_embedding_dim"]
        self.model = ModalitySplitDiT(**dit_kwargs)
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = int(action_config.action_dim)
        self.action_horizon = int(action_config.action_horizon)
        self.num_inference_timesteps = int(action_config.num_inference_timesteps)

        self.state_encoder = MLP(input_dim=int(action_config.state_dim), output_dim=self.input_embedding_dim)
        self.action_encoder = ActionEncoder(action_dim=self.action_dim, hidden_size=self.input_embedding_dim)
        self.action_decoder = MLP(input_dim=self.input_embedding_dim, hidden_dim=1024, output_dim=self.action_dim)
        self.future_tokens = None

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

    def _sequence(self, state_features: Optional[torch.Tensor], action_features: torch.Tensor) -> torch.Tensor:
        """DiT input: the state token (if any) followed by the action tokens."""
        if state_features is None:
            return action_features
        return torch.cat((state_features, action_features), dim=1)

    def _action_features(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        action_features = self.action_encoder(actions, timesteps)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=actions.device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
        return action_features

    def forward(self, vl_embs_list: list, actions: torch.Tensor, state: torch.Tensor = None,
                encoder_attention_mask=None):
        """Velocity-matching loss for one noise level per sample.

        vl_embs_list: layer-wise VLM embeddings, each (B, seq, D).
        actions:      (B, action_horizon, action_dim) normalised target chunk.
        state:        (B, 1, state_dim) anchor, or None.
        encoder_attention_mask: (text_mask, image_mask) bool pair, see ModalitySplitDiT.
        """
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self._action_features(noisy_trajectory, t_discretized)
        state_features = self.state_encoder(state) if state is not None else None

        model_output = self.model(
            hidden_states=self._sequence(state_features, action_features),
            encoder_hidden_states=vl_embs_list,
            timestep=t_discretized,
            encoder_attention_mask=encoder_attention_mask,
            return_pre_output=True,
        )
        pred_actions = self.action_decoder(model_output)[:, -actions.shape[1]:]
        return ((pred_actions - velocity) ** 2).mean()

    @torch.no_grad()
    def predict_action(self, vl_embs_list: list, state: torch.Tensor = None, encoder_attention_mask=None,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Euler-integrate the learned velocity field from noise to an action chunk.

        generator: optional seeded ``torch.Generator`` (on the model device) for the
        initial noise; None draws from the global RNG.
        """
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs_list[0].dtype,
            device=device,
            generator=generator,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = self.state_encoder(state) if state is not None else None

        for step in range(num_steps):
            t_discretized_int = int(step / float(num_steps) * self.num_timestep_buckets)
            timesteps_tensor = torch.full((batch_size,), t_discretized_int, device=device, dtype=torch.long)
            action_features = self._action_features(actions, timesteps_tensor)
            model_output = self.model(
                hidden_states=self._sequence(state_features, action_features),
                encoder_hidden_states=vl_embs_list,
                timestep=timesteps_tensor,
                encoder_attention_mask=encoder_attention_mask,
                return_pre_output=True,
            )
            pred_velocity = self.action_decoder(model_output)[:, -self.action_horizon:]
            actions = actions + dt * pred_velocity
        return actions

    def predict_action_realtime(self, *args, **kwargs):
        raise NotImplementedError("PointmapVLAActionHead does not implement real-time chunking")
