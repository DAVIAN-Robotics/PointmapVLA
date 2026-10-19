# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Qwen3.5 with a pointmap fusion seam in the vision tower.

PointmapVLA encodes a robot-base-frame pointmap with a second copy of the Qwen3.5
vision tower (the pointmap encoder) and adds the resulting merged tokens onto the
RGB merged tokens *before* they are scattered into the language model. The vision
tower output is the only place where visual information enters Qwen3.5 (there is
no deepstack), so this single element-wise add is the whole fusion.

The subclasses below are the stock transformers classes plus that one add:

* ``Qwen3_5VisionModelPointmap.forward`` is a copy of
  ``Qwen3_5VisionModel.forward`` from transformers 5.3.0 with the add inserted
  after ``self.merger``.
* ``Qwen3_5ForConditionalGenerationPointmap`` swaps the vision tower class on
  construction and exposes ``stage_pointmap`` to hand the add tensor over.

Staging contract: the caller sets the add tensor right before the forward that
should consume it; the forward resets it to ``None`` after use so a stale tensor
can never leak into a later call. With nothing staged the forward is numerically
identical to the stock model.

The forward body is copied from a specific transformers version, so this module
refuses to import against any other version rather than silently drifting.
"""

import torch
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    BaseModelOutputWithPooling,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5VisionModel,
)

_SUPPORTED_TRANSFORMERS = "5.3.0"
if transformers.__version__ != _SUPPORTED_TRANSFORMERS:
    raise ImportError(
        f"PointmapVLA copies Qwen3_5VisionModel.forward from transformers {_SUPPORTED_TRANSFORMERS}; "
        f"installed version {transformers.__version__} is untested. Install transformers=={_SUPPORTED_TRANSFORMERS}."
    )


class Qwen3_5VisionModelPointmap(Qwen3_5VisionModel):
    """Qwen3.5 vision tower whose merged output can receive a staged pointmap add."""

    # (total_merged_tokens, out_hidden) tensor added onto the merged RGB tokens, or None.
    _pointmap_merged_add = None
    # Optional (kept_rows,) long index: when a view has no valid pointmap pixel the
    # encoder skips it, so the add only covers the remaining rows (index_add).
    _pointmap_merged_rows = None
    # Expected total merged-row count when rows are given (resolution contract check).
    _pointmap_merged_total = None

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)

        # Pointmap fusion: add the pointmap encoder's merged tokens onto the RGB
        # merged tokens. Both grids come from the same patchify/merge pipeline, so
        # row i of one is the same image location as row i of the other.
        if self._pointmap_merged_add is not None:
            add = self._pointmap_merged_add.to(dtype=merged_hidden_states.dtype, device=merged_hidden_states.device)
            if self._pointmap_merged_rows is not None:
                assert self._pointmap_merged_total == merged_hidden_states.shape[0], (
                    f"pointmap fusion: RGB merged rows {tuple(merged_hidden_states.shape)} vs expected "
                    f"{self._pointmap_merged_total} — pointmap resolution or view count differs from the RGB input"
                )
                assert self._pointmap_merged_rows.shape[0] == add.shape[0], (
                    f"pointmap fusion: rows {tuple(self._pointmap_merged_rows.shape)} vs add {tuple(add.shape)}"
                )
                merged_hidden_states = merged_hidden_states.index_add(
                    0, self._pointmap_merged_rows.to(merged_hidden_states.device), add
                )
            else:
                assert add.shape[0] == merged_hidden_states.shape[0], (
                    f"pointmap fusion: RGB merged rows {tuple(merged_hidden_states.shape)} vs pointmap "
                    f"{tuple(add.shape)} — pointmap resolution or view count differs from the RGB input"
                )
                merged_hidden_states = merged_hidden_states + add
            self._pointmap_merged_add = None
            self._pointmap_merged_rows = None
            self._pointmap_merged_total = None

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


class Qwen3_5ForConditionalGenerationPointmap(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 whose vision tower is ``Qwen3_5VisionModelPointmap``.

    The seam adds no parameters, so checkpoints are interchangeable with the stock
    class. The vision tower is built by the parent constructor, so its class is
    swapped afterwards (same attributes, two extra staging slots).
    """

    @staticmethod
    def _install_pointmap_seam(model):
        visual = model.model.visual
        visual.__class__ = Qwen3_5VisionModelPointmap
        visual._pointmap_merged_add = None
        visual._pointmap_merged_rows = None
        visual._pointmap_merged_total = None
        return model

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls._install_pointmap_seam(super().from_pretrained(*args, **kwargs))

    @classmethod
    def _from_config(cls, config, **kwargs):
        return cls._install_pointmap_seam(super()._from_config(config, **kwargs))

    def forward(self, *args, **kwargs):
        # The action head consumes hidden states only; computing logits for every
        # position would waste (B, L, vocab) of memory. Keep the last row so the
        # output shape stays valid. Hidden states are unaffected.
        kwargs.setdefault("logits_to_keep", 1)
        return super().forward(*args, **kwargs)

    def stage_pointmap(self, merged_add: torch.Tensor, merged_rows: torch.Tensor | None = None,
                       merged_total: int | None = None) -> None:
        """Stage the pointmap add for the next vision-tower forward.

        merged_add:   (rows, out_hidden) pointmap tokens, one row per RGB merged token
                      (or per kept row when ``merged_rows`` is given).
        merged_rows:  (rows,) long — target rows when some views were skipped.
        merged_total: expected RGB merged-row count (required with ``merged_rows``).
        """
        assert (merged_rows is None) == (merged_total is None), "merged_rows and merged_total go together"
        visual = self.model.visual
        visual._pointmap_merged_add = merged_add
        visual._pointmap_merged_rows = merged_rows
        visual._pointmap_merged_total = merged_total
