"""Single source of truth for turning an instruction into the VLM prompt text.

Training (``QwenVLBatchCollator``) and inference (``build_qwenvl_inputs``) must render
exactly the same string, so both go through ``render_prompt`` / ``apply_cot_prompt``.
Two knobs live under ``datasets.vla_data``:

  CoT_prompt: 'Task: {instruction} Mode: <end_effector>'   -- template (absent -> bare instruction)
  instruction_normalize: true                              -- absent/false -> instruction used verbatim

With ``instruction_normalize`` on, the instruction is stripped, sentence-cased and given a
terminal period unless it already ends in ``. ! ?``, e.g.
``'Task: Put the cup on the plate. Mode: <end_effector>'``. Without it, crowd-sourced and
simulator instructions differ randomly in case and punctuation, which would otherwise
become spurious signal for the language model.
"""
from __future__ import annotations

_TERMINAL = ".!?"


def normalize_instruction(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    if s[-1] not in _TERMINAL:
        s = s + "."
    return s


def render_prompt(template, instruction: str, normalize: bool) -> str:
    """``template`` None/'' -> the instruction itself (still normalized when asked)."""
    ins = normalize_instruction(instruction) if normalize else instruction
    return template.replace("{instruction}", ins) if template else ins


def _truthy(v) -> bool:
    return v not in (None, False, 0, "", "False", "false", "0", "None", "null")


def apply_cot_prompt(vla_data_cfg, instruction: str) -> str:
    """Read both knobs from a ``datasets.vla_data`` mapping (DictConfig or dict)."""
    template = vla_data_cfg.get("CoT_prompt", None) if vla_data_cfg is not None else None
    normalize = _truthy(vla_data_cfg.get("instruction_normalize", False)) if vla_data_cfg is not None else False
    return render_prompt(template, instruction, normalize)


def origin_tag(vla_data_cfg, pointmap_fed: bool) -> str:
    """``datasets.vla_data.pointmap_origin_tag`` (e.g. ``' Pointmap_origin: <robot>'``) is
    appended to the rendered prompt only when a pointmap is actually fed for that row, so
    the language model is told which coordinate frame the geometry is expressed in.
    Absent/empty knob -> ''."""
    tag = vla_data_cfg.get("pointmap_origin_tag", None) if vla_data_cfg is not None else None
    return str(tag) if (tag and pointmap_fed) else ""
