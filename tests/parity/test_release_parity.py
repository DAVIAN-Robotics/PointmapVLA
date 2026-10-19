"""Parity of the released PointmapVLA code and checkpoints against reference outputs.

Reference fixtures were captured from the original training code with the paper
checkpoints (one GPU, bf16, SDPA attention, TF32 on, ``torch.manual_seed(1234)``
immediately before each call). Set

    POINTMAPVLA_FIXTURES=<dir with {bench}_train_batch.pt, {bench}_infer.pt, ...>
    POINTMAPVLA_CKPT_DIR=<dir with PointmapVLA-Pretrained/, PointmapVLA-RoboCasa/, PointmapVLA-LIBERO/>

and run ``pytest tests/parity -s``; the tests skip when either variable is unset.

Checks per benchmark:
  keys    converted checkpoint keys == model.state_dict() keys
  loss    forward() on the captured training batch == reference loss
  actions predict_action() on captured inference examples == reference actions
  remap   loading the pretrained checkpoint into the fine-tune model transplants the
          shared leading slots of the 4 embodiment-dependent head tensors and keeps
          the fresh initialisation elsewhere
"""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("STARVLA_FORCE_SDPA", "1")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

FIXTURES = os.environ.get("POINTMAPVLA_FIXTURES")
CKPT_DIR = os.environ.get("POINTMAPVLA_CKPT_DIR")
SEED = 1234
BENCHES = {"robocasa": "PointmapVLA-RoboCasa", "libero": "PointmapVLA-LIBERO"}
PRETRAINED = "PointmapVLA-Pretrained"

pytestmark = pytest.mark.skipif(
    not (FIXTURES and CKPT_DIR and torch.cuda.is_available()),
    reason="set POINTMAPVLA_FIXTURES and POINTMAPVLA_CKPT_DIR and run on a GPU",
)



def _release_keys(rows, model_keys):
    """Map the original checkpoint keys onto release names using the converter's
    ``key_renames.json`` ({"renames": {old_prefix: new_prefix}, "dropped": [keys]})."""
    spec = json.loads(_fixture("key_renames.json").read_text())
    out = {}
    for k, shape, _ in rows:
        if k in spec["dropped"]:
            continue
        for old, new in spec["renames"].items():
            if k.startswith(old):
                k = new + k[len(old):]
        out[k] = tuple(shape)
    return out


def _fixture(name):
    return Path(FIXTURES) / name


_MODELS = {}


def _model(bench):
    """Released model, loaded as the policy server does: strict, bf16, cuda, eval."""
    if bench not in _MODELS:
        from starVLA.model.framework.VLM4A.PointmapVLA import PointmapVLA

        _MODELS.clear()
        torch.cuda.empty_cache()
        _MODELS[bench] = PointmapVLA.from_pretrained(str(Path(CKPT_DIR) / BENCHES[bench])).to(torch.bfloat16).to("cuda").eval()
    return _MODELS[bench]


@pytest.mark.parametrize("bench", sorted(BENCHES))
def test_keys(bench):
    rows = json.loads(_fixture(f"internal_keys_{bench}.json").read_text())
    got = {k: tuple(v.shape) for k, v in _model(bench).state_dict().items()}
    expected = _release_keys(rows, got)
    assert set(got) == set(expected), (sorted(set(got) - set(expected))[:5], sorted(set(expected) - set(got))[:5])
    mismatched = [k for k in expected if expected[k] != got[k]]
    assert not mismatched, mismatched[:5]
    print(f"\n[{bench}] {len(got)} state-dict keys match")


@pytest.mark.parametrize("bench", sorted(BENCHES))
def test_loss(bench):
    from starVLA.dataloader.qwenvl_collator import POINTMAP_INPUTS_KEY, QWEN_INPUTS_KEY

    fx = torch.load(_fixture(f"{bench}_train_batch.pt"), weights_only=False)
    batch = copy.deepcopy(fx["examples"])
    batch[0][QWEN_INPUTS_KEY] = fx[QWEN_INPUTS_KEY]
    batch[0][POINTMAP_INPUTS_KEY] = fx[POINTMAP_INPUTS_KEY]
    ref = json.loads(_fixture(f"{bench}_ref_loss.json").read_text())["run1"]["action_loss"]

    model = _model(bench)
    torch.manual_seed(SEED)
    with torch.no_grad():
        loss = float(model.forward(batch)["action_loss"])
    rel = abs(loss - ref) / max(abs(ref), 1e-12)
    print(f"\n[{bench}] loss release={loss!r} reference={ref!r} rel_diff={rel:.3e}")
    assert rel <= 1e-5


@pytest.mark.parametrize("bench", sorted(BENCHES))
def test_actions(bench):
    infer = torch.load(_fixture(f"{bench}_infer.pt"), weights_only=False)
    ref = np.load(_fixture(f"{bench}_ref_actions.npy"))

    model = _model(bench)
    torch.manual_seed(SEED)
    out = np.asarray(model.predict_action(examples=copy.deepcopy(infer))["normalized_actions"], dtype=np.float32)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    max_abs = float(np.abs(out - ref).max())
    print(f"\n[{bench}] actions {out.shape} max|release - reference| = {max_abs:.3e} (bitwise equal: {max_abs == 0.0})")
    assert max_abs <= 1e-3


@pytest.mark.parametrize("bench", sorted(BENCHES))
def test_remap(bench):
    from omegaconf import OmegaConf

    from starVLA.model.framework.checkpoint_io import load_run_dir_state_dict
    from starVLA.model.framework.VLM4A.PointmapVLA import HEAD_IO_SHARED_SLOTS, PointmapVLA

    ref = torch.load(_fixture(f"{bench}_ref_remap.pt"), weights_only=False)
    assert {k: tuple(v) for k, v in ref["spec"].items()} == HEAD_IO_SHARED_SLOTS

    _MODELS.clear()
    torch.cuda.empty_cache()
    run_dir = Path(CKPT_DIR) / BENCHES[bench]
    cfg = OmegaConf.load(str(run_dir / "config.yaml"))
    cfg.framework.qwenvl.base_vlm = str(run_dir / cfg.framework.qwenvl.base_vlm)
    model = PointmapVLA(cfg)
    fresh = {k: model.state_dict()[k].detach().clone() for k in HEAD_IO_SHARED_SLOTS}

    pretrained = load_run_dir_state_dict(str(Path(CKPT_DIR) / PRETRAINED))
    result = model.load_state_dict(pretrained, strict=False)
    assert not result.missing_keys and not result.unexpected_keys, (result.missing_keys[:5], result.unexpected_keys[:5])
    model.qwen_vl_interface.model.tie_weights()

    sd = model.state_dict()
    for key, (axis, shared) in HEAD_IO_SHARED_SLOTS.items():
        after, src = sd[key], ref["pretrained"][key]
        assert torch.equal(src, pretrained[key]), f"{key}: fixture pretrained tensor != released pretrained tensor"
        assert torch.equal(after.narrow(axis, 0, shared), src.narrow(axis, 0, shared).to(after.dtype)), key
        tail = after.shape[axis] - shared
        if tail > 0:
            assert torch.equal(after.narrow(axis, shared, tail), fresh[key].narrow(axis, shared, tail)), key
        assert torch.equal(after.narrow(axis, 0, shared), ref["after"][key].narrow(axis, 0, shared).to(after.dtype)), key
    others = [k for k in pretrained if k not in HEAD_IO_SHARED_SLOTS]
    differing = [k for k in others if not torch.equal(sd[k].to(pretrained[k].dtype), pretrained[k])]
    assert not differing, differing[:5]
    print(f"\n[{bench}] remap OK: 4 head tensors transplanted, {len(others)} other tensors identical to pretrained")
    del model
    torch.cuda.empty_cache()
