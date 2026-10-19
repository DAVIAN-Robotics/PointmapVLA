"""Parity of the released data layer against reference training batches.

Reference fixtures (``{bench}_train_batch.pt`` + ``.json``) were captured from the
original training code: two samples at fixed ``(dataset, trajectory_id, base_index)``
coordinates, packed by the dataset and collated by the Qwen-VL collator. This test
rebuilds the same batch through the released registry / dataset / collator and
requires every tensor and string to be bitwise identical.

Set

    POINTMAPVLA_FIXTURES=<dir with {bench}_train_batch.pt / .json>
    POINTMAPVLA_BASE_VLM=<processor dir, e.g. <ckpt>/PointmapVLA-RoboCasa/base_vlm>
                         (optional when POINTMAPVLA_CKPT_DIR is set)

with the release datasets downloaded to ``playground/Datasets`` (see
``scripts/download_data.py``), then run ``pytest tests/parity -s``. CPU only.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
FIXTURES = os.environ.get("POINTMAPVLA_FIXTURES")
CKPT_DIR = os.environ.get("POINTMAPVLA_CKPT_DIR")
BASE_VLM = os.environ.get("POINTMAPVLA_BASE_VLM")

pytestmark = pytest.mark.skipif(not FIXTURES, reason="set POINTMAPVLA_FIXTURES")

# Release dataset roots (relative to the repo root) and prompt settings, as shipped in
# the per-benchmark training yamls.
BENCHES = {
    "robocasa": dict(
        data_mix="robocasa_kitchen_mg300_r6_pointmap",
        data_root_dir="playground/Datasets/robocasa_kitchen_mg300_r6",
        pointmap_data_root="playground/Datasets/robocasa-MG_300-pointmap",
        enable_thinking=False,
        ckpt="PointmapVLA-RoboCasa",
    ),
    "libero": dict(
        data_mix="libero_r6_pointmap",
        data_root_dir="playground/Datasets/libero_r6",
        pointmap_data_root="playground/Datasets/libero_r6_pointmap",
        enable_thinking=None,
        ckpt="PointmapVLA-LIBERO",
    ),
}
PROMPT = dict(
    CoT_prompt="Task: {instruction} Mode: <end_effector>",
    instruction_normalize=True,
    pointmap_origin_tag=" Pointmap_origin: <robot>",
)
SAMPLE_KEYS = ("image", "action", "state", "pointmap", "pointmap_valid", "lang", "robot_tag")


def _base_vlm(bench):
    if BASE_VLM:
        return BASE_VLM
    if CKPT_DIR:
        return str(Path(CKPT_DIR) / BENCHES[bench]["ckpt"] / "base_vlm")
    pytest.skip("set POINTMAPVLA_BASE_VLM (or POINTMAPVLA_CKPT_DIR) for the Qwen-VL processor")


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    if isinstance(x, list):  # list of PIL images -> (V, H, W, 3) uint8
        return np.stack([np.asarray(im) for im in x])
    return x


def _assert_same(name, got, ref):
    got, ref = _as_numpy(got), _as_numpy(ref)
    if isinstance(got, np.ndarray):
        assert got.shape == ref.shape and got.dtype == ref.dtype, (name, got.shape, got.dtype, ref.shape, ref.dtype)
        assert np.array_equal(got, ref), f"{name}: max|diff| = {np.abs(got.astype(np.float64) - ref.astype(np.float64)).max()}"
    else:
        assert got == ref, (name, got, ref)


@pytest.mark.parametrize("bench", sorted(BENCHES))
def test_train_batch(bench):
    from omegaconf import OmegaConf

    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from starVLA.dataloader.qwenvl_collator import POINTMAP_INPUTS_KEY, QWEN_INPUTS_KEY, QwenVLBatchCollator

    spec = BENCHES[bench]
    for root in ("data_root_dir", "pointmap_data_root"):
        if not (REPO / spec[root]).exists():
            pytest.skip(f"{spec[root]} not found -- download the {bench} datasets first")
    base_vlm = _base_vlm(bench)

    meta = json.loads((Path(FIXTURES) / f"{bench}_train_batch.json").read_text())
    fixture = torch.load(Path(FIXTURES) / f"{bench}_train_batch.pt", weights_only=False)

    data_cfg = OmegaConf.create(dict(
        dataset_py="lerobot_datasets",
        data_mix=spec["data_mix"],
        data_root_dir=str(REPO / spec["data_root_dir"]),
        pointmap_data_root=str(REPO / spec["pointmap_data_root"]),
        lerobot_version="v3.0",
        include_state=True,
        pointmap_out_hw=[256, 256],
        video_backend="pyav",
        collate_vlm_inputs=True,
        collate_pointmap=True,
        per_device_batch_size=len(meta["samples"]),
        **PROMPT,
    ))
    mixture = get_vla_dataset(data_cfg)
    by_name = {ds.dataset_name: ds for ds in mixture.datasets}

    # Same coordinates as the fixture: dataset.get_step_data -> transforms -> _pack_sample
    # is the deterministic core of LeRobotMixtureDataset.__getitem__.
    samples = []
    for s in meta["samples"]:
        ds = by_name[s["dataset"]]
        samples.append(ds._pack_sample(ds.transforms(ds.get_step_data(s["trajectory_id"], s["base_index"]))))
    for i, (got, ref) in enumerate(zip(samples, fixture["examples"])):
        for key in SAMPLE_KEYS:
            _assert_same(f"examples[{i}].{key}", got[key], ref[key])

    collate = QwenVLBatchCollator(
        base_vlm=base_vlm,
        cot_prompt=PROMPT["CoT_prompt"],
        instruction_normalize=PROMPT["instruction_normalize"],
        pack_pointmap=True,
        enable_thinking=spec["enable_thinking"],
        pointmap_origin_tag=PROMPT["pointmap_origin_tag"],
    )
    batch = collate([dict(s) for s in samples])
    for key, ref in fixture[QWEN_INPUTS_KEY].items():
        _assert_same(f"{QWEN_INPUTS_KEY}.{key}", batch[0][QWEN_INPUTS_KEY][key], ref)
    for key, ref in fixture[POINTMAP_INPUTS_KEY].items():
        _assert_same(f"{POINTMAP_INPUTS_KEY}.{key}", batch[0][POINTMAP_INPUTS_KEY][key], ref)

    n = len(samples) * len(SAMPLE_KEYS) + len(fixture[QWEN_INPUTS_KEY]) + len(fixture[POINTMAP_INPUTS_KEY])
    prompt = collate.processor.tokenizer.decode(
        batch[0][QWEN_INPUTS_KEY]["input_ids"][0][batch[0][QWEN_INPUTS_KEY]["attention_mask"][0].bool()]
    )
    print(f"\n[{bench}] {n} tensors/strings bitwise identical; prompt tail: {prompt[-120:]!r}")
