# Parity tests

These tests check that the released code reproduces the outputs of the original
training code bit-for-bit (data layer) or within tight tolerances (model). They need
reference fixtures that are not part of the repository.

| Test | Needs | Hardware |
|---|---|---|
| `test_data_parity.py` | `POINTMAPVLA_FIXTURES`, the release datasets under `playground/Datasets`, a Qwen-VL processor (`POINTMAPVLA_BASE_VLM` or `POINTMAPVLA_CKPT_DIR`) | CPU |
| `test_release_parity.py` | `POINTMAPVLA_FIXTURES`, `POINTMAPVLA_CKPT_DIR` with the three released checkpoints | 1 GPU |

Tests skip when their variables are unset.

## Obtaining fixtures

The fixtures (`{robocasa,libero}_train_batch.pt/.json`, `{bench}_infer.pt`,
`{bench}_ref_actions.npy`, `{bench}_ref_loss.json`, `{bench}_ref_remap.pt`,
`internal_keys_*.json`, about 40 MB) were captured once from the original training
code with the paper checkpoints. They are published as the Hugging Face dataset
`DAVIAN-Robotics/PointmapVLA-parity-fixtures`; download them into any directory and point
`POINTMAPVLA_FIXTURES` at it:

```bash
hf download DAVIAN-Robotics/PointmapVLA-parity-fixtures --repo-type dataset --local-dir /path/to/fixtures
export POINTMAPVLA_FIXTURES=/path/to/fixtures
export POINTMAPVLA_CKPT_DIR=/path/to/checkpoints      # holds PointmapVLA-{Pretrained,RoboCasa,LIBERO}/
python scripts/download_data.py --bench robocasa      # and/or --bench libero
pytest tests/parity -s
```

`{bench}_train_batch.json` records the `(dataset, trajectory_id, base_index)` coordinates
of each reference sample, so the data-parity test does not depend on the mixture's
random sampling order.
