# Paired dataset for pointmap-VLA training: RGB / state / action from one LeRobot
# dataset, depth + camera features from a second, episode-aligned one.
#
# Why two datasets instead of one: the depth re-render that produced the pointmap
# dataset also re-rendered RGB with regenerated textures, while evaluation renders the
# original textures. Training RGB must therefore come from the ORIGINAL recordings
# and only depth + camera features from the pointmap dataset. The two datasets are
# positionally episode-1:1, so the same (trajectory_id, base_index) coordinate
# addresses the same physical frame in both.
#
# Design:
#   - primary  = original dataset, built by the normal DataConfig path.
#   - secondary = pointmap dataset opened WITHOUT video keys (depth keys + camera
#     feature state keys only): no RGB decode from the re-render at all.
#   - get_step_data() merges the secondary's raw keys into the primary's raw dict; the
#     primary's transform chain (which the pointmap DataConfig starts with
#     PointmapDerive) then produces pointmap.<view> / pointmap_valid.<view>.
#   - _pack_sample() calls super() first and only APPENDS the two pointmap keys.

from pathlib import Path

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform


class PointmapPairedDataset(LeRobotSingleDataset):
    """LeRobotSingleDataset over the ORIGINAL dataset, with depth + camera features
    merged in from a positionally episode-1:1 pointmap dataset.

    Extra args (on top of LeRobotSingleDataset's):
        pm_dataset_path: root of the pointmap dataset.
        pm_views: view base name -> {"depth": key, "intrinsics": key, "extrinsics": key}
            (same mapping the DataConfig hands to PointmapDerive). View ORDER of this
            dict defines the V axis of the packed sample and MUST match the primary's
            RGB view order.
        pm_out_hw: packed pointmap resolution. Downsampling from the native depth
            resolution is stride-N NEAREST subsampling -- never interpolate XYZ across
            invalid/edge pixels.
        pm_extra_state_keys: extra RAW state keys to pull from the pointmap dataset.
            Needed when the primary state vector carries no base pose (LIBERO: the
            11-D state has none, so ``state.base_position`` / ``state.base_rotation``
            come from the pointmap dataset for ``PointmapDerive(frame='base')``).
    """

    def __init__(
        self,
        *args,
        pm_dataset_path: str | Path,
        pm_views: dict[str, dict[str, str]],
        pm_out_hw: tuple[int, int] = (128, 128),
        pm_extra_state_keys: tuple[str, ...] = (),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pm_views = pm_views
        self.pm_out_hw = tuple(pm_out_hw)

        depth_keys = [v["depth"] for v in pm_views.values()]
        cam_keys = [v[w] for v in pm_views.values() for w in ("intrinsics", "extrinsics")]
        cam_keys = cam_keys + [k for k in pm_extra_state_keys if k not in cam_keys]
        pm_modality = {
            "depth": ModalityConfig(delta_indices=[0], modality_keys=depth_keys),
            "state": ModalityConfig(delta_indices=[0], modality_keys=cam_keys),
        }
        self.pm_dataset = LeRobotSingleDataset(
            dataset_path=Path(pm_dataset_path),
            modality_configs=pm_modality,
            # No transforms: the merged dict runs through the PRIMARY's chain
            # (which starts with PointmapDerive); keys here must stay raw.
            transforms=ComposedModalityTransform(transforms=[]),
            embodiment_tag=self.tag,
            video_backend="pyav",  # FFV1 gray16le depth is decoded with pyav only
            data_cfg={"lerobot_version": "v3.0"},
        )

        # Alignment gate: positional episode-1:1 is the load-bearing assumption.
        n_p, n_s = len(self.trajectory_lengths), len(self.pm_dataset.trajectory_lengths)
        assert n_p == n_s, (
            f"paired datasets disagree on episode count: primary {n_p} vs pointmap {n_s} "
            f"({self.dataset_path} vs {pm_dataset_path})"
        )
        assert np.array_equal(
            np.asarray(self.trajectory_lengths), np.asarray(self.pm_dataset.trajectory_lengths)
        ), (
            "paired datasets disagree on per-episode frame counts -- positional 1:1 "
            f"alignment broken ({self.dataset_path} vs {pm_dataset_path})"
        )

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = super().get_step_data(trajectory_id, base_index)
        pm_raw = self.pm_dataset.get_step_data(trajectory_id, base_index)
        overlap = set(data) & set(pm_raw)
        assert not overlap, f"paired key collision (would clobber primary data): {overlap}"
        data.update(pm_raw)
        return data

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)

        pms, valids = [], []
        for view in self.pm_views:  # dict order == RGB view order
            pm = np.asarray(data[f"pointmap.{view}"])[0]        # (H, W, 3) float32, frame 0
            pv = np.asarray(data[f"pointmap_valid.{view}"])[0]  # (H, W) bool
            sh, sw = pm.shape[0] // self.pm_out_hw[0], pm.shape[1] // self.pm_out_hw[1]
            assert sh >= 1 and pm.shape[0] % self.pm_out_hw[0] == 0 and pm.shape[1] % self.pm_out_hw[1] == 0, (
                f"pointmap.{view} {pm.shape} not an integer multiple of {self.pm_out_hw}"
            )
            pms.append(pm[::sh, ::sw, :])
            valids.append(pv[::sh, ::sw])
        sample["pointmap"] = np.stack(pms).astype(np.float16)     # (V, h, w, 3)
        sample["pointmap_valid"] = np.stack(valids)               # (V, h, w) bool
        return sample
