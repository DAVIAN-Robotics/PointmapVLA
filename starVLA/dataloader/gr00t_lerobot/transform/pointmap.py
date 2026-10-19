# Pointmap derivation for the pointmap datasets.
#
# The datasets deliberately do NOT store pointmaps ("derive, don't bake"): they store
# lossless metric depth (uint16 = meters * depth_scale, 0 = invalid) plus per-frame
# camera features (intrinsics fx,fy,cx,cy and extrinsics pos+quat wxyz, cam->world,
# OpenCV optical axes). This transform reconstructs the pointmap at load time from
# those three ingredients.

from typing import Any

import numpy as np
from pydantic import Field

from .base import ModalityTransform

# The pure math (numpy only) lives in pointmap_math.py so the evaluation clients can
# load it by file path inside the simulator environments. Re-exported here so
# training-side callers can import everything from this module.
from .pointmap_math import derive_pointmap, world_to_base


class PointmapDerive(ModalityTransform):
    """Derive pointmaps from depth + camera features, in the frame named by ``frame``:
    'base' (ROBOT BASE -- what every released config uses), 'world', or 'camera'.

    For each configured view this consumes:
      - ``depth.<view>``  (T, H, W) uint16   -- metric depth * depth_scale, 0 invalid
      - intrinsics state key  (T, 4) float   -- fx, fy, cx, cy
      - extrinsics state key  (T, 7) float   -- pos(3) + quat wxyz(4), cam->world OpenCV
    and produces:
      - ``pointmap.<view>``       (T, H, W, 3) float32 -- 3D points; invalid pixels = 0
      - ``pointmap_valid.<view>`` (T, H, W)    bool

    Pixel convention: rays go through pixel CENTERS ((u + 0.5 - cx) / fx). Camera
    feature arrays of length 1 are broadcast over T (fixed cameras / single-frame
    observation windows).

    With ``frame='base'`` the transform reads the RAW world base pose from the state
    keys below, so it must run BEFORE any state normalization in the chain.
    """

    views: dict[str, dict[str, str]] = Field(
        ...,
        description="out view name -> {'depth': depth key, 'intrinsics': state key, "
        "'extrinsics': state key}",
    )
    depth_scale: float = Field(default=10000.0, description="uint16 units per meter")
    frame: str = Field(default="world", description="'world', 'camera', or 'base' (robot base) output frame")
    base_position_key: str = Field(default="state.base_position", description="frame='base': world base pos key")
    base_rotation_key: str = Field(default="state.base_rotation", description="frame='base': world base quat key")
    base_quat_convention: str = Field(default="xyzw", description="quat layout of base_rotation_key")
    apply_to: list[str] = Field(default_factory=list, description="unused; views drives it")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for view, keys in self.views.items():
            depth = np.asarray(data[keys["depth"]])
            assert depth.ndim == 3 and depth.dtype == np.uint16, (
                f"{keys['depth']}: expected (T,H,W) uint16, got {depth.shape} {depth.dtype}"
            )
            intr = np.atleast_2d(np.asarray(data[keys["intrinsics"]], dtype=np.float64))
            extr = np.atleast_2d(np.asarray(data[keys["extrinsics"]], dtype=np.float64))
            T, H, W = depth.shape
            if intr.shape[0] == 1 and T > 1:
                intr = np.repeat(intr, T, axis=0)
            if extr.shape[0] == 1 and T > 1:
                extr = np.repeat(extr, T, axis=0)
            assert intr.shape == (T, 4) and extr.shape == (T, 7), (
                f"{view}: camera features {intr.shape}/{extr.shape} don't match T={T}"
            )

            if self.frame == "base":
                base_pos = np.atleast_2d(np.asarray(data[self.base_position_key], dtype=np.float64))
                base_rot = np.atleast_2d(np.asarray(data[self.base_rotation_key], dtype=np.float64))
                if base_pos.shape[0] == 1 and T > 1:
                    base_pos = np.repeat(base_pos, T, axis=0)
                    base_rot = np.repeat(base_rot, T, axis=0)
                # A state window that disagrees with T would silently pair the wrong
                # base pose with a depth frame; fail loudly instead.
                assert base_pos.shape == (T, 3) and base_rot.shape == (T, 4), (
                    f"{view}: base pose {base_pos.shape}/{base_rot.shape} doesn't match T={T} "
                    f"({self.base_position_key}/{self.base_rotation_key})"
                )

            pm = np.zeros((T, H, W, 3), dtype=np.float32)
            valid = depth > 0
            for t in range(T):
                # depth > 0 stays valid after /scale, so derive_pointmap's own valid
                # mask matches `valid[t]` exactly.
                pm_t, _ = derive_pointmap(
                    depth[t].astype(np.float64) / self.depth_scale,
                    intr[t],
                    extr[t],
                    frame="world" if self.frame == "base" else self.frame,
                )
                if self.frame == "base":
                    pm_t = world_to_base(pm_t, base_pos[t], base_rot[t], self.base_quat_convention)
                    pm_t = np.where(valid[t][..., None], pm_t, 0.0).astype(np.float32)
                pm[t] = pm_t
            data[f"pointmap.{view}"] = pm
            data[f"pointmap_valid.{view}"] = valid
        return data
