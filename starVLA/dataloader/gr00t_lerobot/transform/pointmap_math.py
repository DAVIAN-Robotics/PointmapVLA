# Pure NUMPY pointmap math, deliberately dependency-free.
#
# The evaluation clients run inside the simulator conda environments, which have
# neither pydantic nor the rest of the starVLA import chain. They load this module by
# FILE PATH via importlib, while the training transform imports it normally. One
# implementation, two import routes, so the train/eval conventions cannot drift.
# Keep this file free of any import other than numpy.

import numpy as np


def _quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def derive_pointmap(depth_m, intr, extr, frame: str = "world"):
    """Back-project ONE frame of metric depth into a pointmap.

    Convention: rays go through pixel centers (+0.5), OpenCV pinhole, extrinsics =
    pos(3) + quat wxyz(4) cam->world.

    Args:
        depth_m: (H, W) float meters, <=0 invalid.
        intr: (4,) fx, fy, cx, cy at this resolution.
        extr: (7,) pos + quat wxyz, cam->world, OpenCV optical axes.
        frame: 'world' (apply extrinsics) or 'camera'.
    Returns:
        pointmap (H, W, 3) float32 (invalid = 0), valid (H, W) bool.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    H, W = depth_m.shape
    fx, fy, cx, cy = [float(v) for v in np.asarray(intr).reshape(4)]
    extr = np.asarray(extr, dtype=np.float64).reshape(7)

    u = np.arange(W, dtype=np.float64)[None, :] + 0.5
    v = np.arange(H, dtype=np.float64)[:, None] + 0.5
    valid = depth_m > 0
    x = (u - cx) / fx * depth_m
    y = (v - cy) / fy * depth_m
    pts = np.stack([x, y, depth_m], axis=-1)  # camera frame, OpenCV
    if frame == "world":
        R = _quat_wxyz_to_mat(extr[3:7])
        pts = pts @ R.T + extr[:3]
    pm = np.where(valid[..., None], pts, 0.0).astype(np.float32)
    return pm, valid


def world_to_base(points, base_pos, base_quat, quat_convention: str = "xyzw") -> np.ndarray:
    """Re-express world-frame points in the ROBOT BASE frame: p' = R_base^T (p - t_base).

    ``base_quat`` is the base->world rotation as stored in ``state.base_rotation``; the
    released datasets store it as **xyzw** (the default).
    """
    base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
    q = np.asarray(base_quat, dtype=np.float64).reshape(4)
    if quat_convention == "xyzw":
        q = np.array([q[3], q[0], q[1], q[2]])
    elif quat_convention != "wxyz":
        raise ValueError(f"unknown quat_convention {quat_convention!r}")
    R = _quat_wxyz_to_mat(q)
    return ((np.asarray(points, dtype=np.float64) - base_pos) @ R).astype(np.float32)
