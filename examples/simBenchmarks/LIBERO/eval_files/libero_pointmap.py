"""Online pointmap generation for LIBERO / LIBERO-Plus evaluation.

The training pointmaps were baked offline from re-rendered depth. At evaluation
time the same quantity has to be produced online from the live simulator, with
exactly the same conventions, or the model receives inputs it never saw:

  1. depth buffer -> metres with ``get_real_depth_map``
  2. rot180: the model sees ``rot180(obs)`` RGB (the dataset convention), so the
     pointmap grid is rotated the same way
  3. intrinsics ``[-fx, fy, W - cx, cy]``: the rot180 image is the left-right
     mirror of the geometrically correct orientation (the vertical flip is already
     absorbed by OpenGL's bottom-up rows). A mirror cannot be expressed as a camera
     rotation, so it is encoded in the sign of the focal length. Extrinsics are the
     simulator's cam->world transform unchanged.
  4. ``derive_pointmap(frame="world")`` then ``world_to_base`` (robot base frame)

The math is not copied: ``starVLA/dataloader/gr00t_lerobot/transform/pointmap_math.py``
is loaded by file path (numpy-only; the simulator env cannot import starVLA), so the
training loader and the evaluation client share one implementation.

The env must be created with ``camera_depths=True``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# eval_files -> LIBERO -> simBenchmarks -> examples -> <repo root>
_MATH_PATH = Path(__file__).resolve().parents[4] / "starVLA/dataloader/gr00t_lerobot/transform/pointmap_math.py"

# Same two views, same order as the LIBERO training data (agentview, wrist).
VIEWS = (("agentview", "agentview_depth"),
         ("wrist", "robot0_eye_in_hand_depth"))
CAM_NAMES = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}

_MATH = None


def _math():
    global _MATH
    if _MATH is None:
        spec = importlib.util.spec_from_file_location("pointmap_math_eval", _MATH_PATH)
        _MATH = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MATH)
    return _MATH


def _quat_wxyz_from_mat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> wxyz quaternion (Shepperd's method, as in the dataset bake)."""
    R = np.asarray(R, dtype=np.float64)
    t = float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def base_pose_xyzw(sim):
    """World pose of the robot base body -> (pos(3), quat xyzw(4))."""
    bid = sim.model.body_name2id("robot0_base")
    R = np.asarray(sim.data.body_xmat[bid]).reshape(3, 3)
    pos = np.asarray(sim.data.body_xpos[bid], dtype=np.float64).reshape(3)
    w, x, y, z = _quat_wxyz_from_mat(R)
    return pos, np.array([x, y, z, w], dtype=np.float64)


def derive_from_obs(sim, obs, res: int = 256, out_hw=(256, 256)):
    """obs + sim -> (pointmap (V, h, w, 3) float16, valid (V, h, w) bool) in the robot base frame."""
    import robosuite.utils.camera_utils as CU

    m = _math()
    base_pos, base_quat_xyzw = base_pose_xyzw(sim)
    sh = res // int(out_hw[0])
    sw = res // int(out_hw[1])
    assert sh >= 1 and res % int(out_hw[0]) == 0, f"render res {res} is not a multiple of out_hw {out_hw}"

    pms, pvs = [], []
    for view, depth_key in VIEWS:
        d = np.asarray(obs[depth_key])
        if d.ndim == 2:
            d = d[..., None]
        z = np.asarray(CU.get_real_depth_map(sim, d), dtype=np.float64)[..., 0]
        z = z[::-1, ::-1]  # rot180, same grid as the RGB the model sees

        cam = CAM_NAMES[view]
        K = CU.get_camera_intrinsic_matrix(sim, cam, res, res)
        E = CU.get_camera_extrinsic_matrix(sim, cam)  # 4x4 cam->world, OpenCV axes
        intr = [-K[0, 0], K[1, 1], res - K[0, 2], K[1, 2]]  # mirror encoded as negative fx
        extr7 = np.concatenate([E[:3, 3], _quat_wxyz_from_mat(E[:3, :3])])

        pm, valid = m.derive_pointmap(z, intr, extr7, frame="world")
        pm = m.world_to_base(pm, base_pos, base_quat_xyzw, "xyzw")
        pm = np.where(valid[..., None], pm, 0.0)
        pms.append(pm[::sh, ::sw, :])
        pvs.append(valid[::sh, ::sw])

    return np.stack(pms).astype(np.float16), np.stack(pvs).astype(bool)
