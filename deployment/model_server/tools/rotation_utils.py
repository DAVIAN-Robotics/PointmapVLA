# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Rotation conversions shared by the policy server and the simulator clients.

PointmapVLA stores end-effector rotations as "r6": the first two ROWS of the
rotation matrix, row-major (6 numbers). The simulators emit quaternions for the
state and accept axis-angle for the action, so two conversions are needed at
evaluation time:

  * quaternion (xyzw, robosuite convention) -> r6     (client, state input)
  * r6 -> axis-angle scaled to [-1, 1]                 (server, action output)

Everything here is numpy-only on purpose: the simulator conda environments do
not have torch, and both sides must compute the exact same quantity. All math
runs in float64 and casts back to the caller's dtype.
"""

from __future__ import annotations

import numpy as np

# OSC controller convention: an action rotation of +-1 means +-0.5 rad.
AXIS_ANGLE_UNIT_RAD = 0.5
_EPS = 1e-8


def quat_xyzw_to_r6(q: np.ndarray) -> np.ndarray:
    """(..., 4) quaternion in xyzw order -> (..., 6) first two rows of R."""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.maximum(np.sqrt(w * w + x * x + y * y + z * z), _EPS)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    ], axis=-1)


def r6_to_matrix(r6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt. Never NaN.

    Degenerate inputs fall back deterministically: a zero first row becomes e_x,
    and a second row parallel to the first is replaced by the rejection of e_y
    (or e_z when the first row is close to e_y).
    """
    r = np.asarray(r6, dtype=np.float64)
    a1, a2 = r[..., 0:3], r[..., 3:6]
    e_x = np.zeros_like(a1); e_x[..., 0] = 1.0
    e_y = np.zeros_like(a1); e_y[..., 1] = 1.0
    e_z = np.zeros_like(a1); e_z[..., 2] = 1.0

    n1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    b1 = np.where(n1 < _EPS, e_x, a1 / np.maximum(n1, _EPS))

    u2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    n2 = np.linalg.norm(u2, axis=-1, keepdims=True)
    alt_src = np.where(np.abs((b1 * e_y).sum(-1, keepdims=True)) < 0.9, e_y, e_z)
    alt = alt_src - (b1 * alt_src).sum(-1, keepdims=True) * b1
    alt = alt / np.maximum(np.linalg.norm(alt, axis=-1, keepdims=True), _EPS)
    b2 = np.where(n2 < _EPS, alt, u2 / np.maximum(n2, _EPS))

    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3) axis-angle in radians; stable near 0 and near pi."""
    R = np.asarray(R, dtype=np.float64)
    vee = np.stack([R[..., 2, 1] - R[..., 1, 2],
                    R[..., 0, 2] - R[..., 2, 0],
                    R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    s = 0.5 * np.linalg.norm(vee, axis=-1)                      # sin(angle)
    c = 0.5 * (np.trace(R, axis1=-2, axis2=-1) - 1.0)           # cos(angle)
    angle = np.arctan2(s, np.clip(c, -1.0, 1.0))

    axis_generic = vee / (2.0 * np.maximum(s, 1e-12))[..., None]
    aa_small = 0.5 * vee
    # Near pi the axis comes from the diagonal of (R + I) / 2 = axis axis^T.
    diag = np.maximum(np.diagonal(R, axis1=-2, axis2=-1) + 1.0, 0.0) * 0.5
    axis_pi = np.sqrt(diag)
    largest = diag.argmax(axis=-1)
    sym = np.stack([
        np.stack([np.ones_like(s), R[..., 0, 1] + R[..., 1, 0], R[..., 0, 2] + R[..., 2, 0]], axis=-1),
        np.stack([R[..., 0, 1] + R[..., 1, 0], np.ones_like(s), R[..., 1, 2] + R[..., 2, 1]], axis=-1),
        np.stack([R[..., 0, 2] + R[..., 2, 0], R[..., 1, 2] + R[..., 2, 1], np.ones_like(s)], axis=-1),
    ], axis=-2)
    signs = np.sign(np.take_along_axis(sym, largest[..., None, None], axis=-2)[..., 0, :])
    signs = np.where(signs == 0, 1.0, signs)
    axis_pi = axis_pi * signs
    axis_pi = axis_pi / np.maximum(np.linalg.norm(axis_pi, axis=-1, keepdims=True), 1e-12)

    small = angle < 1e-6
    near_pi = (~small) & (s < 1e-6)
    aa = np.where(near_pi[..., None], axis_pi * angle[..., None], axis_generic * angle[..., None])
    return np.where(small[..., None], aa_small, aa)


def r6_to_axis_angle_unit(r6: np.ndarray, clip: bool = True) -> np.ndarray:
    """(..., 6) r6 -> (..., 3) axis-angle in action units (+-1 == +-0.5 rad)."""
    r6 = np.asarray(r6)
    aa = matrix_to_axis_angle(r6_to_matrix(r6)) / AXIS_ANGLE_UNIT_RAD
    if clip:
        aa = np.clip(aa, -1.0, 1.0)
    return aa.astype(r6.dtype if r6.dtype.kind == "f" else np.float64)


def _self_check() -> None:
    """Round trip: random axis-angle -> R -> r6 -> axis-angle must match."""
    rng = np.random.default_rng(0)
    aa = rng.uniform(-0.5, 0.5, size=(64, 3))
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / angle
    K = np.zeros((64, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    R = np.eye(3) + np.sin(angle)[..., None] * K + (1 - np.cos(angle))[..., None] * (K @ K)
    r6 = R[:, :2, :].reshape(64, 6)
    back = r6_to_axis_angle_unit(r6, clip=False) * AXIS_ANGLE_UNIT_RAD
    assert np.allclose(back, aa, atol=1e-9), np.abs(back - aa).max()
    q = np.array([0.0, 0.0, 0.0, 1.0])            # identity, xyzw
    assert np.allclose(quat_xyzw_to_r6(q), [1, 0, 0, 0, 1, 0])
    assert np.isfinite(r6_to_axis_angle_unit(np.zeros(6))).all()


if __name__ == "__main__":
    _self_check()
    print("rotation_utils self-check passed")
