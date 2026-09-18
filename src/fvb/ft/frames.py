"""Rotate a wrench between the sensor-site frame and the world frame.

A wrench ``[f, t]`` expressed in frame S with rotation ``R`` (S -> world, i.e. columns are
S's axes in world coordinates) becomes ``[R f, R t]`` in world. No lever-arm term because
both are about the same point (the site origin).
"""

from __future__ import annotations

import numpy as np


def rotate_wrench(w: np.ndarray, R: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if w.ndim == 1:
        return np.concatenate([R @ w[:3], R @ w[3:]])
    # batched: w (T,6), R (T,3,3)
    return np.concatenate(
        [np.einsum("tij,tj->ti", R, w[:, :3]), np.einsum("tij,tj->ti", R, w[:, 3:])], axis=1
    )


def site_to_world(w_site: np.ndarray, R_site: np.ndarray) -> np.ndarray:
    return rotate_wrench(w_site, R_site)


def world_to_site(w_world: np.ndarray, R_site: np.ndarray) -> np.ndarray:
    R_site = np.asarray(R_site)
    RT = R_site.T if R_site.ndim == 2 else np.transpose(R_site, (0, 2, 1))
    return rotate_wrench(w_world, RT)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return quat_wxyz_to_xyzw(q)
