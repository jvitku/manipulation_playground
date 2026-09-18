"""Ground-truth contact wrench from ``mujoco.mj_contactForce``.

``mj_contactForce(m, d, i, out)`` returns the 6-vector ``[fn, ft1, ft2, tn, tt1, tt2]`` in the
contact frame whose x-axis is the contact normal, stored row-major in ``d.contact[i].frame``.
MuJoCo's convention: the normal points from ``geom1`` **towards** ``geom2``, and the force
returned is the force applied **by geom1 on geom2** ... no — this is exactly the sort of thing
CLAUDE.md says to verify, so :func:`geom_contact_wrench` has a documented sign convention that
``tests/test_contact_consistency.py`` pins against the F/T sensor:

    ``geom_contact_wrench`` returns the total wrench **acting on** the listed geoms, in the
    world frame, about ``about_point`` (default: world origin).
"""

from __future__ import annotations

from collections.abc import Iterable

import mujoco
import numpy as np


def _geom_ids(m: mujoco.MjModel, geoms: Iterable[str | int]) -> set[int]:
    out = set()
    for g in geoms:
        if isinstance(g, str):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if gid < 0:
                raise KeyError(g)
            out.add(gid)
        else:
            out.add(int(g))
    return out


def body_geom_ids(m: mujoco.MjModel, body: str) -> set[int]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return {i for i in range(m.ngeom) if m.geom_bodyid[i] == bid}


def geom_contact_wrench(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    geoms: Iterable[str | int],
    about_point: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Sum of contact wrenches acting on ``geoms``, world frame.

    Returns ``(wrench6, n_contacts)``. Torque is taken about ``about_point`` (world), so pass
    the F/T site position to compare with the torque sensor.
    """
    ids = _geom_ids(m, geoms)
    about = np.zeros(3) if about_point is None else np.asarray(about_point, dtype=np.float64)
    f_tot = np.zeros(3)
    t_tot = np.zeros(3)
    n = 0
    buf = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        in1, in2 = g1 in ids, g2 in ids
        if in1 == in2:  # neither, or self-contact between two listed geoms
            continue
        mujoco.mj_contactForce(m, d, i, buf)
        frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)  # rows = axes in world
        f_world = frame.T @ buf[:3]
        t_world = frame.T @ buf[3:]
        # mj_contactForce gives the force in the contact frame with the normal pointing from
        # geom1 to geom2; that force is the one acting ON geom2 (equivalently, minus the force
        # on geom1). Verified in tests/test_contact_consistency.py.
        sign = 1.0 if in2 else -1.0
        f = sign * f_world
        t = sign * t_world
        r = np.array(c.pos, dtype=np.float64) - about
        f_tot += f
        t_tot += t + np.cross(r, f)
        n += 1
    return np.concatenate([f_tot, t_tot]), n


def max_penetration(m: mujoco.MjModel, d: mujoco.MjData, geoms: Iterable[str | int]) -> float:
    """Largest penetration depth (m, positive) among contacts involving ``geoms``. 0 if none."""
    ids = _geom_ids(m, geoms)
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if int(c.geom1) in ids or int(c.geom2) in ids:
            worst = max(worst, -float(c.dist))
    return worst
