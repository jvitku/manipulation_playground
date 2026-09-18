"""Track A scripted control of the 3-DoF gantry (position actuators = springs).

The actuators are ``F = kp * (target - q)``, so commanding a target is commanding an
equilibrium; the peg goes where the spring+gravity+contact balance puts it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from fvb.contacts import geom_contact_wrench
from fvb.ft.compensate import LoadParams, compensate
from fvb.ft.frames import mat_to_quat_xyzw, site_to_world
from fvb.ft.read import MujocoFT
from fvb.logging.episode import EpisodeLogger
from fvb.scenes.builder import SceneParams


@dataclass
class Gantry:
    m: mujoco.MjModel
    d: mujoco.MjData
    params: SceneParams
    control_freq: float = 100.0
    load: LoadParams | None = None  # if set, ft_comp is computed at physics rate (see step)

    def __post_init__(self) -> None:
        self._prev_vel: np.ndarray | None = None
        self._comp_acc = np.zeros(6)
        self._comp_n = 0
        self.ft = MujocoFT(self.m, self.d)
        self.peg_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self.peg_geoms = ["peg"]
        self.n_sub = int(round(1.0 / (self.control_freq * self.m.opt.timestep)))
        assert self.n_sub >= 1
        self.jids = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("jx", "jy", "jz")
        ]
        self.qadr = [int(self.m.jnt_qposadr[j]) for j in self.jids]
        self.vadr = [int(self.m.jnt_dofadr[j]) for j in self.jids]

    # -- state -----------------------------------------------------------------
    @property
    def q(self) -> np.ndarray:
        return np.array([self.d.qpos[a] for a in self.qadr])

    @property
    def qd(self) -> np.ndarray:
        return np.array([self.d.qvel[a] for a in self.vadr])

    def ee_pos(self) -> np.ndarray:
        return self.ft.site_pos()

    def ee_rot(self) -> np.ndarray:
        return self.ft.site_rot()

    def ee_vel(self) -> np.ndarray:
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_SITE, self.ft.site_id, v, 0)
        return np.concatenate([v[3:], v[:3]])  # mujoco gives [ang, lin]; we want [lin, ang]

    def contact_wrench(self) -> tuple[np.ndarray, int]:
        return geom_contact_wrench(self.m, self.d, self.peg_geoms, about_point=self.ee_pos())

    # -- stepping ----------------------------------------------------------------
    def settle(self, seconds: float = 1.0) -> None:
        """Let the peg hang at rest on the springs at the current targets."""
        for _ in range(int(seconds / self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)

    def set_target(self, xyz: np.ndarray) -> None:
        self.d.ctrl[:3] = np.asarray(xyz, dtype=np.float64)

    def set_tilt(self, angle_rad: float) -> None:
        """Target for the optional wrist hinge (builder ``tilt_axis``)."""
        assert self.m.nu >= 4, "scene has no tilt joint"
        self.d.ctrl[3] = angle_rad

    def ee_acc_world(self) -> np.ndarray:
        """Site linear acceleration (world) from MuJoCo's ``cacc``.

        NOTE: MuJoCo's ``cacc``/``mj_objectAcceleration`` is computed by ``mj_rnePostConstraint``
        with the world body "accelerating" at −g, so for a body at rest this returns +9.81 ẑ — a
        proper acceleration like an IMU, i.e. ``a − g``. We subtract that back out here and
        return the true kinematic acceleration. Verified in scripts/02_contamination.py.
        """
        res = np.zeros(6)
        mujoco.mj_objectAcceleration(
            self.m, self.d, mujoco.mjtObj.mjOBJ_SITE, self.ft.site_id, res, 0
        )
        return res[3:] + np.asarray(self.m.opt.gravity)

    def step(self, target: np.ndarray, log: EpisodeLogger | None = None) -> None:
        """One control step: set target, run ``n_sub`` physics steps, log hf + one row.

        If ``self.load`` is set, gravity + inertial compensation runs at **physics rate** with a
        finite-difference site acceleration (as a real F/T driver would at ~1 kHz) and the
        control-rate ``ft_comp`` is the mean over the interval. Compensating at control rate
        does not work (see M2 in docs/FINDINGS.md).
        """
        self.set_target(target)
        self._comp_acc[:] = 0.0
        self._comp_n = 0
        for _ in range(self.n_sub):
            mujoco.mj_step(self.m, self.d)
            raw = self.ft.ft_raw()
            if log is not None:
                log.step_hf(self.d.time, raw)
            if self.load is not None:
                v = self.ee_vel()[:3]
                a = (
                    np.zeros(3)
                    if self._prev_vel is None
                    else (v - self._prev_vel) / self.m.opt.timestep
                )
                self._prev_vel = v
                self._comp_acc += compensate(raw, self.ee_rot(), self.load, a)
                self._comp_n += 1
        if log is not None:
            comp = self._comp_acc / self._comp_n if self.load is not None else None
            self.log_row(log, target, ft_comp=comp)

    def ft_comp_last(self) -> np.ndarray | None:
        """Compensated wrench averaged over the last control interval (None if no load set)."""
        return None if self.load is None else self._comp_acc / max(self._comp_n, 1)

    def log_row(self, log: EpisodeLogger, action: np.ndarray, ft_comp: np.ndarray | None = None):
        raw = self.ft.ft_raw()
        R = self.ee_rot()
        cw, n = self.contact_wrench()
        log.step(
            t=self.d.time,
            ee_pos=self.ee_pos(),
            ee_quat=mat_to_quat_xyzw(R),
            ee_vel=self.ee_vel(),
            ft_raw=raw,
            ft_world=site_to_world(raw, R),
            ft_comp=raw if ft_comp is None else ft_comp,
            contact_wrench=cw,
            n_contacts=n,
            action=np.asarray(action, dtype=np.float64),
        )


def linear_descent_targets(
    z_start: float, z_end: float, speed: float, dt: float, xy: np.ndarray, hold: float = 0.5
) -> np.ndarray:
    """(N,3) targets: constant-velocity descent from z_start to z_end, then hold."""
    n_move = int(abs(z_end - z_start) / speed / dt)
    zs = np.linspace(z_start, z_end, max(n_move, 2))
    zs = np.concatenate([zs, np.full(int(hold / dt), z_end)])
    out = np.tile(np.asarray(xy, dtype=np.float64), (len(zs), 1))
    return np.column_stack([out, zs])
