"""Track A scripted control of the 3-DoF gantry (position actuators = springs).

The actuators are ``F = kp * (target - q)``, so commanding a target is commanding an
equilibrium; the peg goes where the spring+gravity+contact balance puts it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from fvb.contacts import geom_contact_wrench
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

    def __post_init__(self) -> None:
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

    def step(self, target: np.ndarray, log: EpisodeLogger | None = None) -> None:
        """One control step: set target, run ``n_sub`` physics steps, log hf + one row."""
        self.set_target(target)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.m, self.d)
            if log is not None:
                log.step_hf(self.d.time, self.ft.ft_raw())
        if log is not None:
            self.log_row(log, target)

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
