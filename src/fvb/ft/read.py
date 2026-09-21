"""Force/torque accessors for raw MuJoCo and robosuite, behind one interface.

Conventions (see PLAN.md §5):
  * A wrench is a length-6 float64 array ``[fx, fy, fz, tx, ty, tz]``.
  * ``ft_raw`` is in the **sensor-site frame**, unmodified from MuJoCo.
  * Force in N, torque in N·m.

A MuJoCo ``force``/``torque`` sensor attached to site ``s`` on body ``b`` reports the
interaction wrench between ``b`` and its parent, expressed in the site frame. Which
sign convention this is (force *on* the child or *by* the child) is derived
experimentally in M1 — do not assume it here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np


class FTSource(Protocol):
    """Anything that can produce a raw sensor-frame wrench and the sensor site pose."""

    def ft_raw(self) -> np.ndarray:  # (6,) sensor-site frame
        ...

    def site_rot(self) -> np.ndarray:  # (3,3) rotation sensor-site -> world
        ...

    def site_pos(self) -> np.ndarray:  # (3,) world
        ...


@dataclass
class MujocoFT:
    """Raw MuJoCo F/T reader over a ``force`` + ``torque`` sensor pair on one site."""

    model: mujoco.MjModel
    data: mujoco.MjData
    force_sensor: str = "ft_force"
    torque_sensor: str = "ft_torque"

    def __post_init__(self) -> None:
        self._f_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, self.force_sensor)
        self._t_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, self.torque_sensor)
        if self._f_id < 0 or self._t_id < 0:
            raise KeyError(f"sensors {self.force_sensor!r}/{self.torque_sensor!r} not in model")
        self._f_adr = int(self.model.sensor_adr[self._f_id])
        self._t_adr = int(self.model.sensor_adr[self._t_id])
        assert self.model.sensor_dim[self._f_id] == 3
        assert self.model.sensor_dim[self._t_id] == 3
        self._site_id = int(self.model.sensor_objid[self._f_id])
        assert self.model.sensor_objtype[self._f_id] == mujoco.mjtObj.mjOBJ_SITE

    @property
    def site_id(self) -> int:
        return self._site_id

    @property
    def site_name(self) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, self._site_id)

    def force(self) -> np.ndarray:
        return np.array(self.data.sensordata[self._f_adr : self._f_adr + 3], dtype=np.float64)

    def torque(self) -> np.ndarray:
        return np.array(self.data.sensordata[self._t_adr : self._t_adr + 3], dtype=np.float64)

    def ft_raw(self) -> np.ndarray:
        return np.concatenate([self.force(), self.torque()])

    def site_rot(self) -> np.ndarray:
        return np.array(self.data.site_xmat[self._site_id], dtype=np.float64).reshape(3, 3)

    def site_pos(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self._site_id], dtype=np.float64)


@dataclass
class RobosuiteFT:
    """F/T reader for a robosuite single-arm env (verified on robosuite 1.5.2).

    Two views of the same sensor are exposed so they can be cross-checked (PLAN §7 M4):
      * ``robot.ee_force["right"]`` / ``robot.ee_torque["right"]`` — robosuite's accessor.
      * raw MuJoCo sensors ``gripper0_right_force_ee`` / ``gripper0_right_torque_ee``.
    Note: ``obs["robot0_eef_force"]`` does NOT exist in 1.5.2.
    """

    env: object  # robosuite env
    arm: str = "right"
    force_sensor: str | None = None  # override for envs without a gripper (fvb.envs.PegInHole)
    torque_sensor: str | None = None

    def __post_init__(self) -> None:
        robot = self.env.robots[0]
        self.robot = robot
        prefix = robot.gripper[self.arm].naming_prefix  # e.g. "gripper0_right_"
        self._raw = MujocoFT(
            self.env.sim.model._model,
            self.env.sim.data._data,
            force_sensor=self.force_sensor or f"{prefix}force_ee",
            torque_sensor=self.torque_sensor or f"{prefix}torque_ee",
        )

    @property
    def raw(self) -> MujocoFT:
        return self._raw

    def ft_raw(self) -> np.ndarray:
        """Raw sensor wrench via MuJoCo sensordata (sensor-site frame)."""
        return self._raw.ft_raw()

    def ft_robosuite(self) -> np.ndarray:
        """Same wrench through robosuite's ``ee_force``/``ee_torque`` accessors.

        Only meaningful when the robot has a gripper with the ``*_force_ee`` sensors; for
        fvb.envs.PegInHole (no gripper) this returns the raw sensor instead.
        """
        if self.force_sensor is not None:
            return self.ft_raw()
        return np.concatenate(
            [np.asarray(self.robot.ee_force[self.arm]), np.asarray(self.robot.ee_torque[self.arm])]
        ).astype(np.float64)

    def site_rot(self) -> np.ndarray:
        return self._raw.site_rot()

    def site_pos(self) -> np.ndarray:
        return self._raw.site_pos()

    def joint_torques(self) -> np.ndarray:
        """Applied arm joint torques (N·m), shape (7,) for Panda.

        ``robot.torques`` is ``None`` in robosuite 1.5.2 (never populated on the
        composite-controller path). The applied torque is what the OSC controller wrote into
        ``sim.data.ctrl`` for the arm actuators (all Panda arm actuators have gear 1, so
        ``ctrl == qfrc_actuator``). Verified in scripts/00_probe_api.py.
        """
        idx = self.robot._ref_arm_joint_actuator_indexes
        return np.asarray(self.env.sim.data.ctrl[idx], dtype=np.float64)
