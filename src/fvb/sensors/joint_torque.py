"""Panda-style joint-torque sensing (PLAN §13, V2).

A real Panda measures the link-side torque tau_J at every joint and estimates the external torque
tau_ext = tau_J - tau_model, where tau_model is the rigid-body model torque M(q) qdd + c(q, qd)
(Franka's ``tau_ext_hat``). The simulated equivalent, all from one forward pass so the terms are
consistent (after ``mj_step`` the derived quantities belong to the pre-integration state):

* tau_J     = actuator torque + passive torque (damping, springs) + the joint's own dry friction
              (``frictionloss`` constraint rows), i.e. everything that acts *inside* the joint;
* tau_model = M qacc + qfrc_bias (armature included, as in the robot's own dynamics model);
* tau_ext   = tau_J - tau_model = -(J^T F_env), with F_env the contact / equality / applied forces
              the environment exerts on the robot.

So tau_ext = J^T F where F is the wrench the robot exerts *on* the environment: holding a hanging
1 kg load gives tau_ext = J^T (0, 0, +9.81 N); pressing down on a table gives a -z force.
``wrench`` maps tau_ext back to a Cartesian estimate F_ext_hat = pinv(J^T) tau_ext at a site.

Noise: white noise per sample plus a bias drawn per episode (``reset``). The defaults are
estimates for a strain-gauge joint sensor, not Franka datasheet values.
"""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

PANDA_ARM_JOINTS = tuple(f"robot0_joint{i}" for i in range(1, 8))


class JointTorqueSensor:
    """Measured (tau_J) and external (tau_ext) joint torques for a set of hinge joints."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joints: Sequence[str] = PANDA_ARM_JOINTS,
        noise_std: float = 0.05,
        bias_std: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.joints = tuple(joints)
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.joints]
        if min(jids) < 0:
            raise ValueError(f"unknown joint in {self.joints}")
        self.dofs = np.array([model.jnt_dofadr[j] for j in jids])
        self.noise_std = noise_std
        self.bias_std = bias_std
        self.rng = np.random.default_rng(seed)
        self.bias = np.zeros(len(self.dofs))
        self._dof_row = np.full(model.nv, -1)
        self._dof_row[self.dofs] = np.arange(len(self.dofs))
        self._mqacc = np.zeros(model.nv)

    def reset(self) -> None:
        """Draw a new per-episode bias."""
        self.bias = self.rng.normal(0.0, self.bias_std, len(self.dofs))

    def _joint_friction(self, data: mujoco.MjData) -> np.ndarray:
        out = np.zeros(len(self.dofs))
        sel = data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
        for i in np.flatnonzero(sel):
            row = self._dof_row[data.efc_id[i]]
            if row >= 0:
                out[row] += data.efc_force[i]
        return out

    def true(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Noise-free (tau_J, tau_ext)."""
        tau_j = (
            data.qfrc_actuator[self.dofs]
            + data.qfrc_passive[self.dofs]
            + self._joint_friction(data)
        )
        mujoco.mj_mulM(self.model, data, self._mqacc, data.qacc)
        tau_model = self._mqacc[self.dofs] + data.qfrc_bias[self.dofs]
        return tau_j, tau_j - tau_model

    def read(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(tau_J, tau_ext) with the sensor's noise and bias (the same draw on both)."""
        tau_j, tau_ext = self.true(data)
        err = self.bias
        if self.noise_std > 0:
            err = err + self.rng.normal(0.0, self.noise_std, len(self.dofs))
        return tau_j + err, tau_ext + err

    def jacobian(self, data: mujoco.MjData, site: str) -> np.ndarray:
        """6 x n spatial Jacobian (linear; angular) of ``site`` w.r.t. the sensed joints."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jp, jr, sid)
        return np.vstack([jp, jr])[:, self.dofs]

    def wrench(self, data: mujoco.MjData, tau_ext: np.ndarray, site: str) -> np.ndarray:
        """F_ext_hat (force; torque about the site, world frame) = pinv(J^T) tau_ext."""
        return np.linalg.pinv(self.jacobian(data, site).T) @ tau_ext
