"""Panda-style joint-torque sensing (PLAN §13, V2).

A real Panda measures the link-side torque tau_J at every joint and estimates the external torque
tau_ext = tau_J - tau_model, where tau_model is the rigid-body model torque M(q) qdd + c(q, qd)
(Franka's ``tau_ext_hat``). The simulated equivalent, all from one forward pass so the terms are
consistent (after ``mj_step`` the derived quantities belong to the pre-integration state):

* tau_J     = actuator torque + passive torque (damping, springs) + the joint's own dry friction
              (``frictionloss`` rows) and its end stops (joint-limit rows), i.e. everything that
              acts *inside* the joint (a limit reaction otherwise reads as hundreds of N);
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

    def _joint_internal(self, data: mujoco.MjData) -> np.ndarray:
        """Joint-space force of each joint's own friction-loss and limit constraint rows."""
        out = np.zeros(len(self.dofs))
        types = data.efc_type[: data.nefc]
        for i in np.flatnonzero(types == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF):
            row = self._dof_row[data.efc_id[i]]
            if row >= 0:
                out[row] += data.efc_force[i]
        for i in np.flatnonzero(types == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT):
            jnt = data.efc_id[i]
            if self.model.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            row = self._dof_row[self.model.jnt_dofadr[jnt]]
            if row >= 0:
                # limit rows have J = +-1 at the dof; J^T f is the force on the joint
                out[row] += self._efc_j(data, i, self.model.jnt_dofadr[jnt]) * data.efc_force[i]
        return out

    def _efc_j(self, data: mujoco.MjData, row: int, dof: int) -> float:
        if mujoco.mj_isSparse(self.model):
            a, n = data.efc_J_rowadr[row], data.efc_J_rownnz[row]
            cols = data.efc_J_colind[a : a + n]
            vals = data.efc_J[a : a + n]
            hit = np.flatnonzero(cols == dof)
            return float(vals[hit[0]]) if len(hit) else 0.0
        return float(data.efc_J[row * self.model.nv + dof])

    def true(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Noise-free (tau_J, tau_ext)."""
        tau_j = (
            data.qfrc_actuator[self.dofs]
            + data.qfrc_passive[self.dofs]
            + self._joint_internal(data)
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
