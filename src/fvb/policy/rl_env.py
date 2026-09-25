"""Stage 1 RL: the arm hidden-hole task (``ArmTask``) as a reinforcement-learning environment.

* Observation: the last ``history`` ARM observations (fvb.policy.spec.ARM, 20 each), z-scored
  with fixed statistics (from the expert dataset), stacked oldest-first. With ``use_force=False``
  the 9 force/torque channels are zeroed *after* normalisation, in every stacked frame.
* Action: 3 numbers in [-1, 1], scaled to a peg-tip target delta of ±xy_scale / ±z_scale (m).
* Reward (privileged: it may use the true hole; the policy never observes it):
    depth progress [mm] + potential-based lateral shaping gamma*phi' - phi with
    phi = -lat_coef * |tip - hole|_xy [mm] (does not change the optimal policy, Ng et al. 1999)
    - time_cost per step - force_coef * max(0, |F| - force_free_N)
    + success_bonus on success, - abort_penalty on a force abort; all times reward_scale.
* ``terminated`` on success / force abort; ``truncated`` on timeout (TD3 bootstraps through it).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from fvb.policy.arm_task import ArmTask, ArmTaskParams, ArmWorld
from fvb.policy.data import Norm
from fvb.policy.spec import ARM


@dataclass
class RewardParams:
    gamma: float = 0.99
    lat_coef: float = 2.0
    time_cost: float = 0.1
    force_free_N: float = 20.0
    force_coef: float = 0.05
    success_bonus: float = 50.0
    abort_penalty: float = 20.0
    reward_scale: float = 0.1


class ArmRLEnv:
    def __init__(
        self,
        p: ArmTaskParams,
        world: ArmWorld,
        norm: Norm,
        use_force: bool = True,
        history: int = 4,
        xy_scale: float = 0.001,
        z_scale: float = 0.003,
        rp: RewardParams | None = None,
    ):
        self.p, self.world, self.norm = p, world, norm
        self.use_force, self.history = use_force, history
        self.scale = np.array([xy_scale, xy_scale, z_scale])
        self.rp = rp or RewardParams()
        self.obs_dim = ARM.obs_dim * history
        self.act_dim = 3
        self.task: ArmTask | None = None
        self._frames: deque = deque(maxlen=history)

    # -- observation ------------------------------------------------------------------------
    def _frame(self) -> np.ndarray:
        o = (self.task.observe() - self.norm.obs_mean) / self.norm.obs_std
        if not self.use_force:
            o[ARM.force_idx] = 0.0
        return o.astype(np.float32)

    def _obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames))

    # -- privileged quantities for the reward -----------------------------------------------
    def _lat_err_mm(self) -> float:
        t = self.task
        return 1e3 * float(np.linalg.norm(t.env.peg_tip_pos()[:2] - t.env.hole_center_world[:2]))

    def _phi(self) -> float:
        return -self.rp.lat_coef * self._lat_err_mm()

    # -- gym-like API -------------------------------------------------------------------------
    def reset(self, seed: int) -> np.ndarray:
        self.task = ArmTask(self.p, seed, self.world)
        f = self._frame()
        self._frames.clear()
        for _ in range(self.history):
            self._frames.append(f)
        self._depth = self.task.depth()
        self._phi_prev = self._phi()
        return self._obs()

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        done, reason = self.task.step(a * self.scale)
        rp = self.rp
        depth = self.task.depth()
        phi = self._phi()
        f = self.task.force_norm()
        r = 1e3 * (depth - self._depth)
        r += rp.gamma * phi - self._phi_prev
        r -= rp.time_cost + rp.force_coef * max(0.0, f - rp.force_free_N)
        if reason == "success":
            r += rp.success_bonus
        elif reason == "force_abort":
            r -= rp.abort_penalty
        self._depth, self._phi_prev = depth, phi
        self._frames.append(self._frame())
        terminated = reason in ("success", "force_abort")
        truncated = reason == "timeout"
        info = {"reason": reason, "force_N": f, "depth_mm": 1e3 * depth}
        return self._obs(), rp.reward_scale * r, terminated, truncated, info
