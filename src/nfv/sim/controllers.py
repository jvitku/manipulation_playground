"""Position and hybrid force-position control of the nfv gantry, and the control-rate clock.

The clock ladder (asserted in tests/nfv/test_nfv_timing.py):

    physics     2 ms   (500 Hz)   PHYS_DT
    controller  5 ms   (200 Hz)   CTRL_DT   = 2.5 physics steps -> alternating 2/3 substeps
    policy    100 ms   ( 10 Hz)   POLICY_DT = 20 controller ticks = 50 physics steps

200 Hz does not divide the 500 Hz physics clock, so one controller tick advances physics by
``PHYS_PER_CTRL[i % 2]`` steps — 2, 3, 2, 3, … (4 ms / 6 ms, 5 ms on average) — and 20 ticks
cover exactly one 100 ms policy step. The 200 Hz wrench stream is sampled once per tick.

Position mode (f_target is None, or on unselected axes):
    The 10 Hz chunk action Δp (metres, world = joint axes, clamped to ±DP_MAX per step) is
    linearly interpolated to 200 Hz position targets, which drive MuJoCo position actuators:
    a PD with stiffness K_p (``GantryParams.kp``, default 800 N/m) and joint damping as the D
    term. Steady contact force under position mode is therefore ~K_p times the target overshoot
    — the "a position policy commands a penetration depth, not a force" lesson.

Hybrid mode — ForceVLA2 leaves its controller unspecified, so THIS is the course's
written-down law (plan §4; every number below is a deliberate choice, not from the paper):
    * Per-axis selection, evaluated once per policy step from the commanded force target:
      S_i = 1[|f_target,i| > 0.5 N], with 0.1 N of hysteresis — an axis already selected stays
      selected until |f_target,i| < 0.4 N. f_target is a (6,) wrench but only the force part
      (x, y, z) can select; the gantry has no compliant rotation, torque targets are ignored
      and yaw + gripper always run position mode.
    * Selected axes move by admittance at 200 Hz:
          xdot_i = clip(K_a * (f_target,i − f_meas,i), ±V_MAX),   p_cmd_i += xdot_i * CTRL_DT
      with K_a = 0.01 m/(s·N) and V_MAX = 0.05 m/s (5 cm/s). f_meas is the compensated,
      filtered wrench in the course sign convention (push up = negative F_z), so "press down
      with 5 N" is f_target_z = −5 and the integrator descends until contact balances it.
      The admittance is integral action on the force error: steady-state force tracking is
      exact regardless of K_p, surface height or tool weight.
    * Unselected axes follow position mode; their Δp component applies, the Δp component of a
      selected axis is ignored for that step.
    * Safety: |f_meas[:3]| > F_ABORT (25 N) aborts the episode — the env flags
      ``force_abort`` and the episode counts as a failure, so digging in can never score.

All commanded targets are clamped to the joint ranges, so a force error can never wind the
command out of the workspace.
"""

from __future__ import annotations

import mujoco
import numpy as np

# --- the clock ladder (see module docstring) ---
PHYS_DT = 0.002  # s
CTRL_DT = 0.005  # s
POLICY_DT = 0.1  # s
CTRL_PER_POLICY = 20  # controller ticks per policy step
PHYS_PER_POLICY = 50  # physics steps per policy step
PHYS_PER_CTRL = (2, 3)  # alternating physics substeps per controller tick (mean 2.5)

# --- the hybrid law's constants ---
DP_MAX = 0.05  # m, max |Δp| per axis per policy step (0.5 m/s commanded ceiling)
SELECT_ON_N = 0.5  # N, |f_target,i| above this selects axis i for force control
SELECT_HYST_N = 0.1  # N, hysteresis: deselect only below SELECT_ON_N − SELECT_HYST_N
K_ADMITTANCE = 0.01  # m/(s·N)
V_MAX = 0.05  # m/s, admittance velocity clamp
F_ABORT = 25.0  # N, |force| above this aborts the episode (failure)

_RANGE_MARGIN = 1e-4  # m, keep commands strictly inside the joint range


class GantryController:
    """Drives the gantry's position actuators at 200 Hz from 10 Hz policy actions.

    Usage per policy step (the env does this):
        ctl.begin_policy_step(dp, gripper, f_target)
        for i in range(CTRL_PER_POLICY):
            ctl.tick(f_meas)                      # writes data.ctrl from the current wrench
            ... advance physics PHYS_PER_CTRL[i % 2] steps, sample the wrench ...
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        act = {  # actuator name -> id
            n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ("ax", "ay", "az", "ayaw", "agl", "agr")
        }
        if min(act.values()) < 0:
            raise KeyError(f"gantry actuators missing: {act}")
        self._a_xyz = np.array([act["ax"], act["ay"], act["az"]])
        self._a_yaw = act["ayaw"]
        self._a_grip = np.array([act["agl"], act["agr"]])
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("jx", "jy", "jz")]
        self._q_xyz = model.jnt_qposadr[jid]
        self._xyz_lo = model.jnt_range[jid, 0] + _RANGE_MARGIN  # (3,)
        self._xyz_hi = model.jnt_range[jid, 1] - _RANGE_MARGIN  # (3,)
        self.kp = float(model.actuator_gainprm[act["ax"], 0])  # informational: the exposed K_p
        self.grip_travel = float(model.actuator_ctrlrange[act["agl"], 1])
        self.reset()

    def reset(self) -> None:
        """Re-anchor the command state on the CURRENT joint positions (call after env reset)."""
        self._p_cmd = np.array(self.data.qpos[self._q_xyz], dtype=np.float64)  # (3,) xyz targets
        self._grip_cmd = 0.0
        self._selected = np.zeros(3, dtype=bool)  # per-axis force-control selection (hysteresis)
        self._p0 = self._p_cmd.copy()
        self._dp = np.zeros(3)
        self._g0 = self._g1 = 0.0
        self._ft = np.zeros(3)
        self._tick_i = 0
        self._write_ctrl()

    @property
    def selected(self) -> np.ndarray:
        """Copy of the per-axis force-selection state (3,) bool — for logging/teaching."""
        return self._selected.copy()

    @property
    def p_cmd(self) -> np.ndarray:
        """Copy of the current 200 Hz position command (3,) m."""
        return self._p_cmd.copy()

    def begin_policy_step(
        self, dp: np.ndarray, gripper: float, f_target: np.ndarray | None
    ) -> None:
        """Latch one 10 Hz action: clamp Δp, evaluate the per-axis selection with hysteresis."""
        self._p0 = self._p_cmd.copy()
        self._dp = np.clip(np.asarray(dp, dtype=np.float64), -DP_MAX, DP_MAX)
        self._g0 = self._grip_cmd
        self._g1 = float(np.clip(gripper, 0.0, 1.0)) * self.grip_travel
        if f_target is None:
            self._selected[:] = False
            self._ft = np.zeros(3)
        else:
            ft = np.asarray(f_target, dtype=np.float64).ravel()  # (6,) commanded wrench
            self._ft = ft[:3].copy()  # torque components ignored (no compliant rotation)
            for i in range(3):
                # hysteresis: select above 0.5 N, an already-selected axis stays down to 0.4 N
                on = SELECT_ON_N - (SELECT_HYST_N if self._selected[i] else 0.0)
                self._selected[i] = abs(self._ft[i]) > on
        self._tick_i = 0

    def tick(self, f_meas: np.ndarray) -> None:
        """One 200 Hz update: move the position command, write data.ctrl.

        f_meas: the latest filtered contact wrench (6,) in the course sign convention.
        """
        i = self._tick_i
        frac = (i + 1) / CTRL_PER_POLICY  # linear interpolation reaches p0 + dp on the last tick
        f3 = np.asarray(f_meas, dtype=np.float64)[:3]  # (3,) force part of the (6,) wrench
        for ax in range(3):
            if self._selected[ax]:
                # admittance: integrate the force error into the position command (m/s, clamped)
                v = np.clip(K_ADMITTANCE * (self._ft[ax] - f3[ax]), -V_MAX, V_MAX)
                self._p_cmd[ax] += v * CTRL_DT
            else:
                self._p_cmd[ax] = self._p0[ax] + self._dp[ax] * frac
        np.clip(self._p_cmd, self._xyz_lo, self._xyz_hi, out=self._p_cmd)  # (3,) joint ranges
        self._grip_cmd = self._g0 + (self._g1 - self._g0) * frac
        self._tick_i = i + 1
        self._write_ctrl()

    def _write_ctrl(self) -> None:
        self.data.ctrl[self._a_xyz] = self._p_cmd
        self.data.ctrl[self._a_yaw] = 0.0  # yaw position-held at 0 in M0
        self.data.ctrl[self._a_grip] = self._grip_cmd

    @staticmethod
    def abort(f_meas: np.ndarray) -> bool:
        """The 25 N safety: True if the force magnitude exceeds F_ABORT."""
        return bool(np.linalg.norm(np.asarray(f_meas)[:3]) > F_ABORT)
