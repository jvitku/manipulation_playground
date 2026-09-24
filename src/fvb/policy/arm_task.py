"""Stage 1 G1: the hidden-hole insertion on the Panda (``fvb.envs.PegInHole``).

Same task as ``fvb.policy.task.GantryTask`` on the arm: the hole is displaced by a random,
unobserved (x, y) each episode; the peg starts above the *nominal* hole centre. The policy
outputs peg-tip target deltas; a fixed low-level layer (``ArmTask.low_level``) turns the tip
target into OSC_POSE actions, servoing the peg axis to vertical and integrating the xy tip
error (M6: a zero rotation delta does not hold orientation, and the OSC settles ~1 mm off its
goal). Observation layout: ``fvb.policy.spec.ARM``.

``ArmWorld`` owns the (slow to build) robosuite env and is reused across episodes; the hole is
moved by editing the model's ``body_pos``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from fvb.control.osc_scripts import ArmRig, identify_load, make_controller_config, make_env
from fvb.ft.frames import site_to_world
from fvb.policy.spec import ARM
from fvb.policy.task import decode_action, encode_action

CONTROL_FREQ = 20.0
DT = 1.0 / CONTROL_FREQ
ACT_STEP_MAX = 0.005  # m per control step


@dataclass
class ArmTaskParams:
    clearance: float = 0.0005  # m per side
    offset_sigma: float = 0.0015  # m, per axis
    offset_max: float = 0.004
    kp: float = 150.0  # OSC kp (effective stiffness ~ Lambda * kp, M5)
    speed: float = 0.05  # m/s expert descent in free space (2.5 mm per step)
    contact_speed: float = 0.01  # m/s once the tip is within slow_zone of the rim
    slow_zone: float = 0.005  # m above the (known, offset-independent) rim height
    jam_force_N: float = 3.0
    retract: float = 0.003
    correct_step: float = 0.0005
    start_height_above_rim: float = 0.02
    success_depth: float = 0.025
    force_abort_N: float = 60.0
    max_steps: int = 200  # 10 s
    settle_steps_max: int = 150
    settle_tol: float = 0.0002
    xy_int_gain: float = 0.3
    # anti-windup: stop integrating the xy tip error while in contact. Without it the integrator
    # winds up against rim friction and pushes ~8 N sideways (along x, the arm's reach), and via
    # the 100 mm peg lever that force swamps the torque that says where the hole is.
    freeze_int_in_contact: bool = True
    contact_threshold_N: float = 1.0
    # reference point of the observed wrench: "tip" (tool centre point) or "flange" (sensor)
    wrench_ref: str = "tip"
    correct_at: str = "after_retract"  # or "jam" (see fvb.policy.task.TaskParams)
    action_mode: str = "delta"  # or "xy_abs" (see fvb.policy.task.TaskParams)
    xy_int_max: float = 0.005
    rot_gain: float = 1.0


class ArmWorld:
    """One PegInHole env + ArmRig with an identified load; reused across episodes."""

    def __init__(self, p: ArmTaskParams, seed: int = 0, offscreen: bool = False):
        self.p = p
        self.env = make_env(
            "PegInHole",
            controller_config=make_controller_config(kp=p.kp),
            offscreen=offscreen,
            seed=seed,
            clearance=p.clearance,
        )
        self.rig = ArmRig(self.env)
        self.rig.reset(seed)
        load, self.load_diag = identify_load(self.rig)
        ref = self.rig.model_load()
        # identification is exact in sim (M4/M6); fall back to the model if it ever is not
        self.rig.load = load if abs(load.mass - ref.mass) / ref.mass < 0.2 else ref
        m = self.rig.m
        self.hole_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hole")
        assert self.hole_body >= 0, "PegInHole has no 'hole' body"
        self.hole_pos0 = np.array(m.body_pos[self.hole_body])
        self.hole_xy0 = np.array(self.env.hole_xy)

    def place_hole(self, offset_xy: np.ndarray) -> None:
        self.rig.m.body_pos[self.hole_body] = self.hole_pos0 + np.array([*offset_xy, 0.0])
        self.env.hole_xy = self.hole_xy0 + np.asarray(offset_xy)
        mujoco.mj_forward(self.rig.m, self.rig.d)

    def close(self) -> None:
        self.env.close()


class ArmTask:
    """One episode. Same interface as GantryTask: observe / expert_action / step / depth /
    force_norm / hole_xy / k."""

    def __init__(
        self,
        p: ArmTaskParams,
        seed: int,
        world: ArmWorld,
        hole_offset: np.ndarray | None = None,
    ):
        self.p, self.world = p, world
        self.env, self.rig = world.env, world.rig
        rng = np.random.default_rng(seed)
        off = np.clip(rng.normal(0, p.offset_sigma, 2), -p.offset_max, p.offset_max)
        if hole_offset is not None:
            off = np.asarray(hole_offset, dtype=float)
        self.hole_xy = off  # the hidden variable (m, relative to the nominal centre)
        self.rig.reset(seed)
        world.place_hole(off)
        self._bias = np.zeros(3)
        # start: tip above the NOMINAL hole centre (not the displaced one)
        nominal_top = world.hole_pos0 + np.array([0, 0, self.env.hole_depth / 2])
        self.target = nominal_top + np.array([0, 0, p.start_height_above_rim])
        self.settle_steps = 0
        for _ in range(p.settle_steps_max):
            self.rig.step(self.low_level(self.target))
            self.settle_steps += 1
            if (
                np.linalg.norm(self.env.peg_tip_pos() - self.target) < p.settle_tol
                and np.linalg.norm(self.rig.ee_vel()[:3]) < 2e-3
            ):
                break
        self.pos0 = self.env.peg_tip_pos().copy()
        self.target_xy0 = self.target[:2].copy()
        self.rim_z = float(nominal_top[2])  # same for every offset: not privileged
        self.prev_action = np.zeros(3)
        self.k = 0
        self._phase = "descend"
        self._retract_z: float | None = None
        self._last_depth = -1.0

    # -- low-level controller ---------------------------------------------------------------
    def _tilt_rotvec(self) -> np.ndarray:
        a = self.env.peg_axis_world()
        d = np.array([0.0, 0.0, -1.0])
        r = np.cross(a, d)
        ang = float(np.arctan2(np.linalg.norm(r), np.dot(a, d)))
        return r / max(np.linalg.norm(r), 1e-9) * ang

    def low_level(self, tip_target: np.ndarray) -> np.ndarray:
        """Tip target (world) -> 6-D OSC_POSE action: xy integral on the tip error, peg axis
        servoed to vertical."""
        p = self.p
        tip = self.env.peg_tip_pos()
        err = np.asarray(tip_target) - tip
        if not (p.freeze_int_in_contact and self.force_norm() > p.contact_threshold_N):
            self._bias[:2] = np.clip(
                self._bias[:2] + p.xy_int_gain * err[:2], -p.xy_int_max, p.xy_int_max
            )
        site_target = np.asarray(tip_target) + self._bias + (self.rig.ee_pos() - tip)
        rot = np.clip(p.rot_gain * self._tilt_rotvec() / 0.5, -1, 1)
        return self.rig.action_towards(site_target, rot=rot)

    # -- observation --------------------------------------------------------------------------
    def depth(self) -> float:
        return float(self.env.insertion_depth())

    def force_norm(self) -> float:
        return float(np.linalg.norm(self.rig.ft_comp_last()[:3]))

    def observe(self) -> np.ndarray:
        hf = np.array(self.rig._hf_ft) if self.rig._hf_ft else np.zeros((1, 6))
        fmag = np.linalg.norm(hf[:, :3], axis=1)
        tmag = np.linalg.norm(hf[:, 3:], axis=1)
        comp_w = site_to_world(self.rig.ft_comp_last(), self.rig.ee_rot())
        if self.p.wrench_ref == "tip":
            # torque about the peg tip: T_tip = T_site - (tip - site) x F  (removes the lever arm)
            r = self.env.peg_tip_pos() - self.rig.ee_pos()
            comp_w[3:] = comp_w[3:] - np.cross(r, comp_w[:3])
        obs = np.concatenate(
            [
                self.env.peg_tip_pos() - self.pos0,
                self.rig.ee_vel()[:3],
                self.env.peg_axis_world()[:2],
                comp_w,
                [fmag.max(), tmag.max(), fmag.std()],
                self.prev_action,
            ]
        ).astype(np.float32)
        assert obs.shape == (ARM.obs_dim,)
        return obs

    # -- stepping ------------------------------------------------------------------------------
    def encode(self, delta: np.ndarray) -> np.ndarray:
        return encode_action(self.p.action_mode, delta, self.target, self.target_xy0)

    def step(self, action: np.ndarray, log=None) -> tuple[bool, str | None]:
        delta = decode_action(self.p.action_mode, action, self.target, self.target_xy0)
        delta = np.clip(delta, -ACT_STEP_MAX, ACT_STEP_MAX)
        self.target = self.target + delta
        self.rig.step(self.low_level(self.target), log)
        self.prev_action = delta.astype(np.float32)
        self.k += 1
        if self.depth() >= self.p.success_depth:
            return True, "success"
        if self.force_norm() > self.p.force_abort_N:
            return True, "force_abort"
        if self.k >= self.p.max_steps:
            return True, "timeout"
        return False, None

    # -- privileged expert ---------------------------------------------------------------------
    def expert_action(self) -> np.ndarray:
        p = self.p
        tip = self.env.peg_tip_pos()
        depth = self.depth()
        jammed = self.force_norm() > p.jam_force_N and depth < self._last_depth + 2e-4
        self._last_depth = depth
        a = np.zeros(3)
        if self._phase == "descend":
            if jammed:
                self._phase = "retract"
                self._retract_z = tip[2] + p.retract
                if p.correct_at == "jam":
                    err = self.env.hole_center_world[:2] - tip[:2]  # privileged
                    a[:2] = np.clip(err, -p.correct_step, p.correct_step)
            else:
                near_rim = tip[2] - self.rim_z < p.slow_zone
                a[2] = -(p.contact_speed if near_rim else p.speed) * DT
                return a
        if self._phase == "retract":
            if tip[2] < self._retract_z - 5e-4:
                # retract the *target* to just above the tip: a wound-up spring must unload
                a[2] = min(p.speed * DT, max(self._retract_z - self.target[2], 0.0))
                return a
            self._phase = "correct"
        if self._phase == "correct" and p.correct_at == "jam":
            self._phase = "descend"
            self._last_depth = -1.0
            near_rim = tip[2] - self.rim_z < p.slow_zone
            a[2] = -(p.contact_speed if near_rim else p.speed) * DT
            return a
        if self._phase == "correct":
            err = self.env.hole_center_world[:2] - tip[:2]  # privileged: the true hole
            a[:2] = np.clip(err, -p.correct_step, p.correct_step)
            self._phase = "descend"
            self._last_depth = -1.0
            return a
        return a
