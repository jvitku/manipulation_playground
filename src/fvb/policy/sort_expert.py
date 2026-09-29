"""Scripted expert for SortBoltsNuts (PLAN §13, V4): the synthetic stand-in for human demos.

The expert reads privileged state (part poses, hole and bucket positions) and tracks grip-site
pose targets with robosuite's OSC delta actions (world frame, 0.05 m / 0.5 rad per unit action).
Per part, nearest first:

* nut  : top-down grasp across two flats (or across the faces when it lies on its edge), lift,
         carry above bucket A, release;
* bolt : top-down grasp across the shank just below the head, lift, rotate the hand so the bolt
         stands head-up (the hand then points horizontally, away from the robot), servo the real
         shank tip over the hole estimate, lower it ~20 mm into the hole, release and back off: the
         bolt drops the rest of the way and seats. If the tip stalls on the fixture's top face the
         expert runs an outward spiral search around its hole estimate.

A grasp is checked by lifting: if the part does not follow the hand it re-grasps (at most 3
attempts per part, then the part is skipped).

"Human-like" noise (``ExpertNoise``) makes the demos imperfect the way a teleoperator is: a hole
estimate off by ~1.5 mm, grasp-point offsets, per-episode speed, action jitter and short pauses.
Demos made with it are labelled synthetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from fvb.envs.fasteners import DEFAULT_BOLT
from fvb.envs.sort_bolts_nuts import BIN_SIZE, BIN_WALL_H, BUCKET_H, BUCKET_R

POS_SCALE, ROT_SCALE = 0.05, 0.5  # OSC output_max: action 1.0 = 5 cm / 0.5 rad target offset
DOWN = np.array([0.0, 0.0, -1.0])
OPEN, CLOSE = -1.0, 1.0
# robosuite's PandaGripper integrates sign(action) in 0.2 steps into a level c in [-1, 1]; finger
# q = 0.02 (1 + c), pads meet at q ~ 2 mm. Pre-grasp levels leave ~10 mm per side around the part
# (a fully open hand is 80 mm wide and hits the bin walls next to a part).
PREGRASP_LEVEL = {"bolt": 0.0, "nut": 0.2}
FINGER_BELOW_PAD = 0.013  # m, the finger hull reaches this far below the pad centre
FLOOR_T = 0.005  # tray floor thickness (fasteners.bin_geoms default)


@dataclass
class ExpertNoise:
    hole_xy_std: float = 0.0015  # m, the operator's estimate of where the hole is
    grasp_std: float = 0.002  # m, grasp point offset along the part
    action_std: float = 0.05  # additive jitter on position/rotation actions
    speed_range: tuple = (0.7, 1.0)  # per-episode speed factor
    pause_prob: float = 0.01  # chance per step of a short pause
    pause_steps: tuple = (2, 8)


NO_NOISE = ExpertNoise(0.0, 0.0, 0.0, (1.0, 1.0), 0.0, (0, 0))


def frame(close_axis: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """Grip-site rotation with x = finger closing axis and z = approach direction."""
    z = approach / np.linalg.norm(approach)
    x = close_axis - z * (close_axis @ z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _horizontal(v: np.ndarray) -> np.ndarray:
    h = np.array([v[0], v[1], 0.0])
    n = np.linalg.norm(h)
    return h / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])


class SortExpert:
    """Call ``reset(env)`` after ``env.reset()``, then ``act()`` every control step."""

    def __init__(self, noise: ExpertNoise = NO_NOISE, seed: int = 0) -> None:
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    # -- env access ----------------------------------------
    def reset(self, env) -> None:
        self.env = env
        d = env.sim.data._data
        m = env.sim.model._model
        R = d.site_xmat[env.grip_site].reshape(3, 3)
        pads = [
            m.site(f"{env.robots[0].gripper['right'].naming_prefix}{s}_pad").id
            for s in ("left", "right")
        ]
        mid = d.site_xpos[pads].mean(axis=0)
        # pad centre in the grip-site frame (the point that should sit on the grasp point)
        self.pad_offset = R.T @ (mid - d.site_xpos[env.grip_site])
        self.speed = self.rng.uniform(*self.noise.speed_range)
        self.hole_est = env.hole_top.copy()
        self.hole_est[:2] += self.rng.normal(0, self.noise.hole_xy_std, 2)
        self.attempts = np.zeros(len(env.parts), int)
        self.grasp_noise = self.rng.normal(0, self.noise.grasp_std, len(env.parts))
        self.skipped: set[int] = set()
        self.part: int | None = None
        self.phase = "select"
        self.t_phase = 0
        self.pause = 0
        self.gripper = OPEN
        self.log: list[tuple[int, str, int | None]] = []
        self.step_i = 0

    def grip_pose(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.env.sim.data._data
        return d.site_xpos[self.env.grip_site].copy(), d.site_xmat[self.env.grip_site].reshape(
            3, 3
        ).copy()

    def pad_point(self) -> np.ndarray:
        p, R = self.grip_pose()
        return p + R @ self.pad_offset

    # -- control primitives ----------------------------------------
    def track(
        self, pad_target: np.ndarray, R_target: np.ndarray, gain: float = 1.0
    ) -> tuple[np.ndarray, float, float]:
        """Action moving the pad centre to ``pad_target`` and the hand to ``R_target``."""
        p, R = self.grip_pose()
        pad = p + R @ self.pad_offset
        # the pad point moves with the grip site for translation; rotation pivots about the
        # grip site, so aim the grip site at target - R_target @ offset
        grip_target = pad_target - R_target @ self.pad_offset
        e_p = grip_target - p
        e_r = Rotation.from_matrix(R_target @ R.T).as_rotvec()
        a = np.zeros(7)
        a[:3] = np.clip(gain * self.speed * e_p / POS_SCALE * 4.0, -1, 1)
        a[3:6] = np.clip(2.0 * e_r / ROT_SCALE, -1, 1)
        a[6] = self._gripper_action()
        return a, float(np.linalg.norm(pad_target - pad)), float(np.linalg.norm(e_r))

    def _gripper_action(self) -> float:
        """Drive robosuite's gripper level towards ``self.gripper`` (-1 open ... +1 closed)."""
        if self.gripper >= 1.0:
            return CLOSE
        level = -self.env.robots[0].gripper["right"].current_action[0]  # +1 = closed
        if abs(self.gripper - level) < 0.1:
            return 0.0
        return CLOSE if self.gripper > level else OPEN

    def _go(self, phase: str) -> None:
        self.phase = phase
        self.t_phase = 0
        self.log.append((self.step_i, phase, self.part))

    # -- task logic ----------------------------------------
    def _next_part(self) -> int | None:
        env = self.env
        pad = self.pad_point()
        best, best_d = None, np.inf
        for i, p in enumerate(env.parts):  # nearest pickable part
            if i in self.skipped or p.status in ("vanished", "in_bucket", "dropped", "seated"):
                continue
            # parts that left the tray (on the table, by the fixture) are not picked up again
            if not env._in_bin(i):
                continue
            # a nut still sliding inside the bucket is sorted, not a part to pick (V4 seeds 4, 7)
            rel = env.part_pose(i)[0] - env.bucket_floor
            if np.hypot(rel[0], rel[1]) < BUCKET_R and rel[2] < BUCKET_H:
                continue
            dist = np.linalg.norm(env.part_pose(i)[0] - pad)
            if dist < best_d:
                best, best_d = i, dist
        return best

    def _clearance(self, point: np.ndarray, R: np.ndarray, aperture: float) -> float:
        """Smallest distance from sample points on the open fingers and the palm to the tray
        walls (inf for points above the wall tops)."""
        env = self.env
        c, a = R[:, 0], R[:, 2]
        floor = env.bin_floor
        lx, ly = BIN_SIZE[0] / 2 - 0.005, BIN_SIZE[1] / 2 - 0.005
        pts = []
        for side in (-1, 1):
            for s in np.linspace(-0.01, 0.045, 6):  # finger, pad bottom up to the palm
                pts.append(point + side * c * (aperture / 2 + 0.012) - a * s)
            for s in (0.05, 0.08):  # palm corners, ~100 mm to each side along the closing axis
                pts.append(point + side * c * 0.105 - a * s)
        best = np.inf
        for q in pts:
            rel = q - floor
            if rel[2] > BIN_WALL_H + 0.005:
                continue
            best = min(best, lx - abs(rel[0]), ly - abs(rel[1]))
        return best

    def _grasp_plan(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(pad target, hand rotation) for a grasp of part ``i`` as it lies now.

        Candidates are top-down grasps (nuts: across each pair of flats; bolts: across the shank
        just below the head) and, for bolts, grasps tilted about the shank axis by up to 50 deg;
        the first candidate whose fingers and palm clear the tray walls by 10 mm wins, otherwise
        the one with the most clearance.
        """
        env = self.env
        pos, Rp = env.part_pose(i)
        axis = Rp[:, 2]
        _, R_now = self.grip_pose()
        kind = env.parts[i].kind
        aperture = 2 * (0.02 * (1 + PREGRASP_LEVEL[kind]) - 0.002)
        cands = []  # (point, R)
        if kind == "bolt":
            u = _horizontal(axis)
            # by the hex head (pads on two flats): with the hand horizontal over the hole the
            # palm spans +-32 mm around the grasp point, so a shank grasp put it on the fixture
            point = pos + axis * (0.005 + self.grasp_noise[i]) + [0, 0, 0.004]
            c0 = self._yaw_sign(np.cross([0, 0, 1.0], u), R_now)
            for deg in (0, 25, -25, 50, -50):
                rot = Rotation.from_rotvec(u * math.radians(deg)).as_matrix()
                c = rot @ c0
                cands.append((point, frame(c, rot @ DOWN)))
        else:
            point = pos + [0, 0, 0.004]
            if abs(axis[2]) > 0.7:  # lying flat: across two flats (flats at 30 + 60k deg)
                dirs = [
                    Rp @ [math.cos(t), math.sin(t), 0] for t in np.deg2rad(30 + 60 * np.arange(3))
                ]
                dirs.sort(key=lambda c: -abs(_horizontal(c) @ R_now[:, 0]))  # least yaw first
            else:  # on its edge: pads on the two faces
                dirs = [axis]
            for c in dirs:
                cands.append((point, frame(self._yaw_sign(_horizontal(c), R_now), DOWN)))
        # never drive the fingertips into the tray floor
        min_z = env.bin_floor[2] + FLOOR_T + FINGER_BELOW_PAD + 0.002
        cands = [(np.r_[p[:2], max(p[2], min_z)], R) for p, R in cands]
        scored = [(self._clearance(p, R, aperture), p, R) for p, R in cands]
        for clear, p, R in scored:
            if clear > 0.010:
                return p, R
        _, p, R = max(scored, key=lambda t: t[0])
        return p, R

    def _yaw_sign(self, c: np.ndarray, R_now: np.ndarray) -> np.ndarray:
        """Pick c or -c (the grasp is symmetric) so joint 7 stays away from its +-2.9 rad limit.
        For a downward hand a world yaw of +d turns joint 7 by about -d."""
        q7 = float(self.env.robots[0]._joint_positions[6])
        best, best_q = c, np.inf
        for cc in (c, -c):
            x = R_now[:, 0]
            d = math.atan2(x[0] * cc[1] - x[1] * cc[0], x @ cc)
            q_new = abs(q7 - d)
            if q_new < best_q:
                best, best_q = cc, q_new
        return best

    def _held(self, i: int) -> bool:
        return np.linalg.norm(self.env.part_pose(i)[0] - self.pad_point()) < 0.045

    def act(self) -> np.ndarray:
        self.step_i += 1
        self.t_phase += 1
        a = self._act()
        if self.pause > 0:
            self.pause -= 1
            a[:6] = 0
        elif self.rng.random() < self.noise.pause_prob:
            self.pause = (
                int(self.rng.integers(*self.noise.pause_steps)) if self.noise.pause_steps[1] else 0
            )
        if self.noise.action_std:
            a[:6] = np.clip(a[:6] + self.rng.normal(0, self.noise.action_std, 6), -1, 1)
        return a

    def _act(self) -> np.ndarray:
        env = self.env
        safe_z = env.table_offset[2] + BIN_WALL_H + 0.12
        p, R = self.grip_pose()
        i = self.part

        if self.phase == "select":
            self.part = self._next_part()
            self.gripper = OPEN
            if self.part is None:
                self._go("done")
            else:
                self._go("above")
            return self.track(self.pad_point(), R)[0]

        if self.phase == "done":
            self.gripper = OPEN
            return self.track(self.pad_point(), R)[0]

        kind = env.parts[i].kind
        if self.phase == "above":
            self.gripper = -PREGRASP_LEVEL[kind]
            target, Rg = self._grasp_plan(i)
            a, ep, er = self.track(np.r_[target[:2], max(safe_z, self.pad_point()[2])], Rg)
            if ep < 0.01 and er < 0.05 or self.t_phase > 120:
                self._go("descend")
            return a
        if self.phase == "descend":
            if self.t_phase == 1:
                self.plan = self._grasp_plan(i)
            target, Rg = self.plan
            a, ep, er = self.track(target, Rg, gain=0.7)
            # moderate speed down into the tray, the last 3 cm slowly (~2 cm/s): steel on steel
            # at 0.2 m/s is > 80 N before the touch check can react
            cap = 0.08 if ep < 0.03 else 0.4
            a[:3] = np.clip(a[:3], -cap, cap)
            # touched something (a leaning part, the floor): stop pushing and grasp here
            touched = self.t_phase > 3 and np.linalg.norm(env.f_ext_hat()[:3]) > 15.0
            if ep < 0.004 or touched or self.t_phase > 80:
                self._go("close")
            return a
        if self.phase == "close":
            self.gripper = CLOSE
            a, _, _ = self.track(self.pad_point(), R)
            if self.t_phase > 12:
                self._go("lift")
            return a
        if self.phase == "lift":
            a, ep, _ = self.track(np.r_[self.pad_point()[:2], safe_z], R)
            if self.t_phase > 10 and not self._held(i):
                return self._fail_grasp()
            if ep < 0.01 or self.t_phase > 80:
                self._go("carry" if kind == "nut" else "reorient")
            return a

        if kind == "nut":
            bucket = env.bucket_floor + [0, 0, BUCKET_H + 0.06]
            if self.phase == "carry":
                a, ep, _ = self.track(bucket, frame(R[:, 0], DOWN))
                if not self._held(i):
                    return self._fail_grasp()
                if ep < 0.02 or self.t_phase > 150:
                    self._go("release")
                return a
            if self.phase == "release":
                self.gripper = OPEN
                a, _, _ = self.track(bucket, R)
                if self.t_phase > 12:
                    self._go("select")
                return a

        # bolt
        pos, Rp = env.part_pose(i)
        u_world = Rp[:, 2]
        tip = pos - u_world * DEFAULT_BOLT.length
        if self.phase == "reorient":
            # bolt head-up, hand horizontal; of the horizontal approach directions (away from the
            # robot +-90 deg) take the one that needs the least rotation (large wrist rotations
            # ran joints 4/5 into their limits)
            u_hand = R.T @ u_world
            H = np.column_stack([u_hand, [0, 0, 1.0], np.cross(u_hand, [0, 0, 1.0])])
            best = None
            for deg in (0, 45, -45):  # +-90 put the elbow over bucket A
                t = math.radians(deg)
                approach = np.array([math.cos(t), math.sin(t), 0.0])
                W = np.column_stack([[0, 0, 1.0], approach, np.cross([0, 0, 1.0], approach)])
                U, _, Vt = np.linalg.svd(W @ np.linalg.inv(H))  # re-orthonormalise
                Rc = U @ Vt
                ang = np.linalg.norm(Rotation.from_matrix(Rc @ R.T).as_rotvec())
                if best is None or ang < best[0]:
                    best = (ang, Rc)
            self.R_insert = best[1]
            self.i_err = np.zeros(2)
            self._go("rotate")
        if self.phase == "rotate":
            # slerp the target so the OSC is never asked for a huge rotation at once
            rel = Rotation.from_matrix(self.R_insert @ R.T).as_rotvec()
            ang = np.linalg.norm(rel)
            step = Rotation.from_rotvec(rel * min(1.0, 0.3 / max(ang, 1e-9))).as_matrix() @ R
            a, _, _ = self.track(np.r_[self.pad_point()[:2], safe_z + 0.02], step)
            if not self._held(i):
                return self._fail_grasp()
            if ang < 0.05 or self.t_phase > 150:
                self._go("to_hole")
            return a
        if self.phase in ("to_hole", "hover"):
            if not self._held(i):
                return self._fail_grasp()
            goal_xy = self.hole_est[:2]
            top = self.hole_est[2]
            lat_err = goal_xy - tip[:2]
            lat = float(np.linalg.norm(lat_err))
            if self.phase == "to_hole":
                # travel high (tip 10 cm above the fixture): a low transit hit bucket A and the
                # fixture with the arm
                tip_goal = np.r_[goal_xy, top + 0.10]
                pad_goal = self.pad_point() + (tip_goal - tip)
                a, _, _ = self.track(pad_goal, self.R_insert)
                if lat < 0.01 and abs(tip[2] - tip_goal[2]) < 0.01 or self.t_phase > 200:
                    self.i_err = np.zeros(2)
                    self.ok_steps = 0
                    self._go("hover")
                return a
            # hover: settle the upright bolt 4 mm above the hole, then drop it in. Pushing it in
            # against the chamfer made the bolt pivot in the grasp (tilt 3 -> 54 deg) and the
            # arm dragged along the fixture. Low gain + an integral that only acts near the goal
            # (it wound up to 10 mm during the transit and overshot +-5 mm).
            if lat < 0.003:
                self.i_err = np.clip(self.i_err + 0.1 * lat_err, -0.006, 0.006)
            z_goal = max(top + 0.004, tip[2] - 0.005) if lat < 0.003 else tip[2]
            tip_goal = np.r_[goal_xy, z_goal]
            pad_goal = self.pad_point() + (tip_goal - tip) + np.r_[self.i_err, 0.0]
            a, _, _ = self.track(pad_goal, self.R_insert, gain=0.3)
            tilt = math.degrees(math.acos(np.clip(u_world[2], -1, 1)))
            h = tip[2] - top
            good = lat < 0.0007 and 0.002 < h < 0.007 and tilt < 2.0
            self.ok_steps = self.ok_steps + 1 if good else 0
            timed_out = self.t_phase > 200 and lat < 0.0015 and h < 0.01
            if self.ok_steps >= 3 or timed_out:
                self._go("let_go")
            elif self.t_phase > 300:
                return self._fail_grasp()
            return a
        if self.phase == "let_go":
            self.gripper = OPEN
            a, _, _ = self.track(self.pad_point(), R)
            level = -self.env.robots[0].gripper["right"].current_action[0]
            if level < -0.9 and self.t_phase > 4:  # fully open, or backing off drags the bolt out
                self._go("up")
            return a
        if self.phase == "up":
            # straight up first (the open fingers clear the 24 mm head), then turn the hand
            # back to pointing down at safe height
            if self.t_phase <= 25:
                a, _, _ = self.track(self.pad_point() + [0, 0, 0.03], R)
                return a
            rel = Rotation.from_matrix(frame(R[:, 0], DOWN) @ R.T).as_rotvec()
            ang = np.linalg.norm(rel)
            step = Rotation.from_rotvec(rel * min(1.0, 0.3 / max(ang, 1e-9))).as_matrix() @ R
            a, _, _ = self.track(np.r_[self.pad_point()[:2], safe_z + 0.03], step)
            if ang < 0.1 or self.t_phase > 100:
                self._go("select")
            return a
        raise RuntimeError(self.phase)

    def _fail_grasp(self) -> np.ndarray:
        i = self.part
        self.attempts[i] += 1
        if self.attempts[i] >= 3:
            self.skipped.add(i)
        self.gripper = OPEN
        self._go("regrasp_up")
        self.phase = "select"
        _, R = self.grip_pose()
        a, _, _ = self.track(self.pad_point() + [0, 0, 0.02], R)
        return a
