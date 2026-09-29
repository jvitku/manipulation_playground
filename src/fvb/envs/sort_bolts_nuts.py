"""SortBoltsNuts: sort M16 bolts into a clearance hole and M16 nuts into bucket A (PLAN §13, V3).

Scene (table frame, the robot base sits at x ~ -0.56): a source bin with ``n_bolts`` ISO 4017
bolts and ``n_nuts`` ISO 4032 nuts (``fvb.envs.fasteners``), a steel fixture with a 17 mm ISO 273
fine hole, and bucket A. The robot is a Panda with the tactile Franka Hand
(``fvb.sensors.tactile``, 70 N) and joint-torque sensing (``fvb.sensors.joint_torque``).

Physics runs at 1 kHz (implicitfast, elliptic cones; the stiff steel contacts need the 1 ms step),
control at 20 Hz. Sensors are sampled every physics step into 2 s ring buffers.

Success rules (geometric, so they cannot be gamed by lowering a part next to the target):

* a bolt is *seated* when its shank tip is >= ``seat_depth`` below the fixture's top face, both its
  tip and its head are laterally inside the hole radius, its axis is < ``max_tilt_deg`` from
  vertical (head up) and it touches no gripper geom; after ``vanish_after`` s seated it *vanishes*
  (its free joint is parked at rest on the floor 5 m away; MuJoCo cannot delete bodies);
* a nut is *in the bucket* while it lies inside the bucket wall, below the rim, slower than
  ``rest_speed``;
* the episode succeeds when every bolt has vanished and every nut is in the bucket.

``info`` from ``step`` carries per-part statuses and the episode outcome (success, force_abort,
timeout or running) and failure counts (dropped, jammed).

Registered with robosuite on import: ``suite.make("SortBoltsNuts", robots="Panda", ...)``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco
import numpy as np
from robosuite.environments.base import register_env
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.observables import Observable, sensor

import fvb.sensors.tactile  # noqa: F401  (registers TactilePandaGripper)
from fvb.envs.fasteners import (
    DEFAULT_BOLT,
    DEFAULT_HOLE,
    STEEL_SOLIMP,
    STEEL_SOLREF,
    bin_geoms,
    bolt_assets,
    bolt_body,
    bucket_geoms,
    hole_assets,
    hole_geoms,
    nut_assets,
    nut_body,
)
from fvb.sensors.joint_torque import JointTorqueSensor
from fvb.sensors.tactile import TactilePads

INSTRUCTION = "sort the parts: bolts into the hole, nuts into bucket A"
SUB_INSTRUCTIONS = {"bolt": "put the bolt in the hole", "nut": "put the nut in bucket A"}

BIN_SIZE = (0.30, 0.22)
BIN_WALL_H = 0.08
BUCKET_R, BUCKET_H = 0.10, 0.12

SORTED = ("vanished", "in_bucket")
# front camera framing the bin, the fixture and bucket A (robosuite's agentview crops both ends)
SORTVIEW_POS = (0.95, 0.08, 1.55)
SORTVIEW_TARGET = (-0.02, 0.08, 0.82)


def look_at_xyaxes(pos, target) -> str:
    """MuJoCo camera ``xyaxes`` for a camera at ``pos`` looking at ``target`` (world z up)."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return array_to_string(np.r_[x, y])


@dataclass
class SortTaskParams:
    n_bolts: int = 3
    n_nuts: int = 3
    seat_depth: float = 0.025  # m, shank tip below the fixture's top face
    max_tilt_deg: float = 5.0
    vanish_after: float = 1.0  # s seated and released
    rest_speed: float = 0.02  # m/s, a nut in the bucket must be at rest
    force_abort_N: float = 80.0  # |F_ext_hat|
    settle_s: float = 1.0  # spawn settling time
    bin_xy: tuple = (0.0, -0.22)  # table frame
    hole_xy: tuple = (0.05, 0.15)
    bucket_xy: tuple = (-0.12, 0.38)


@dataclass
class PartState:
    kind: str  # "bolt" | "nut"
    body: str
    status: str = "in_bin"  # in_bin, moving, grasped, seated, vanished, in_bucket, dropped, jammed
    seated_since: float | None = None
    history: list = field(default_factory=list)


class RingBuffer:
    def __init__(self, n: int, shape: tuple) -> None:
        self.buf = np.zeros((n, *shape))
        self.i = 0
        self.full = False

    def push(self, x) -> None:
        self.buf[self.i] = x
        self.i = (self.i + 1) % len(self.buf)
        self.full = self.full or self.i == 0

    def last(self, k: int) -> np.ndarray:
        """The last ``k`` samples, oldest first (zeros before the buffer has filled)."""
        idx = (self.i - k + np.arange(k)) % len(self.buf)
        return self.buf[idx]

    def clear(self) -> None:
        self.buf[:] = 0
        self.i = 0
        self.full = False


class SortBoltsNuts(ManipulationEnv):
    """Panda + tactile Franka Hand sorting M16 bolts and nuts (see module docstring)."""

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="TactilePandaGripper",
        initialization_noise="default",
        table_full_size=(0.8, 1.1, 0.05),
        table_friction=(0.5, 0.005, 0.0001),
        task: SortTaskParams | None = None,
        joint_torque: bool = True,
        tactile: bool = True,
        torque_noise_std: float = 0.05,
        torque_bias_std: float = 0.1,
        torque_hist_frames: int = 10,
        torque_hist_s: float = 2.0,
        use_camera_obs=False,
        use_object_obs=True,
        camera_names=("sortview", "robot0_eye_in_hand"),
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
        horizon=3600,
        reward_scale=1.0,
        physics_dt: float = 0.001,
        **kwargs,
    ):
        self.task = task or SortTaskParams()
        self.table_full_size = tuple(table_full_size)
        self.table_friction = tuple(table_friction)
        self.table_offset = np.array((0, 0, 0.8))
        self.use_joint_torque = joint_torque
        self.use_tactile = tactile and gripper_types == "TactilePandaGripper"
        self.torque_noise_std = torque_noise_std
        self.torque_bias_std = torque_bias_std
        self.torque_hist_frames = torque_hist_frames
        self.torque_hist_s = torque_hist_s
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.physics_dt = physics_dt
        self.parts: list[PartState] = []
        self.outcome = "running"
        self.instruction = INSTRUCTION
        n_buf = int(round(torque_hist_s / physics_dt))
        self.tau_ext_buf = RingBuffer(n_buf, (7,))
        self.tau_j_buf = RingBuffer(n_buf, (7,))
        self.taxel_buf = RingBuffer(n_buf, (2, 4, 4))
        self.pad_force_buf = RingBuffer(n_buf, (2, 3))
        self._n_since_obs = 0
        self._last_interval = 1
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            camera_names=list(camera_names),
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            control_freq=control_freq,
            horizon=horizon,
            **kwargs,
        )

    # -- time ----------------------------------------
    def initialize_time(self, control_freq):
        super().initialize_time(control_freq)
        self.model_timestep = self.physics_dt
        # robosuite runs int(control_timestep / model_timestep) substeps; make it exact
        n = int(round(self.control_timestep / self.model_timestep))
        self.control_timestep = n * self.model_timestep + 1e-12

    # -- geometry helpers ----------------------------------------
    def _table_to_world(self, xy, z=0.0) -> np.ndarray:
        return np.array([xy[0], xy[1], self.table_offset[2] + z])

    @property
    def hole_top(self) -> np.ndarray:
        return self._table_to_world(self.task.hole_xy, DEFAULT_HOLE.height)

    @property
    def bucket_floor(self) -> np.ndarray:
        return self._table_to_world(self.task.bucket_xy)

    @property
    def bin_floor(self) -> np.ndarray:
        return self._table_to_world(self.task.bin_xy)

    # -- model ----------------------------------------
    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        ET.SubElement(
            arena.worldbody,
            "camera",
            name="sortview",
            mode="fixed",
            pos=array_to_string(SORTVIEW_POS),
            xyaxes=look_at_xyaxes(SORTVIEW_POS, SORTVIEW_TARGET),
            fovy="50",
        )
        static = []
        b = ET.SubElement(arena.worldbody, "body", name="bin", pos=array_to_string(self.bin_floor))
        bin_geoms(b, "bin", size=BIN_SIZE, wall_h=BIN_WALL_H)
        static.append(b)
        k = ET.SubElement(
            arena.worldbody, "body", name="bucket", pos=array_to_string(self.bucket_floor)
        )
        bucket_geoms(k, "bucket", radius=BUCKET_R, height=BUCKET_H)
        static.append(k)
        h = ET.SubElement(
            arena.worldbody,
            "body",
            name="fixture",
            pos=array_to_string(self._table_to_world(self.task.hole_xy)),
        )
        hole_geoms(h, "fixture")
        static.append(h)
        # every surface a steel part lands on is stiff too (contact parameters mix, V1)
        for body in static:
            for g in body.iter("geom"):
                g.set("solref", STEEL_SOLREF)
                g.set("solimp", STEEL_SOLIMP)
        for g in arena.worldbody.iter("geom"):
            if g.get("name") == "table_collision":
                g.set("solref", STEEL_SOLREF)
                g.set("solimp", STEEL_SOLIMP)

        assets = arena.root.find("asset")
        for e in hole_assets("fixture"):
            assets.append(e)
        self.parts = []
        for i in range(self.task.n_bolts):
            name = f"bolt{i}"
            for e in bolt_assets(name):
                assets.append(e)
            arena.worldbody.append(bolt_body(name, pos=(1.5 + 0.1 * i, 1.5, 0.1)))
            self.parts.append(PartState("bolt", name))
        for i in range(self.task.n_nuts):
            name = f"nut{i}"
            for e in nut_assets(name):
                assets.append(e)
            arena.worldbody.append(nut_body(name, pos=(1.5 + 0.1 * i, 1.7, 0.1)))
            self.parts.append(PartState("nut", name))

        # robosuite's renderer hides geom group 0 (its collision group) and its compiler takes
        # inertia from group 0 only. Static geoms simply move to group 1 (contacts depend on
        # contype, not group); every part geom gets a visual twin in group 1 that does not collide.
        for body in static:
            for g in body.iter("geom"):
                g.set("group", "1")
        for p in self.parts:
            body = arena.worldbody.find(f"body[@name='{p.body}']")
            for g in list(body.findall("geom")):
                v = ET.SubElement(body, "geom", dict(g.attrib))
                v.set("name", g.get("name") + "_vis")
                v.set("group", "1")
                v.set("contype", "0")
                v.set("conaffinity", "0")

        self.model = ManipulationTask(
            mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots]
        )
        opt = self.model.root.find("option")  # robosuite's base.xml carries the <option>
        opt.set("timestep", f"{self.physics_dt:g}")
        opt.set("integrator", "implicitfast")
        opt.set("cone", "elliptic")

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model._model
        self._m = m
        self.part_body = [m.body(p.body).id for p in self.parts]
        self.part_qadr = [m.jnt_qposadr[m.joint(f"{p.body}_joint").id] for p in self.parts]
        self.part_dadr = [m.jnt_dofadr[m.joint(f"{p.body}_joint").id] for p in self.parts]
        self.part_geoms = [
            {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid} for bid in self.part_body
        ]
        grip = self.robots[0].gripper["right"]
        self.gripper_geoms = {m.geom(n).id for n in grip.contact_geoms}
        self.grip_site = self.robots[0].eef_site_id["right"]
        self.jts = JointTorqueSensor(
            m,
            joints=self.robots[0].robot_joints,
            noise_std=self.torque_noise_std,
            bias_std=self.torque_bias_std,
            seed=int(self.rng.integers(2**31)) if hasattr(self, "rng") else None,
        )
        self.pads = TactilePads(m, prefix=grip.naming_prefix) if self.use_tactile else None

    # -- spawning ----------------------------------------
    def _sample_spawn(self, rng) -> list[tuple[np.ndarray, np.ndarray]]:
        """Rejection-sample non-overlapping poses above the bin (bounding spheres)."""
        radius = {"bolt": 0.032, "nut": 0.017}
        lx, ly = BIN_SIZE[0] / 2 - 0.035, BIN_SIZE[1] / 2 - 0.035
        c = self.bin_floor
        poses: list[tuple[np.ndarray, np.ndarray]] = []
        for p in self.parts:
            for _ in range(1000):
                pos = c + [rng.uniform(-lx, lx), rng.uniform(-ly, ly), rng.uniform(0.04, 0.16)]
                ok = all(
                    np.linalg.norm(pos - q[0]) > radius[p.kind] + radius[o.kind] + 0.004
                    for q, o in zip(poses, self.parts, strict=False)
                )
                if ok:
                    break
            else:
                raise RuntimeError("could not place parts in the bin")
            quat = rng.normal(size=4)
            poses.append((pos, quat / np.linalg.norm(quat)))
        return poses

    def _hold_robot_step(self, qpos_robot: np.ndarray) -> None:
        d = self.sim.data._data
        d.qpos[self._robot_q] = qpos_robot
        d.qvel[self._robot_v] = 0
        mujoco.mj_step(self._m, d)

    def _settle(self, rng, attempts: int = 5) -> None:
        d = self.sim.data._data
        r = self.robots[0]
        self._robot_q = np.r_[r._ref_joint_pos_indexes, r._ref_gripper_joint_pos_indexes["right"]]
        self._robot_v = np.r_[r._ref_joint_vel_indexes, r._ref_gripper_joint_vel_indexes["right"]]
        q_robot = d.qpos[self._robot_q].copy()
        for _ in range(attempts):
            for (pos, quat), qa, da in zip(
                self._sample_spawn(rng), self.part_qadr, self.part_dadr, strict=True
            ):
                d.qpos[qa : qa + 3] = pos
                d.qpos[qa + 3 : qa + 7] = quat
                d.qvel[da : da + 6] = 0
            mujoco.mj_forward(self._m, d)
            for _ in range(int(round(self.task.settle_s / self.physics_dt))):
                self._hold_robot_step(q_robot)
            if all(self._in_bin(i) for i in range(len(self.parts))) and all(
                self._speed(i) < 0.05 for i in range(len(self.parts))
            ):
                break
        d.qpos[self._robot_q] = q_robot
        d.qvel[self._robot_v] = 0
        d.time = 0.0
        mujoco.mj_forward(self._m, d)

    # -- part state ----------------------------------------
    def part_pose(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        d = self.sim.data._data
        b = self.part_body[i]
        return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()

    def _speed(self, i: int) -> float:
        da = self.part_dadr[i]
        return float(np.linalg.norm(self.sim.data._data.qvel[da : da + 3]))

    def _in_bin(self, i: int) -> bool:
        pos, _ = self.part_pose(i)
        rel = pos - self.bin_floor
        return bool(
            abs(rel[0]) < BIN_SIZE[0] / 2 and abs(rel[1]) < BIN_SIZE[1] / 2 and rel[2] < BIN_WALL_H
        )

    def bolt_geometry(self, i: int) -> dict:
        """Tip depth below the fixture top, tilt from vertical, lateral offsets of tip and head."""
        pos, R = self.part_pose(i)
        axis = R[:, 2]  # head side up when upright; the shank points along -axis
        tip = pos - axis * DEFAULT_BOLT.length
        top = self.hole_top
        return {
            "depth": float(top[2] - tip[2]),
            "tilt_deg": float(math.degrees(math.acos(np.clip(axis[2], -1, 1)))),
            "tip_lat": float(np.linalg.norm(tip[:2] - top[:2])),
            "head_lat": float(np.linalg.norm(pos[:2] - top[:2])),
        }

    def _touching_gripper(self, i: int) -> bool:
        d = self.sim.data._data
        mine = self.part_geoms[i]
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in mine and g2 in self.gripper_geoms) or (
                g2 in mine and g1 in self.gripper_geoms
            ):
                return True
        return False

    def bolt_seated(self, i: int) -> bool:
        g = self.bolt_geometry(i)
        r = DEFAULT_HOLE.diameter / 2
        return (
            g["depth"] >= self.task.seat_depth
            and g["tilt_deg"] < self.task.max_tilt_deg
            and g["tip_lat"] < r
            and g["head_lat"] < r
            and not self._touching_gripper(i)
        )

    def nut_in_bucket(self, i: int) -> bool:
        pos, _ = self.part_pose(i)
        rel = pos - self.bucket_floor
        return bool(
            np.hypot(rel[0], rel[1]) < BUCKET_R - 0.004
            and 0.0 < rel[2] < BUCKET_H
            and self._speed(i) < self.task.rest_speed
        )

    def _park(self, i: int) -> None:
        d = self.sim.data._data
        qa, da = self.part_qadr[i], self.part_dadr[i]
        # lying on the floor plane (z = 0), far outside every camera, spaced apart
        d.qpos[qa : qa + 3] = [5.0 + 0.2 * i, 5.0, 0.012]
        d.qpos[qa + 3 : qa + 7] = [0.7071068, 0.7071068, 0, 0]
        d.qvel[da : da + 6] = 0

    def _update_parts(self) -> None:
        t = float(self.sim.data._data.time)
        table_z = self.table_offset[2]
        for i, p in enumerate(self.parts):
            if p.status == "vanished":
                continue
            pos, _ = self.part_pose(i)
            if pos[2] < table_z - 0.05:
                p.status = "dropped"
                continue
            if p.kind == "bolt":
                if self.bolt_seated(i):
                    if p.seated_since is None:
                        p.seated_since = t
                    p.status = "seated"
                    if t - p.seated_since >= self.task.vanish_after - 1e-9:
                        self._park(i)
                        p.status = "vanished"
                    continue
                p.seated_since = None
                g = self.bolt_geometry(i)
                if g["tip_lat"] < 1.5 * DEFAULT_HOLE.diameter and g["depth"] > 0:
                    p.status = "jammed" if self._speed(i) < 0.005 else "moving"
                    continue
            elif self.nut_in_bucket(i):
                p.status = "in_bucket"
                continue
            if self._touching_gripper(i):
                p.status = "grasped"
            elif self._in_bin(i):
                p.status = "in_bin"
            else:
                p.status = "moving"

    # -- sensors ----------------------------------------
    def _pre_action(self, action, policy_step=False):
        # data holds a fresh forward pass here (robosuite calls sim.forward() first)
        d = self.sim.data._data
        if self.use_joint_torque:
            tau_j, tau_ext = self.jts.read(d)
            self.tau_j_buf.push(tau_j)
            self.tau_ext_buf.push(tau_ext)
        if self.pads is not None:
            self.taxel_buf.push(self.pads.taxels(d))
            self.pad_force_buf.push(self.pads.pad_force(d))
        self._n_since_obs += 1
        super()._pre_action(action, policy_step)

    def tactile_summary(self, taxels: np.ndarray, pad_force: np.ndarray) -> np.ndarray:
        """(2, 4): normal force, CoP x, CoP y, shear per pad from raw arrays."""
        out = np.zeros((2, 4))
        x, y = self.pads.x, self.pads.y
        for k in range(2):
            n = taxels[k].sum()
            if n > 1e-6:
                out[k] = [
                    n,
                    (taxels[k].sum(axis=0) * x).sum() / n,
                    (taxels[k].sum(axis=1) * y).sum() / n,
                    np.hypot(pad_force[k, 0], pad_force[k, 1]),
                ]
        return out

    def torque_history(self) -> np.ndarray:
        """(frames, 15): tau_ext (7) + tactile summary (2 x 4) sampled evenly over the last 2 s."""
        n = len(self.tau_ext_buf.buf)
        k = self.torque_hist_frames
        idx = np.linspace(n / k - 1, n - 1, k).round().astype(int)
        tau = self.tau_ext_buf.last(n)[idx]
        if self.pads is None:
            tac = np.zeros((k, 8))
        else:
            tx = self.taxel_buf.last(n)[idx]
            pf = self.pad_force_buf.last(n)[idx]
            tac = np.stack(
                [self.tactile_summary(a, b).ravel() for a, b in zip(tx, pf, strict=True)]
            )
        return np.hstack([tau, tac])

    def f_ext_hat(self) -> np.ndarray:
        """Cartesian wrench estimate at the grip site from the latest measured tau_ext."""
        tau = self.tau_ext_buf.last(1)[0]
        site = self._m.site(self.grip_site).name
        return self.jts.wrench(self.sim.data._data, tau, site)

    # -- episode ----------------------------------------
    def _reset_internal(self):
        super()._reset_internal()
        for p in self.parts:
            p.status, p.seated_since = "in_bin", None
        self.outcome = "running"
        self._settle(self.rng)
        for b in (self.tau_ext_buf, self.tau_j_buf, self.taxel_buf, self.pad_force_buf):
            b.clear()
        self.jts.reset()
        self._n_since_obs = 0

    def n_sorted(self) -> int:
        return sum(p.status in SORTED for p in self.parts)

    def _check_success(self):
        return all(
            (p.kind == "bolt" and p.status == "vanished")
            or (p.kind == "nut" and p.status == "in_bucket")
            for p in self.parts
        )

    def reward(self, action=None):
        if not self.parts:
            return 0.0
        return float(self.reward_scale * self.n_sorted() / len(self.parts))

    def _post_action(self, action):
        self._update_parts()
        reward, done, info = super()._post_action(action)
        if self._check_success():
            self.outcome = "success"
        elif (
            self.use_joint_torque and np.linalg.norm(self.f_ext_hat()[:3]) > self.task.force_abort_N
        ):
            self.outcome = "force_abort"
        elif done:
            self.outcome = "timeout"
        if self.outcome != "running" and not self.ignore_done:
            done = True
        info = dict(info)
        info.update(
            outcome=self.outcome,
            parts={p.body: p.status for p in self.parts},
            n_sorted=self.n_sorted(),
            dropped=sum(p.status == "dropped" for p in self.parts),
            jammed=sum(p.status == "jammed" for p in self.parts),
        )
        self._last_interval = max(1, self._n_since_obs)
        self._n_since_obs = 0
        return reward, done, info

    # -- observations ----------------------------------------
    def _setup_observables(self):
        observables = super()._setup_observables()
        rate = self.control_freq

        def add(fn, modality):
            fn = sensor(modality=modality)(fn)
            observables[fn.__name__] = Observable(name=fn.__name__, sensor=fn, sampling_rate=rate)

        if self.use_object_obs:

            def parts_pos(obs_cache):
                return np.concatenate([self.part_pose(i)[0] for i in range(len(self.parts))])

            def parts_axis(obs_cache):
                return np.concatenate([self.part_pose(i)[1][:, 2] for i in range(len(self.parts))])

            def parts_sorted(obs_cache):
                return np.array([float(p.status in SORTED) for p in self.parts])

            def targets(obs_cache):
                return np.concatenate([self.hole_top, self.bucket_floor + [0, 0, BUCKET_H]])

            for fn in (parts_pos, parts_axis, parts_sorted, targets):
                add(fn, "object")

        if self.use_joint_torque:

            def tau_ext(obs_cache):
                return self.tau_ext_buf.last(1)[0].copy()

            def torque_hist(obs_cache):
                return self.torque_history().ravel()

            add(tau_ext, "force")
            add(torque_hist, "force")

        if self.pads is not None:

            def tactile(obs_cache):
                k = self._last_interval  # max over the last control interval
                return self.taxel_buf.last(k).max(axis=0).ravel()

            add(tactile, "force")
        return observables


register_env(SortBoltsNuts)
