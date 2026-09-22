"""The Stage 1 task: Track A gantry insertion with a *hidden* hole offset.

The hole is placed at a random (x, y) each episode; the peg starts above the origin. Neither the
policy's observation nor the episode config given to the policy reveals the offset — only the
scripted expert reads it from the simulator. Observations are built here so that data
collection, training and closed-loop evaluation share one definition (PLAN §9 hook).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fvb.control.gantry import Gantry
from fvb.ft.compensate import LoadParams
from fvb.scenes.builder import SceneParams, build_model, insertion_depth

CONTROL_FREQ = 20.0
DT = 1.0 / CONTROL_FREQ
MAX_STEPS = 160  # 8 s
SUCCESS_DEPTH = 0.030  # m
FORCE_ABORT_N = 60.0
OBS_DIM = 18  # rel pos 3, vel 3, ft_comp 6, hf summary 3, prev action 3
FORCE_IDX = slice(6, 15)  # the channels zeroed in the no-force ablation
ACT_DIM = 3
ACT_STEP_MAX = 0.005  # m per control step (clip for the policy)


@dataclass
class TaskParams:
    clearance: float = 0.0005
    offset_sigma: float = 0.0015  # m, per axis
    offset_max: float = 0.004
    kp: float = 800.0
    timestep: float = 0.001  # M3 default
    speed: float = 0.05  # m/s expert descent
    jam_force_N: float = 3.0
    retract: float = 0.003  # m
    correct_step: float = 0.0005  # m per correction (= clearance)
    start_height_above_rim: float = 0.02
    force_noise_N: float = 0.0  # Gaussian sensor noise std per physics step, force channels
    torque_noise_Nm: float = 0.0  # same, torque channels


class GantryTask:
    """One episode's simulator + observation builder. `expert_action` is privileged."""

    def __init__(self, p: TaskParams, seed: int):
        self.p = p
        rng = np.random.default_rng(seed)
        off = np.clip(rng.normal(0, p.offset_sigma, 2), -p.offset_max, p.offset_max)
        self.hole_xy = off  # the hidden variable
        self.scene = SceneParams(
            kp=p.kp,
            timestep=p.timestep,
            clearance=p.clearance,
            hole_xy=(float(off[0]), float(off[1])),
        )
        self.m, self.d = build_model(self.scene)
        load = LoadParams(mass=self.scene.peg_mass, com=np.array([0, 0, -self.scene.peg_len / 2]))
        noise = None
        if p.force_noise_N > 0 or p.torque_noise_Nm > 0:
            noise = np.array([p.force_noise_N] * 3 + [p.torque_noise_Nm] * 3)
        self.g = Gantry(
            self.m,
            self.d,
            self.scene,
            CONTROL_FREQ,
            load=load,
            noise_std=noise,
            noise_seed=seed + 7919,
        )
        # start: peg tip start_height_above_rim above the rim, over the origin (not the hole)
        z0 = (
            -(self.scene.start_height - 0.01 - self.scene.peg_len - self.scene.hole_depth)
            + p.start_height_above_rim
        )
        self.target = np.array([0.0, 0.0, z0])
        self.g.set_target(self.target)
        self.g.settle(1.0)
        self.pos0 = self.g.ee_pos().copy()
        self.prev_action = np.zeros(3)
        self.k = 0
        # expert state machine
        self._phase = "descend"
        self._retract_z = None
        self._last_depth = -1.0

    # -- observation ------------------------------------------------------------------------
    def depth(self) -> float:
        return insertion_depth(self.scene, self.g.q[2])

    def observe(self) -> np.ndarray:
        raw_hf = self.g.last_hf_ft  # (n_sub, 6) physics-rate raw wrench of the last interval
        fmag = np.linalg.norm(raw_hf[:, :3], axis=1)
        tmag = np.linalg.norm(raw_hf[:, 3:], axis=1)
        comp = self.g.ft_comp_last()
        return np.concatenate(
            [
                self.g.ee_pos() - self.pos0,
                self.g.ee_vel()[:3],
                comp,
                [fmag.max(), raw_hf[:, 2].min(), tmag.max()],
                self.prev_action,
            ]
        ).astype(np.float32)

    # -- stepping -----------------------------------------------------------------------------
    def step(self, delta: np.ndarray, log=None) -> tuple[bool, str | None]:
        delta = np.clip(np.asarray(delta, dtype=np.float64), -ACT_STEP_MAX, ACT_STEP_MAX)
        self.target = self.target + delta
        self.g.step(self.target, log)
        self.prev_action = delta.astype(np.float32)
        self.k += 1
        f = np.linalg.norm(self.g.ft_comp_last()[:3])
        if self.depth() >= SUCCESS_DEPTH:
            return True, "success"
        if f > FORCE_ABORT_N:
            return True, "force_abort"
        if self.k >= MAX_STEPS:
            return True, "timeout"
        return False, None

    # -- privileged expert --------------------------------------------------------------------
    def expert_action(self) -> np.ndarray:
        p = self.p
        pos = self.g.ee_pos()
        depth = self.depth()
        f = self.g.ft_comp_last()
        jammed = np.linalg.norm(f[:3]) > p.jam_force_N and depth < self._last_depth + 2e-4
        self._last_depth = depth
        a = np.zeros(3)
        if self._phase == "descend":
            if jammed:
                self._phase = "retract"
                self._retract_z = pos[2] + p.retract
            else:
                a[2] = -p.speed * DT
                return a
        if self._phase == "retract":
            if pos[2] < self._retract_z - 5e-4:
                a[2] = min(p.speed * DT, self._retract_z - pos[2])
                return a
            self._phase = "correct"
        if self._phase == "correct":
            err = self.hole_xy - pos[:2]  # privileged: true hole position
            step = np.clip(err, -p.correct_step, p.correct_step)
            a[:2] = step
            self._phase = "descend"
            self._last_depth = -1.0
            return a
        return a


@dataclass
class EpisodeResult:
    success: bool
    reason: str
    steps: int
    peak_F: float
    n_jams: int
    hole_xy: list = field(default_factory=list)
