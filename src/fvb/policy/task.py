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
from fvb.policy.spec import GANTRY
from fvb.scenes.builder import SceneParams, build_model, insertion_depth

CONTROL_FREQ = 20.0
DT = 1.0 / CONTROL_FREQ
MAX_STEPS = 160  # 8 s
SUCCESS_DEPTH = 0.030  # m
FORCE_ABORT_N = 60.0
OBS_DIM = GANTRY.obs_dim  # rel pos 3, vel 3, ft_comp 6, hf summary 3, prev action 3
FORCE_IDX = GANTRY.force_idx  # the channels zeroed in the no-force ablation
ACT_DIM = GANTRY.act_dim
ACT_STEP_MAX = 0.005  # m per control step (clip for the policy)
XY_ABS_MAX = 0.008  # m, lateral target range in action_mode "xy_abs"


PHYS_TIMESTEPS = (0.0005, 0.001, 0.002)  # s
PHYS_SOLREF_RANGE = (0.002, 0.02)  # s
PHYS_KP_RANGE = (200.0, 3200.0)  # N/m


def encode_action(mode: str, delta, target, target_xy0) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    if mode == "delta":
        return delta
    return np.array([*(target[:2] + delta[:2] - target_xy0), delta[2]])


def decode_action(mode: str, action, target, target_xy0) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    if mode == "delta":
        return action
    xy = target_xy0 + np.clip(action[:2], -XY_ABS_MAX, XY_ABS_MAX)
    return np.array([*(xy - target[:2]), action[2]])


@dataclass
class TaskParams:
    clearance: float = 0.0005
    offset_sigma: float = 0.0015  # m, per axis
    offset_max: float = 0.004
    kp: float = 800.0
    timestep: float = 0.001  # M3 default
    solref_tc: float = 0.005  # contact time constant, s (M3 default)
    # Stage 1 G2: per-episode physics randomisation (timestep from PHYS_TIMESTEPS, solref time
    # constant log-uniform in PHYS_SOLREF_RANGE), drawn from an RNG independent of the hole's
    phys_rand: bool = False
    kp_rand: bool = False  # also draw kp log-uniform in kp_range (controller stiffness)
    kp_range: tuple[float, float] = PHYS_KP_RANGE  # N/m
    obs_kp: bool = False  # append log(kp / 800) to the observation (spec gantry_kp)
    # when the expert makes its lateral correction: "after_retract" (Stage 1 original: one step
    # at the end of the retract, whose length depends on hidden kp/physics) or "jam" (on the
    # jam step itself, while the force/torque cue is in the current observation)
    correct_at: str = "after_retract"
    # policy action format: "delta" (target delta, m) or "xy_abs" (lateral target relative to
    # the episode start + z delta). With deltas a one-step lateral correction whose timing is
    # uncertain regresses to zero; as a position it is a persistent step and survives.
    action_mode: str = "delta"
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
        timestep, tc = p.timestep, p.solref_tc
        if p.phys_rand:
            prng = np.random.default_rng(seed + 104729)
            timestep = float(prng.choice(PHYS_TIMESTEPS))
            tc = float(np.exp(prng.uniform(*np.log(PHYS_SOLREF_RANGE))))
        kp = p.kp
        if p.kp_rand:
            krng = np.random.default_rng(seed + 130363)
            kp = float(np.exp(krng.uniform(*np.log(p.kp_range))))
        self.phys = {"timestep": timestep, "solref_tc": tc, "kp": kp}
        self.scene = SceneParams(
            kp=kp,
            timestep=timestep,
            solref=(tc, 1.0),
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
        self.target_xy0 = self.target[:2].copy()
        self.prev_action = np.zeros(3)
        self.k = 0
        # expert state machine
        self._phase = "descend"
        self._retract_z = None
        self._last_depth = -1.0

    # -- observation ------------------------------------------------------------------------
    def depth(self) -> float:
        return insertion_depth(self.scene, self.g.q[2])

    def force_norm(self) -> float:
        return float(np.linalg.norm(self.g.ft_comp_last()[:3]))

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
                [np.log(self.scene.kp / 800.0)] if self.p.obs_kp else [],
            ]
        ).astype(np.float32)

    # -- stepping -----------------------------------------------------------------------------
    def encode(self, delta: np.ndarray) -> np.ndarray:
        """Expert target delta -> this task's action format (see TaskParams.action_mode)."""
        return encode_action(self.p.action_mode, delta, self.target, self.target_xy0)

    def step(self, action: np.ndarray, log=None) -> tuple[bool, str | None]:
        delta = decode_action(self.p.action_mode, action, self.target, self.target_xy0)
        delta = np.clip(delta, -ACT_STEP_MAX, ACT_STEP_MAX)
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
                if p.correct_at == "jam":
                    a[:2] = np.clip(self.hole_xy - pos[:2], -p.correct_step, p.correct_step)
            else:
                a[2] = -p.speed * DT
                return a
        if self._phase == "retract":
            if pos[2] < self._retract_z - 5e-4:
                a[2] = min(p.speed * DT, self._retract_z - pos[2])
                return a
            self._phase = "correct"
        if self._phase == "correct" and p.correct_at == "jam":
            self._phase = "descend"
            self._last_depth = -1.0
            a[2] = -p.speed * DT
            return a
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
