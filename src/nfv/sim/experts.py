"""Privileged scripted experts for T0-T3, plus DART action noise and the progress labels.

Privileged: each expert reads the task's HIDDEN quantities (click threshold, surface
height/tilt, true hole centre) straight off the task object — that is the point: the demos
contain behaviour a camera-only policy cannot reproduce without reading the force channel.

Experts are small FSMs emitting one 10 Hz :class:`~nfv.sim.env.Action` per call, in the
patterns the plan names: two-speed approach everywhere (fast > 5 cm from the goal, slow
below), a position-mode press on T1 (1 mm/step descent carried entirely by the dp labels —
the wrench is the only observable stop cue; back off after the click), wipe with the hybrid
controller holding the commanded force on T2, and a spiral search on a missed insertion
on T3.

Per step each expert also emits (``ExpertStep``):
  * ``subtask_id`` — its FSM phase mapped into the task's SUBTASKS (the demo label; the env's
    ``Obs.subtask_id`` is a state heuristic, the expert's own phase is the cleaner teacher);
  * ``s_fv2`` — the ForceVLA2 progress label (paper eq. 10, alpha = lambda = 2):
        s = theta^alpha * exp(-lambda * l) * clip((f - n) / (m - n), 0, 1)
    with, on the gantry: theta ≡ 1 (the yaw is position-held, no orientation misalignment
    exists), l = remaining distance to the current phase goal (m), f = a phase-relevant force
    magnitude (N) and (n, m) the phase's force range — (0, 0) in free-space phases, where the
    force factor is defined as 1. Every choice the paper leaves open is written down here.
  * ``s_lin`` — a simple linear phase progress: (phase_index + within-phase fraction) /
    n_phases, the within-phase fraction being 1 − l/l0 (l0 = l at phase entry).

Action noise (DART-style, the convention of lab/tasks/task4_force_token.py): while recording,
the EXECUTED action is the expert action plus N(0, sigma) per dp axis, but the stored label
stays the clean expert action; the observations see the noisy motion. ``noisy()`` builds the
executed action; the recorder stores both (``dp`` = clean label, ``dp_exec`` = executed).
Default sigma 0.5 mm.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from nfv.sim.controllers import POLICY_DT
from nfv.sim.env import Action, Env
from nfv.sim.gantry_xml import START_HEIGHT
from nfv.sim.tasks import TIP_TIP_Z, PegInsert, PressButton, ReachGrasp, Wipe

DART_SIGMA_MM = 0.5  # default std of the executed-action noise, mm per dp axis

SPEED_FAST = 0.30  # m/s, far from the goal
SPEED_SLOW = 0.10  # m/s, within 5 cm of the goal
TWO_SPEED_DIST = 0.05  # m


def fv2_progress(
    theta: float, dist: float, f: float, n: float, m: float, alpha: float = 2.0, lam: float = 2.0
) -> float:
    """ForceVLA2 progress label; see the module docstring for the on-gantry conventions."""
    force = 1.0 if m <= n else float(np.clip((f - n) / (m - n), 0.0, 1.0))
    return float(np.clip(theta, 0.0, 1.0)) ** alpha * float(np.exp(-lam * dist)) * force


def noisy(action: Action, rng: np.random.Generator, sigma_mm: float = DART_SIGMA_MM) -> Action:
    """The EXECUTED action: expert dp plus per-axis N(0, sigma). The label stays clean."""
    if sigma_mm <= 0:
        return action
    dp = np.asarray(action.dp, dtype=np.float64) + rng.normal(0.0, sigma_mm * 1e-3, size=3)
    return replace(action, dp=dp)


@dataclass
class ExpertStep:
    """One 10 Hz expert decision: the clean action plus the demo labels."""

    action: Action
    subtask_id: int  # index into the task's SUBTASKS
    s_fv2: float
    s_lin: float


class Expert:
    """Base FSM: phase bookkeeping, two-speed moves, progress labels."""

    PHASES: tuple[str, ...] = ()
    PHASE_SUBTASK: tuple[int, ...] = ()  # PHASES[i] -> task SUBTASKS index

    def __init__(self, env: Env):
        self.env = env
        self.task = env.task
        self._phase = 0
        self._l0: float | None = None

    # --- shared geometry helpers ---
    @property
    def cur(self) -> np.ndarray:
        """Current prismatic joint positions (3,) = tool position in gantry coordinates."""
        return self.env.qpos_xyz

    def tool_z_for(self, world_z: float) -> float:
        """jz that puts the TOOL ORIGIN at the given world z."""
        return world_z - START_HEIGHT

    def dp_to(self, target: np.ndarray, speed: float | None = None) -> np.ndarray:
        """Per-axis step toward ``target`` (3,), two-speed unless ``speed`` is given."""
        err = np.asarray(target, dtype=np.float64) - self.cur
        if speed is None:
            speed = SPEED_FAST if np.linalg.norm(err) > TWO_SPEED_DIST else SPEED_SLOW
        lim = speed * POLICY_DT
        return np.clip(err, -lim, lim)

    def to_phase(self, idx: int) -> None:
        self._phase = idx
        self._l0 = None

    def labels(self, dist: float, f: float = 0.0, n: float = 0.0, m: float = 0.0) -> tuple:
        """(s_fv2, s_lin) for the current phase; captures l0 on first call in the phase."""
        if self._l0 is None:
            self._l0 = max(dist, 1e-6)
        frac = float(np.clip(1.0 - dist / self._l0, 0.0, 1.0))
        s_lin = (self._phase + frac) / len(self.PHASES)
        return fv2_progress(1.0, dist, f, n, m), s_lin

    def step(self, action: Action, dist: float, f: float = 0.0, n: float = 0.0, m: float = 0.0):
        s_fv2, s_lin = self.labels(dist, f, n, m)
        return ExpertStep(action, self.PHASE_SUBTASK[self._phase], s_fv2, s_lin)

    def act(self) -> ExpertStep:
        raise NotImplementedError


class ReachGraspExpert(Expert):
    """T0: two-speed reach above the cube, descend, close, lift. Pure position mode."""

    PHASES = ("reach", "descend", "grasp", "lift")
    PHASE_SUBTASK = (0, 0, 1, 2)  # task SUBTASKS: reach, grasp, lift
    GRASP_STEPS = 5  # dwell while the gripper closes (0.5 s)

    def __init__(self, env: Env):
        super().__init__(env)
        assert isinstance(self.task, ReachGrasp)
        self._grasp_i = 0

    def act(self) -> ExpertStep:
        cube = self.task.cube_pos(self.env)
        cur = self.cur
        if self._phase == 0:  # reach: xy above the cube at the start height
            tgt = np.array([cube[0], cube[1], cur[2]])
            if np.linalg.norm(tgt[:2] - cur[:2]) < 0.004:
                self.to_phase(1)
            else:
                return self.step(Action(dp=self.dp_to(tgt)), float(np.linalg.norm(tgt - cur)))
        if self._phase == 1:  # descend: finger pads around the cube, pad bottoms ~4 mm clear
            jz_t = self.tool_z_for(cube[2] + 0.049)  # pads span tool -0.06..-0.01
            tgt = np.array([cube[0], cube[1], jz_t])
            if abs(jz_t - cur[2]) < 0.003:
                self.to_phase(2)
            else:
                return self.step(
                    Action(dp=self.dp_to(tgt, SPEED_SLOW)), float(np.linalg.norm(tgt - cur))
                )
        if self._phase == 2:  # grasp: hold position, close the gripper
            self._grasp_i += 1
            if self._grasp_i > self.GRASP_STEPS:
                self.to_phase(3)
            else:
                dist = (self.GRASP_STEPS - self._grasp_i) / self.GRASP_STEPS * 0.01
                return self.step(Action(dp=np.zeros(3), gripper=1.0), dist)
        # lift: straight up, keep the grip
        lift = float(self.env.data.xpos[self.task._cube_bid][2]) - self.task.cube_z0
        return self.step(
            Action(dp=np.array([0.0, 0.0, 0.012]), gripper=1.0), max(0.0, 0.055 - lift)
        )


class PressButtonExpert(Expert):
    """T1: approach the instructed button, press in POSITION mode (1 mm/step), back off.

    Privileged: reads the hidden threshold (for the progress labels) and the click latch
    (its stop signal). The press itself never touches the hybrid controller: every press
    step is ``Action(dp=(0, 0, -PRESS_DP))``, so the demos carry the whole descent in the
    dp labels and a position-head policy has a press to clone. When to STOP is observable
    only through the wrench: the quasi-static force ramps at K_eff * PRESS_DP ≈ 0.38 N per
    step (K_p, pad and cap spring in series) and the click comes at the hidden threshold.
    The slow descent bounds the post-click overshoot to under one step of ramp plus the cap
    weight, so the episode peak stays well below the 1.5x-threshold success bound even at
    the lowest threshold (3 N) — measured, see tests/nfv/measure_t1_overshoot.py.
    """

    PHASES = ("approach", "descend", "press", "back_off")
    PHASE_SUBTASK = (0, 0, 1, 2)  # task SUBTASKS: approach, press, done
    HOVER_M = 0.006  # m, descend target: tip this far above the cap top (still free space)
    PRESS_DP = 0.001  # m per 10 Hz step = 1 cm/s position-mode press descent

    def __init__(self, env: Env):
        super().__init__(env)
        assert isinstance(self.task, PressButton)

    def act(self) -> ExpertStep:
        task: PressButton = self.task
        bxy = task.button_pos(self.env, task.target)[:2]
        thr = task.threshold[task.target]
        cur = self.cur
        if self._phase == 0:  # approach: xy over the cap
            tgt = np.array([bxy[0], bxy[1], cur[2]])
            if np.linalg.norm(tgt[:2] - cur[:2]) < 0.003:
                self.to_phase(1)
            else:
                return self.step(Action(dp=self.dp_to(tgt)), float(np.linalg.norm(tgt - cur)))
        if self._phase == 1:  # descend: tip HOVER_M above the cap top, still free of contact
            jz_t = self.tool_z_for(task.CAP_TOP_Z + self.HOVER_M - TIP_TIP_Z)
            tgt = np.array([bxy[0], bxy[1], jz_t])
            if abs(jz_t - cur[2]) < 0.002:
                self.to_phase(2)
            else:
                return self.step(
                    Action(dp=self.dp_to(tgt, SPEED_SLOW)), float(np.linalg.norm(tgt - cur))
                )
        if self._phase == 2:  # press: position-mode descent, PRESS_DP per step, no f_target
            if task.clicked[task.target]:
                self.to_phase(3)
            else:
                d_rem = max(0.0, thr / task.K_SPRING - task.depression(self.env, task.target))
                f_now = abs(float(self.env.wrench[2]))
                return self.step(
                    Action(dp=np.array([0.0, 0.0, -self.PRESS_DP])), d_rem, f_now, 0.0, thr
                )
        # back off: position-mode retreat (in practice unreached: the env latches the episode
        # verdict — success or over_force — at the end of the step the click fired in)
        return self.step(Action(dp=np.array([0.0, 0.0, 0.02])), 0.0, 0.0, 0.0, thr)


class WipeExpert(Expert):
    """T2: hybrid wipe — approach the strip start, engage f_z = −target, sweep along x.

    The hybrid z-axis handles the hidden height and tilt by itself (the admittance descends
    until the measured normal force matches the commanded one); the expert never reads the
    surface height. It sweeps at 0.05 m/s so the admittance tracks the tilt well inside the
    1.5 N band (measured; at 0.1 m/s the sliding contact chatters),
    and reverses for a second pass if any cell was missed.
    """

    PHASES = ("approach", "engage", "wipe")
    PHASE_SUBTASK = (0, 0, 1)  # task SUBTASKS: approach, wipe, done
    X_START, X_END = -0.095, 0.098  # sweep span in the world (plate frame ≈ world xy)
    SWEEP_DP = 0.005  # m per policy step = 0.05 m/s
    ENGAGE_MAX_STEPS = 25  # 2.5 s safety on the engage phase

    PROBE_F = 3.0  # N commanded until first touch: caps the descent at 3 cm/s, so the
    # impact transient of a "firmly" (8 N) episode stays well under the 25 N abort

    def __init__(self, env: Env):
        super().__init__(env)
        assert isinstance(self.task, Wipe)
        self._engage_i = 0
        self._dir = +1.0
        self._touched = False

    def _ft(self, probe: bool = False) -> np.ndarray:
        f = min(self.PROBE_F, self.task.f_target_n) if probe else self.task.f_target_n
        return np.array([0.0, 0.0, -f, 0.0, 0.0, 0.0])

    def act(self) -> ExpertStep:
        task: Wipe = self.task
        cur = self.cur
        tgt_f = task.f_target_n
        if self._phase == 0:  # approach: xy over the strip start, stay at the start height
            tgt = np.array([self.X_START, 0.0, cur[2]])
            if np.linalg.norm(tgt[:2] - cur[:2]) < 0.004:
                self.to_phase(1)
            else:
                return self.step(Action(dp=self.dp_to(tgt)), float(np.linalg.norm(tgt - cur)))
        if self._phase == 1:  # engage: probe gently down to the surface, then load to target
            self._engage_i += 1
            f_n = task.normal_force(self.env.wrench)
            self._touched = self._touched or f_n > task.CONTACT_N
            if f_n >= tgt_f - 1.0 or self._engage_i > self.ENGAGE_MAX_STEPS:
                self.to_phase(2)
            else:
                ft = self._ft(probe=not self._touched)
                return self.step(Action(dp=np.zeros(3), f_target=ft), tgt_f - f_n, f_n, 0.0, tgt_f)
        # wipe: hold the force, sweep x, reverse at the end if cells were missed
        end = self.X_END if self._dir > 0 else -self.X_END
        if (end - cur[0]) * self._dir < 0.002 and task.coverage < task.COVER_FRAC:
            self._dir = -self._dir
            end = -end
        dp = np.array([np.clip(end - cur[0], -self.SWEEP_DP, self.SWEEP_DP), -cur[1] * 0.3, 0.0])
        f_n = task.normal_force(self.env.wrench)
        return self.step(Action(dp=dp, f_target=self._ft()), abs(end - cur[0]), f_n, 0.0, tgt_f)


class PegInsertExpert(Expert):
    """T3: two-speed approach over the TRUE hole, descend, spiral search on a miss, insert.

    Privileged: aims at marker + hidden offset. The spiral exists for the noisy executed
    actions — a missed insertion shows contact at near-zero depth, and the expert then walks
    an outward spiral under a gentle hybrid push (f_z = −3 N) until the peg drops in.
    """

    PHASES = ("approach", "descend", "search", "insert")
    PHASE_SUBTASK = (0, 1, 2, 3)
    DESCEND_FAST_TIP_Z = 0.055  # m, switch to the slow descend above the hole top: the fast
    # step is 20 mm, so even the worst fast step stops short of the 0.040 block top
    SEARCH_F = 3.0  # N held during the spiral
    INSERT_F = 6.0  # N held while seating

    def __init__(self, env: Env):
        super().__init__(env)
        assert isinstance(self.task, PegInsert)
        self._angle = 0.0

    def act(self) -> ExpertStep:
        task: PegInsert = self.task
        hole = task.hole_xy
        cur = self.cur
        tip = task.peg_tip(self.env)
        depth = task.depth_mm(self.env)
        l_all = float(np.linalg.norm(tip[:2] - hole)) + max(0.0, tip[2] - task.SEAT_TIP_Z)
        f_now = float(np.linalg.norm(self.env.wrench[:3]))
        if self._phase == 0:  # approach: xy over the true hole centre
            tgt = np.array([hole[0], hole[1], cur[2]])
            if np.linalg.norm(tgt[:2] - cur[:2]) < 0.0015:
                self.to_phase(1)
            else:
                return self.step(Action(dp=self.dp_to(tgt)), l_all)
        if self._phase == 1:  # descend: fast to just above the top, then creep down
            if depth > 3.0:
                self.to_phase(3)
            elif f_now > 1.5:
                self.to_phase(2)  # contact with no depth: missed -> spiral
            else:
                dz = -0.02 if tip[2] > self.DESCEND_FAST_TIP_Z else -0.004
                return self.step(Action(dp=np.array([0.0, 0.0, dz])), l_all)
        if self._phase == 2:  # spiral search under a gentle push until the peg drops in
            if depth > 3.0:
                self.to_phase(3)
            else:
                self._angle += 0.9
                r = 0.0006 + 0.0005 * self._angle / (2 * np.pi)
                tgt_xy = hole + r * np.array([np.cos(self._angle), np.sin(self._angle)])
                dp_xy = np.clip(tgt_xy - cur[:2], -0.002, 0.002)
                ft = np.array([0.0, 0.0, -self.SEARCH_F, 0.0, 0.0, 0.0])
                return self.step(
                    Action(dp=np.array([*dp_xy, 0.0]), f_target=ft),
                    l_all,
                    f_now,
                    0.0,
                    self.INSERT_F,
                )
        # insert: compliant push down, xy held on the true centre
        dp_xy = np.clip(hole - cur[:2], -0.001, 0.001)
        ft = np.array([0.0, 0.0, -self.INSERT_F, 0.0, 0.0, 0.0])
        return self.step(
            Action(dp=np.array([*dp_xy, 0.0]), f_target=ft), l_all, f_now, 0.0, self.INSERT_F
        )


EXPERTS: dict[str, type[Expert]] = {
    "t0": ReachGraspExpert,
    "t1": PressButtonExpert,
    "t2": WipeExpert,
    "t3": PegInsertExpert,
}


def make_expert(env: Env) -> Expert:
    """The scripted expert matching the env's current task. Build AFTER env.reset()."""
    return EXPERTS[env.task_name](env)


def rollout_expert(
    env: Env,
    seed: int,
    noise_mm: float = DART_SIGMA_MM,
    task: str | None = None,
    hidden_overrides: dict | None = None,
) -> dict:
    """Roll one scripted-expert episode (no recording) and return the final info dict.

    The noise rng is seeded from the episode seed (offset so it never mirrors the env's
    placement stream) — the same convention nfv.data.record uses, so an evaluation episode
    and a recorded demo of the same seed see identical noise.
    """
    env.reset(seed, task=task, hidden_overrides=hidden_overrides)
    expert = make_expert(env)
    rng = np.random.default_rng(seed + 7_777_777)
    while True:
        es = expert.act()
        _, info = env.step(noisy(es.action, rng, noise_mm))
        if info["done"]:
            return info
