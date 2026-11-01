"""Lesson 19's simulated deployment: an operator who corrects the robot, and how RL learns from it.

Everything here is OFF unless ``run_residual_rl(interventions="expert", ...)`` or
``deploy_mode=True`` is passed; with both off the loop runs the exact lesson-19 code path.

**The simulated operator** (:class:`PegOperator`, T3 only). It watches every training episode
with privileged sight of the true hole (the T3 expert's knowledge, ``nfv.sim.experts``) and
takes over when it sees one of the two failures the base's T3 anatomy shows (measured on the
committed base: 67/100 episodes land on the rim and sit there until the timeout, 31/100 dig in
past the 25 N abort, and the in-hole cases stall at ~15 mm depth because the position-head base
emits dp_z ~ 0 where the demos' hybrid controller carried the push):

    trigger  stall: in contact (|F| > 1.5 N, the expert's contact cue) for >= STALL_STEPS = 10
                    policy steps (1 s) without gaining DEPTH_GAIN_MM = 1 mm of insertion depth
             force: |F| > F_WARN = 12 N (about half the 25 N abort)
    segment  once triggered the operator keeps the robot until the episode ends or
             SEGMENT_STEPS = 40 steps (4 s) have passed; then its stall counter restarts, so
             the robot gets at least 1 s of autonomy before the next take-over
    budget   a segment starts only while operator steps <= budget(t) x env steps so far, with
             budget(t) = intervention_budget x max(0, 1 - intervention_decay x t / env_steps)
             (decay 0 = an operator who never tires; 1 = one who stops helping by the end)

While intervening it issues a CORRECTION in residual space, like a human nudging a compliant
robot (CR-DAgger's delta corrections): the residual the env executes is the operator's, not the
actor's, and the base keeps planning underneath.

    operator_dp  (position mode, the expert's privileged targets; the residual cannot set f_target)
        lateral  e = hole_xy - tip_xy; dp_xy = clip(e, +-2 mm)  (1 mm once over the hole)
        vertical over the hole (|e|_inf < 0.4 mm, inside the 0.5 mm clearance) or already in
                 it (depth > 1 mm): dp_z = -2 mm (push it home);
                 on the rim in contact: dp_z = clip((push - 3 N) / K_p, -1 mm, +2 mm) - relieve
                 the press down to the expert's 3 N search preload;
                 in free space: dp_z = -1 mm (creep down)
    correction   = clip((operator_dp - base_dp) / dp_res, -1, 1)    [A = 3] residual units

**Learning from it, ForceRFT-style** (:class:`DeployBuffer` + the two views):

* the actor's BC term pulls toward the operator's correction on intervention samples and toward
  0 (with the lesson-12 floor) on autonomous ones (:class:`CorrectionBCView`: a fixed share
  ``corr_frac`` of every BC batch comes from intervention samples - symmetric sampling as in
  HIL-SERL/RLPD, so a 5-20 % intervention share is not drowned by the autonomous majority);
* the twin critics train only on autonomous transitions, and bootstrapping is cut at every
  intervention boundary: the autonomous transition whose next step was handed to the operator
  is not a critic sample at all (:class:`CriticView`). Its o2 is where the operator took over,
  so a target r + gamma Q(o2) would credit the robot with the operator's segment; cutting it as
  a terminal (target r alone) would make "get the operator to take over" look free. Excluding
  it means no TD target ever reads a state whose future belonged to the operator.

**Deployment realism** (:class:`PolicySnapshot`): the acting policy is a frozen copy of the
learner's actor, refreshed every ``swap_every`` TD3 updates - a background learner on a real
robot never acts with weights mid-update. "Robot minutes" = env steps / 10 Hz / 60.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from fvb.policy.td3 import ReplayBuffer
from nfv.sim.controllers import POLICY_DT

# ---- the operator's trigger and segment rule (see the module docstring) -------------------
CONTACT_N = 1.5  # N, the T3 expert's "contact with no depth" cue (PegInsertExpert.act)
STALL_STEPS = 10  # policy steps in contact without depth progress before the operator steps in
DEPTH_GAIN_MM = 1.0  # mm of new insertion depth that counts as progress (resets the counter)
F_WARN = 12.0  # N, take over at once above this (the abort is at 25 N)
SEGMENT_STEPS = 40  # policy steps per take-over at most (4 s), unless the episode ends first
# ---- the operator's motion (position-mode version of the T3 expert) ------------------------
OP_LAT_MAX = 0.002  # m per step toward the true hole while on the rim
OP_LAT_FINE = 0.001  # m per step once over (or in) the hole
OP_ALIGN = 0.0004  # m, per-axis lateral error that counts as "over the hole" (clearance 0.5 mm)
OP_PUSH_DP = -0.002  # m per step: push home once over/in the hole
OP_CREEP_DP = -0.001  # m per step in free space
OP_PRELOAD_N = 3.0  # N, the expert's search preload (PegInsertExpert.SEARCH_F)
OP_RELIEF_MAX = 0.002  # m per step of press relief
OP_PRESS_MAX = 0.001  # m per step of extra press while below the preload
ROBOT_HZ = 1.0 / POLICY_DT  # 10 Hz policy steps


def robot_minutes(env_steps: int) -> float:
    """Wall time a real robot would spend on ``env_steps`` 10 Hz steps (minutes)."""
    return float(env_steps) / ROBOT_HZ / 60.0


def budget_at(t: int, env_steps: int, budget: float, decay: float) -> float:
    """The operator's allowed share of steps at env step ``t`` (linear decay, never < 0)."""
    return float(budget) * max(0.0, 1.0 - float(decay) * t / max(int(env_steps), 1))


class PegOperator:
    """The simulated T3 operator for ONE live episode (privileged: reads the true hole).

    ``observe(env)`` is called once per step BEFORE acting and updates the stall counter;
    ``wants_in(env)`` says whether a trigger fires; ``operator_dp(env)`` is what the operator
    would command; :func:`correction` turns it into residual space.
    """

    def __init__(self, kp: float):
        self.kp = float(kp)
        self.reset()

    def reset(self) -> None:
        self.stall = 0  # consecutive in-contact steps without DEPTH_GAIN_MM of progress
        self.best_depth = 0.0  # mm, deepest insertion seen in the current stall window
        self.active = False  # intervening now
        self.seg_len = 0  # steps of the current segment
        self.trigger = ""  # "stall" | "force" (the reason of the current/last segment)

    def observe(self, env) -> None:
        f = float(np.linalg.norm(np.asarray(env.wrench, np.float64)[:3]))
        depth = float(env.task.depth_mm(env))
        if f > CONTACT_N:
            if depth > self.best_depth + DEPTH_GAIN_MM:
                self.best_depth, self.stall = depth, 0
            else:
                self.stall += 1
        else:
            self.stall, self.best_depth = 0, depth

    def wants_in(self, env) -> str:
        """The trigger on the current observation: "force", "stall", or "" (none)."""
        if float(np.linalg.norm(np.asarray(env.wrench, np.float64)[:3])) > F_WARN:
            return "force"
        return "stall" if self.stall >= STALL_STEPS else ""

    def begin(self, why: str) -> None:
        self.active, self.seg_len, self.trigger = True, 0, why

    def tick(self) -> None:
        """One intervened step done; ends the segment after SEGMENT_STEPS."""
        self.seg_len += 1
        if self.seg_len >= SEGMENT_STEPS:
            self.active = False
            self.stall = 0

    def operator_dp(self, env) -> np.ndarray:
        """The operator's own position-mode step (m, [3]); see the module docstring."""
        task = env.task
        tip = task.peg_tip(env)
        e = np.asarray(task.hole_xy, np.float64) - tip[:2]  # [2] m, true hole - peg tip
        depth = float(task.depth_mm(env))
        w = np.asarray(env.wrench, np.float64)
        push = max(0.0, -float(w[2]))  # N pressing down (course sign: pushing up = -F_z)
        over = bool(np.max(np.abs(e)) < OP_ALIGN) or depth > 1.0
        lat = OP_LAT_FINE if over else OP_LAT_MAX
        dp = np.zeros(3)
        dp[:2] = np.clip(e, -lat, lat)
        if over:
            dp[2] = OP_PUSH_DP
        elif float(np.linalg.norm(w[:3])) > CONTACT_N:
            dp[2] = np.clip((push - OP_PRELOAD_N) / self.kp, -OP_PRESS_MAX, OP_RELIEF_MAX)
        else:
            dp[2] = OP_CREEP_DP
        return dp


# ---- snippet: the correction (residual space) ----------------------------------------------
def correction(operator_dp: np.ndarray, base_dp: np.ndarray, dp_res: float) -> np.ndarray:
    """What the operator's nudge is in the residual's own units.

    legend: A = 3 residual dims (dp xyz); dp_res = the residual bound (m per axis per step)
        operator_dp [3] m   what the operator wants executed this step
        base_dp     [3] m   what the frozen base is about to execute
        returns     [A]     float32 in [-1, 1]: executed dp = base_dp + dp_res * correction
    """
    c = (np.asarray(operator_dp, np.float64) - np.asarray(base_dp, np.float64)) / dp_res  # [3]
    return np.clip(c, -1.0, 1.0).astype(np.float32)  # [3] -> [A], bounded like any residual


# ---- the replay buffer with intervention flags ---------------------------------------------
class DeployBuffer(ReplayBuffer):
    """fvb's ReplayBuffer + parallel arrays (nothing else changes). Legend, N = capacity:

    * intervened [N] bool: the operator chose this transition's action (a = its correction)
    * critic_ok  [N] bool: the critics may train on it (autonomous AND not a boundary)
    * ep_id, ep_t [N] int: episode id and step index inside it (for audits and tests)
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        super().__init__(obs_dim, act_dim, size)
        self.intervened = np.zeros(size, bool)
        self.critic_ok = np.zeros(size, bool)
        self.ep_id = np.full(size, -1, np.int64)
        self.ep_t = np.full(size, -1, np.int64)

    def add(self, o, a, r, o2, terminal: bool, intervened: bool = False, ep_id: int = -1,
            ep_t: int = -1) -> int:  # fmt: skip
        """Store one transition; -> its index (the loop keeps it to mark a boundary later)."""
        i = self.i
        super().add(o, a, r, o2, terminal)
        self.intervened[i] = bool(intervened)
        self.critic_ok[i] = not intervened
        self.ep_id[i], self.ep_t[i] = ep_id, ep_t
        return i

    # ---- snippet: the boundary cut ---------------------------------------------------------
    def mark_boundary(self, idx: int) -> None:
        """The autonomous transition ``idx`` led into an operator take-over: its o2 is the
        state the operator took over in, so no TD target may bootstrap from it -> not a
        critic sample (it still feeds the actor's BC-toward-0)."""
        if idx >= 0 and not self.intervened[idx]:
            self.critic_ok[idx] = False

    def critic_idx(self) -> np.ndarray:
        """Indices the twin critics may sample: autonomous, not cut at a boundary."""
        return np.flatnonzero(self.critic_ok[: self.n])

    def take(self, idx: np.ndarray, device):
        t = lambda x: torch.as_tensor(x[idx], device=device)  # noqa: E731
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d), t(self.p), t(self.p2)


class CriticView:
    """What TD3.update samples as ``buf``: only critic_ok transitions (falls back to every
    autonomous one while none is eligible yet). Duck-types ReplayBuffer.sample/.n."""

    def __init__(self, buf: DeployBuffer):
        self.buf = buf

    @property
    def n(self) -> int:
        return int(self.buf.critic_ok[: self.buf.n].sum())

    def sample(self, batch: int, rng: np.random.Generator, device):
        idx = self.buf.critic_idx()
        if len(idx) == 0:  # only before the first autonomous transition: nothing to learn
            idx = np.arange(self.buf.n)
        return self.buf.take(idx[rng.integers(0, len(idx), batch)], device)


class CorrectionBCView:
    """What TD3.update samples as ``demo`` (its BC term): ``corr_frac`` of the batch from
    intervention samples with target = the operator's correction (the stored action), the rest
    from autonomous samples with target 0 (BC toward the base, with the lesson-12 floor).
    Before the first intervention the whole batch is autonomous (= ZeroActionView)."""

    def __init__(self, buf: DeployBuffer, corr_frac: float = 0.5):
        self.buf, self.corr_frac = buf, float(corr_frac)
        self.last_n_corr = 0

    @property
    def n(self) -> int:
        return self.buf.n

    def sample(self, batch: int, rng: np.random.Generator, device):
        b = self.buf
        iv = b.intervened[: b.n]
        corr_i, auto_i = np.flatnonzero(iv), np.flatnonzero(~iv)
        k = int(round(self.corr_frac * batch)) if len(corr_i) else 0
        if len(auto_i) == 0:
            k = batch
        idx = np.concatenate([corr_i[rng.integers(0, len(corr_i), k)] if k else
                              np.zeros(0, np.int64),
                              auto_i[rng.integers(0, len(auto_i), batch - k)]])  # fmt: skip
        o, a, *rest = b.take(idx, device)
        # ---- snippet: the BC target ----
        # legend: B = batch (k correction rows first, then B-k autonomous rows), A = 3
        target = a.clone()  # [B, A] the stored residual: the operator's correction on rows < k
        target[k:] = 0.0  # [B-k, A] autonomous rows: BC toward residual 0 = toward the base
        self.last_n_corr = k
        return (o, target, *rest)


class PolicySnapshot:
    """The acting policy in deploy mode: a frozen copy of the learner's actor, swapped in only
    by :meth:`refresh` (the loop calls it every ``swap_every`` updates). Duck-types TD3.act."""

    def __init__(self, agent):
        self.agent = agent
        self.device = agent.device
        self.actor = copy.deepcopy(agent.actor).eval()
        self.swaps = 0

    def refresh(self) -> None:
        # ---- snippet: the swap ----
        self.actor.load_state_dict(self.agent.actor.state_dict())  # learner -> robot, atomically
        self.swaps += 1

    @torch.no_grad()
    def act(self, o: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(o, dtype=torch.float32, device=self.device)[None]  # [O] -> [1, O]
        return self.actor(x)[0].cpu().numpy()  # [1, O] -> [1, A] -> [A]
