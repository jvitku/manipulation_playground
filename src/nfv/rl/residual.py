"""Lesson 19's residual actor: a small TD3 correction on top of a FROZEN nano(Force)VLA.

The acting policy, per 10 Hz env step (``j`` = position inside the base's current chunk):

    base chunk  = base.act(obs, instruction)            every n_exec steps (as in lessons 16-18)
    a_base      = (chunk.dp[j], chunk.gripper[j])       the base's NEXT action
    r           = tanh(actor(features(obs, a_base)))    in [-1, 1]^A, A = 3 (or 6 with force)
    dp          = a_base.dp + DP_RES * r[:3]            what the env executes
    f_target    = a_base.f_target + F_RES * r[3:6]      only with ``residual_force`` (hybrid head)

The base never receives a gradient and is never re-trained; it keeps planning chunks from the
real (residual-perturbed) observations, so the residual corrects a closed loop, not a replay.

Design choices (each one is quoted by lesson 19):

* **Bound ±DP_RES = 2 mm per step per axis**, not v1's ±5 mm. On T1 the base presses at about
  1 mm/step, and the button spring is K = 1000 N/m, so 1 mm of travel is 1 N. ±5 mm would let one
  exploratory residual step add 5 N on top of a 3-9 N click threshold, i.e. jump straight through
  the 1.5x over-force band: the residual would dominate the press instead of correcting it.
  ±2 mm is enough to cancel the press entirely and back off at 1 mm/step (the "stop on force"
  behaviour BC never learned), while keeping every residual step a bounded deviation (2 N).
* **±F_RES = 2 N on f_target** (v1's value) behind ``residual_force``; it needs a base whose
  chunk carries ``f_target`` (the lesson-18 hybrid head). Today's bases are position-head, so the
  flag raises on them instead of silently learning a correction nothing applies.
* **Zero start**: the actor's last layer weight is multiplied by 0.01 and its bias set to 0 (the
  task-13 recipe), so at step 0 the residual is ~1e-3 of its bound (micrometres) and the policy
  IS the base. ``zero_last_layer`` makes it exactly zero (tanh(0) = 0, base + 0.0 = base
  bit-for-bit) — the step-0 equivalence the unit test asserts.
* **Inputs** (fixed scales, nothing fitted, like nfv.model.build's action scaling): the 8-D
  state (positions and velocities x10, i.e. decimetres), the latest wrench /10 N, its change
  since the previous policy step /5 N (the "is the force rising?" signal a press needs), the
  base's next action (dp / 1 cm, gripper in [-1, 1]), and the task one-hot (4). 28 inputs; +3
  (base f_target /10 N) with ``residual_force``. No images: the base already looked; the
  residual reads what changes at contact, which images at 10 Hz barely show.
* **Stored TD3 action = the residual r** (not the summed action): the critic learns
  Q(features, r), and the BC term pulls r toward 0, i.e. toward the base.

Shape legend: F = feature dims (28, or 31 with force), A = residual dims (3, or 6 with force).
"""

from __future__ import annotations

import numpy as np
import torch

from fvb.policy.td3 import TD3, TD3Config
from nfv.seeds import TASK_NAMES
from nfv.sim.controllers import F_ABORT

DP_RES = 0.002  # m, residual bound per axis per policy step (see the module docstring)
F_RES = 2.0  # N, residual bound on f_target per force axis (hybrid-head bases only)
POS_SCALE = 10.0  # state positions/velocities m -> dm
WRENCH_SCALE = 10.0  # N
DWRENCH_SCALE = 5.0  # N per policy step
BASE_DP_SCALE = 0.01  # m: the base's dp in centimetres
LAST_LAYER_INIT = 0.01  # last-layer weight multiplier (task-13 recipe); bias -> 0
CONTACT_N = 0.5  # N: |F| above this counts as contact (= T1's "press" subtask threshold and
#                  the controller's SELECT_ON_N), used only to split the residual-magnitude log

# BC toward residual = 0 (lesson 12: anneal, never to zero — at 0 the actor drifts off the base
# as soon as the critic is wrong, which it is early on)
BC_START = 2.5
BC_FLOOR = 1.0
BC_FLOOR_MIN = 1.0  # the schedule refuses floors below this (the lesson-12 failure)
HOLE_SHAPING_MM = 4.0  # T3's hidden-offset bound: the shaping potential's length scale


def feature_dim(residual_force: bool = False) -> int:
    return 8 + 6 + 6 + 4 + len(TASK_NAMES) + (3 if residual_force else 0)


def act_dim(residual_force: bool = False) -> int:
    return 6 if residual_force else 3


def features(
    state: np.ndarray,
    wrench: np.ndarray,
    prev_wrench: np.ndarray,
    base_dp: np.ndarray,
    base_gripper: float,
    task: str,
    base_f_target: np.ndarray | None = None,
    residual_force: bool = False,
) -> np.ndarray:
    """One residual-actor input vector, float32 [F] (see the module docstring for the scales).

    ``prev_wrench`` is the wrench of the previous policy step (= ``wrench`` at an episode's first
    step, so the change starts at 0).
    """
    s = np.asarray(state, dtype=np.float32).copy()
    s[:3] *= POS_SCALE
    s[5:8] *= POS_SCALE
    w = np.asarray(wrench, dtype=np.float32)
    dw = (w - np.asarray(prev_wrench, dtype=np.float32)) / DWRENCH_SCALE
    onehot = np.zeros(len(TASK_NAMES), np.float32)
    onehot[TASK_NAMES.index(task)] = 1.0
    parts = [
        s,
        w / WRENCH_SCALE,
        dw,
        np.asarray(base_dp, dtype=np.float32) / BASE_DP_SCALE,
        np.array([2.0 * float(base_gripper) - 1.0], np.float32),
        onehot,
    ]
    if residual_force:
        ft = np.zeros(3, np.float32) if base_f_target is None else base_f_target[:3]
        parts.append(np.asarray(ft, dtype=np.float32) / WRENCH_SCALE)
    return np.concatenate(parts).astype(np.float32)


def compose(
    base_dp: np.ndarray,
    base_gripper: float,
    base_f_target: np.ndarray | None,
    r: np.ndarray,
    residual_force: bool = False,
    dp_res: float = DP_RES,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Base action + bounded residual -> (dp float64 [3], gripper, f_target [6] | None).

    The gripper is never corrected (lesson 19 is about contact force, not grasping). With
    ``r == 0`` the returned dp equals ``base_dp.astype(float64)`` bit-for-bit. ``dp_res`` is
    the bound (default DP_RES); the lesson's wrong exercise raises it to v1's 5 mm.
    """
    r = np.asarray(r, dtype=np.float64)
    dp = np.asarray(base_dp, dtype=np.float64) + dp_res * r[:3]
    ft = None if base_f_target is None else np.asarray(base_f_target, dtype=np.float64).copy()
    if residual_force:
        if ft is None:
            raise ValueError("residual_force needs a base whose chunk carries f_target")
        ft[:3] = ft[:3] + F_RES * r[3:6]
    return dp, float(base_gripper), ft


def bc_weight_at(t: int, decay_steps: int, start: float = BC_START, floor: float = BC_FLOOR):
    """BC weight lambda(t) = floor + (start - floor) * max(0, 1 - t / decay_steps).

    Lesson 12's schedule (lab/tasks/task7_td3bc.bc_weight_at, same formula): a linear anneal
    from 2.5 to a floor of 1.0 over ``decay_steps`` environment steps, then flat — never to
    zero. Floors below BC_FLOOR_MIN are refused, so a config cannot reintroduce that failure.
    """
    if floor < BC_FLOOR_MIN:
        raise ValueError(f"bc floor {floor} < {BC_FLOOR_MIN}: the BC term must never anneal away")
    if start < floor:
        raise ValueError("bc start must be >= the floor (the schedule only decays)")
    return floor + (start - floor) * max(0.0, 1.0 - t / max(int(decay_steps), 1))


def make_agent(
    residual_force: bool = False,
    hidden: int = 256,
    batch: int = 256,
    device: str = "cpu",
    zero_last_layer: bool = False,
) -> TD3:
    """The fvb TD3 (twin critics, delayed actor, target smoothing) sized for the residual.

    Two hidden layers of ``hidden`` (v1: 256); the actor's last layer is scaled by
    LAST_LAYER_INIT with bias 0 (``zero_last_layer``: weight 0 too, so r == 0 exactly).
    """
    cfg = TD3Config(hidden=hidden, batch=batch)
    agent = TD3(feature_dim(residual_force), act_dim(residual_force), cfg, device=device)
    with torch.no_grad():
        for net in (agent.actor, agent.actor_t):
            last = net.net[-1]
            if zero_last_layer:
                last.weight.zero_()
            else:
                last.weight.mul_(LAST_LAYER_INIT)
            last.bias.zero_()
    return agent


class ZeroActionView:
    """The BC "demo" buffer of lesson 19: the replay buffer's own observations with target
    action 0, i.e. BC toward residual = 0 on the states the policy actually visits.

    v1 put the BC term "on the demonstrations"; for a residual whose target is always 0 the
    demonstrations' only contribution would be their states, and those would need the frozen
    base re-run on every demo frame (images) to build the base-next-action feature. The visited
    states are the ones where drifting off the base matters, and cost nothing extra.
    Duck-types the two things TD3.update reads from ``demo``: ``.n`` and ``.sample()``.
    """

    def __init__(self, buf):
        self.buf = buf

    @property
    def n(self) -> int:
        return self.buf.n

    def sample(self, batch: int, rng: np.random.Generator, device):
        o, a, *rest = self.buf.sample(batch, rng, device)
        return (o, torch.zeros_like(a), *rest)


def reward(
    info: dict,
    peak_prev: float = 0.0,
    step_cost: float = 0.01,
    peak_force_penalty: float = 0.0,
    fail_penalty: float = 0.0,
    hole_shaping: float = 0.0,
) -> tuple[float, bool]:
    """Sparse reward of one env step -> (r, terminal).

    +1 on success, -1 on a force_abort or T1's over_force (pressing through the 1.5x band),
    -``fail_penalty`` (default 0) on any other task failure (wrong_button, …), -``step_cost``
    every step, and an optional peak-force penalty ``peak_force_penalty`` x the increase of the
    episode's peak |F| this step / F_ABORT — summed over an episode that is exactly
    ``peak_force_penalty * peak / F_ABORT``, a dense way to state "gentler" as a goal.
    ``terminal`` is True on every end except timeout: the features carry no clock, so a timeout
    is not a property of the state and bootstraps (as in task 13).

    ``hole_shaping`` (default 0 = the plan's sparse reward) adds lesson 13's privileged
    potential shaping toward the TRUE hole on T3: k x (d_prev - d) / HOLE_SHAPING_MM, with d the
    peg tip's lateral distance to the hidden hole (mm, ``info["hole_dist_mm"]`` and
    ``info["hole_prev_mm"]``, written by the loop, never seen by the actor or the critic).
    Potential-based, so it changes how fast the optimum is found, not which policy is optimal.
    """
    r = -step_cost
    hd, hp = info.get("hole_dist_mm"), info.get("hole_prev_mm")
    if hole_shaping and hd is not None and hp is not None:
        r += hole_shaping * (hp - hd) / HOLE_SHAPING_MM
    if peak_force_penalty:
        r -= peak_force_penalty * max(0.0, float(info["peak_force"]) - peak_prev) / F_ABORT
    out = info.get("outcome")
    if info.get("done"):
        if info.get("success"):
            r += 1.0
        elif info.get("force_abort") or out == "over_force":
            r -= 1.0
        elif out not in (None, "timeout"):
            r -= fail_penalty
    terminal = bool(info.get("done")) and out != "timeout"
    return r, terminal
