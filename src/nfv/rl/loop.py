"""Lesson 19's runtime loop: TD3+BC residual RL on a frozen nano(Force)VLA, in the simulator.

    run_residual_rl(base_ckpt, task="t1", seed=0, env_steps=20_000, ctx=None, ...) -> dict

Loop (all in one process, CPU physics + osmesa rendering):

* ``n_envs`` independent Env instances (default 4) stepped round-robin: each keeps its own
  episode, base chunk, chunk position and flow-noise generator. The env has no cheap state copy
  (MuJoCo model + renderer + task state per instance), so "vectorised" here means N live
  episodes interleaved, which decorrelates consecutive replay entries — not batched physics.
* the base plans a chunk every ``n_exec`` steps from its trained config (``base.cfg.n_exec``),
  its flow noise seeded per episode exactly like task_nfv1's evaluator, so the base alone on an
  eval seed reproduces the lesson-16/17 rollout of that seed.
* exploration: Gaussian noise ``expl_noise`` (in residual units, 0.2 = 0.4 mm) on the actor's
  output, clipped to [-1, 1]; from step 0 on (the zero-start actor means "warm-up" is the base
  plus noise, not uniform random actions, which at ±2 mm would just be a shaky base).
* learning starts after ``warmup`` transitions; then UTD 0.5 (one TD3 update per two env
  transitions, the task-13 ratio), BC toward residual = 0 on visited states with weight
  bc_weight_at(t): 2.5 -> floor 1.0 over ``bc_decay`` env steps (never zero).
* eval every ``eval_every`` env steps (and at 0 with the residual OFF = the frozen base) on fixed
  held-out seeds ``nfv.seeds.eval_seeds(task, n_eval)``, deterministic residual (no noise).
* RL training episodes draw seeds from RL_SEED0 + seed * RL_SEED_BLOCK + k, beyond every
  nfv.seeds block (demos < 41 000, eval < 90 000), so one RL seed never sees an eval placement
  and two RL seeds never share a training episode.

Task default **T3** (v1's and the plan's choice; the hole sits up to ±4 mm off the visible
marker, so the missing information is in the wrench). T1 was tried first and is kept as a
parameter: measured on 40 held-out seeds, 55-58 % of two frozen T1 bases' episodes end in
wrong_button, which a residual without images or instruction cannot fix, and only 1-3 of 40 in
a force failure — the residual's whole T1 headroom (``success_cond`` reports it).

Simulated deployment (``interventions="expert"``, ``deploy_mode``; both default off, and off
they leave every line of the lesson-19 path above untouched): a privileged operator corrects
the robot in residual space, the actor imitates the corrections, the critics learn only from
autonomous transitions with bootstrapping cut at intervention boundaries, and the acting
policy is a snapshot swapped every ``swap_every`` updates — see :mod:`nfv.rl.deploy`.

Returned dict (also the shape the lab wrapper writes): see :func:`run_residual_rl`.

RunContext duck-type (the same four optional methods nfv.train.fit uses): ``should_continue()``,
``log(step, **series)``, ``set_status(**kw)``, ``save_ckpt(step, payload, metrics=...)``;
``ctx.params`` (if present) is stored in the checkpoint meta.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from fvb.policy.td3 import ReplayBuffer
from nfv.model.config import ModelConfig, build
from nfv.model.text import BLANK_INSTRUCTION
from nfv.rl.deploy import (
    CorrectionBCView,
    CriticView,
    DeployBuffer,
    PegOperator,
    PolicySnapshot,
    budget_at,
    correction,
    robot_minutes,
)
from nfv.rl.residual import (
    BC_FLOOR,
    BC_START,
    CONTACT_N,
    DP_RES,
    F_RES,
    ZeroActionView,
    act_dim,
    bc_weight_at,
    compose,
    features,
    make_agent,
    reward,
)
from nfv.seeds import eval_seeds
from nfv.sim.env import Action, Env

RL_SEED0 = 100_000  # RL training episodes: beyond every nfv.seeds block
RL_SEED_BLOCK = 100_000  # episodes per RL seed
MA_WINDOW = 50  # training-episode moving average
_CURVE_KEYS = (
    "env_step", "success", "success_lo", "success_hi", "success_cond", "force_abort",
    "over_force", "peak_force_mean", "return_mean", "res_mm_contact", "res_mm_free",
)  # fmt: skip
_CKPT_EVAL_KEYS = (
    "success",
    "force_abort",
    "over_force",
    "peak_force_mean",
    "res_mm_contact",
    "res_mm_free",
)


def _call(ctx, method: str, *args, **kw):
    fn = getattr(ctx, method, None)
    return fn(*args, **kw) if callable(fn) else None


def wilson95(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


# ------------------------------------------------------------------------------- base loading
def policy_from_payload(payload: dict, device: str = "cpu"):
    """Rebuild a frozen Policy from one nfv.train.fit payload {model, ema, arch, ...}
    (task_nfv1._policy_from without the act-time overrides). Acts through its EMA."""
    policy = build(ModelConfig(**dict(payload["arch"])))
    policy.net.load_state_dict(payload["model"])
    policy.ema.load_state_dict(payload["ema"])
    policy = policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def load_base(base_ckpt, task: str, arm: str | None = None, device: str = "cpu"):
    """-> (frozen Policy, info dict). ``base_ckpt``: a Policy, a payload dict or a .pt path.

    A claims-suite payload ({"arms": [labels], label: payload, ...}) is resolved by ``arm``
    (its label); without one, the single non-blank-text arm trained on ``task`` is used, and an
    ambiguous choice raises with the labels. A single-model payload is used as is.
    """
    if hasattr(base_ckpt, "act") and hasattr(base_ckpt, "cfg"):
        base_ckpt.eval()
        return base_ckpt, {"source": "policy", "arm": arm, "task": task}
    src = "payload"
    ck = base_ckpt
    if not isinstance(ck, dict):
        src = str(base_ckpt)
        ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    if "arms" in ck:
        labels = [str(x) for x in ck["arms"]]
        if arm is None:
            cands = [
                lb for lb in labels
                if str(ck[lb].get("task")) == task and not ck[lb].get("blank_text")
            ]  # fmt: skip
            if len(cands) != 1:
                raise ValueError(f"pick the base arm by label (arm=...) from {labels}")
            arm = cands[0]
        if arm not in labels:
            raise ValueError(f"arm {arm!r} not in the suite's arms {labels}")
        pl = ck[arm]
        if pl.get("blank_text"):
            raise ValueError(f"arm {arm!r} was trained on the blank instruction")
        return policy_from_payload(pl, device), {"source": src, "arm": arm, "task": task}
    return policy_from_payload(ck, device), {"source": src, "arm": None, "task": task}


def base_part(policy) -> dict:
    """The frozen base embedded in a residual checkpoint: its EMA (as both model and ema, so
    policy_from_payload rebuilds it; torch.save stores the shared tensors once) and arch."""
    from dataclasses import asdict

    ema = {k: v.detach().cpu().clone() for k, v in policy.ema.state_dict().items()}
    return {"model": ema, "ema": ema, "arch": asdict(policy.cfg)}


# ------------------------------------------------------------------------------- one episode
@dataclass
class _Slot:
    """One live episode in one Env (the round-robin unit)."""

    env: Env
    task: str
    seed: int = 0
    instr: str = ""
    obs: object = None
    gen: torch.Generator | None = None
    chunk: dict | None = None
    j: int = 0
    prev_wrench: np.ndarray | None = None
    feat: np.ndarray | None = None
    ep_ret: float = 0.0
    ep_len: int = 0
    pending: tuple | None = None  # (o, a, r, terminal) waiting for its o2
    hole_mm: float | None = None  # privileged peg-to-hole distance (reward shaping only)
    trace: dict = field(default_factory=dict)
    # simulated deployment only (interventions="expert"; untouched otherwise)
    op: PegOperator | None = None  # this episode's operator
    last_idx: int = -1  # buffer index of this episode's previous transition (boundary cut)
    ep_id: int = -1
    ep_iv: int = 0  # operator steps in this episode


class ResidualActor:
    """The acting policy: frozen base chunks + the residual, one env step at a time.

    ``lazy_render`` (default on): the env renders its two cameras only on the steps whose
    observation the base will plan from (every n_exec-th), because the residual reads no
    images. Rendering is ~26 ms of a ~33 ms CPU step (osmesa, 2 x 96x96; physics is ~1 ms),
    so this is the single biggest throughput lever; it never touches the physics, which the
    step-0 equivalence test checks against a base rollout that renders every step.
    """

    def __init__(
        self, base, agent, residual_force: bool = False, device: str = "cpu",
        lazy_render: bool = True,
        dp_res: float = DP_RES,
    ):  # fmt: skip
        self.base, self.agent, self.residual_force = base, agent, residual_force
        self.device = device
        self.lazy_render = lazy_render
        self.dp_res = float(dp_res)
        self.n_exec = max(1, int(base.cfg.n_exec))
        self.blank = False

    def start(self, slot: _Slot, seed: int) -> None:
        slot.seed = seed
        slot.obs = slot.env.reset(seed, task=slot.task)
        slot.instr = BLANK_INSTRUCTION if self.blank else slot.env.instruction
        slot.gen = torch.Generator(device=self.device)
        slot.gen.manual_seed(seed)  # = task_nfv1's evaluator: the base is seed-deterministic
        slot.chunk, slot.j = None, 0
        slot.prev_wrench = np.asarray(slot.obs.wrench, np.float32).copy()
        slot.ep_ret, slot.ep_len, slot.pending = 0.0, 0, None
        slot.hole_mm = _hole_mm(slot.env)
        slot.feat = self._features(slot)

    def _base_next(self, slot: _Slot, replan: bool = True):
        if replan and (slot.chunk is None or slot.j >= self.n_exec):
            slot.chunk = self.base.act(slot.obs, slot.instr, generator=slot.gen)
            slot.j = 0
            if self.residual_force and slot.chunk.get("f_target") is None:
                raise ValueError(
                    "residual_force needs a hybrid-head base (chunk f_target); this base is "
                    f"head={getattr(self.base.cfg, 'head', '?')!r}"
                )
        j = min(slot.j, len(slot.chunk["dp"]) - 1)
        ft = slot.chunk.get("f_target")
        return slot.chunk["dp"][j], float(slot.chunk["gripper"][j]), (None if ft is None else ft[j])

    def _features(self, slot: _Slot, replan: bool = True) -> np.ndarray:
        dp, g, ft = self._base_next(slot, replan)
        o = slot.obs
        return features(o.state, o.wrench, slot.prev_wrench, dp, g, slot.task, ft,
                        self.residual_force)  # fmt: skip

    def residual(self, feat: np.ndarray, noise: float, rng, off: bool = False) -> np.ndarray:
        if off:
            return np.zeros(act_dim(self.residual_force), np.float32)
        r = self.agent.act(feat)
        if noise > 0:
            r = np.clip(r + rng.normal(0.0, noise, size=r.shape), -1.0, 1.0)
        return np.asarray(r, np.float32)

    def step(self, slot: _Slot, r: np.ndarray):
        """Execute base-next + residual r; -> (obs, info, contact flag of the acted-on obs)."""
        dp, g, ft = self._base_next(slot, replan=False)
        dp, g, ft = compose(dp, g, ft, r, self.residual_force, self.dp_res)
        contact = float(np.linalg.norm(slot.obs.wrench[:3])) > CONTACT_N
        slot.prev_wrench = np.asarray(slot.obs.wrench, np.float32).copy()
        env = slot.env
        render0 = env.render_enabled
        if self.lazy_render:  # render only if the base re-plans from this step's obs
            env.render_enabled = render0 and slot.j + 1 >= self.n_exec
        try:
            obs, info = env.step(Action(dp=dp, gripper=g, f_target=ft))
        finally:
            env.render_enabled = render0  # reset() always renders (the first plan)
        slot.obs = obs
        slot.j += 1
        slot.ep_len += 1
        prev, slot.hole_mm = slot.hole_mm, _hole_mm(env)
        if prev is not None:
            info["hole_dist_mm"], info["hole_prev_mm"] = slot.hole_mm, prev
        return obs, info, contact


def _hole_mm(env) -> float | None:
    """Privileged lateral peg-tip-to-hole distance in mm (T3 only; reward shaping only)."""
    task = env.task
    if not (hasattr(task, "hole_xy") and hasattr(task, "peg_tip")):
        return None
    return float(np.linalg.norm(task.peg_tip(env)[:2] - task.hole_xy) * 1e3)


def _residual_mm(r: np.ndarray, dp_res: float = DP_RES) -> float:
    return float(np.linalg.norm(np.asarray(r[:3], np.float64)) * dp_res * 1e3)


def evaluate(actor: ResidualActor, env: Env, task: str, seeds: list[int], off: bool = False,
             trace_first: bool = False, reward_kw: dict | None = None) -> dict:  # fmt: skip
    """Closed-loop eval on fixed seeds, deterministic residual (``off``: the base alone).

    Returns success/force_abort/over_force rates (+ Wilson 95 % CI on success), mean peak |F|,
    mean return, outcome counts, the residual magnitude (mm/step, |r_dp| * DP_RES) split by
    contact (|F| > CONTACT_N on the observation the residual acted on) vs free space, and with
    ``trace_first`` the first seed's per-step trace {t, fz, res_mm, contact}.
    """
    reward_kw = reward_kw or {}
    rows, mag = [], {"contact": [], "free": []}
    trace = None
    for i, s in enumerate(seeds):
        slot = _Slot(env=env, task=task)
        actor.start(slot, s)
        tr = {"t": [], "fz": [], "res_mm": [], "contact": []}
        info, ret, peak_prev = {"done": False}, 0.0, 0.0
        while not info["done"]:
            r = actor.residual(slot.feat, 0.0, None, off=off)
            obs, info, contact = actor.step(slot, r)
            rr, _ = reward(info, peak_prev, **reward_kw)
            peak_prev = float(info["peak_force"])
            ret += rr
            m = _residual_mm(r, actor.dp_res)
            mag["contact" if contact else "free"].append(m)
            if trace_first and i == 0:
                tr["t"].append(round(float(info["t"]), 2))
                tr["fz"].append(round(float(obs.wrench[2]), 3))
                tr["res_mm"].append(round(m, 4))
                tr["contact"].append(bool(contact))
            if not info["done"]:
                slot.feat = actor._features(slot)
        if trace_first and i == 0:
            trace = tr
        rows.append({"seed": s, "outcome": info["outcome"] or "timeout",
                     "success": bool(info["success"]), "peak_force": float(info["peak_force"]),
                     "return": ret})  # fmt: skip
    n = len(rows)
    k = sum(r["success"] for r in rows)
    outcomes: dict[str, int] = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    n_force = outcomes.get("force_abort", 0) + outcomes.get("over_force", 0)
    out = {
        "n": n,
        "success": k / max(n, 1),
        "success_ci95": wilson95(k, n),
        "force_abort": outcomes.get("force_abort", 0) / max(n, 1),
        "over_force": outcomes.get("over_force", 0) / max(n, 1),
        # the residual's reachable headroom: success among episodes NOT lost to a non-force
        # task failure (wrong_button, …) — a residual without images/instruction cannot fix
        # those (lesson-19 measurement on T1: 55-70 % of base failures are wrong_button)
        "success_cond": k / max(1, k + n_force + outcomes.get("timeout", 0)),
        "peak_force_mean": float(np.mean([r["peak_force"] for r in rows])) if rows else 0.0,
        "return_mean": float(np.mean([r["return"] for r in rows])) if rows else 0.0,
        "outcomes": outcomes,
        "res_mm_contact": float(np.mean(mag["contact"])) if mag["contact"] else None,
        "res_mm_free": float(np.mean(mag["free"])) if mag["free"] else None,
        "n_steps_contact": len(mag["contact"]),
        "n_steps_free": len(mag["free"]),
        "episodes": rows,
    }
    if trace is not None:
        out["trace"] = trace
    return out


# ---------------------------------------------------------------------------------- the run
def run_residual_rl(
    base_ckpt,
    task: str = "t3",
    seed: int = 0,
    env_steps: int = 20_000,
    ctx=None,
    *,
    arm: str | None = None,
    n_envs: int = 4,
    warmup: int = 1_000,
    utd: float = 0.5,
    eval_every: int = 2_000,
    n_eval: int = 20,
    eval_offset: int = 0,
    bc_start: float = BC_START,
    bc_floor: float = BC_FLOOR,
    bc_decay: int | None = None,
    expl_noise: float = 0.2,
    step_cost: float = 0.01,
    peak_force_penalty: float = 0.0,
    fail_penalty: float = 0.0,
    hole_shaping: float = 0.0,
    residual_force: bool = False,
    hidden: int = 256,
    batch: int = 256,
    log_every: int = 200,
    device: str | None = None,
    zero_last_layer: bool = False,
    torch_threads: int | None = 1,
    dp_res: float = DP_RES,
    interventions: str = "off",
    intervention_budget: float = 0.2,
    intervention_decay: float = 0.0,
    corr_frac: float = 0.5,
    deploy_mode: bool = False,
    swap_every: int = 500,
) -> dict:
    """Train one residual on a frozen base for ``env_steps`` env transitions; one RL seed.

    ``seed`` fixes the residual's init, the exploration noise, the replay sampling and the
    training episode seeds (RL_SEED0 + seed * RL_SEED_BLOCK + k); the eval seeds are the same
    for every RL seed, so per-seed curves are comparable and their spread is the seed lottery.
    ``bc_decay`` defaults to env_steps // 2 (anneal over the first half, floor afterwards).
    ``torch_threads`` (default 1; None = leave torch's setting alone) is set process-wide: the
    base's flow outputs differ at ~1e-6 between 1 and 4 intra-op threads, enough to change a
    peak force in the 6th digit, so a reproducible seed needs a fixed count; and the step is
    render/physics-bound (base.act 48 ms at 1 thread vs 41 ms at 4), so 1 thread per process
    with seeds in parallel processes is the efficient layout.

    Returns::

        {"curves": {"env_step": [...], "success", "success_lo", "success_hi", "success_cond",
                    "force_abort",
                    "over_force", "peak_force_mean", "return_mean", "res_mm_contact",
                    "res_mm_free"},           # one point per eval; point 0 = the frozen base
         "train": [{"env_step", "ep_return_ma", "train_success_ma", "train_force_fail_ma",
                    "bc_weight",
                    "critic_loss", "q1_mean", "actor_loss", "bc_loss", "res_mm_ma"}, ...],
         "base_eval": eval dict (residual off), "final_eval": eval dict (incl. "trace"),
         "residual_trace": {"contact_mm", "free_mm", "trace"} of the final eval,
         "env_steps", "updates", "env_steps_per_s", "wall_s", "seed", "task", "base": info,
         "config": {...the knobs...}, "agent": the TD3 object}

    Simulated deployment (:mod:`nfv.rl.deploy`; all off by default): ``interventions``
    "off" | "expert" (the privileged T3 operator; T3 only), ``intervention_budget`` (max share
    of env steps the operator may take, default 0.2), ``intervention_decay`` (0 = constant
    budget, 1 = linearly to 0 by the end), ``corr_frac`` (share of each BC batch drawn from
    intervention samples), ``deploy_mode`` (act with a snapshot of the actor refreshed every
    ``swap_every`` updates). With either on, the train rows add ``robot_min`` (and with
    interventions ``intervention_rate_ma``, ``operator_steps``, ``operator_min``,
    ``corr_share``, ``n_segments``) and the dict adds ``"deploy"``: {"robot_min",
    "operator_steps", "operator_min", "operator_share", "first20_rate", "last20_rate",
    "curve": {"env_step", "rate"} (intervened share per bin of env steps), "episodes": [...],
    "swaps", "n_segments", "triggers"}.
    """
    if interventions not in ("off", "expert"):
        raise ValueError(f"interventions must be 'off' or 'expert', got {interventions!r}")
    op_on = interventions == "expert"
    if op_on and task != "t3":
        raise ValueError("the simulated operator (interventions='expert') exists for T3 only")
    if deploy_mode and int(swap_every) < 1:
        raise ValueError("swap_every must be >= 1")
    device = device or "cpu"
    if bc_decay is None:
        bc_decay = max(1, env_steps // 2)
    bc_weight_at(0, bc_decay, bc_start, bc_floor)  # validate the schedule up front
    if torch_threads:
        torch.set_num_threads(int(torch_threads))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    base, base_info = load_base(base_ckpt, task, arm, device)
    agent = make_agent(residual_force, hidden, batch, device, zero_last_layer)
    actor = ResidualActor(base, agent, residual_force, device, dp_res=dp_res)
    if op_on:  # flags per transition; BC toward corrections; critics on autonomous data only
        buf = DeployBuffer(agent.obs_dim, agent.act_dim, max(env_steps, 1))
        bc_view, critic_buf = CorrectionBCView(buf, corr_frac), CriticView(buf)
    else:
        buf = ReplayBuffer(agent.obs_dim, agent.act_dim, max(env_steps, 1))
        bc_view, critic_buf = ZeroActionView(buf), buf
    # the policy that ACTS in training: the learner itself, or (deploy mode) its snapshot
    snapshot = PolicySnapshot(agent) if deploy_mode else None
    train_actor = actor if snapshot is None else ResidualActor(
        base, snapshot, residual_force, device, dp_res=dp_res)  # fmt: skip
    kp = None
    reward_kw = {"step_cost": step_cost, "peak_force_penalty": peak_force_penalty,
                 "fail_penalty": fail_penalty, "hole_shaping": hole_shaping}  # fmt: skip
    config = {
        "task": task, "seed": seed, "env_steps": env_steps, "arm": arm, "n_envs": n_envs,
        "warmup": warmup, "utd": utd, "eval_every": eval_every, "n_eval": n_eval,
        "eval_offset": eval_offset, "bc_start": bc_start, "bc_floor": bc_floor,
        "bc_decay": bc_decay, "expl_noise": expl_noise, "dp_res_m": float(dp_res), "f_res_n": F_RES,
        "residual_force": residual_force, "hidden": hidden, "batch": batch, **reward_kw,
        "n_exec": actor.n_exec,
        "torch_threads": torch.get_num_threads(),
        "interventions": interventions, "intervention_budget": float(intervention_budget),
        "intervention_decay": float(intervention_decay), "corr_frac": float(corr_frac),
        "deploy_mode": bool(deploy_mode), "swap_every": int(swap_every),
    }  # fmt: skip
    ev_seeds = eval_seeds(task, n_eval, eval_offset)
    eval_env = Env(task=task, render=True)
    envs = [Env(task=task, render=True) for _ in range(max(1, n_envs))]
    slots = [_Slot(env=e, task=task) for e in envs]
    ep_counter = 0

    curves: dict[str, list] = {k: [] for k in _CURVE_KEYS}
    train_log: list[dict] = []

    def record_eval(step: int, ev: dict) -> None:
        curves["env_step"].append(step)
        curves["success"].append(ev["success"])
        curves["success_lo"].append(ev["success_ci95"][0])
        curves["success_hi"].append(ev["success_ci95"][1])
        for k in ("success_cond", "force_abort", "over_force", "peak_force_mean",
                  "return_mean", "res_mm_contact", "res_mm_free"):  # fmt: skip
            curves[k].append(ev[k])
        series = {f"eval_{k}": float(ev[k]) for k in (
            "success", "success_cond", "force_abort", "over_force", "peak_force_mean",
            "return_mean")
            if ev[k] is not None}  # fmt: skip
        for k in ("res_mm_contact", "res_mm_free"):
            if ev[k] is not None:
                series[f"eval_{k}"] = float(ev[k])
        _call(ctx, "log", step, **series)

    t_start = time.monotonic()
    base_eval = evaluate(actor, eval_env, task, ev_seeds, off=True, reward_kw=reward_kw)
    record_eval(0, base_eval)
    t_train0 = time.monotonic()

    def start_train(sl: _Slot) -> None:
        nonlocal ep_counter
        train_actor.start(sl, RL_SEED0 + seed * RL_SEED_BLOCK + ep_counter)
        if op_on:
            sl.op = PegOperator(kp)
            sl.last_idx, sl.ep_id, sl.ep_iv = -1, ep_counter, 0
        ep_counter += 1

    if op_on:
        kp = float(envs[0].controller.kp)
    for sl in slots:
        start_train(sl)
    rets: deque = deque(maxlen=MA_WINDOW)
    succ: deque = deque(maxlen=MA_WINDOW)
    ffail: deque = deque(maxlen=MA_WINDOW)  # force_abort / over_force among training episodes
    res_ma: deque = deque(maxlen=500)
    iv_ma: deque = deque(maxlen=MA_WINDOW)  # per-episode operator share (interventions only)
    iv_steps = np.zeros(max(env_steps, 1), bool)  # per env step: did the operator act?
    ep_log: list[dict] = []
    triggers = {"stall": 0, "force": 0}
    swap_log: list[int] = []
    op_steps = 0
    loss_acc: dict[str, list] = {}
    peak_prev = [0.0] * len(slots)
    budget, updates, t, last = 0.0, 0, 0, None
    best = -1.0
    while t < env_steps:
        if ctx is not None and _call(ctx, "should_continue") is False:
            break
        for si, sl in enumerate(slots):
            if t >= env_steps:
                break
            o = sl.feat
            iv = False
            if op_on:  # the operator watches the observation the robot is about to act on
                sl.op.observe(sl.env)
                why = "" if sl.op.active else sl.op.wants_in(sl.env)
                allowed = budget_at(t, env_steps, intervention_budget, intervention_decay)
                if why and op_steps <= allowed * t:
                    sl.op.begin(why)
                    triggers[why] += 1
                    buf.mark_boundary(sl.last_idx)  # no TD target may cross into the take-over
                iv = sl.op.active
            if iv:
                # ---- snippet: the operator's correction replaces the actor's residual ----
                base_dp = train_actor._base_next(sl, replan=False)[0]  # [3] m, base's next dp
                r = correction(sl.op.operator_dp(sl.env), base_dp, train_actor.dp_res)  # [A]
            else:
                r = train_actor.residual(o, expl_noise, rng)
            obs, info, _ = train_actor.step(sl, r)
            if iv:
                sl.op.tick()
                sl.ep_iv += 1
                op_steps += 1
                iv_steps[t] = True
            rr, terminal = reward(info, peak_prev[si], **reward_kw)
            peak_prev[si] = float(info["peak_force"])
            sl.ep_ret += rr
            if not iv:
                res_ma.append(_residual_mm(r, actor.dp_res))
            t += 1
            extra = {"intervened": iv, "ep_id": sl.ep_id, "ep_t": sl.ep_len - 1} if op_on else {}
            if info["done"]:
                # o2 for the (timeout) bootstrap: the base's plan continuation, no re-plan
                o2 = train_actor._features(sl, replan=False)
                buf.add(o, r, rr, o2, terminal, **extra)
                rets.append(sl.ep_ret)
                succ.append(float(info["success"]))
                ffail.append(float(info["force_abort"] or info["outcome"] == "over_force"))
                if op_on or deploy_mode:
                    ep_log.append({"env_step": t, "len": sl.ep_len, "operator_steps": sl.ep_iv,
                                   "outcome": info["outcome"] or "timeout"})  # fmt: skip
                    iv_ma.append(sl.ep_iv / max(sl.ep_len, 1))
                peak_prev[si] = 0.0
                start_train(sl)
            else:
                sl.feat = train_actor._features(sl)
                idx = buf.add(o, r, rr, sl.feat, terminal, **extra)
                if op_on:
                    sl.last_idx = idx
            if t > warmup and buf.n >= batch:
                budget += utd
                while budget >= 1.0:
                    budget -= 1.0
                    w = bc_weight_at(t, bc_decay, bc_start, bc_floor)
                    out = agent.update(critic_buf, rng, demo=bc_view, bc_weight=w)
                    updates += 1
                    if snapshot is not None and updates % int(swap_every) == 0:
                        snapshot.refresh()  # deploy mode: the robot gets the new weights now
                        swap_log.append(updates)
                    for k, v in out.items():
                        loss_acc.setdefault(k, []).append(v)
            if t % log_every == 0:
                row = {
                    "env_step": t,
                    "ep_return_ma": float(np.mean(rets)) if rets else None,
                    "train_success_ma": float(np.mean(succ)) if succ else None,
                    "train_force_fail_ma": float(np.mean(ffail)) if ffail else None,
                    "bc_weight": bc_weight_at(t, bc_decay, bc_start, bc_floor),
                    "res_mm_ma": float(np.mean(res_ma)) if res_ma else 0.0,
                    "updates": updates,
                    "env_steps_per_s": round(t / max(time.monotonic() - t_train0, 1e-9), 2),
                }
                if op_on or deploy_mode:
                    row["robot_min"] = round(robot_minutes(t), 3)
                    row["swaps"] = len(swap_log)
                if op_on:
                    row.update({
                        "intervention_rate_ma": float(np.mean(iv_ma)) if iv_ma else None,
                        "operator_steps": op_steps,
                        "operator_min": round(robot_minutes(op_steps), 3),
                        "corr_share": float(buf.intervened[: buf.n].mean()) if buf.n else 0.0,
                        "n_segments": sum(triggers.values()),
                    })  # fmt: skip
                row.update({k: float(np.mean(v)) for k, v in loss_acc.items()})
                loss_acc = {}
                train_log.append(row)
                _call(ctx, "log", t, **{k: v for k, v in row.items()
                                        if k != "env_step" and v is not None})  # fmt: skip
                _call(ctx, "set_status", step=t, total_steps=env_steps)
            if eval_every and t % eval_every == 0 and t < env_steps:
                ev = evaluate(actor, eval_env, task, ev_seeds, reward_kw=reward_kw)
                record_eval(t, ev)
                last = ev
                if ev["success"] > best and ctx is not None:
                    best = ev["success"]
                    _save(ctx, agent, base, base_info, config, t, ev)
    train_wall = time.monotonic() - t_train0
    final = evaluate(actor, eval_env, task, ev_seeds, trace_first=True, reward_kw=reward_kw)
    record_eval(t, final)
    if ctx is not None:
        _save(ctx, agent, base, base_info, config, t, final)
    _call(ctx, "set_status", step=t, total_steps=env_steps)
    for e in (*envs, eval_env):
        e.close()
    del last
    out = {}
    if op_on or deploy_mode:
        out["deploy"] = _deploy_summary(t, iv_steps[:t], op_steps, ep_log, swap_log, triggers,
                                        op_on)  # fmt: skip
    return {
        **out,
        "curves": curves,
        "train": train_log,
        "base_eval": base_eval,
        "final_eval": final,
        "residual_trace": {
            "contact_mm": final["res_mm_contact"],
            "free_mm": final["res_mm_free"],
            "trace": final.get("trace"),
        },
        "env_steps": t,
        "updates": updates,
        "env_steps_per_s": round(t / max(train_wall, 1e-9), 3),
        "wall_s": round(time.monotonic() - t_start, 2),
        "seed": seed,
        "task": task,
        "base": base_info,
        "config": config,
        "agent": agent,
    }


DEPLOY_BINS = 40  # intervention-rate curve: env steps split into this many equal bins
EDGE_FRAC = 0.2  # "first / last 20 % of training" for m4.autonomy_rises


def _deploy_summary(t, iv_steps, op_steps, ep_log, swap_log, triggers, op_on) -> dict:
    """Robot time, operator cost and the intervention-rate curve of one training run.

    Rates are STEP-weighted (operator steps / env steps inside a window), so a long autonomous
    timeout counts as 200 steps of autonomy, not as one episode.
    """
    iv = np.asarray(iv_steps, bool)
    edges = np.linspace(0, t, DEPLOY_BINS + 1).round().astype(int) if t else np.zeros(2, int)
    rate = [float(iv[a:b].mean()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:],
                                                                    strict=True)]  # fmt: skip
    k = max(1, int(round(EDGE_FRAC * t)))
    return {
        "robot_min": round(robot_minutes(t), 3),
        "operator_steps": int(op_steps),
        "operator_min": round(robot_minutes(op_steps), 3),
        "operator_share": float(op_steps / max(t, 1)),
        "first20_rate": float(iv[:k].mean()) if op_on and t else None,
        "last20_rate": float(iv[-k:].mean()) if op_on and t else None,
        "curve": {"env_step": [int(b) for b in edges[1:]], "rate": rate},
        "n_segments": int(sum(triggers.values())),
        "triggers": dict(triggers),
        "swaps": list(swap_log),
        "episodes": ep_log,
    }


def _save(ctx, agent, base, base_info, config, step, ev) -> None:
    """Residual checkpoint = TD3.state(meta) + the frozen base embedded (standalone, task-13)."""
    meta = {"algo": "residual-td3bc", "config": config, "base_info": base_info,
            "params": getattr(ctx, "params", {}),
            "eval": {k: ev[k] for k in _CKPT_EVAL_KEYS}}  # fmt: skip
    payload = agent.state(meta)
    payload["base"] = base_part(base)
    payload["arch"] = {"kind": "nfv_residual", "dp_res_m": config["dp_res_m"], "f_res_n": F_RES,
                       "residual_force": config["residual_force"]}  # fmt: skip
    _call(ctx, "save_ckpt", step, payload, metrics={"eval_success": ev["success"]})
