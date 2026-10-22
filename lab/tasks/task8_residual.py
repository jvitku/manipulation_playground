"""Task 8 - "Residual RL on a flow policy" (id `residual-rl`): TD3 learns a correction on top of a
FROZEN task-3 flow policy (the contact2d T3 peg search).

The base is a `flow-bc` checkpoint picked by the `base_ckpt` param (kind "ckpt", source task
"flow-bc"; the server resolves the id to a file path before the run starts). The acting policy is

    a = clip(a_base + residual_scale * delta),    delta = tanh(actor([obs_norm, a_base]))

with `a_base` the base flow policy's action (chunked exactly as in task 3, K sampling steps) and
`actor` a plain TD3 actor (`fvb.policy.td3`; twin critics, delayed actor, target smoothing; no
demonstrations, no BC term). The base never receives a gradient; the actor starts at delta ~ 0, so
training begins AT the base policy. The residual actor and the critics see the base's normalised
22-D observation plus the base action (24-D); the transition action is `delta` (not the sum).

Reward (own definition on `fvb.ta_tutorial.contact2d`, structured like the arm task's
`fvb.policy.rl_env.ArmRLEnv.step` that task 6 trains on: depth progress + potential shaping toward
the true hole, time cost, force penalty, terminal bonus/penalty). Per live step:
    r = dd/(WC+INSERT_DEPTH) + [shaping*(-|x-hole|/0.08)' - shaping*(-|x-hole|/0.08)]
        - time_cost - force_penalty * max(0, |F| - F_ABORT/2)/F_ABORT*10  (+10 success, -5 abort)
with dd the change of depth below the local surface level z0 (privileged, reward only). A timeout
bootstraps (the observation is not time-aware); success/abort are terminal.

Metrics (`metrics.jsonl`, step = environment transitions): per finished training round
{ep_return, ep_return_ma, train_success_ma, delta_abs_ma}; every LOSS_EVERY updates {critic_loss,
q1_mean, target_mean, actor_loss}; step 0 {base_success}; every eval {eval_success,
eval_force_abort, eval_return, base_success (flat reference), eval_s}. `base_success` is the FROZEN
base's success on the same fixed eval scenes (EVAL_SEED0, `eval_episodes`, est_sigma 0.03, sensor
noise 0.05), measured once at the start.

Checkpoint (`ckpt/`, best-by-eval + newest, as task 6): `TD3.state(meta)` = {"actor", "critic",
"cfg", "obs_dim", "act_dim", "priv_dim", "meta"} plus
    "base":  {"ema", "norm", "arch", "params"}  the task-3 part EMBEDDED (so the residual ckpt is
             standalone: deleting the task-3 run does not break it; ~5 MB at hidden 512 x depth 4),
    "residual_scale", "arch" ({"kind": "residual", "base": base arch}).
meta = {algo "residual-td3", args, reward, td3, base_success, base_ckpt (source path, provenance),
t, eval}.

result.json written by `evaluate` (task 3's shape, `contact2d` viewer) plus
    "residual_off", "residual_scale", "base_success" (the training-time reference, null if absent),
    episodes[i]["act"] = the applied (summed, clipped) action, ["base_act"] = the base's action,
    ["delta"] = the applied residual scale*delta, each 3 decimals, len = len(pos)-1.
With `residual_off` the residual is zero, so act == base_act: the base alone, for comparison.
"""

from __future__ import annotations

import time
from collections import Counter, deque

import numpy as np
import torch

from fvb.ta_tutorial import contact2d as c2d
from lab.core.runctx import atomic_write_json
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.task1_bc_mse import OVERHEAD_S, _load_part, wilson95
from lab.tasks.task3_flow_bc import MAX_VIEW_EPISODES, OUTCOMES, _obs_tensor, make_gen
from lab.tasks.task6_td3 import _save

N_ENVS = 16  # parallel scenes per training round
UTD = 0.5  # gradient updates per environment transition
LOSS_EVERY, MA_RETURN, MA_SUCCESS = 200, 20, 100
EVAL_SEED0, EVAL_EPISODES_MAX = 30_000, 400  # disjoint from task 3's eval (20 000+) and training
SUCCESS_BONUS, ABORT_PENALTY, LAT_SCALE = 10.0, 5.0, 0.08
DEPTH_SCALE = c2d.WC + c2d.INSERT_DEPTH
FALLBACK_STEPS_PER_S, FALLBACK_EVAL_S = 150.0, 3.0
EST_SIGMA, SENSOR_NOISE = 0.03, 0.05


def _dev() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def base_part(ck: dict) -> dict:
    """The part of a task-3 checkpoint the residual run needs (EMA weights, norm, arch)."""
    if not isinstance(ck, dict) or not {"ema", "norm", "arch"} <= set(ck):
        raise ValueError("not a flow-bc checkpoint (needs ema, norm, arch)")
    if ck["arch"].get("obs_dim") != 2 + 2 * c2d.HIST or ck["arch"].get("act_dim") != 2:
        raise ValueError("not a task-3 (contact2d T3) policy")
    cpu = lambda t: t.detach().cpu().clone()  # noqa: E731
    return {
        "ema": {k: cpu(v) for k, v in ck["ema"].items()},
        "norm": {k: (cpu(m), cpu(s)) for k, (m, s) in ck["norm"].items()},
        "arch": dict(ck["arch"]),
        "params": dict(ck.get("params") or {}),
    }


class _Base:
    """The frozen task-3 flow policy, rebuilt from the embedded part."""

    def __init__(self, part: dict, dev: str, K: int | None = None):
        self.model, self.ds, self.arch = _load_part({"b": part}, "b", dev)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dev = dev
        self.use_force = bool(self.arch.get("use_force", True))
        self.n_exec = max(1, min(int(self.arch.get("n_exec", 8)), self.arch["horizon"]))
        self.K = int(K or part.get("params", {}).get("sample_steps") or 8)
        self.mean, self.std = (t.to(dev) for t in part["norm"]["obs"])

    def policy(self, seed: int):
        from fvb.ta_tutorial.train import ChunkPolicy

        return ChunkPolicy(
            self.model, make_gen(), self.ds,
            lambda o: {"obs": _obs_tensor(o, self.dev, self.use_force)},
            self.n_exec, self.K, seed=seed,
        )  # fmt: skip

    def feats(self, raw: dict, a_base: np.ndarray) -> np.ndarray:
        """(n, 24): the base's normalised observation + the base action."""
        o = (_obs_tensor(raw, self.dev, self.use_force) - self.mean) / self.std
        return np.concatenate([o.cpu().numpy(), a_base], 1).astype(np.float32)


def _make_actor_agent(obs_dim: int, device: str = "cpu"):
    from fvb.policy.td3 import TD3, TD3Config

    cfg = TD3Config()
    agent = TD3(obs_dim, 2, cfg, device)
    with torch.no_grad():  # start AT the base policy: delta ~ 0
        agent.actor.net[-1].weight.mul_(0.01)
        agent.actor.net[-1].bias.zero_()
    agent.actor_t.load_state_dict(agent.actor.state_dict())
    return agent, cfg


@torch.no_grad()
def _delta(actor, feats: np.ndarray) -> np.ndarray:
    return actor(torch.as_tensor(feats)).numpy()


def _rollout(base: _Base, actor, scale, n, seed, est_sigma=EST_SIGMA, sensor_noise=SENSOR_NOISE,
             record=0, ret_params=None):  # fmt: skip
    """Closed-loop rollouts of base (+ residual when `actor` is given) on `n` fresh scenes.
    Returns (env, rec) with rec = per-episode {pos, force, act, base_act, delta} for `record`."""
    env = c2d.t3_scenes(n, seed, est_sigma)
    env.noise = sensor_noise
    pol = base.policy(seed)
    record = min(record, n)
    rec = [{"pos": [env.pos[i].tolist()], "force": [[0.0, 0.0]], "act": [], "base_act": [],
            "delta": []} for i in range(record)]  # fmt: skip
    ret = np.zeros(n)
    rew = _Reward(env, **(ret_params or {}))
    while not env.done.all():
        live = ~env.done.copy()
        raw = c2d.t3_obs(env)
        ab = pol.act(raw)
        d = scale * _delta(actor, base.feats(raw, ab)) if actor is not None else np.zeros_like(ab)
        a = np.clip(ab + d, -1, 1)
        env.step(a)
        ret += rew.step(live)
        for i in range(record):
            if live[i]:
                r = rec[i]
                r["act"].append(a[i].tolist())
                r["base_act"].append(np.asarray(ab[i], float).tolist())
                r["delta"].append(d[i].tolist())
                r["pos"].append(env.pos[i].tolist())
                r["force"].append(env.hist[i, -1].tolist())
    return env, rec, ret


class _Reward:
    """Per-step reward from the env's own signals (see the module docstring)."""

    def __init__(self, env, shaping=1.0, time_cost=0.01, force_penalty=0.05, gamma=0.99):
        self.env, self.shaping, self.time_cost = env, shaping, time_cost
        self.force_penalty, self.gamma = force_penalty, gamma
        self.depth = self._depth()
        self.phi = self._phi()

    def _depth(self):
        return np.maximum(self.env.z0 - self.env.pos[:, 1], 0.0)

    def _phi(self):
        return -self.shaping * np.abs(self.env.pos[:, 0] - self.env.holes[:, 0]) / LAT_SCALE

    def step(self, live: np.ndarray) -> np.ndarray:
        """Reward of the step just taken by the episodes that were `live` before it."""
        e = self.env
        depth, phi = self._depth(), self._phi()
        f = np.linalg.norm(e.force, axis=1)
        over = np.maximum(0.0, f - 0.5 * c2d.F_ABORT) / c2d.F_ABORT * 10.0
        r = (depth - self.depth) / DEPTH_SCALE + self.gamma * phi - self.phi
        r = r - self.time_cost - self.force_penalty * over
        r = (
            r
            + SUCCESS_BONUS * (e.outcome == "success")
            - ABORT_PENALTY * (e.outcome == "force_abort")
        )
        self.depth, self.phi = depth, phi
        return np.where(live, r, 0.0)


def _eval(base, actor, scale, n, seed0=EVAL_SEED0):
    """Success / force-abort share and mean return of the policy on the fixed eval scenes."""
    env, _, ret = _rollout(base, actor, scale, n, seed0)
    return {"success": float(env.success.mean()),
            "force_abort": float((env.outcome == "force_abort").mean()),
            "return": float(ret.mean())}  # fmt: skip


class _AgentView:
    """`task6._save` calls `agent.state(meta)`; this adds the embedded base to the payload."""

    def __init__(self, agent, extra):
        self.agent, self.extra = agent, extra

    def state(self, meta):
        return {**self.agent.state(meta), **self.extra}


def train(ctx) -> None:
    P = ctx.params
    path = str(P.get("base_ckpt") or "")
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        part = base_part(ck)
    except (OSError, ValueError, RuntimeError, EOFError) as e:
        ctx.fail(f"base checkpoint unusable ({path or 'none given'}): {e}")
        return
    torch.set_num_threads(2)
    steps, eval_every = int(P["env_steps"]), int(P["eval_every"])
    n_eval = min(int(P.get("eval_episodes", 100)), EVAL_EPISODES_MAX)
    seed, expl, scale = int(P.get("seed", 0)), float(P["expl_noise"]), float(P["residual_scale"])
    rwd = {"shaping": float(P["shaping_weight"]), "time_cost": float(P["time_cost"]),
           "force_penalty": float(P["force_penalty"])}  # fmt: skip
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    base = _Base(part, _dev())
    ctx.set_status(step=0, total_steps=steps, message="measuring the frozen base policy")
    b = _eval(base, None, scale, n_eval)
    base_success = b["success"]
    ctx.log(0, base_success=base_success)

    agent, cfg = _make_actor_agent(2 + 2 * c2d.HIST + 2)
    from fvb.policy.td3 import ReplayBuffer

    buf = ReplayBuffer(agent.obs_dim, 2, min(1_000_000, max(steps, 1000)))
    learn_start = min(1000, steps // 5)
    meta = {"algo": "residual-td3", "args": dict(P), "reward": rwd, "td3": vars(cfg).copy(),
            "base_success": base_success, "base_ckpt": path,
            "residual_scale": scale}  # fmt: skip
    extra = {
        "base": part,
        "residual_scale": scale,
        "arch": {"kind": "residual", "base": part["arch"]},
    }
    view = _AgentView(agent, extra)
    ctx.set_status(message=f"TD3 residual on the frozen base (base success {base_success:.2f})")

    rets: deque = deque(maxlen=MA_RETURN)
    succ: deque = deque(maxlen=MA_SUCCESS)
    dabs: deque = deque(maxlen=MA_RETURN)
    acc: dict[str, list[float]] = {}
    best = (-1.0, -np.inf)
    t = n_upd = rnd = 0
    saved_t = -1
    next_eval = eval_every
    owed = 0.0

    def flush(at):
        nonlocal acc
        if acc:
            ctx.log(at, **{k: round(float(np.mean(v)), 4) for k, v in acc.items()})
            acc = {}

    while t < steps:
        env = c2d.t3_scenes(N_ENVS, 1_000_000 * (seed + 1) + rnd, EST_SIGMA)
        env.noise = SENSOR_NOISE
        rnd += 1
        pol = base.policy(seed * 7919 + rnd)
        rew = _Reward(env, **rwd)
        raw = c2d.t3_obs(env)
        ab = pol.act(raw)
        o = base.feats(raw, ab)
        ep_ret, ep_d, ep_n = np.zeros(N_ENVS), np.zeros(N_ENVS), np.zeros(N_ENVS)
        while not env.done.all() and t < steps:
            if not ctx.should_continue():
                flush(t)
                if t > 0 and saved_t != t:
                    _save(ctx, view, {**meta, "t": t}, t, "last")
                ctx.set_status(state="stopped")
                return
            live = ~env.done.copy()
            d = np.clip(_delta(agent.actor, o) + rng.normal(0, expl, (N_ENVS, 2)), -1, 1)
            env.step(np.clip(ab + scale * d, -1, 1))
            r = rew.step(live)
            raw = c2d.t3_obs(env)
            ab = pol.act(raw)
            o2 = base.feats(raw, ab)
            term = np.isin(env.outcome, ("success", "force_abort"))
            for i in np.flatnonzero(live):
                buf.add(o[i], d[i].astype(np.float32), float(r[i]), o2[i], bool(term[i]))
            ep_ret += r
            ep_d += live * np.abs(d).mean(1)
            ep_n += live
            o = o2
            t += int(live.sum())
            if t >= learn_start:
                owed += UTD * int(live.sum())
                while owed >= 1.0:
                    owed -= 1.0
                    for k, v in agent.update(buf, rng).items():
                        acc.setdefault(k, []).append(v)
                    n_upd += 1
                    if n_upd % LOSS_EVERY == 0:
                        flush(t)
            if t % 100 < live.sum():
                ctx.set_status(step=min(t, steps))
            if t >= next_eval or t >= steps:
                next_eval += eval_every
                flush(t)
                t0 = time.perf_counter()
                agent.actor.eval()
                ev = _eval(base, agent.actor, scale, n_eval)
                ctx.log(t, eval_success=ev["success"], eval_force_abort=ev["force_abort"],
                        eval_return=round(ev["return"], 3), base_success=base_success,
                        eval_s=round(time.perf_counter() - t0, 2))  # fmt: skip
                key = (ev["success"], ev["return"])
                improved = key > best
                best = max(best, key)
                _save(
                    ctx,
                    view,
                    {**meta, "t": t, "eval": ev},
                    t,
                    "best" if improved else "last",
                    {"eval_success": ev["success"], "base_success": base_success},
                )
                saved_t = t
                ctx.set_status(step=min(t, steps))
        for i in range(N_ENVS):  # finished episodes of this round
            if env.done[i]:
                rets.append(float(ep_ret[i]))
                succ.append(bool(env.success[i]))
                dabs.append(float(ep_d[i] / max(1.0, ep_n[i])))
        if rets:
            ctx.log(t, ep_return=round(float(np.mean(rets)), 3),
                    ep_return_ma=round(float(np.mean(rets)), 3),
                    train_success_ma=round(float(np.mean(succ)), 3),
                    delta_abs_ma=round(float(np.mean(dabs)), 4))  # fmt: skip
    flush(t)
    ctx.done(f"trained {steps} steps, best eval success {best[0]:.2f} (base {base_success:.2f})")


def evaluate(ctx) -> None:
    P = ctx.params
    n = int(P.get("n_episodes", 400))
    seed = int(P.get("eval_seed", 0))
    est_sigma = float(P.get("est_sigma", EST_SIGMA))
    sensor_noise = float(P.get("sensor_noise", SENSOR_NOISE))
    off = bool(P.get("residual_off", False))
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    scale = float(ck["residual_scale"])
    K = int(P.get("eval_steps") or ck["base"].get("params", {}).get("sample_steps") or 8)
    dev = _dev()
    base = _Base(ck["base"], dev, K)
    actor = None
    if not off:
        agent, _ = _make_actor_agent(ck["obs_dim"])
        agent.actor.load_state_dict(ck["actor"])
        actor = agent.actor.eval()
    env, rec, _ = _rollout(base, actor, scale, n, 20_000 + seed, est_sigma, sensor_noise,
                           MAX_VIEW_EPISODES)  # fmt: skip
    k = int(env.success.sum())
    cnt = Counter(env.outcome.tolist())
    r4 = lambda v: np.round(np.asarray(v, float), 4).tolist()  # noqa: E731
    r3 = lambda v: np.round(np.asarray(v, float), 3).tolist()  # noqa: E731
    episodes = [
        {
            "z0": r4(env.z0[i]), "holes": r4(env.holes[i]), "target": int(env.target[i]),
            "est": r4(env.est[i]), "pos": r4(rec[i]["pos"]), "force": r4(rec[i]["force"]),
            "act": r3(rec[i]["act"]), "base_act": r3(rec[i]["base_act"]),
            "delta": r3(rec[i]["delta"]), "outcome": str(env.outcome[i]),
        }
        for i in range(min(n, MAX_VIEW_EPISODES))
    ]  # fmt: skip
    atomic_write_json(
        ctx.dir / "result.json",
        {
            "n_episodes": n, "success": k / n, "counts": {o: int(cnt.get(o, 0)) for o in OUTCOMES},
            "wilson95": wilson95(k, n), "force_zeroed": False,
            "trained_with_force": bool(base.arch.get("use_force", True)), "sample_steps": K,
            "est_sigma": est_sigma, "sensor_noise": sensor_noise, "residual_off": off,
            "residual_scale": scale, "base_success": (ck.get("meta") or {}).get("base_success"),
            "episodes": episodes,
        },
    )  # fmt: skip
    ctx.done("evaluated")


def estimate(params: dict) -> float:
    cal = registry.calib("residual-rl")
    steps = int(params["env_steps"])
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    eval_s = cal.get("eval_s") or FALLBACK_EVAL_S
    n_eval = int(params.get("eval_episodes", 100)) / 100
    n_evals = steps // max(1, int(params["eval_every"])) + 2  # + the base measurement
    return steps / sps + n_evals * eval_s * n_eval + cal.get("overhead_s", OVERHEAD_S)


SPEC = TaskSpec(
    id="residual-rl",
    order=9,
    title="Residual RL on a flow policy",
    subtitle="Freeze a flow-matching policy and let TD3 learn only a small correction on top.",
    params=[
        Param("base_ckpt", "Base policy (a Flow-BC checkpoint)", "ckpt", "", ckpt_task="flow-bc",
              help="the frozen policy the residual is added to; train one in the Flow-BC task"),
        Param("env_steps", "Environment steps", "int", 60_000, min=2000, max=300_000, step=2000,
              long=True, help="environment transitions; the base is never updated"),
        Param("residual_scale", "Residual scale", "float", 0.3, min=0.05, max=1.0, step=0.05,
              help="action = clip(base + scale x delta): how far the residual may push"),
        Param("seed", "Seed", "int", 0, min=0, max=99),
        Param("expl_noise", "Exploration noise", "float", 0.1, min=0.0, max=0.5, step=0.05,
              help="Gaussian noise on the residual's output delta in [-1, 1]"),
        Param("force_penalty", "Force penalty", "float", 0.05, min=0.0, max=0.5, step=0.01,
              help="reward minus this x (excess force above half the abort level)"),
        Param("time_cost", "Time cost", "float", 0.01, min=0.0, max=0.1, step=0.005,
              help="reward minus this every step"),
        Param("shaping_weight", "Hole-shaping weight", "float", 1.0, min=0.0, max=5.0, step=0.5,
              help="potential-based shaping toward the true hole (the policy never sees it)"),
        Param("eval_every", "Evaluate every (steps)", "int", 5000, min=500, max=50_000, step=500),
        Param("eval_episodes", "Scenes per evaluation", "int", 100, min=20, max=400, step=20,
              help="the same fixed scenes also give the frozen base's success"),
    ],
    eval_params=[
        Param("n_episodes", "Episodes", "int", 400, min=50, max=2000, step=50),
        Param("residual_off", "Residual off (base alone)", "bool", False,
              help="evaluate the frozen base without the correction, for comparison"),
        Param("est_sigma", "Hole-estimate error (sigma)", "float", 0.03, min=0.01, max=0.06,
              step=0.01),
        Param("sensor_noise", "Force sensor noise", "float", 0.05, min=0.0, max=0.2, step=0.01),
        Param("eval_steps", "Base sampling steps K", "choice", 8, choices=[1, 2, 4, 8, 16, 32]),
        Param("eval_seed", "Eval seed", "int", 0, min=0, max=9999),
    ],
    charts=[
        {"id": "reward", "title": "Training reward (rolling mean of 20 episodes)",
         "series": [{"key": "ep_return_ma", "label": "episode return"}], "log_y": False},
        {"id": "success", "title": "Success on the fixed eval scenes vs the frozen base",
         "series": [{"key": "eval_success", "label": "base + residual"},
                    {"key": "base_success", "label": "frozen base (reference)"},
                    {"key": "train_success_ma", "label": "train (last 100 episodes)"}],
         "log_y": False},
        {"id": "force", "title": "Eval force aborts",
         "series": [{"key": "eval_force_abort", "label": "force abort"}], "log_y": False},
        {"id": "critic", "title": "Critic loss (both Q heads)",
         "series": [{"key": "critic_loss", "label": "critic loss"}], "log_y": True},
        {"id": "q", "title": "Q estimate vs TD target (diverging = unhealthy)",
         "series": [{"key": "q1_mean", "label": "q1 mean"},
                    {"key": "target_mean", "label": "target mean"}], "log_y": False},
        {"id": "actor", "title": "Residual actor loss (-Q)",
         "series": [{"key": "actor_loss", "label": "actor loss"}], "log_y": False},
        {"id": "delta", "title": "Residual size (mean |delta| per step, before scaling)",
         "series": [{"key": "delta_abs_ma", "label": "|delta|"}], "log_y": False},
    ],
    viewer="contact2d",
    vecgrids=True,
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="residual-rl.html",
)  # fmt: skip
registry.register(SPEC)
