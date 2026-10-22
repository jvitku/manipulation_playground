"""Task 1 - "Behaviour cloning, the floor" (id `bc-mse`).

One run trains two regression (MSE) policies sequentially: T1 (two-mode obstacle, one-step
"episodes", horizon 1) then T2 (closed-loop point mass around an obstacle, chunk `horizon`).
Metrics are prefixed `t1_` / `t2_`.

Checkpoint payload (`last.pt`; later tasks copy this pattern):
    {"t1": P1, "t2": P2,
     "arch": {"t1": arch1, "t2": arch2}, "gen": "mse", "params": <train params>}
with each Pn = {"model": sd, "ema": sd, "norm": {"obs": (mean, std)}, "arch": arch_n,
"params": ..., "gen": "mse"} (the dict `lab.core.loop.train_loop` writes) and
arch_n = {"act_dim", "horizon", "obs_dim", "hidden", "depth"}. Mid-T1 checkpoints hold only "t1".
Evaluation rebuilds every model from `arch`, never from the current form params.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from fvb.ta_tutorial import toy
from fvb.ta_tutorial.data import ChunkDataset, split_episodes
from fvb.ta_tutorial.generative import Regression
from fvb.ta_tutorial.nets import MLPDenoiser
from fvb.ta_tutorial.train import ChunkPolicy
from lab.core.loop import train_loop
from lab.core.runctx import atomic_write_json
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec

HIDDEN, DEPTH = 512, 4
MAX_POINTS, MAX_VIEW_EPISODES = 500, 24
# Fallbacks used when lab/tasks/calib.json has no "bc-mse" entry yet (conservative CPU-ish guesses).
FALLBACK_STEPS_PER_S, FALLBACK_COLLECT_S_PER_EP, OVERHEAD_S = 100.0, 0.05, 15.0


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _arch(horizon: int, obs_dim: int) -> dict:
    return {"act_dim": 2, "horizon": horizon, "obs_dim": obs_dim, "hidden": HIDDEN, "depth": DEPTH}


def _build(arch: dict) -> MLPDenoiser:
    return MLPDenoiser(
        arch["act_dim"], arch["horizon"], arch["obs_dim"], arch["hidden"], arch["depth"]
    )


class _Norm:
    """Minimal stand-in for ChunkDataset that `ChunkPolicy` needs at evaluation time."""

    def __init__(self, horizon, norm, device):
        self.H, self.norm, self.device = horizon, norm, device
        self.t = {"x0": torch.zeros(1, horizon, 2)}

    def cond(self, raw):
        out = {}
        for k, v in raw.items():
            if k in self.norm:
                m, s = self.norm[k]
                out[k] = (v.float() - m.to(v.device)) / s.to(v.device)
            else:
                out[k] = v
        return out


def _collect(n: int, seed: int, noise: float) -> list[dict]:
    """Like `toy.t2_collect` (must track it) but with the expert noise as a parameter."""
    env = toy.ObstacleReach(n, seed)
    ex = toy.ObstacleExpert(env, seed, noise)
    obs, act, alive = [], [], []
    while not env.done.all():
        alive.append(~env.done)
        obs.append(env.obs())
        a = ex.act(env)
        act.append(a)
        env.step(a)
    obs, act, alive = np.stack(obs, 1), np.stack(act, 1), np.stack(alive, 1)
    return [{"obs": obs[i, alive[i]], "action": act[i, alive[i]]} for i in range(n)]


def _t1_eval(
    model, ds_like, gen, n: int, grid: bool, seed: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    if grid:
        c = np.linspace(-0.4, 0.4, n).astype(np.float32)
    else:
        c = np.random.default_rng(seed).uniform(-0.4, 0.4, n).astype(np.float32)
    dev = ds_like.device
    cond = ds_like.cond({"obs": torch.as_tensor(c[:, None], device=dev)})
    g = torch.Generator(device=dev).manual_seed(seed)
    with torch.no_grad():
        a = gen.sample(model.eval(), (n, 1, 2), cond, 1, g).clamp(-1, 1)[:, 0].cpu().numpy()
    return toy.t1_metrics(c, a), c, a


def _run_t2(model, ds_like, gen, n, seed, n_exec, radius=None, record=0):
    env = toy.ObstacleReach(n, seed)
    if radius is not None:
        env.RADIUS = radius
    pol = ChunkPolicy(model, gen, ds_like, lambda o: {"obs": o}, n_exec, 1, seed=seed)
    paths = [[env.p[i].tolist()] for i in range(record)]
    while not env.done.all():
        live = ~env.done.copy()
        a = pol.act(torch.as_tensor(env.obs(), device=ds_like.device))
        env.step(a)
        for i in range(record):
            if live[i]:
                paths[i].append(env.p[i].tolist())
    return env, paths


def wilson95(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def train(ctx) -> None:
    P = ctx.params
    steps, demos = int(P["steps"]), int(P["demos"])
    horizon, noise, seed = int(P["horizon"]), float(P["expert_noise"]), int(P.get("seed", 0))
    batch, lr = int(P["batch"]), float(P["lr"])
    eval_every = int(P.get("eval_every", max(1, steps // 8)))
    ckpt_every = int(P.get("ckpt_every", max(1, steps // 4)))
    warmup = max(1, min(500, steps // 10))
    dev = _device()
    gen = Regression()
    total = 2 * steps
    common = dict(
        steps=steps, batch=batch, lr=lr, warmup=warmup, eval_every=eval_every,
        ckpt_every=ckpt_every, log_every=max(1, min(50, steps // 4)), seed=seed,
    )  # fmt: skip

    # ---- part 1: two-mode obstacle, one-step episodes
    c, a = toy.t1_data(demos, np.random.default_rng(seed), noise)
    eps = [{"obs": c[i : i + 1], "action": a[i : i + 1]} for i in range(len(c))]
    tr, va = split_episodes(eps, 0.1, seed)
    ds = ChunkDataset(tr, 1, device=dev)
    vds = ChunkDataset(va, 1, device=dev, norm=ds.norm)
    arch1 = _arch(1, 1)
    m1 = _build(arch1)
    n_eval = 200

    def t1_eval(model):
        m, _, _ = _t1_eval(model, ds, gen, n_eval, True, seed)
        return {"t1_collision": m["collision"], "t1_mode_err": m["mode_err"]}

    holder: dict = {}

    def wrap1(p):
        holder["t1"] = p  # the last call is the final T1 save
        return {"t1": p, "arch": {"t1": p["arch"]}, "gen": p["gen"], "params": ctx.params}

    done1 = train_loop(
        ctx, m1, gen, ds, vds, prefix="t1_", total_steps=total, eval_fn=t1_eval,
        arch=arch1, payload_wrap=wrap1, **common,
    )  # fmt: skip
    if not done1:
        ctx.set_status(state="stopped")
        return

    # ---- part 2: closed-loop obstacle reach
    eps2 = _collect(max(8, demos // 4), seed, noise)
    tr2, va2 = split_episodes(eps2, 0.1, seed)
    ds2 = ChunkDataset(tr2, horizon, device=dev)
    vds2 = ChunkDataset(va2, horizon, device=dev, norm=ds2.norm)
    arch2 = _arch(horizon, 6)
    m2 = _build(arch2)

    def t2_eval(model):
        env, _ = _run_t2(model, ds2, gen, 64, 10_000 + seed, horizon)
        return {
            "t2_success": float(env.success.mean()),
            "t2_collision": float(env.collided.mean()),
        }

    def wrap2(p):
        return {
            "t1": holder["t1"],
            "t2": p,
            "arch": {"t1": arch1, "t2": p["arch"]},
            "gen": p["gen"],
            "params": ctx.params,
        }

    done2 = train_loop(
        ctx, m2, gen, ds2, vds2, prefix="t2_", step_offset=steps, total_steps=total,
        eval_fn=t2_eval, arch=arch2, payload_wrap=wrap2, **common,
    )  # fmt: skip
    if done2:
        ctx.done("trained t1 + t2")
    else:
        ctx.set_status(state="stopped")


def evaluate(ctx) -> None:
    P = ctx.params
    n = int(P.get("n_episodes", 300))
    grid = bool(P.get("c_grid", True))
    radius = float(P.get("obstacle_radius", 0.3))
    seed = int(P.get("seed", 0))
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    dev = _device()
    gen = Regression()

    def load(part):
        arch = ck[part]["arch"]
        m = _build(arch)
        m.load_state_dict(ck[part]["ema"])
        norm = {k: (mm.to(dev), s.to(dev)) for k, (mm, s) in ck[part]["norm"].items()}
        return m.to(dev).eval(), _Norm(arch["horizon"], norm, dev), arch

    m1, n1, _ = load("t1")
    metrics, c, a = _t1_eval(m1, n1, gen, n, grid, seed)
    sel = np.linspace(0, len(c) - 1, min(len(c), MAX_POINTS)).astype(int)
    points = [[float(c[i]), float(a[i, 0]), float(a[i, 1])] for i in sel]

    m2, n2, arch2 = load("t2")
    env, paths = _run_t2(m2, n2, gen, n, 20_000 + seed, arch2["horizon"], radius, MAX_VIEW_EPISODES)
    k = int(env.success.sum())
    episodes = []
    for i in range(min(n, MAX_VIEW_EPISODES)):
        outcome = "success" if env.success[i] else "collision" if env.collided[i] else "timeout"
        episodes.append(
            {"p": paths[i], "g": env.g[i].tolist(), "o": env.o[i].tolist(), "outcome": outcome}
        )
    result = {
        "t1": {**metrics, "points": points},
        "t2": {
            "success": k / n,
            "collision": float(env.collided.mean()),
            "timeout": float((~env.success & ~env.collided).mean()),
            "wilson95": wilson95(k, n),
            "episodes": episodes,
        },
    }
    atomic_write_json(ctx.dir / "result.json", result)
    ctx.done("evaluated")


def estimate(params: dict) -> float:
    cal = registry.calib("bc-mse")
    steps_per_s = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    collect = cal.get("collect_s_per_ep") or FALLBACK_COLLECT_S_PER_EP
    return params["steps"] / steps_per_s + params["demos"] * collect + OVERHEAD_S


SPEC = TaskSpec(
    id="bc-mse",
    order=1,
    title="Behaviour cloning, the floor",
    subtitle="Regress the expert's action with MSE and watch it average two modes into a crash.",
    params=[
        Param("demos", "Demonstrations", "int", 1000, min=50, max=5000, step=50),
        Param("steps", "Training steps", "int", 4000, min=500, max=20000, step=500),
        Param("lr", "Learning rate", "float", 3e-4, min=1e-4, max=3e-3),
        Param("batch", "Batch size", "int", 256, min=64, max=512, step=64),
        Param("horizon", "Action chunk (T2)", "choice", 8, choices=[1, 8, 16]),
        Param("expert_noise", "Expert noise", "float", 0.1, min=0.02, max=0.3, step=0.01),
    ],
    eval_params=[
        Param("n_episodes", "Episodes", "int", 300, min=50, max=2000, step=50),
        Param("c_grid", "Sweep obstacle position on a grid", "bool", True),
        Param("obstacle_radius", "Obstacle radius (T2)", "float", 0.3, min=0.1, max=0.5, step=0.05),
        Param("seed", "Seed", "int", 0, min=0, max=9999),
    ],
    charts=[
        {"id": "t1_loss", "title": "T1 loss",
         "series": [{"key": "t1_loss", "label": "train"}, {"key": "t1_val_loss", "label": "val"}],
         "log_y": True},
        {"id": "t2_loss", "title": "T2 loss",
         "series": [{"key": "t2_loss", "label": "train"}, {"key": "t2_val_loss", "label": "val"}],
         "log_y": True},
        {"id": "closedloop", "title": "Closed-loop evaluation",
         "series": [{"key": "t1_collision", "label": "T1 collision"},
                    {"key": "t2_success", "label": "T2 success"},
                    {"key": "t2_collision", "label": "T2 collision"}],
         "log_y": False},
    ],
    viewer="t1",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="bc-mse.html",
)  # fmt: skip
registry.register(SPEC)
