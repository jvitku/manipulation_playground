"""Task 2 - "Flow matching fixes it" (id `flow`).

Same two problems and data as task 1 (T1 two-mode obstacle, T2 closed-loop point mass), but the
head is generative: `head` = flow matching (default) or DDPM/DDIM, sampled with `sample_steps`
integration steps. Helpers (`_collect`, `_Norm`, `_arch`, `_build`, `wilson95`) are task 1's.

Checkpoint payload (`last.pt`), same shape as task 1:
    {"t1": P1, "t2": P2, "arch": {"t1": arch1, "t2": arch2}, "gen": "flow"|"ddpm", "params": ...}
with Pn = the `train_loop` dict {"model","ema","norm","arch","params","gen"}. `gen` (and every
arch) is read from the checkpoint at evaluation time, never from the form. The T2 chunk
execution length is `params["n_exec"]` (clamped to the horizon).

result.json written by `evaluate`:
    {"head", "sample_steps",
     "t1": {collision, left_frac, mode_err, points: [[c, a_x, a_y]...]},     # at K=sample_steps
     "t2": {success, collision, timeout, wilson95, episodes: [...]},         # K=sample_steps
     "sweep": {"kind": "euler"|"ddim", "euler": {"1": T1 collision, ... "32": ...}},
     "trace": {"c": [...<=64], "xs": [step][sample] -> [a_x, a_y]}}          # noise -> actions
The K-sweep is shown by the `flow` viewer (the spec's charts only plot training metrics).
"""

from __future__ import annotations

import numpy as np
import torch

from fvb.ta_tutorial import toy
from fvb.ta_tutorial.data import ChunkDataset, split_episodes
from fvb.ta_tutorial.generative import make
from fvb.ta_tutorial.train import ChunkPolicy
from lab.core.loop import train_loop
from lab.core.runctx import atomic_write_json
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.task1_bc_mse import (
    MAX_POINTS,
    MAX_VIEW_EPISODES,
    OVERHEAD_S,
    _arch,
    _build,
    _collect,
    _device,
    _Norm,
    wilson95,
)

SWEEP = [1, 2, 4, 8, 16, 32]
MAX_TRACE = 64
FALLBACK_STEPS_PER_S, FALLBACK_COLLECT_S_PER_EP = 100.0, 0.05
FALLBACK_EVAL_S_PER_K = 0.03  # seconds per sampling step per periodic evaluation


def _t1_sample(model, ds_like, gen, n, grid, seed, steps):
    if grid:
        c = np.linspace(-0.4, 0.4, n).astype(np.float32)
    else:
        c = np.random.default_rng(seed).uniform(-0.4, 0.4, n).astype(np.float32)
    dev = ds_like.device
    cond = ds_like.cond({"obs": torch.as_tensor(c[:, None], device=dev)})
    g = torch.Generator(device=dev).manual_seed(seed)
    with torch.no_grad():
        a = gen.sample(model.eval(), (n, 1, 2), cond, steps, g).clamp(-1, 1)[:, 0].cpu().numpy()
    return toy.t1_metrics(c, a), c, a


def _run_t2(model, ds_like, gen, n, seed, n_exec, steps, radius=None, record=0):
    env = toy.ObstacleReach(n, seed)
    if radius is not None:
        env.RADIUS = radius
    pol = ChunkPolicy(model, gen, ds_like, lambda o: {"obs": o}, n_exec, steps, seed=seed)
    record = min(record, n)
    paths = [[env.p[i].tolist()] for i in range(record)]
    while not env.done.all():
        live = ~env.done.copy()
        env.step(pol.act(torch.as_tensor(env.obs(), device=ds_like.device)))
        for i in range(record):
            if live[i]:
                paths[i].append(env.p[i].tolist())
    return env, paths


def train(ctx) -> None:
    P = ctx.params
    steps, demos = int(P["steps"]), int(P["demos"])
    horizon, noise, seed = int(P["horizon"]), float(P["expert_noise"]), int(P.get("seed", 0))
    n_exec = max(1, min(int(P.get("n_exec", horizon)), horizon))
    batch, lr, K = int(P["batch"]), float(P["lr"]), int(P.get("sample_steps", 8))
    eval_every = int(P.get("eval_every", max(1, steps // 8)))
    ckpt_every = int(P.get("ckpt_every", max(1, steps // 4)))
    warmup = max(1, min(500, steps // 10))
    dev = _device()
    gen = make(P.get("head", "flow"))
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

    def t1_eval(model):
        m, _, _ = _t1_sample(model, ds, gen, 200, True, seed, K)
        return {"t1_collision": m["collision"], "t1_mode_err": m["mode_err"]}

    holder: dict = {}

    def wrap1(p):
        holder["t1"] = p
        return {"t1": p, "arch": {"t1": p["arch"]}, "gen": p["gen"], "params": ctx.params}

    if not train_loop(
        ctx, m1, gen, ds, vds, prefix="t1_", total_steps=total, eval_fn=t1_eval,
        arch=arch1, payload_wrap=wrap1, **common,
    ):  # fmt: skip
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
        env, _ = _run_t2(model, ds2, gen, 64, 10_000 + seed, n_exec, K)
        return {
            "t2_success": float(env.success.mean()),
            "t2_collision": float(env.collided.mean()),
        }

    def wrap2(p):
        return {
            "t1": holder["t1"], "t2": p,
            "arch": {"t1": arch1, "t2": p["arch"]}, "gen": p["gen"], "params": ctx.params,
        }  # fmt: skip

    if train_loop(
        ctx, m2, gen, ds2, vds2, prefix="t2_", step_offset=steps, total_steps=total,
        eval_fn=t2_eval, arch=arch2, payload_wrap=wrap2, **common,
    ):  # fmt: skip
        ctx.done(f"trained t1 + t2 ({gen.name})")
    else:
        ctx.set_status(state="stopped")


def evaluate(ctx) -> None:
    P = ctx.params
    n = int(P.get("n_episodes", 300))
    grid = bool(P.get("c_grid", True))
    radius = float(P.get("obstacle_radius", 0.3))
    seed = int(P.get("seed", 0))
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    K = int(P.get("eval_steps") or (ck.get("params") or {}).get("sample_steps") or 8)
    dev = _device()
    gen = make(ck["gen"])
    n_exec_ck = int((ck.get("params") or {}).get("n_exec", 0)) or ck["t2"]["arch"]["horizon"]

    def load(part):
        arch = ck[part]["arch"]
        m = _build(arch)
        m.load_state_dict(ck[part]["ema"])
        norm = {k: (mm.to(dev), s.to(dev)) for k, (mm, s) in ck[part]["norm"].items()}
        return m.to(dev).eval(), _Norm(arch["horizon"], norm, dev), arch

    m1, n1, _ = load("t1")
    metrics, c, a = _t1_sample(m1, n1, gen, n, grid, seed, K)
    sel = np.linspace(0, len(c) - 1, min(len(c), MAX_POINTS)).astype(int)
    points = [[float(c[i]), float(a[i, 0]), float(a[i, 1])] for i in sel]

    sweep = {}
    for k in SWEEP:
        sweep[str(k)] = _t1_sample(m1, n1, gen, n, grid, seed, k)[0]["collision"]

    # noise -> actions transport for up to MAX_TRACE evenly spaced obstacle positions
    ct = np.linspace(-0.4, 0.4, MAX_TRACE).astype(np.float32)
    cond = n1.cond({"obs": torch.as_tensor(ct[:, None], device=dev)})
    g = torch.Generator(device=dev).manual_seed(seed)
    with torch.no_grad():
        _, xs = gen.sample(m1, (len(ct), 1, 2), cond, K, g, trace=True)
    xs = xs[:, :, 0].cpu().numpy()  # (K+1, n, 2)
    trace = {"c": [float(x) for x in ct], "xs": xs.round(4).tolist()}

    m2, n2, arch2 = load("t2")
    n_exec = max(1, min(n_exec_ck, arch2["horizon"]))
    env, paths = _run_t2(m2, n2, gen, n, 20_000 + seed, n_exec, K, radius, MAX_VIEW_EPISODES)
    k = int(env.success.sum())
    episodes = []
    for i in range(min(n, MAX_VIEW_EPISODES)):
        outcome = "success" if env.success[i] else "collision" if env.collided[i] else "timeout"
        episodes.append(
            {"p": paths[i], "g": env.g[i].tolist(), "o": env.o[i].tolist(), "outcome": outcome}
        )
    result = {
        "head": gen.name,
        "sample_steps": K,
        "t1": {**metrics, "points": points},
        "t2": {
            "success": k / n,
            "collision": float(env.collided.mean()),
            "timeout": float((~env.success & ~env.collided).mean()),
            "wilson95": wilson95(k, n),
            "episodes": episodes,
        },
        "sweep": {"kind": "euler" if gen.name == "flow" else "ddim", "euler": sweep},
        "trace": trace,
    }
    atomic_write_json(ctx.dir / "result.json", result)
    ctx.done("evaluated")


def estimate(params: dict) -> float:
    cal = registry.calib("flow")
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    collect = cal.get("collect_s_per_ep") or FALLBACK_COLLECT_S_PER_EP
    overhead = cal.get("overhead_s", OVERHEAD_S)
    per_k = cal.get("eval_s_per_k", FALLBACK_EVAL_S_PER_K)
    # steps_per_s is per-model (two models are trained); 8 periodic evals per model, each
    # costing roughly sample_steps network passes of the T2 rollout.
    evals = 2 * 8 * int(params.get("sample_steps", 8)) * per_k
    return params["steps"] / sps + params["demos"] * collect + overhead + evals


SPEC = TaskSpec(
    id="flow",
    order=2,
    title="Flow matching fixes it",
    subtitle="Same data, same net: learn a velocity field, sample by integrating from noise.",
    params=[
        Param("head", "Generative head", "choice", "flow", choices=["flow", "ddpm"]),
        Param("sample_steps", "Sampling steps K", "choice", 8, choices=SWEEP),
        Param("demos", "Demonstrations", "int", 1000, min=50, max=5000, step=50),
        Param("steps", "Training steps", "int", 4000, min=500, max=20000, step=500),
        Param("lr", "Learning rate", "float", 1e-3, min=1e-4, max=3e-3),
        Param("batch", "Batch size", "int", 256, min=64, max=512, step=64),
        Param("horizon", "Action chunk (T2)", "choice", 8, choices=[1, 8, 16]),
        Param("n_exec", "Actions executed per chunk (T2)", "int", 8, min=1, max=16, step=1),
        Param("expert_noise", "Expert noise", "float", 0.1, min=0.02, max=0.3, step=0.01),
    ],
    eval_params=[
        Param("eval_steps", "Sampling steps K", "choice", 8, choices=SWEEP),
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
    viewer="flow",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="flow.html",
)  # fmt: skip
registry.register(SPEC)
