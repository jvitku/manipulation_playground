"""Task 3 - "Flow-BC from demos" (id `flow-bc`): the 2-D contact peg search.

A flow-matching policy (MLPDenoiser over the flattened observation) learns from scripted-expert
demonstrations on `fvb.ta_tutorial.contact2d` (`t3_scenes` + `Expert` + `t3_obs`). The observation
is 22-D: state (peg x relative to the noisy hole estimate, peg z; 2) + the last 10 force readings
(Fx, Fz; 20). `use_force=false` zeroes the 20 torque channels (the zeroed twin). Successful
demonstrations only are used. Every `eval_every` steps the EMA model is rolled out closed loop on
200 fresh scenes - the "success keeps climbing while the loss flattens" chart.

Checkpoint payload (`last.pt`): the `train_loop` dict
    {"model", "ema", "norm": {"obs": (mean, std)}, "arch", "params", "gen": "flow"}
with arch = {"act_dim": 2, "horizon", "obs_dim": 22, "hidden", "depth", "use_force", "n_exec"}.
Evaluation rebuilds the model from `arch` (`use_force`/`n_exec` included), never from the form.

result.json written by `evaluate`:
    {"n_episodes", "success", "counts": {outcome: n}, "wilson95", "force_zeroed",
     "trained_with_force", "sample_steps",
     "est_sigma", "sensor_noise",
     "episodes": [{"z0", "holes", "target", "est", "pos": [[x, z]...], "force": [[fx, fz]...],
                   "act": [[ax, az]...] (action applied at step s, 3 decimals; len = len(pos)-1),
                   "outcome"}]}     # <= 16, floats rounded to 4 decimals; pos[0] is the start
`pos[i]` and `force[i]` are the peg position / sensed force after step i (force[0] = [0, 0]).
`force_zeroed` is true when the torque channels were zeroed at evaluation (the form flag, or a
checkpoint trained with use_force=false).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from fvb.ta_tutorial import contact2d as c2d
from fvb.ta_tutorial.data import ChunkDataset, split_episodes
from fvb.ta_tutorial.generative import make
from fvb.ta_tutorial.train import ChunkPolicy
from lab.core.loop import train_loop
from lab.core.runctx import atomic_write_json
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.task1_bc_mse import (
    DEPTH,
    HIDDEN,
    OVERHEAD_S,
    _build,
    _device,
    _Norm,
    wilson95,
)

OBS_DIM = 2 + 2 * c2d.HIST
MAX_VIEW_EPISODES = 16
TRAIN_EVAL_N = 200
EXPERT_NOISE = 0.05
OUTCOMES = ("success", "force_abort", "wrong_hole", "timeout")
FALLBACK_STEPS_PER_S, FALLBACK_COLLECT_S_PER_EP = 100.0, 0.05
FALLBACK_EVAL_S = 8.0  # one 200-scene closed-loop evaluation at sample_steps=8


def _obs_tensor(raw: dict, dev, use_force: bool) -> torch.Tensor:
    """env obs dict (state 2 + torque 20) -> (n, 22); torque zeroed when `use_force` is false."""
    tq = raw["torque"] if use_force else np.zeros_like(raw["torque"])
    return torch.as_tensor(np.concatenate([raw["state"], tq], 1), dtype=torch.float32, device=dev)


def _rollout(model, ds_like, n, seed, n_exec, K, use_force, est_sigma=0.03, sensor_noise=0.05,
             record=0):  # fmt: skip
    """Closed-loop rollouts on `n` fresh scenes; returns (env, recorded pos/force per episode)."""
    env = c2d.t3_scenes(n, seed, est_sigma)
    env.noise = sensor_noise
    dev = ds_like.device
    pol = ChunkPolicy(
        model, make_gen(), ds_like, lambda o: {"obs": _obs_tensor(o, dev, use_force)}, n_exec, K,
        seed=seed,
    )  # fmt: skip
    record = min(record, n)
    pos = [[env.pos[i].tolist()] for i in range(record)]
    force = [[[0.0, 0.0]] for _ in range(record)]
    act = [[] for _ in range(record)]
    while not env.done.all():
        live = ~env.done.copy()
        a = pol.act(c2d.t3_obs(env))
        env.step(a)
        for i in range(record):
            if live[i]:
                act[i].append(np.asarray(a[i], float).tolist())
                pos[i].append(env.pos[i].tolist())
                force[i].append(env.hist[i, -1].tolist())
    return env, pos, force, act


_GEN = {}


def make_gen():
    if "flow" not in _GEN:
        _GEN["flow"] = make("flow")
    return _GEN["flow"]


def train(ctx) -> None:
    P = ctx.params
    steps, demos = int(P["steps"]), int(P["demos"])
    horizon, seed = int(P["horizon"]), int(P.get("seed", 0))
    n_exec = max(1, min(int(P.get("n_exec", 8)), horizon))
    batch, lr, K = int(P["batch"]), float(P["lr"]), int(P.get("sample_steps", 8))
    use_force = bool(P.get("use_force", True))
    eval_every = int(P.get("eval_every", 1000))
    ckpt_every = int(P.get("ckpt_every", max(1, steps // 4)))
    dev = _device()
    gen = make_gen()

    eps = c2d.collect("t3", demos, seed * 1000 + 1, EXPERT_NOISE)
    ok = [e for e in eps if e["success"]]
    ctx.set_status(message=f"demonstrations: {len(ok)} of {demos} episodes succeeded")
    if len(ok) < 2:
        ctx.fail(f"only {len(ok)} successful demonstrations out of {demos}")
        return
    for e in ok:
        tq = e["torque"] if use_force else np.zeros_like(e["torque"])
        e["obs"] = np.concatenate([e["state"], tq], 1).astype(np.float32)
    tr, va = split_episodes(ok, 0.1, seed)
    ds = ChunkDataset(tr, horizon, device=dev)
    vds = ChunkDataset(va, horizon, device=dev, norm=ds.norm)
    arch = {"act_dim": 2, "horizon": horizon, "obs_dim": OBS_DIM, "hidden": HIDDEN, "depth": DEPTH,
            "use_force": use_force, "n_exec": n_exec}  # fmt: skip
    model = _build(arch)

    def eval_fn(m):
        env, *_ = _rollout(m, ds, TRAIN_EVAL_N, 10_000 + seed, n_exec, K, use_force)
        return {
            "success": float(env.success.mean()),
            "force_abort": float((env.outcome == "force_abort").mean()),
        }

    done = train_loop(
        ctx, model, gen, ds, vds, steps=steps, batch=batch, lr=lr,
        warmup=max(1, min(500, steps // 10)), eval_every=eval_every, ckpt_every=ckpt_every,
        eval_fn=eval_fn, log_every=max(1, min(50, steps // 4)), seed=seed, arch=arch,
    )  # fmt: skip
    if done:
        ctx.done(f"trained on {len(ok)} successful demos ({'with' if use_force else 'no'} force)")
    else:
        ctx.set_status(state="stopped")


def evaluate(ctx) -> None:
    P = ctx.params
    n = int(P.get("n_episodes", 400))
    seed = int(P.get("seed", 0))
    est_sigma, sensor_noise = float(P.get("est_sigma", 0.03)), float(P.get("sensor_noise", 0.05))
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    arch = ck["arch"]
    K = int(P.get("eval_steps") or (ck.get("params") or {}).get("sample_steps") or 8)
    zeroed = bool(P.get("force_zeroed", False)) or not arch.get("use_force", True)
    dev = _device()
    model = _build(arch)
    model.load_state_dict(ck["ema"])
    model = model.to(dev).eval()
    norm = {k: (m.to(dev), s.to(dev)) for k, (m, s) in ck["norm"].items()}
    ds_like = _Norm(arch["horizon"], norm, dev)
    n_exec = max(1, min(int(arch.get("n_exec", 8)), arch["horizon"]))
    env, pos, force, act = _rollout(
        model, ds_like, n, 20_000 + seed, n_exec, K, not zeroed, est_sigma, sensor_noise,
        MAX_VIEW_EPISODES,
    )  # fmt: skip
    k = int(env.success.sum())
    cnt = Counter(env.outcome.tolist())
    r4 = lambda v: np.round(np.asarray(v, float), 4).tolist()  # noqa: E731
    episodes = [
        {
            "z0": r4(env.z0[i]), "holes": r4(env.holes[i]), "target": int(env.target[i]),
            "est": r4(env.est[i]), "pos": r4(pos[i]), "force": r4(force[i]),
            "act": np.round(np.asarray(act[i], float), 3).tolist(),
            "outcome": str(env.outcome[i]),
        }
        for i in range(min(n, MAX_VIEW_EPISODES))
    ]  # fmt: skip
    atomic_write_json(
        ctx.dir / "result.json",
        {
            "n_episodes": n, "success": k / n, "counts": {o: int(cnt.get(o, 0)) for o in OUTCOMES},
            "wilson95": wilson95(k, n), "force_zeroed": zeroed,
            "trained_with_force": bool(arch.get("use_force", True)),
            "sample_steps": K,
            "est_sigma": est_sigma, "sensor_noise": sensor_noise, "episodes": episodes,
        },
    )  # fmt: skip
    ctx.done("evaluated")


def estimate(params: dict) -> float:
    cal = registry.calib("flow-bc")
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    collect = cal.get("collect_s_per_ep") or FALLBACK_COLLECT_S_PER_EP
    overhead = cal.get("overhead_s", OVERHEAD_S)
    eval_s = cal.get("eval_s", FALLBACK_EVAL_S) * int(params.get("sample_steps", 8)) / 8
    n_evals = max(1, -(-int(params["steps"]) // max(1, int(params.get("eval_every", 1000)))))
    return params["steps"] / sps + params["demos"] * collect + overhead + n_evals * eval_s


SPEC = TaskSpec(
    id="flow-bc",
    order=4,
    title="Flow-BC from demos",
    subtitle="Search for a hole by feel: a flow policy reads the force history while it slides.",
    params=[
        Param("demos", "Demonstrations", "int", 400, min=50, max=2000, step=50),
        Param("steps", "Training steps", "int", 8000, min=1000, max=40000, step=1000),
        Param("lr", "Learning rate", "float", 1e-3, min=1e-4, max=3e-3),
        Param("batch", "Batch size", "int", 256, min=64, max=512, step=64),
        Param("horizon", "Action chunk", "choice", 16, choices=[8, 16]),
        Param("n_exec", "Actions executed per chunk", "choice", 8, choices=[4, 8, 16]),
        Param("use_force", "Feed the force history", "bool", True),
        Param("sample_steps", "Sampling steps K", "choice", 8, choices=[1, 2, 4, 8, 16, 32]),
        Param("eval_every", "Closed-loop eval every N steps", "int", 1000, min=100, max=10000,
              step=100),
    ],
    eval_params=[
        Param("n_episodes", "Episodes", "int", 400, min=50, max=2000, step=50),
        Param("est_sigma", "Hole-estimate error (sigma)", "float", 0.03, min=0.01, max=0.06,
              step=0.01),
        Param("sensor_noise", "Force sensor noise", "float", 0.05, min=0.0, max=0.2, step=0.01),
        Param("force_zeroed", "Zero the force input (twin)", "bool", False),
        Param("eval_steps", "Sampling steps K", "choice", 8, choices=[1, 2, 4, 8, 16, 32]),
        Param("seed", "Seed", "int", 0, min=0, max=9999),
    ],
    charts=[
        {"id": "loss", "title": "Flow loss",
         "series": [{"key": "loss", "label": "train"}, {"key": "val_loss", "label": "val"}],
         "log_y": True},
        {"id": "success", "title": "Closed-loop success (200 fresh scenes)",
         "series": [{"key": "success", "label": "success"},
                    {"key": "force_abort", "label": "force abort"}],
         "log_y": False},
        {"id": "lr", "title": "Learning rate", "series": [{"key": "lr", "label": "lr"}],
         "log_y": False},
    ],
    viewer="contact2d",
    vecgrids=True,
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="flow-bc.html",
)  # fmt: skip
registry.register(SPEC)
