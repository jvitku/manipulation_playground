#!/usr/bin/env python
"""TA-VLA / diffusion-policy tutorial experiments T1-T4 (VLA image; GPU optional).

    python scripts/40_ta_tutorial.py --exp t1 --gen flow --out outputs/ta_tutorial/t1/flow_s0
    python scripts/40_ta_tutorial.py --exp t2 --gen flow --horizon 16 --n-exec 8 --demos 200
    python scripts/40_ta_tutorial.py --exp t3 --torque suffix --aux 0.1 --demos 300
    python scripts/40_ta_tutorial.py --exp t4 --torque suffix

Writes ``<out>/history.json`` (training loss, held-out loss, learning rate, closed-loop
evaluations during training), ``<out>/final.json`` and visualisation extras. Code in
``src/fvb/ta_tutorial``; walkthrough in ``docs/ta_tutorial.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from fvb.ta_tutorial import contact2d as c2d
from fvb.ta_tutorial import generative, toy
from fvb.ta_tutorial.data import ChunkDataset, split_episodes
from fvb.ta_tutorial.nets import MLPDenoiser, TAVLAMini
from fvb.ta_tutorial.train import Cfg, ChunkPolicy, train

EVAL_SEED = 10_000  # every arm is evaluated on the same scenes (paired comparison)


class RenderingTAVLA(TAVLAMini):
    """T4: renders the camera image from the scene state on the GPU inside the prefix encoder
    (so the dataset stores 5 numbers per frame instead of a 32x32 image)."""

    def encode_prefix(self, cond):
        if "image" not in cond:
            cond = dict(cond, image=c2d.render(cond["scene"]))
        return super().encode_prefix(cond)


def as_t(x, dev):
    return {k: torch.as_tensor(v, device=dev) for k, v in x.items()}


# --------------------------------------------------------------------------------------- T1
def run_t1(a, dev, out):
    r = np.random.default_rng(a.seed)
    c, act = toy.t1_data(20_000, r)
    eps = [{"obs": c[i : i + 1], "action": act[i : i + 1]} for i in range(len(c))]
    tr, va = split_episodes(eps, 0.1, a.seed)
    ds, vds = ChunkDataset(tr, 1, device=dev), None
    vds = ChunkDataset(va, 1, device=dev, norm=ds.norm)
    gen = generative.make(a.gen)
    model = MLPDenoiser(2, 1, 1, hidden=256, depth=3)
    ce = torch.as_tensor(
        np.random.default_rng(99).uniform(-0.4, 0.4, (2000, 1)), dtype=torch.float32
    )

    def sample(m, cc, steps, trace=False, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        cond = ds.cond({"obs": cc.to(dev)})
        return gen.sample(m, (len(cc), 1, 2), cond, steps, g, trace=trace)

    def eval_fn(m):
        x = sample(m, ce, a.sample_steps)
        return toy.t1_metrics(ce[:, 0].numpy(), x[:, 0].cpu().numpy())

    hist, ema = train(model, gen, ds, cfg(a, eval_every=250, log_every=50), vds, eval_fn)
    extras = {"sweep": {}}
    for s in (1, 2, 4, 8, 16, 32, 64):
        x = sample(ema, ce, s)
        extras["sweep"][s] = toy.t1_metrics(ce[:, 0].numpy(), x[:, 0].cpu().numpy())
    for cv in (-0.3, 0.0, 0.3):
        cc = torch.full((400, 1), cv)
        x, tr_ = sample(ema, cc, a.sample_steps, trace=True)
        extras[f"samples_c{cv}"] = x[:, 0].cpu().numpy().round(4).tolist()
        if cv == 0.0:
            extras["trace_c0"] = tr_[:, :48, 0].cpu().numpy().round(4).tolist()  # (steps+1, 48, 2)
    if a.gen == "flow":  # the learned velocity field at three noise levels, for c = 0
        gx = torch.linspace(-2, 2, 13)
        X, Y = torch.meshgrid(gx, gx, indexing="xy")
        pts = torch.stack([X.flatten(), Y.flatten()], 1)[:, None].to(dev)
        cond = ds.cond({"obs": torch.zeros(len(pts), 1, device=dev)})
        field = {}
        for t in (0.9, 0.5, 0.1):
            with torch.no_grad():
                v, _ = ema(pts, torch.full((len(pts),), t, device=dev), cond)
            field[t] = {
                "xy": pts[:, 0].cpu().numpy().round(3).tolist(),
                "v": v[:, 0].cpu().numpy().round(4).tolist(),
            }
        extras["field_c0"] = field
    data_c0 = act[np.abs(c[:, 0]) < 0.05][:400]
    extras["data_c0"] = data_c0.round(4).tolist()
    final = eval_fn(ema)
    return hist, final, extras


# --------------------------------------------------------------------------------------- T2
def run_t2(a, dev, out):
    eps = toy.t2_collect(a.demos, a.seed * 1000 + 1)
    tr, va = split_episodes(eps, 0.1, a.seed) if len(eps) >= 10 else (eps, eps[:1])
    ds = ChunkDataset(tr, a.horizon, device=dev)
    vds = ChunkDataset(va, a.horizon, device=dev, norm=ds.norm)
    gen = generative.make(a.gen)
    model = MLPDenoiser(2, a.horizon, 6, hidden=512, depth=4)

    def evaluate(m, steps=None, n=200, keep=0):
        env = toy.ObstacleReach(n, EVAL_SEED)
        pol = ChunkPolicy(m, gen, ds, lambda e: {"obs": torch.as_tensor(e.obs(), device=dev)},
                          a.n_exec, steps or a.sample_steps, seed=a.seed)  # fmt: skip
        paths = [env.p.copy()]
        while not env.done.all():
            env.step(pol.act(env))
            paths.append(env.p.copy())
        r = {"success": float(env.success.mean()), "collision": float(env.collided.mean())}
        if keep:
            P = np.stack(paths, 1)[:keep]
            r["paths"] = P.round(3).tolist()
            r["obstacle"] = env.o[:keep].round(3).tolist()
            r["goal"] = env.g[:keep].round(3).tolist()
        return r

    hist, ema = train(model, gen, ds, cfg(a), vds, lambda m: evaluate(m))
    final = evaluate(ema, keep=24)
    extras = {
        "paths": final.pop("paths"),
        "obstacle": final.pop("obstacle"),
        "goal": final.pop("goal"),
    }
    if a.gen != "mse":
        extras["sweep"] = {s: evaluate(ema, steps=s) for s in (1, 2, 4, 10, 32)}
    extras["demo_pairs"] = int(ds.n)
    return hist, final, extras


# --------------------------------------------------------------------------------------- T3/T4
def contact_model(a, state_dim, image=False):
    kw = dict(image_hw=32, vocab=len(c2d.VOCAB), lang_len=c2d.LANG_LEN) if image else {}
    cls = RenderingTAVLA if image else TAVLAMini
    return cls(
        act_dim=2,
        horizon=a.horizon,
        state_dim=state_dim,
        torque_dim=2 * c2d.HIST,
        torque_pos="suffix" if a.torque == "zero" else a.torque,
        zero_torque=a.torque == "zero",
        aux_dim=16 if a.aux > 0 else 0,
        **kw,
    )


def eval_contact(kind, m, gen, ds, a, dev, n, keep=0):
    env = c2d.t3_scenes(n, EVAL_SEED) if kind == "t3" else c2d.t4_scenes(n, EVAL_SEED)
    obs_fn = c2d.t3_obs if kind == "t3" else c2d.t4_obs
    pol = ChunkPolicy(m, gen, ds, lambda o: as_t(o, dev), a.n_exec, a.sample_steps, seed=a.seed)
    rec = []
    while not env.done.all():
        o = obs_fn(env)
        env.step(pol.act(o))
        if keep:
            rec.append(np.concatenate([env.pos[:keep], env.hist[:keep, -1]], 1))
    oc = Counter(env.outcome.tolist())
    r = {"success": float(env.success.mean()), **{f"n_{k}": int(v) for k, v in oc.items()}}
    if kind == "t4":
        for t in range(4):
            sel = env.task == t
            r[f"task{t}"] = float(env.success[sel].mean())
    if keep:
        r["traj"] = np.stack(rec, 1).round(4).tolist()
        r["scene"] = {
            "z0": env.z0[:keep].round(4).tolist(),
            "holes": env.holes[:keep].round(4).tolist(),
            "outcome": env.outcome[:keep].tolist(),
        }
        if kind == "t3":
            r["scene"]["est"] = env.est[:keep].round(4).tolist()
    return r


def run_contact(kind, a, dev, out):
    eps = c2d.collect(kind, a.demos, a.seed * 1000 + 1)
    ok = [e for e in eps if e["success"]]
    tr, va = split_episodes(ok, 0.1, a.seed)
    ds = ChunkDataset(tr, a.horizon, device=dev)
    vds = ChunkDataset(va, a.horizon, device=dev, norm=ds.norm)
    gen = generative.make(a.gen)
    model = contact_model(a, 2, image=kind == "t4")
    n_eval = 200 if kind == "t3" else 400
    hist, ema = train(model, gen, ds, cfg(a), vds,
                      lambda m: eval_contact(kind, m, gen, ds, a, dev, n_eval))  # fmt: skip
    final = eval_contact(kind, ema, gen, ds, a, dev, n_eval, keep=12)
    extras = {"traj": final.pop("traj"), "scene": final.pop("scene")}
    extras.update(demo_episodes=len(ok), demo_success=len(ok) / len(eps), demo_pairs=int(ds.n),
                  params=sum(p.numel() for p in model.parameters()))  # fmt: skip
    return hist, final, extras


def cfg(a, **over):
    c = Cfg(
        steps=a.steps,
        batch=a.batch,
        lr=a.lr,
        aux_weight=a.aux,
        seed=a.seed,
        eval_every=a.eval_every,
    )
    for k, v in over.items():
        setattr(c, k, v)
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", choices=("t1", "t2", "t3", "t4"), required=True)
    ap.add_argument("--gen", choices=("flow", "ddpm", "mse"), default="flow")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--n-exec", type=int, default=8)
    ap.add_argument("--demos", type=int, default=200)
    ap.add_argument("--torque", choices=("zero", "prefix", "suffix"), default="suffix")
    ap.add_argument("--aux", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--sample-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    fn = {
        "t1": run_t1,
        "t2": run_t2,
        "t3": lambda *x: run_contact("t3", *x),
        "t4": lambda *x: run_contact("t4", *x),
    }
    hist, final, extras = fn[a.exp](a, dev, out)
    final["wall_min"] = round((time.time() - t0) / 60, 2)
    (out / "history.json").write_text(json.dumps(hist))
    (out / "final.json").write_text(json.dumps(final, indent=2))
    (out / "extras.json").write_text(json.dumps(extras))
    print(json.dumps(final))


if __name__ == "__main__":
    main()
