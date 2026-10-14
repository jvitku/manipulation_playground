#!/usr/bin/env python
"""Tutorial T5: the TA diffusion policy on the real insertion skill (train image: MuJoCo +
robosuite + torch).

``SortInsertEnv`` (randomised in-hand offsets) - the same set-ups as the BC experiment
(``scripts/30_bc_insert.py``: demos from seeds 800000+i, evaluation on 50000+k), so the result
sits next to BC-MLP's 28.8 % (force) vs 6.8 % (no force).

Observation per step: 4 stacked frames of [pad - hole estimate (3, cm), gripper width (1)] =
16-D state, and 4 frames of [tau_ext (7), tactile summary (8)] = 60-D torque history. Policy:
``TAVLAMini`` with a flow-matching head, chunks of 8, re-planned every 4 steps; arms: torque
token in the expert suffix vs the same model with the token zeroed (+ optional aux loss).

    python scripts/42_ta_tutorial_t5.py collect --shard 0 --shards 6
    python scripts/42_ta_tutorial_t5.py train --torque suffix --seed 0 \
        --out outputs/ta_tutorial/t5/suffix_s0
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

DEMO_DIR = Path("outputs/ta_tutorial/t5/demos")
FRAME, HIST = 19, 4
STATE_COLS = np.concatenate([np.arange(f * FRAME, f * FRAME + 4) for f in range(HIST)])
TORQUE_COLS = np.concatenate([np.arange(f * FRAME + 4, (f + 1) * FRAME) for f in range(HIST)])


def collect(shard: int, shards: int, n: int) -> None:
    from fvb.policy.sort_skill_env import SortInsertEnv

    env = SortInsertEnv(use_force=True, grasp_rand=True)
    out = DEMO_DIR / f"shard{shard}.npz"
    O, A, L, OK = [], [], [], []
    for i in range(shard, n, shards):
        o = env.reset(800_000 + i)
        ex = env.expert
        obs, act = [], []
        for _ in range(150):
            p0 = ex.pad_point().copy()
            ob, _, _, _ = env.env.step(ex.act())
            env.obs = ob
            act.append(np.clip((ex.pad_point() - p0) / env.scale, -1, 1))
            obs.append(o)
            env._frames.append(env._frame())
            o = env._obs()
            if env.depth() >= 0.008 or ex.phase in ("let_go", "up", "select"):
                break
            if env.env.outcome == "force_abort":
                break
        O += obs
        A += act
        L.append(len(obs))
        OK.append(env.depth() >= 0.008)
    np.savez(
        out, O=np.array(O, np.float32), A=np.array(A, np.float32), L=np.array(L), ok=np.array(OK)
    )
    print(json.dumps({"shard": shard, "episodes": len(L), "success": int(np.sum(OK))}), flush=True)


def load_episodes() -> list[dict]:
    eps = []
    for f in sorted(DEMO_DIR.glob("shard*.npz")):
        d = np.load(f)
        s = 0
        for n, ok in zip(d["L"], d["ok"], strict=True):
            o, a = d["O"][s : s + n], d["A"][s : s + n]
            s += n
            if ok:
                eps.append(
                    {
                        "state": o[:, STATE_COLS],
                        "torque": o[:, TORQUE_COLS],
                        "action": a,
                        "force": o[:, -15:-8].copy(),
                    }
                )  # fmt: skip  latest tau_ext (aux target)
    return eps


def train_eval(a) -> None:
    import torch

    from fvb.policy.sort_skill_env import SortInsertEnv
    from fvb.ta_tutorial import generative
    from fvb.ta_tutorial.data import ChunkDataset, split_episodes
    from fvb.ta_tutorial.nets import TAVLAMini
    from fvb.ta_tutorial.train import Cfg, ChunkPolicy, train

    torch.set_num_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    eps = load_episodes()
    tr, va = split_episodes(eps, 0.1, a.seed)
    ds = ChunkDataset(tr, a.horizon, device=dev)
    vds = ChunkDataset(va, a.horizon, device=dev, norm=ds.norm)
    gen = generative.make("flow")
    model = TAVLAMini(act_dim=3, horizon=a.horizon, state_dim=16, torque_dim=60,
                      torque_pos="suffix" if a.torque == "zero" else a.torque,
                      zero_torque=a.torque == "zero", aux_dim=56 if a.aux > 0 else 0)  # fmt: skip
    env = SortInsertEnv(use_force=True, grasp_rand=True)

    def evaluate(m, n):
        rows = []
        for k in range(n):
            o = env.reset(50_000 + k)
            pol = ChunkPolicy(m, gen, ds, None, a.n_exec, a.sample_steps, seed=a.seed * 100 + k)
            pol.make_cond = lambda ob: {
                "state": torch.as_tensor(ob[None, STATE_COLS], device=dev),
                "torque": torch.as_tensor(ob[None, TORQUE_COLS], device=dev),
            }
            reason, peak = "timeout", 0.0
            for _ in range(env.max_steps):
                o, _, term, trunc, info = env.step(pol.act(o)[0])
                peak = max(peak, info["force_N"])
                if term or trunc:
                    reason = info["reason"] or "timeout"
                    break
            rows.append({"seed": 50_000 + k, "reason": reason, "peak_F_N": round(peak, 2)})
        c = Counter(r["reason"] for r in rows)
        return {"success": c["success"] / n, **{f"n_{k}": v for k, v in c.items()}}, rows

    t0 = time.time()
    hist, ema = train(model, gen, ds, Cfg(steps=a.steps, lr=a.lr, aux_weight=a.aux, seed=a.seed,
                      eval_every=a.eval_every), vds, lambda m: evaluate(m, a.eval_mid)[0])  # fmt: skip
    final, rows = evaluate(ema, a.eval)
    final.update(
        demo_episodes=len(eps), demo_pairs=int(ds.n), wall_min=round((time.time() - t0) / 60, 1)
    )
    (out / "history.json").write_text(json.dumps(hist))
    (out / "final.json").write_text(json.dumps({**final, "episodes": rows}, indent=2))
    print(json.dumps(final))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--shard", type=int, default=0)
    c.add_argument("--shards", type=int, default=1)
    c.add_argument("--demos", type=int, default=300)
    t = sub.add_parser("train")
    t.add_argument("--torque", choices=("zero", "prefix", "suffix"), default="suffix")
    t.add_argument("--aux", type=float, default=0.0)
    t.add_argument("--horizon", type=int, default=8)
    t.add_argument("--n-exec", type=int, default=4)
    t.add_argument("--steps", type=int, default=20_000)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--sample-steps", type=int, default=10)
    t.add_argument("--eval", type=int, default=50)
    t.add_argument("--eval-mid", type=int, default=20)
    t.add_argument("--eval-every", type=int, default=5000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--cpu", action="store_true")
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "collect":
        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        collect(a.shard, a.shards, a.demos)
    else:
        train_eval(a)


if __name__ == "__main__":
    main()
