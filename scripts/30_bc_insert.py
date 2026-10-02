#!/usr/bin/env python
"""V9 (skill level): behaviour cloning of the insertion skill with vs without force/tactile.

Whole-task imitation (V5/V7) never reaches a part, so the force comparison is made where force
matters: the insertion skill (``SortInsertEnv`` with randomised in-hand offsets).

* Demonstrations: the scripted expert's *own* insertion behaviour from the hand-over point - it
  aims at its noisy hole estimate, notices contact (> 20 N) and searches on a spiral - so the
  demos contain force-driven decisions (a privileged demonstrator that goes straight to the true
  hole never touches anything and its demos would carry no force information). Actions are the
  pad's displacement per step in the skill's action units (clipped to [-1, 1]).
* Policy: MLP on the skill observation (4 frames x 19); ``--use-force 0`` zeroes the 15 force /
  tactile channels in training and evaluation.
* Evaluation: closed loop in the skill env on ``--eval`` unseen set-ups.

    python scripts/30_bc_insert.py --demos 300 --use-force 1 --seed 0 \
        --out outputs/bc_insert/force_s0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def collect(env, n: int, seed0: int, max_steps: int = 150):
    """Expert insertion episodes from the skill set-ups: (obs, action) pairs + outcome."""
    O, A, ok = [], [], 0
    for i in range(n):
        o = env.reset(seed0 + i)
        ex, sk = env.expert, env
        for _ in range(max_steps):
            p0 = ex.pad_point().copy()
            obs, _, _, _ = env.env.step(ex.act())
            sk.obs = obs
            a = np.clip((ex.pad_point() - p0) / sk.scale, -1, 1)
            O.append(o)
            A.append(a)
            sk._frames.append(sk._frame())
            o = sk._obs()
            if sk.depth() >= 0.008 or ex.phase in ("let_go", "up", "select"):
                ok += sk.depth() >= 0.008
                break
            if env.env.outcome == "force_abort":
                break
    return np.array(O, np.float32), np.array(A, np.float32), ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", type=int, default=300)
    ap.add_argument("--use-force", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval", type=int, default=50)
    ap.add_argument("--data", default="outputs/bc_insert/demos.npz", help="cached demos")
    ap.add_argument("--out", default="outputs/bc_insert/force_s0")
    args = ap.parse_args()

    import torch

    from fvb.policy.sort_skill_env import SortInsertEnv

    torch.set_num_threads(2)  # 10 parallel runs with torch's default (all cores) pushed the
    # 24-core host to a load of ~137 and stalled every run
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = SortInsertEnv(use_force=True, grasp_rand=True)  # force channels always recorded
    data = Path(args.data)
    if data.exists():
        d = np.load(data)
        O, A, demo_ok = d["O"], d["A"], int(d["ok"])
    else:
        t0 = time.time()
        O, A, demo_ok = collect(env, args.demos, 800_000)
        data.parent.mkdir(parents=True, exist_ok=True)
        np.savez(data, O=O, A=A, ok=demo_ok)
        print(
            json.dumps(
                {
                    "demos": args.demos,
                    "success": demo_ok,
                    "pairs": len(O),
                    "min": round((time.time() - t0) / 60, 1),
                }
            ),
            flush=True,
        )
    force_cols = np.zeros(env.frame_dim, bool)
    force_cols[4:] = True
    mask = np.tile(~force_cols, env.history) if not args.use_force else np.ones(env.obs_dim, bool)
    mu, sd = O.mean(0), O.std(0) + 1e-6
    X = torch.as_tensor(((O - mu) / sd) * mask, dtype=torch.float32)
    Y = torch.as_tensor(A, dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(env.obs_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 3),
        torch.nn.Tanh(),
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    g = torch.Generator().manual_seed(args.seed)
    for _step in range(args.steps):
        idx = torch.randint(0, len(X), (256,), generator=g)
        loss = torch.nn.functional.mse_loss(net(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    rows = []
    for k in range(args.eval):
        o = env.reset(50_000 + k)
        reason, peak = None, 0.0
        for _ in range(env.max_steps):
            x = torch.as_tensor(((o - mu) / sd) * mask, dtype=torch.float32)[None]
            with torch.no_grad():
                a = net(x)[0].numpy()
            o, _, term, trunc, info = env.step(a)
            peak = max(peak, info["force_N"])
            if term or trunc:
                reason = info["reason"]
                break
        rows.append({"seed": 50_000 + k, "reason": reason, "peak_F_N": peak})
    sr = float(np.mean([r["reason"] == "success" for r in rows]))
    res = {
        "use_force": bool(args.use_force),
        "seed": args.seed,
        "demo_success": demo_ok,
        "demo_pairs": int(len(O)),
        "final_loss": float(loss),
        "success_rate": sr,
        "reasons": {k: sum(r["reason"] == k for r in rows) for k in {r["reason"] for r in rows}},
        "episodes": rows,
    }
    (out / "final.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "episodes"}))


if __name__ == "__main__":
    main()
