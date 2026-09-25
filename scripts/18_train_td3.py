#!/usr/bin/env python
"""Stage 1 RL: TD3 on the arm hidden-hole task, with or without the wrist force/torque.

Trains on episode seeds disjoint from evaluation, evaluates the deterministic actor every
--eval-every steps on --eval-episodes fixed held-out seeds, keeps best.pt (by eval success, then
return), and finishes with a --final-episodes evaluation on the BC evaluation seeds (50000...).
Writes progress.json (episodes, losses, evals) and final.json to --out.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


def run_eval(env, agent, seeds: list[int]) -> dict:
    rows = []
    for s in seeds:
        o, ret, done, peak, steps = env.reset(s), 0.0, False, 0.0, 0
        info = {}
        while not done:
            o, r, term, trunc, info = env.step(agent.act(o))
            ret += r
            steps += 1
            peak = max(peak, info["force_N"])
            done = term or trunc
        rows.append(
            {
                "seed": s,
                "reason": info["reason"],
                "success": info["reason"] == "success",
                "return": ret,
                "steps": steps,
                "peak_F_N": peak,
                "hole_xy_mm": (1e3 * env.task.hole_xy).round(3).tolist(),
                "final_depth_mm": info["depth_mm"],
            }
        )
    ok = [r["success"] for r in rows]
    return {
        "success_rate": float(np.mean(ok)),
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_steps_success": float(np.mean([r["steps"] for r in rows if r["success"]]))
        if any(ok)
        else None,
        "mean_peak_F_N": float(np.mean([r["peak_F_N"] for r in rows])),
        "reasons": {k: sum(r["reason"] == k for r in rows) for k in {r["reason"] for r in rows}},
        "episodes": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-force", type=int, default=1)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-steps", type=int, default=5000, help="uniform-random actions first")
    ap.add_argument("--expl-noise", type=float, default=0.2)
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--final-episodes", type=int, default=50)
    ap.add_argument("--norm", default="outputs/s1_arm/force/norm.json")
    ap.add_argument("--out", default="outputs/s1_td3/force_s0")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch

    from fvb.policy.arm_task import ArmTaskParams, ArmWorld
    from fvb.policy.data import Norm
    from fvb.policy.rl_env import ArmRLEnv, RewardParams
    from fvb.policy.td3 import TD3, ReplayBuffer, TD3Config

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = ArmTaskParams()
    rp = RewardParams()
    cfg = TD3Config()
    norm = Norm.from_json(json.loads(Path(args.norm).read_text()))
    world = ArmWorld(p, seed=args.seed)
    env = ArmRLEnv(p, world, norm, use_force=bool(args.use_force), history=args.history, rp=rp)
    agent = TD3(env.obs_dim, env.act_dim, cfg, device)
    buf = ReplayBuffer(env.obs_dim, env.act_dim, min(args.buffer, args.steps))
    meta = {
        "algo": "td3",
        "task": "arm",
        "use_force": bool(args.use_force),
        "history": args.history,
        "xy_scale": float(env.scale[0]),
        "z_scale": float(env.scale[2]),
        "norm": args.norm,
        "args": vars(args),
        "task_params": asdict(p),
        "reward": asdict(rp),
        "td3": asdict(cfg),
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2))

    eval_seeds = [70000 + i for i in range(args.eval_episodes)]
    train_seed = 1_000_000 * (args.seed + 1)
    prog: dict = {"episodes": [], "losses": [], "evals": [], "meta": meta}
    loss_acc: dict = {}
    best = (-1.0, -np.inf)
    t0 = time.perf_counter()
    o = env.reset(train_seed)
    ep_ret, ep_len, ep_peak, n_ep = 0.0, 0, 0.0, 0
    for t in range(1, args.steps + 1):
        if t <= args.start_steps:
            a = rng.uniform(-1, 1, env.act_dim).astype(np.float32)
        else:
            a = agent.act(o) + rng.normal(0, args.expl_noise, env.act_dim)
            a = np.clip(a, -1, 1).astype(np.float32)
        o2, r, term, trunc, info = env.step(a)
        buf.add(o, a, r, o2, term)
        o = o2
        ep_ret += r
        ep_len += 1
        ep_peak = max(ep_peak, info["force_N"])
        if t > args.start_steps:
            for k, v in agent.update(buf, rng).items():
                loss_acc.setdefault(k, []).append(v)
        if term or trunc:
            prog["episodes"].append(
                {
                    "t": t,
                    "return": ep_ret,
                    "len": ep_len,
                    "reason": info["reason"],
                    "peak_F_N": ep_peak,
                }
            )
            n_ep += 1
            o = env.reset(train_seed + n_ep)
            ep_ret, ep_len, ep_peak = 0.0, 0, 0.0
        if t % 1000 == 0 and loss_acc:
            prog["losses"].append({"t": t, **{k: float(np.mean(v)) for k, v in loss_acc.items()}})
            loss_acc = {}
        if t % args.eval_every == 0:
            ev = run_eval(env, agent, eval_seeds)
            ev_row = {k: v for k, v in ev.items() if k != "episodes"}
            prog["evals"].append({"t": t, "wall_s": time.perf_counter() - t0, **ev_row})
            recent = prog["episodes"][-50:]
            print(
                json.dumps(
                    {
                        "t": t,
                        "eval_success": ev["success_rate"],
                        "eval_return": round(ev["mean_return"], 2),
                        "train_success_last50": float(
                            np.mean([e["reason"] == "success" for e in recent])
                        )
                        if recent
                        else None,
                        "episodes": n_ep,
                        "wall_min": round((time.perf_counter() - t0) / 60, 1),
                    }
                ),
                flush=True,
            )
            key = (ev["success_rate"], ev["mean_return"])
            if key > best:
                best = key
                torch.save(agent.state({**meta, "t": t, "eval": ev_row}), out / "best.pt")
            torch.save(agent.state({**meta, "t": t}), out / "last.pt")
            (out / "progress.json").write_text(json.dumps(prog))
            o = env.reset(train_seed + n_ep)  # eval used the env: restart the training episode
            ep_ret, ep_len, ep_peak = 0.0, 0, 0.0

    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    agent.actor.load_state_dict(ck["actor"])
    final = run_eval(env, agent, [50000 + i for i in range(args.final_episodes)])
    final["best_t"] = ck["meta"]["t"]
    final["wall_s"] = time.perf_counter() - t0
    (out / "final.json").write_text(json.dumps(final, indent=2))
    (out / "progress.json").write_text(json.dumps(prog))
    print(
        f"final ({args.final_episodes} unseen seeds, best checkpoint @ {final['best_t']}): "
        f"success {final['success_rate']:.2f} {final['reasons']}"
    )
    world.close()


if __name__ == "__main__":
    main()
