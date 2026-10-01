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
                "hole_xy_mm": (1e3 * env.task.hole_xy).round(3).tolist()
                if hasattr(env, "task")
                else None,
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


def collect_demos(env, n: int, seed0: int, bufs) -> dict:
    """Run the privileged expert through the RL env (real observations and rewards) and add its
    transitions to every buffer in ``bufs``. Expert target deltas are scaled to [-1, 1]."""
    ok = 0
    for i in range(n):
        o, done = env.reset(seed0 + i), False
        p, info = env.privileged(), {}
        while not done:
            if hasattr(env, "expert_action"):  # SortInsertEnv: already in [-1, 1]
                a = env.expert_action()
            else:
                a = env.task.encode(env.task.expert_action()) / env.scale  # delta or xy_abs
            a = np.clip(a, -1, 1).astype(np.float32)
            o2, r, term, trunc, info = env.step(a)
            for b in bufs:
                b.add(o, a, r, o2, term, p, info["priv"])
            o, p, done = o2, info["priv"], term or trunc
        ok += info["reason"] == "success"
    return {"episodes": n, "success": ok, "transitions": bufs[0].n}


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
    ap.add_argument("--demos", type=int, default=0, help="expert episodes pre-filled into replay")
    ap.add_argument("--bc-weight", type=float, default=0.0, help="TD3+BC weight at the start")
    ap.add_argument(
        "--bc-decay-steps", type=int, default=100_000, help="linear decay of the BC weight to 0"
    )
    ap.add_argument(
        "--bc-final", type=float, default=0.0, help="BC weight after the decay (a floor)"
    )
    ap.add_argument(
        "--pretrain-steps",
        type=int,
        default=0,
        help="offline TD3+BC updates on the demos alone before any interaction",
    )
    ap.add_argument(
        "--pretrain-q-weight",
        type=float,
        default=1.0,
        help="weight of the Q term during pretraining (0 = pure BC actor, critic still trained)",
    )
    ap.add_argument(
        "--rim-speed-limit-mm",
        type=float,
        default=0.0,
        help="cap the downward step within 5 mm of the rim (mm/step, 0 = off)",
    )
    ap.add_argument(
        "--action-mode",
        choices=["delta", "xy_abs"],
        default="delta",
        help="xy_abs: the lateral action is a target relative to the start (z stays a delta)",
    )
    ap.add_argument("--xy-scale-mm", type=float, default=1.0, help="lateral action scale")
    ap.add_argument(
        "--expl-noise-xy", type=float, default=None, help="lateral exploration noise (default: =)"
    )
    ap.add_argument(
        "--asym-critic",
        type=int,
        default=0,
        help="1: the critic also sees the privileged tip->hole vector and depth (actor does not)",
    )
    ap.add_argument(
        "--task",
        choices=["arm", "sort_insert", "sort_pick"],
        default="arm",
        help="sort_insert: the Stage 2 insertion skill (fvb.policy.sort_skill_env)",
    )
    ap.add_argument("--cache", type=int, default=200, help="sort_insert: cached expert set-ups")
    ap.add_argument(
        "--grasp-rand", type=int, default=0, help="sort_insert: randomised in-hand bolt offsets"
    )
    ap.add_argument(
        "--centred", type=int, default=0, help="sort_pick: bolt picks must be centred to count"
    )
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
    p = ArmTaskParams(rim_speed_limit=1e-3 * args.rim_speed_limit_mm, action_mode=args.action_mode)
    rp = RewardParams()
    cfg = TD3Config()
    world = None
    if args.task in ("sort_insert", "sort_pick"):
        from fvb.policy.sort_skill_env import SortInsertEnv, SortPickEnv

        env_cls = SortPickEnv if args.task == "sort_pick" else SortInsertEnv
        env = env_cls(
            use_force=bool(args.use_force),
            history=args.history,
            **({"xy_scale": 1e-3 * args.xy_scale_mm} if args.task == "sort_insert" else {}),
            rp=rp,
            cache=args.cache,
            **({"grasp_rand": bool(args.grasp_rand)} if args.task == "sort_insert" else {}),
            **({"centred": bool(args.centred)} if args.task == "sort_pick" else {}),
        )
    else:
        norm = Norm.from_json(json.loads(Path(args.norm).read_text()))
        world = ArmWorld(p, seed=args.seed)
        env = ArmRLEnv(
            p,
            world,
            norm,
            use_force=bool(args.use_force),
            history=args.history,
            rp=rp,
            xy_scale=1e-3 * args.xy_scale_mm,
        )
    nxy = args.expl_noise if args.expl_noise_xy is None else args.expl_noise_xy
    noise_std = np.array([nxy, nxy, args.expl_noise] + [args.expl_noise] * (env.act_dim - 3))
    priv_dim = 3 if args.asym_critic else 0
    agent = TD3(env.obs_dim, env.act_dim, cfg, device, priv_dim=priv_dim)
    buf = ReplayBuffer(env.obs_dim, env.act_dim, min(args.buffer, args.steps), priv_dim)
    meta = {
        "algo": "td3",
        "task": args.task,
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

    demo_buf = None
    if args.demos:
        demo_buf = ReplayBuffer(env.obs_dim, env.act_dim, args.demos * 250, priv_dim)
        meta["demo_stats"] = collect_demos(env, args.demos, 900_000, [buf, demo_buf])
        print(json.dumps({"demos": meta["demo_stats"]}), flush=True)
        if args.pretrain_steps:
            # offline warm start: actor starts at demo level, critic calibrated on demo returns
            pre: dict = {}
            for _ in range(args.pretrain_steps):
                upd = agent.update(
                    demo_buf, rng, demo_buf, args.bc_weight, q_weight=args.pretrain_q_weight
                )
                for k, v in upd.items():
                    pre.setdefault(k, []).append(v)
            meta["pretrain"] = {k: float(np.mean(v[-1000:])) for k, v in pre.items()}
            print(json.dumps({"pretrain": meta["pretrain"]}), flush=True)
        (out / "config.json").write_text(json.dumps(meta, indent=2))
    eval_seeds = [70000 + i for i in range(args.eval_episodes)]
    train_seed = 1_000_000 * (args.seed + 1)
    prog: dict = {"episodes": [], "losses": [], "evals": [], "meta": meta}
    loss_acc: dict = {}
    best = (-1.0, -np.inf)
    t0 = time.perf_counter()
    o = env.reset(train_seed)
    p = env.privileged()
    ep_ret, ep_len, ep_peak, n_ep = 0.0, 0, 0.0, 0
    for t in range(1, args.steps + 1):
        if t <= args.start_steps:
            a = rng.uniform(-1, 1, env.act_dim).astype(np.float32)
        else:
            a = agent.act(o) + rng.normal(0, 1, env.act_dim) * noise_std
            a = np.clip(a, -1, 1).astype(np.float32)
        o2, r, term, trunc, info = env.step(a)
        buf.add(o, a, r, o2, term, p, info["priv"])
        o, p = o2, info["priv"]
        ep_ret += r
        ep_len += 1
        ep_peak = max(ep_peak, info["force_N"])
        if t > args.start_steps:
            frac = max(0.0, 1.0 - t / max(args.bc_decay_steps, 1))
            bc_w = args.bc_final + (args.bc_weight - args.bc_final) * frac
            for k, v in agent.update(buf, rng, demo_buf, bc_w).items():
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
            p = env.privileged()
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
            p = env.privileged()
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
    if world is not None:
        world.close()


if __name__ == "__main__":
    main()
