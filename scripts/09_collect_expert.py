#!/usr/bin/env python
"""Stage 1: collect privileged-expert episodes on the hidden-hole task (gantry or arm).

Writes PLAN §5 episodes (+ the per-step policy observation as `obs` alongside, and the expert
delta actions as `action`) to --out. The hole offset is stored in the .json config for analysis
only; the observation never contains it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fvb.logging.episode import EpisodeLogger
from fvb.policy.task import GantryTask, TaskParams


def task_factory(name: str, args):
    """(params, make_task(seed), close) for --task. The arm env is built once and reused."""
    if name == "gantry":
        p = TaskParams(
            clearance=args.clearance_mm * 1e-3,
            offset_sigma=args.offset_sigma_mm * 1e-3,
            force_noise_N=args.force_noise_N,
            torque_noise_Nm=args.torque_noise_Nm,
            phys_rand=args.phys_rand,
            kp_rand=args.kp_rand,
            kp_range=tuple(args.kp_range),
            obs_kp=args.obs_kp,
            action_mode=args.action_mode,
            correct_at=args.correct_at,
        )
        return p, (lambda seed: GantryTask(p, seed)), (lambda: None)
    from fvb.policy.arm_task import ArmTask, ArmTaskParams, ArmWorld

    p = ArmTaskParams(
        clearance=args.clearance_mm * 1e-3,
        offset_sigma=args.offset_sigma_mm * 1e-3,
        correct_at=args.correct_at,
        action_mode=args.action_mode,
    )
    world = ArmWorld(p, seed=args.seed)
    return p, (lambda seed: ArmTask(p, seed, world)), world.close


def run_episode(seed: int, p, make_task, out: Path, i: int) -> dict:
    task = make_task(seed)
    log = EpisodeLogger(
        config={
            "task": asdict(p),
            "hole_xy": task.hole_xy.tolist(),
            "phys": getattr(task, "phys", None),
            "expert": True,
        },
        seed=seed,
    )
    obs_rows, act_rows = [], []
    peak, n_jams, done, reason = 0.0, 0, False, None
    while not done:
        o = task.observe()
        a = task.encode(task.expert_action())
        if task._phase == "retract" and a[2] > 0 and len(act_rows) and act_rows[-1][2] <= 0:
            n_jams += 1
        obs_rows.append(o)
        act_rows.append(a.astype(np.float32))
        done, reason = task.step(a, log)
        peak = max(peak, task.force_norm())
    log.success = reason == "success"
    arrs = log.arrays()
    # the §5 `action` key holds the absolute target (Gantry.log_row); store the policy-facing
    # delta action and observation next to the episode in a sibling .npz
    path = log.save(out / f"ep{i:04d}.npz")
    np.savez_compressed(
        out / f"ep{i:04d}_policy.npz",
        obs=np.stack(obs_rows),
        action=np.stack(act_rows),
        success=np.bool_(log.success),
    )
    return {
        "seed": seed,
        "success": log.success,
        "reason": reason,
        "steps": int(len(arrs["t"])),
        "n_jams": n_jams,
        "peak_F_N": peak,
        "hole_xy_mm": (task.hole_xy * 1e3).tolist(),
        "phys": getattr(task, "phys", None),
        "path": str(path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["gantry", "arm"], default="gantry")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="data/expert_trackA")
    ap.add_argument("--offset-sigma-mm", type=float, default=1.5)
    ap.add_argument("--clearance-mm", type=float, default=0.5)
    ap.add_argument("--force-noise-N", type=float, default=0.0)
    ap.add_argument("--torque-noise-Nm", type=float, default=0.0)
    ap.add_argument("--phys-rand", action="store_true", help="gantry: randomise dt + solref")
    ap.add_argument("--kp-rand", action="store_true", help="gantry: randomise the spring kp")
    ap.add_argument("--kp-range", type=float, nargs=2, default=[200.0, 3200.0], help="N/m")
    ap.add_argument("--obs-kp", action="store_true", help="gantry: kp in the observation")
    ap.add_argument("--correct-at", choices=["after_retract", "jam"], default="after_retract")
    ap.add_argument("--action-mode", choices=["delta", "xy_abs"], default="delta")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p, make_task, close = task_factory(args.task, args)
    spec_name = "gantry_kp" if args.task == "gantry" and args.obs_kp else args.task
    cfg = {**vars(args), "task_spec": spec_name, "task": asdict(p)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    t0 = time.perf_counter()
    rows = [run_episode(args.seed + i, p, make_task, out, i) for i in range(args.n)]
    close()
    ok = sum(r["success"] for r in rows)
    jams = np.array([r["n_jams"] for r in rows])
    summary = {
        "n": len(rows),
        "n_success": int(ok),
        "wall_s": time.perf_counter() - t0,
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "jam_episodes": int((jams > 0).sum()),
        "mean_jams": float(jams.mean()),
        "reasons": {
            k: int(sum(r["reason"] == k for r in rows)) for k in {r["reason"] for r in rows}
        },
        "episodes": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
