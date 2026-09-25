#!/usr/bin/env python
"""Stage 1: closed-loop evaluation of trained BC checkpoints vs the privileged expert.

`--task arm` evaluates on the Panda PegInHole hidden-hole task (one env, reused).
`--untrained` adds a randomly initialised copy of the first checkpoint as a baseline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fvb.policy.rollout import evaluate, load_policy
from fvb.policy.task import TaskParams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["gantry", "arm"], default="gantry")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument(
        "--correct-at", choices=["after_retract", "jam"], default="after_retract", help="expert"
    )
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=50000, help="eval seeds are disjoint from data")
    ap.add_argument("--offset-sigma-mm", type=float, default=1.5)
    ap.add_argument("--clearance-mm", type=float, default=0.5)
    ap.add_argument("--force-noise-N", type=float, default=0.0)
    ap.add_argument("--torque-noise-Nm", type=float, default=0.0)
    ap.add_argument("--out", default="outputs/s1/eval")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    make_task, close = None, lambda: None
    if args.task == "arm":
        from fvb.policy.arm_task import ArmTask, ArmTaskParams, ArmWorld

        p = ArmTaskParams(
            clearance=args.clearance_mm * 1e-3,
            offset_sigma=args.offset_sigma_mm * 1e-3,
            correct_at=args.correct_at,
        )
        world = ArmWorld(p, seed=args.seed)
        make_task, close = (lambda p_, s: ArmTask(p_, s, world)), world.close
    else:
        p = TaskParams(
            clearance=args.clearance_mm * 1e-3,
            offset_sigma=args.offset_sigma_mm * 1e-3,
            force_noise_N=args.force_noise_N,
            torque_noise_Nm=args.torque_noise_Nm,
            correct_at=args.correct_at,
        )
    seeds = [args.seed + i for i in range(args.n)]
    results = {"task": asdict(p), "task_spec": args.task, "seeds": [seeds[0], seeds[-1]]}
    results["policies"] = {}
    print("expert ...")
    results["policies"]["expert"] = evaluate(None, p, seeds, make_task)
    pols = []
    if args.untrained:
        pols.append(("untrained", load_policy(args.ckpt[0], args.device, untrained_seed=0)))
    for ck in args.ckpt:
        pol = load_policy(ck, args.device)
        name = (
            f"{pol.meta['model']}_{'force' if pol.use_force else 'noforce'}_{Path(ck).parent.name}"
        )
        pols.append((name, pol))
    for name, pol in pols:
        print(name, "...")
        results["policies"][name] = evaluate(pol, p, seeds, make_task)
    close()
    for name, r in results["policies"].items():
        print(
            f"{name:32s} success {r['success_rate']:.2f}  steps {r['mean_steps_success']:.1f}  "
            f"peak|F| {r['mean_peak_F_N']:.1f} N  recoveries {r['mean_recoveries']:.2f}  "
            f"{r['reasons']}"
        )
    (out / "eval.json").write_text(json.dumps(results, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results["policies"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].bar(names, [results["policies"][n]["success_rate"] for n in names])
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(0, 1)
    axes[1].bar(names, [results["policies"][n]["mean_peak_F_N"] for n in names])
    axes[1].set_ylabel("mean episode peak |F| [N]")
    # success vs |offset|
    for n in names:
        eps = results["policies"][n]["episodes"]
        r = np.array([np.linalg.norm(e["hole_xy_mm"]) for e in eps])
        s = np.array([e["success"] for e in eps], float)
        bins = np.array([0, 0.5, 1.0, 2.0, 3.0, 6.0])
        rate = [
            s[(r >= a) & (r < b)].mean() if ((r >= a) & (r < b)).any() else np.nan
            for a, b in zip(bins[:-1], bins[1:], strict=True)
        ]
        axes[2].plot(0.5 * (bins[:-1] + bins[1:]), rate, "o-", label=n)
    axes[2].set_xlabel("|hole offset| [mm]  (clearance 0.5 mm)")
    axes[2].set_ylabel("success rate")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7, rotation=20)
        ax.grid(alpha=0.3)
    fig.suptitle(f"closed-loop eval, {args.n} episodes, σ_offset {args.offset_sigma_mm} mm")
    fig.tight_layout()
    fig.savefig(out / "eval.png", dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
