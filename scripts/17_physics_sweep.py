#!/usr/bin/env python
"""Stage 1 G2: does the gantry force policy survive a change of timestep / contact stiffness?

Also sweeps the gantry spring stiffness kp (M3/M5: the knob that moves the control-rate force
most) at nominal solver settings, without retraining.

1. Collect expert episodes with per-episode physics randomisation (timestep, solref) and train
   ACT-lite + force on them (subprocesses: scripts 09 and 10).
2. Evaluate the privileged expert, the nominal policy (Stage 1, trained at dt 1 ms / solref
   5 ms) and the randomised-physics policy on a timestep x solref grid, 50 unseen seeds per
   cell, one worker process per cell.
Writes sweep.json and heatmaps to --out. Needs the train image (torch).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TIMESTEPS = (0.0005, 0.001, 0.002)
SOLREFS = (0.002, 0.005, 0.02)
KPS = (200.0, 400.0, 800.0, 1600.0, 3200.0)  # gantry spring stiffness, N/m (nominal 800)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def eval_cell(job: dict) -> dict:
    import torch

    from fvb.policy.rollout import TorchPolicy, evaluate
    from fvb.policy.task import TaskParams

    torch.set_num_threads(1)
    p = replace(
        TaskParams(), timestep=job["timestep"], solref_tc=job["solref_tc"], kp=job.get("kp", 800.0)
    )
    seeds = [job["seed"] + i for i in range(job["n"])]
    row = {"timestep": job["timestep"], "solref_tc": job["solref_tc"], "kp": p.kp}
    row["expert"] = evaluate(None, p, seeds)
    for name, ck in job["ckpts"].items():
        row[name] = evaluate(TorchPolicy(ck), p, seeds)
    for k in ["expert", *job["ckpts"]]:
        row[k].pop("episodes")
    print(
        f"dt {job['timestep']:g} solref {job['solref_tc']:g} kp {p.kp:g}: "
        + "  ".join(f"{k} {row[k]['success_rate']:.2f}" for k in ["expert", *job["ckpts"]]),
        flush=True,
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/s1_phys")
    ap.add_argument("--data", default="data/expert_trackA_physrand")
    ap.add_argument("--nominal-ckpt", default="outputs/s1/force/best.pt")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--extra-ckpt", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    ap.add_argument(
        "--with-kp-rand",
        action="store_true",
        help="also train on dt + solref + kp randomisation and evaluate it",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    rand_ckpt = out / "train_physrand" / "best.pt"
    if not args.skip_train:
        py = sys.executable
        run(
            [
                py,
                "scripts/09_collect_expert.py",
                "--n",
                str(args.n_train),
                "--phys-rand",
                "--out",
                args.data,
            ]
        )
        run(
            [
                py,
                "scripts/10_train_bc.py",
                "--data",
                args.data,
                "--force",
                "1",
                "--out",
                str(rand_ckpt.parent),
            ]
        )
    ckpts = {"nominal": args.nominal_ckpt, "physrand": str(rand_ckpt)}
    if args.with_kp_rand:
        kp_ckpt = out / "train_physkprand" / "best.pt"
        if not kp_ckpt.exists():
            py, d = sys.executable, args.data + "_kp"
            run(
                [
                    py,
                    "scripts/09_collect_expert.py",
                    "--n",
                    str(args.n_train),
                    "--phys-rand",
                    "--kp-rand",
                    "--out",
                    d,
                ]
            )
            run(
                [
                    py,
                    "scripts/10_train_bc.py",
                    "--data",
                    d,
                    "--force",
                    "1",
                    "--out",
                    str(kp_ckpt.parent),
                ]
            )
        ckpts["physkprand"] = str(kp_ckpt)
    for name, path in args.extra_ckpt:
        ckpts[name] = path
    jobs = [
        {"timestep": dt, "solref_tc": tc, "n": args.n_eval, "seed": args.seed, "ckpts": ckpts}
        for dt in TIMESTEPS
        for tc in SOLREFS
    ]
    kp_jobs = [
        {
            "timestep": 0.001,
            "solref_tc": 0.005,
            "kp": kp,
            "n": args.n_eval,
            "seed": args.seed,
            "ckpts": ckpts,
        }
        for kp in KPS
    ]
    with Pool(args.workers) as pool:
        rows = pool.map(eval_cell, jobs)
        kp_rows = pool.map(eval_cell, kp_jobs)
    (out / "sweep.json").write_text(json.dumps(rows, indent=2))
    (out / "kp_sweep.json").write_text(json.dumps(kp_rows, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["expert", *ckpts]
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.8))
    for ax, name in zip(axes, names, strict=True):
        M = np.array(
            [[r[name]["success_rate"] for r in rows if r["timestep"] == dt] for dt in TIMESTEPS]
        )
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        for i in range(len(TIMESTEPS)):
            for j in range(len(SOLREFS)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w")
        ax.set_xticks(range(len(SOLREFS)), [f"{s:g}" for s in SOLREFS])
        ax.set_yticks(range(len(TIMESTEPS)), [f"{t:g}" for t in TIMESTEPS])
        ax.set_xlabel("solref time constant [s]")
        ax.set_ylabel("timestep [s]")
        ax.set_title(name)
    fig.colorbar(im, ax=axes, label="closed-loop success rate")
    fig.suptitle(f"Gantry hidden-hole task: success over physics settings ({args.n_eval} seeds)")
    fig.savefig(out / "sweep.png", dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
