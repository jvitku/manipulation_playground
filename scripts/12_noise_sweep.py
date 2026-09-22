#!/usr/bin/env python
"""Stage 1b: sensor-noise sweep. For each noise level: collect noisy expert data, train ACT-lite
with force, evaluate closed-loop at that noise; also evaluate the clean-trained policy there.
Runs the existing scripts as subprocesses (needs the train image: torch)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LEVELS = [  # (torque noise N·m, force noise N) per physics step; signal at a jam ≈ 0.018 N·m
    (0.0, 0.0),
    (0.002, 0.05),
    (0.005, 0.1),
    (0.01, 0.2),
    (0.02, 0.5),
    (0.05, 1.0),
]


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/s1_noise")
    ap.add_argument("--clean-ckpt", default="outputs/s1/force/best.pt")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--levels", type=int, nargs="*", default=None, help="indices into LEVELS")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sweep_path = out / "sweep.json"
    rows = json.loads(sweep_path.read_text()) if sweep_path.exists() else []
    done = {(r["torque_noise_Nm"], r["force_noise_N"]) for r in rows}
    levels = LEVELS if args.levels is None else [LEVELS[i] for i in args.levels]
    for tn, fn in levels:
        if (tn, fn) in done:
            continue
        tag = f"t{tn:g}_f{fn:g}"
        d = out / tag
        data = f"data/expert_noise_{tag}"
        run(
            [
                sys.executable,
                "scripts/09_collect_expert.py",
                "--n",
                str(args.n_train),
                "--out",
                data,
                "--torque-noise-Nm",
                str(tn),
                "--force-noise-N",
                str(fn),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/10_train_bc.py",
                "--data",
                data,
                "--out",
                str(d / "train"),
                "--force",
                "1",
                "--device",
                args.device,
            ]
        )
        run(
            [
                sys.executable,
                "scripts/11_eval_bc.py",
                "--ckpt",
                str(d / "train" / "best.pt"),
                args.clean_ckpt,
                "--n",
                str(args.n_eval),
                "--out",
                str(d / "eval"),
                "--torque-noise-Nm",
                str(tn),
                "--force-noise-N",
                str(fn),
                "--device",
                args.device,
            ]
        )
        ev = json.loads((d / "eval" / "eval.json").read_text())["policies"]
        names = list(ev)
        rows.append(
            {
                "torque_noise_Nm": tn,
                "force_noise_N": fn,
                "expert": ev["expert"]["success_rate"],
                "trained_at_noise": ev[names[1]]["success_rate"],
                "trained_clean": ev[names[2]]["success_rate"],
                "expert_recoveries": ev["expert"]["mean_recoveries"],
                "noisy_recoveries": ev[names[1]]["mean_recoveries"],
                "clean_recoveries": ev[names[2]]["mean_recoveries"],
            }
        )
        print(json.dumps(rows[-1]), flush=True)
        rows.sort(key=lambda r: r["torque_noise_Nm"])
        sweep_path.write_text(json.dumps(rows, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [max(r["torque_noise_Nm"], 5e-4) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, [r["expert"] for r in rows], "k--", label="privileged expert")
    ax.plot(
        x,
        [r["trained_at_noise"] for r in rows],
        "o-",
        label="ACT-lite + force, trained at this noise",
    )
    ax.plot(
        x, [r["trained_clean"] for r in rows], "s-", label="ACT-lite + force, trained noise-free"
    )
    ax.axvline(0.018, color="C3", ls=":", label="torque signal at a 3 N jam (0.018 N·m)")
    ax.set_xscale("log")
    ax.set_xlabel(
        "torque noise std per 1 kHz sample [N·m]  (force noise = 20× in N; 0 plotted at 5e-4)"
    )
    ax.set_ylabel("closed-loop success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("Sensor-noise floor for the force-token policy")
    fig.tight_layout()
    fig.savefig(out / "noise_sweep.png", dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
