#!/usr/bin/env python
"""Plot TD3 learning curves (training reward + evaluation success) for all runs in --runs.

Blue = with wrist force/torque, orange = F/T channels zeroed; one line per seed. Runs still in
progress are drawn up to their latest checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def rolling(x, w: int) -> np.ndarray:
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    lo = np.maximum(np.arange(1, len(x) + 1) - w, 0)
    return (c[1:] - c[lo]) / (np.arange(1, len(x) + 1) - lo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=["outputs/s1_td3"], help="one or more run dirs")
    ap.add_argument(
        "--title",
        default="TD3 on the Panda hidden-hole insertion: with vs without wrist force/torque",
    )
    ap.add_argument("--out", default="docs/img/td3_training_reward.png")
    ap.add_argument("--window", type=int, default=100, help="episodes in the rolling mean")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, height_ratios=[3, 2])
    dirs = sorted(d for r in args.runs for d in glob.glob(str(Path(r) / "*_s*")))
    for d in dirs:
        pf = Path(d) / "progress.json"
        if not pf.exists():
            continue
        prog = json.loads(pf.read_text())
        force = prog["meta"]["use_force"]
        seed = prog["meta"]["args"]["seed"]
        color = "#1f67c9" if force else "#d0561c"
        label = f"{'F/T' if force else 'no F/T'}, seed {seed}"
        t = np.array([e["t"] for e in prog["episodes"]]) / 1e3
        r = rolling([e["return"] for e in prog["episodes"]], args.window)
        ax1.plot(t, r, color=color, lw=1.1, alpha=0.75, label=label)
        ev = prog["evals"]
        ax2.plot(
            [e["t"] / 1e3 for e in ev],
            [e["success_rate"] for e in ev],
            "o-",
            color=color,
            lw=1.1,
            ms=3,
            alpha=0.85,
        )
    ax1.set_ylabel(f"training episode return\n(rolling mean, {args.window} episodes)")
    ax1.set_title(args.title)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=6, ncol=4, loc="lower right")
    ax2.set_ylabel("eval success\n(20 held-out holes)")
    ax2.set_xlabel("environment steps [k]")
    ax2.set_ylim(-0.03, 1.03)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
