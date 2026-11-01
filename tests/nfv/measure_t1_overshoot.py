"""Measure the T1 position-press overshoot against the 1.5x-threshold success bound.

Not a pytest file — run it in the image and read the table:

    docker run --rm -u 1001:1001 -v "$PWD":/workspace -w /workspace -e HOME=/tmp \
        -e MUJOCO_GL=osmesa fvb-lab:latest python tests/nfv/measure_t1_overshoot.py

The expert presses in position mode at PRESS_DP per 10 Hz step, so the episode peak force is
threshold + (what accrues between the 200 Hz click latch and the end of that policy step):
up to one step of quasi-static ramp (K_eff * PRESS_DP ≈ 0.38 N) plus the cap weight (~0.2 N)
plus noise. The worst case is the LOWEST threshold (3 N), where the absolute margin
0.5 * threshold = 1.5 N is smallest — measured here with the threshold overridden to 3 N on
every seed (overrides never shift the rng stream), DART noise at the demo default.

Exit 0 iff every episode succeeds and the worst peak/threshold ratio stays under 1.5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from nfv.seeds import eval_seeds  # noqa: E402
from nfv.sim.env import Env  # noqa: E402
from nfv.sim.experts import rollout_expert  # noqa: E402


def table(env: Env, seeds: list[int], noise_mm: float, overrides: dict | None) -> dict:
    peaks, ratios, outcomes = [], [], {}
    for seed in seeds:
        info = rollout_expert(env, seed, noise_mm=noise_mm, task="t1",
                              hidden_overrides=overrides)  # fmt: skip
        thr = env.task.threshold[env.task.target]
        peaks.append(info["peak_force"])
        ratios.append(info["peak_force"] / thr)
        outcomes[info["outcome"]] = outcomes.get(info["outcome"], 0) + 1
    peaks, ratios = np.array(peaks), np.array(ratios)
    return {
        "n": len(seeds), "outcomes": outcomes,
        "peak_mean": float(peaks.mean()), "peak_max": float(peaks.max()),
        "ratio_mean": float(ratios.mean()), "ratio_max": float(ratios.max()),
    }  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=100, help="held-out T1 seeds per row")
    ap.add_argument("--noise-mm", type=float, default=0.5, help="DART action noise std")
    args = ap.parse_args()

    env = Env(task="t1", render=False)
    seeds = eval_seeds("t1", args.seeds)
    rows = {
        "threshold 3 N (worst case)": {"threshold": 3.0},
        "threshold U(3, 9) N (default)": None,
    }
    ok = True
    print(f"T1 position-press overshoot over {args.seeds} held-out seeds "
          f"(DART noise {args.noise_mm} mm; success bound: peak < 1.5x threshold):")  # fmt: skip
    for name, ov in rows.items():
        r = table(env, seeds, args.noise_mm, ov)
        good = r["outcomes"].get("success", 0) == r["n"] and r["ratio_max"] < 1.5
        ok &= good
        print(
            f"  {name:31s}  peak mean {r['peak_mean']:5.2f} N  max {r['peak_max']:5.2f} N"
            f"   peak/thr mean {r['ratio_mean']:5.3f}  max {r['ratio_max']:5.3f}"
            f"   {'PASS' if good else 'FAIL'}  outcomes {r['outcomes']}"
        )
    env.close()
    print(f"T1 OVERSHOOT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
