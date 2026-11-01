"""Measure the M0 gate (plan §8): expert success over 100 held-out seeds per task, the hybrid
5 N hold at three hidden surface heights, and the cost of demo recording.

Not a pytest file — run it in the image and read the table:

    docker run --rm -u 1001:1001 -v "$PWD":/workspace -w /workspace -e HOME=/tmp \
        -e MUJOCO_GL=osmesa fvb-lab:latest python tests/nfv/measure_m0_gate.py

Gate: every task >= 95 % expert success (with the default 0.5 mm DART noise active), and the
hybrid controller's mean F_n within 5 +- 1 N at dz = -8/0/+8 mm. Exit code 0 iff the gate
passes. Success rates carry 95 % Wilson intervals.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from nfv.seeds import TASK_NAMES, eval_seeds  # noqa: E402
from nfv.sim.env import Action, Env  # noqa: E402
from nfv.sim.experts import rollout_expert  # noqa: E402

GATE_RATE = 0.95
FT_5N = np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0])


def wilson95(k: int, n: int) -> tuple[float, float]:
    """95 % Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.959964, k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def expert_table(env: Env, n_seeds: int, noise_mm: float) -> tuple[bool, list[str]]:
    ok, lines = True, []
    for task in TASK_NAMES:
        t0 = time.monotonic()
        outcomes: dict[str, int] = {}
        peaks = []
        for seed in eval_seeds(task, n_seeds):
            info = rollout_expert(env, seed, noise_mm=noise_mm, task=task)
            outcomes[info["outcome"]] = outcomes.get(info["outcome"], 0) + 1
            peaks.append(info["peak_force"])
        k = outcomes.get("success", 0)
        lo, hi = wilson95(k, n_seeds)
        rate = k / n_seeds
        ok &= rate >= GATE_RATE
        fails = {o: c for o, c in outcomes.items() if o != "success"}
        lines.append(
            f"  {task}  {k:3d}/{n_seeds}  = {rate:5.1%}  wilson95 [{lo:.1%}, {hi:.1%}]"
            f"  peak_F mean {np.mean(peaks):5.2f} N  "
            f"{'PASS' if rate >= GATE_RATE else 'FAIL'}"
            f"  ({time.monotonic() - t0:.0f}s){'  fails: ' + str(fails) if fails else ''}"
        )
    return ok, lines


def hybrid_table(env: Env) -> tuple[bool, list[str]]:
    ok, lines = True, []
    seed = eval_seeds("t2", 1)[0]
    for dz_mm in (-8.0, 0.0, 8.0):
        env.reset(
            seed=seed,
            task="t2",
            hidden_overrides={"dz": dz_mm * 1e-3, "tilt_x": 0.0, "tilt_y": 0.0},
        )
        f_n: list[float] = []
        for i in range(30):  # 3 s: descend + build, average the final second at 200 Hz
            obs, _ = env.step(Action(dp=np.zeros(3), f_target=FT_5N))
            if i >= 20:
                f_n.extend(-obs.wrench_window[:, 2])
        mean, worst = float(np.mean(f_n)), float(np.max(np.abs(np.array(f_n) - 5.0)))
        good = abs(mean - 5.0) < 1.0
        ok &= good
        lines.append(
            f"  dz {dz_mm:+5.1f} mm   mean F_n {mean:5.2f} N   max |err| {worst:4.2f} N   "
            f"{'PASS' if good else 'FAIL'}"
        )
    return ok, lines


def demo_timing(n: int, noise_mm: float) -> str:
    from nfv.data.record import record_demos

    with tempfile.TemporaryDirectory() as tmp:
        env = Env(task="t2", render=True)
        t0 = time.monotonic()
        index = record_demos("t2", n=n, out_dir=tmp, noise_mm=noise_mm, env=env)
        wall = time.monotonic() - t0
        env.close()
        size_mb = sum(f.stat().st_size for f in (Path(tmp) / "t2").glob("*.npz")) / 1e6
    return (
        f"  {index['n']} t2 demos ({index['tried']} tried) in {wall:.1f} s "
        f"= {wall / index['n']:.2f} s/demo, {size_mb / index['n']:.1f} MB/demo compressed"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=100, help="held-out seeds per task")
    ap.add_argument("--noise-mm", type=float, default=0.5, help="DART action noise std")
    ap.add_argument("--demo-n", type=int, default=20, help="rendered t2 demos to time (0 = skip)")
    args = ap.parse_args()

    env = Env(task="t0", render=False)
    print(
        f"M0 gate - expert success over {args.seeds} held-out seeds/task "
        f"(DART noise {args.noise_mm} mm):"
    )
    ok_e, lines = expert_table(env, args.seeds, args.noise_mm)
    print("\n".join(lines))
    print("hybrid controller: hold 5 N on the hidden-height plate (gate: mean within +-1 N):")
    ok_h, lines = hybrid_table(env)
    print("\n".join(lines))
    env.close()
    if args.demo_n:
        print("demo recording cost (rendered, both cameras, 96x96 osmesa):")
        print(demo_timing(args.demo_n, args.noise_mm))
    verdict = ok_e and ok_h
    print(
        f"M0 GATE: {'PASS' if verdict else 'FAIL'} "
        f"(expert >= {GATE_RATE:.0%} per task: {ok_e}; 5 N +- 1 N hold: {ok_h})"
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
