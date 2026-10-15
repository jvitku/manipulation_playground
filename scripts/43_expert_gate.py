#!/usr/bin/env python
"""V4 gate: the scripted expert WITH its human-like noise on unseen seeds (PLAN §13).

    python scripts/43_expert_gate.py --seeds 9000-9049 --noise realistic \\
        --out outputs/v4_gate/gate_realistic.jsonl

One line per episode: outcome, parts sorted, final part statuses and the expert's failed
attempts (step, part, phase), so a run doubles as a failure census. Episodes run in parallel
processes (no cameras).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_episode(seed: int, noise_name: str, presentation: str, max_steps: int) -> dict:
    import robosuite as suite

    import fvb.envs  # noqa: F401
    from fvb.envs.sort_bolts_nuts import SortTaskParams
    from fvb.policy.sort_expert import NOISE_PRESETS, SortExpert

    t0 = time.time()
    env = suite.make(
        "SortBoltsNuts",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        seed=seed,
        horizon=max_steps,
        task=SortTaskParams(bolt_presentation=presentation),
    )
    env.reset()
    expert = SortExpert(NOISE_PRESETS[noise_name], seed)
    expert.reset(env)
    info = {"outcome": "running", "n_sorted": 0}
    steps = 0
    while steps < max_steps:
        steps += 1
        _, _, done, info = env.step(expert.act())
        if done:
            break
    out = {
        "seed": seed,
        "noise": noise_name,
        "outcome": info["outcome"],
        "sorted": int(info["n_sorted"]),
        "steps": steps,
        "status": [f"{p.kind}:{p.status}" for p in env.parts],
        "failed_attempts": [list(f) for f in expert.fail_log],
        "skipped": sorted(int(i) for i in expert.skipped),
        "wall_s": round(time.time() - t0, 1),
    }
    env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="9000-9049")
    ap.add_argument("--noise", default="realistic", help="preset in sort_expert.NOISE_PRESETS")
    ap.add_argument("--presentation", choices=("tray", "rack"), default="rack")
    ap.add_argument("--max-steps", type=int, default=3600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".config.json")).write_text(json.dumps(vars(args), indent=2))
    seeds = parse_seeds(args.seeds)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex, out.open("w") as f:
        futs = [
            ex.submit(run_episode, s, args.noise, args.presentation, args.max_steps) for s in seeds
        ]
        for fu in futs:
            r = fu.result()
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
    n = len(rows)
    ok = sum(r["outcome"] == "success" for r in rows)
    parts = sum(r["sorted"] for r in rows)
    by = {}
    for r in rows:
        by[r["outcome"]] = by.get(r["outcome"], 0) + 1
    print(f"{args.noise}: {ok}/{n} episodes ({100 * ok / n:.0f} %), {parts}/{6 * n} parts, {by}")


if __name__ == "__main__":
    main()
