#!/usr/bin/env python
"""Closed-loop evaluation of a served policy on SortBoltsNuts (sim image).

    python scripts/25_eval_sort_policy.py --host policy_srv --seeds 5000-5049 --out outputs/eval_act

Start the server first (``scripts/24_serve_policy.py`` in the VLA image, same compose network).
``--task`` overrides the instruction, e.g. a per-part sub-instruction for per-part policies.
Writes one JSON line per episode to ``<out>/results.jsonl`` and prints a summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6010)
    ap.add_argument("--seeds", default="5000-5009")
    ap.add_argument("--out", default="outputs/eval_sort")
    ap.add_argument("--task", default=None)
    ap.add_argument("--max-steps", type=int, default=3600)
    ap.add_argument("--camera-size", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    import robosuite as suite

    import fvb.envs  # noqa: F401
    from fvb.vla.sort_io import run_episode
    from fvb.vla.transport import PolicyClient

    sys_path = Path(__file__).resolve().parent
    import sys

    sys.path.insert(0, str(sys_path))
    from importlib import import_module

    seeds = import_module("23_record_sort_demos").parse_seeds(args.seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = PolicyClient(args.host, args.port, timeout=args.timeout)
    rows = []
    try:
        for seed in seeds:
            env = suite.make(
                "SortBoltsNuts",
                robots="Panda",
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_names=["sortview", "robot0_eye_in_hand"],
                camera_heights=args.camera_size,
                camera_widths=args.camera_size,
                seed=seed,
                horizon=args.max_steps,
            )
            r = {"seed": seed, **run_episode(env, client, args.max_steps, args.task)}
            env.close()
            rows.append(r)
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)
    finally:
        client.close()
    n = len(rows)
    print(
        json.dumps(
            {
                "episodes": n,
                "success": sum(r["outcome"] == "success" for r in rows),
                "parts": sum(r["n_sorted"] for r in rows),
                "parts_total": 6 * n,
            }
        )
    )


if __name__ == "__main__":
    main()
