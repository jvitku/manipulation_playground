#!/usr/bin/env python
"""V9: closed-loop insertion-skill evaluation of a served image policy (sim image).

    python scripts/35_eval_insert_skill.py --host srv_ta --seeds 0-49 --out outputs/eval_ta_insert

Start ``scripts/24_serve_policy.py`` (``--ta ta|ta_zero`` for TA-SmolVLA checkpoints) first.
Set-ups are seeds 800_000 + s (the demos used 700_000 + s), randomised in-hand offsets.
One JSON line per episode in ``<out>/results.jsonl``; ``<out>/final.json`` holds the summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6010)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out", default="outputs/eval_insert_skill")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    from fvb.policy.sort_skill_env import SortInsertEnv
    from fvb.vla.skill_eval import run_insert_episode
    from fvb.vla.transport import PolicyClient

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module

    seeds = import_module("23_record_sort_demos").parse_seeds(args.seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sk = SortInsertEnv(use_force=True, grasp_rand=True, offscreen=True)
    client = PolicyClient(args.host, args.port, timeout=args.timeout)
    rows = []
    try:
        for s in seeds:
            r = run_insert_episode(
                sk,
                client,
                800_000 + s,
                "put the bolt in the hole",
                args.max_steps,
                args.size,
                args.gain,
            )
            rows.append(r)
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)
    finally:
        client.close()
    n = len(rows)
    k = sum(r["outcome"] == "success" for r in rows)
    final = {"episodes": n, "success": k, "success_rate": k / n if n else float("nan")}
    for o in ("force_abort", "dropped", "timeout"):
        final[o] = sum(r["outcome"] == o for r in rows)
    (out / "final.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final))


if __name__ == "__main__":
    main()
