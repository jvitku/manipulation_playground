#!/usr/bin/env python
"""V9: insertion-skill demos with camera images, for SmolVLA / TA-SmolVLA (sim image).

From the insertion hand-over (``SortInsertEnv`` with randomised in-hand offsets) the expert runs
its own insertion (aim at its estimate, contact -> spiral search). Recorded per step through
``fvb.logging.sort_episode`` (so ``scripts/21_to_lerobot.py --mode episode --only-success``
converts it): front + wrist images, 16-D state, torque history (10 x 15), tactile, and the
action = pad displacement in the skill's units in dims 0-2 (dims 3-6 zero). Outcome "success"
when the tip is >= 8 mm in the hole.

    python scripts/33_record_insert_demos.py --seeds 0-24 --out outputs/insert_demos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CAMS = ("sortview", "robot0_eye_in_hand")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-3")
    ap.add_argument("--out", default="outputs/insert_demos")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=150)
    args = ap.parse_args()

    from fvb.logging.sort_episode import SortEpisodeWriter
    from fvb.policy.sort_skill_env import SortInsertEnv

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module

    seeds = import_module("23_record_sort_demos").parse_seeds(args.seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sk = SortInsertEnv(use_force=True, grasp_rand=True, offscreen=True)
    task = "put the bolt in the hole"
    for seed in seeds:
        sk.reset(700_000 + seed)
        env, ex = sk.env, sk.expert
        w = SortEpisodeWriter(
            out / f"ep_{seed}",
            seed,
            cameras=CAMS,
            meta={"instruction": task, "synthetic": True, "skill": "insert", "action_dims_used": 3},
        )
        obs = dict(sk.obs)
        outcome = "timeout"
        for _ in range(args.max_steps):
            for c in CAMS:
                obs[f"{c}_image"] = env.sim.render(camera_name=c, width=args.size, height=args.size)
            p0 = ex.pad_point().copy()
            a7 = np.zeros(7)
            nxt, _, _, _ = env.step(ex.act())
            a7[:3] = np.clip((ex.pad_point() - p0) / sk.scale, -1, 1)
            w.add(float(env.sim.data._data.time), obs, a7, env)
            sk.obs = nxt
            obs = dict(nxt)
            if sk.depth() >= 0.008:
                outcome = "success"
                break
            if env.outcome == "force_abort" or ex.phase in ("let_go", "up", "select"):
                outcome = "failed"
                break
        w.close({"outcome": outcome, "segments": []})
        print(json.dumps({"seed": seed, "outcome": outcome, "steps": len(w.rows["t"])}), flush=True)


if __name__ == "__main__":
    main()
