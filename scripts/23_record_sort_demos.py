#!/usr/bin/env python
"""Record scripted-expert demos of SortBoltsNuts (PLAN §13, V4).

    python scripts/23_record_sort_demos.py --seeds 1000-1049 --out outputs/sort_demos --noise

Each episode goes to ``<out>/ep_<seed>/`` (see ``fvb.logging.sort_episode``); a line per episode
is appended to ``<out>/index.jsonl``. Demos are synthetic (scripted expert + human-like noise).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1000-1003")
    ap.add_argument("--out", default="outputs/sort_demos")
    ap.add_argument("--noise", action="store_true", help="human-like expert noise")
    ap.add_argument("--camera-size", type=int, default=256)
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--max-steps", type=int, default=3600)
    args = ap.parse_args()

    import robosuite as suite

    import fvb.envs  # noqa: F401
    from fvb.logging.sort_episode import SortEpisodeWriter, part_segments
    from fvb.policy.sort_expert import NO_NOISE, ExpertNoise, SortExpert

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cams = () if args.no_cameras else ("sortview", "robot0_eye_in_hand")
    noise = ExpertNoise() if args.noise else NO_NOISE
    for seed in parse_seeds(args.seeds):
        t0 = time.time()
        env = suite.make(
            "SortBoltsNuts",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=bool(cams),
            use_camera_obs=bool(cams),
            camera_names=list(cams) or ["sortview"],
            camera_heights=args.camera_size,
            camera_widths=args.camera_size,
            seed=seed,
            horizon=args.max_steps,
        )
        obs = env.reset()
        expert = SortExpert(noise, seed)
        expert.reset(env)
        w = SortEpisodeWriter(
            out / f"ep_{seed}",
            seed,
            cameras=cams,
            meta={
                "instruction": env.instruction,
                "synthetic": True,
                "expert_noise": noise.__dict__,
                "env": "SortBoltsNuts",
            },
        )
        info = {"outcome": "running", "n_sorted": 0}
        for _ in range(args.max_steps):
            a = expert.act()
            w.add(float(env.sim.data._data.time), obs, a, env)
            obs, _, done, info = env.step(a)
            if done:
                break
        status = np.asarray(w.rows["part_status"])
        segs = part_segments(expert.log, status, env.parts)
        w.close(
            {
                "outcome": info["outcome"],
                "n_sorted": info["n_sorted"],
                "segments": segs,
                "expert_failures": expert.fail_log,
            }
        )
        row = {
            "seed": seed,
            "outcome": info["outcome"],
            "n_sorted": info["n_sorted"],
            "steps": len(w.rows["t"]),
            "ok_segments": sum(s["success"] for s in segs),
            "wall_s": round(time.time() - t0, 1),
        }
        with open(out / "index.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        env.close()


if __name__ == "__main__":
    main()
