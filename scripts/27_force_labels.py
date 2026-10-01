#!/usr/bin/env python
"""V8: collect force/tactile features with grasp-stable and seated labels (PLAN §13 V8).

The rack expert (with human-like noise) runs SortBoltsNuts without cameras. Per attempt it is
perturbed so both outcomes occur: a random extra grasp height (pads -6 ... +12 mm off their
normal height) and, for bolts, a random release depth (-4 ... +10 mm of the tip relative to the
fixture top). Physics decides the outcome. Events recorded:

* ``grasp``: features at the end of the "close" phase; label 1 if the part is still held when
  the lift finishes (the expert does not fail the grasp), else 0;
* ``release``: features at the step the bolt is let go; label 1 if it is seated / vanished within
  2 s, else 0.

Features: torque_hist (10 x 15 = tau_ext + tactile summary over 2 s), tactile (2 x 4 x 4, max over
the step), tau_ext (7). Writes <out>/events.npz (X, y, kind, seed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="9000-9019")
    ap.add_argument("--out", default="outputs/v8_labels/part0")
    args = ap.parse_args()

    import sys

    import robosuite as suite

    import fvb.envs  # noqa: F401
    from fvb.envs.sort_bolts_nuts import SortTaskParams
    from fvb.policy.sort_expert import ExpertNoise, SortExpert

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module

    seeds = import_module("23_record_sort_demos").parse_seeds(args.seeds)
    X, y, kind, sd = [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        env = suite.make(
            "SortBoltsNuts",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            seed=seed,
            task=SortTaskParams(bolt_presentation="rack"),
        )
        obs = env.reset()
        ex = SortExpert(ExpertNoise(), seed)
        ex.reset(env)
        pending: list = []  # (kind, step, features, part)
        n_fail = 0
        for k in range(3600):
            prev = ex.phase
            if prev == "above" and ex.t_phase == 0:
                ex.grasp_dz = float(rng.uniform(-0.006, 0.012))
            a = ex.act()
            if ex.phase == "enter" and prev != "enter":
                ex.release_depth = float(rng.uniform(-0.004, 0.010))
            feats = np.concatenate([obs["torque_hist"], obs["tactile"], obs["tau_ext"]])
            obs, _, done, info = env.step(a)
            if prev == "close" and ex.phase == "lift":
                pending.append(["grasp", k, feats, ex.part, n_fail])
            if prev == "enter" and ex.phase == "let_go":
                pending.append(["release", k, feats, ex.part, None])
            # resolve pending events
            for ev in list(pending):
                knd, k0, f, part, nf0 = ev
                if knd == "grasp":
                    failed = len(ex.fail_log) > nf0 and ex.fail_log[-1][1] == part
                    if failed or ex.phase not in ("lift", "close"):
                        label = 0 if failed else 1
                        X.append(f), y.append(label), kind.append(0), sd.append(seed)
                        pending.remove(ev)
                else:
                    st = env.parts[part].status
                    if st in ("seated", "vanished"):
                        X.append(f), y.append(1), kind.append(1), sd.append(seed)
                        pending.remove(ev)
                    elif k - k0 > 40:
                        X.append(f), y.append(0), kind.append(1), sd.append(seed)
                        pending.remove(ev)
            n_fail = len(ex.fail_log)
            if done or ex.phase == "done":
                break
        env.close()
        print(seed, "events", len(y), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "events.npz", X=np.array(X), y=np.array(y), kind=np.array(kind), seed=np.array(sd)
    )
    print("saved", len(y), "events; grasp", kind.count(0), "release", kind.count(1))


if __name__ == "__main__":
    main()
