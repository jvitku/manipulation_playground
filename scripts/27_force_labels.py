#!/usr/bin/env python
"""V8: collect force/tactile features with grasp-stable and seated labels (PLAN §13 V8).

The rack expert (with human-like noise) runs SortBoltsNuts without cameras. Per attempt it is
perturbed so both outcomes occur: a random extra grasp height (pads -6 ... +12 mm off their
normal height) and, for bolts, a random release depth (+1 ... +10 mm of the tip below the
fixture top). Physics decides the outcome. Events recorded:

* ``grasp``: features at the end of the "close" phase; label 1 if the part is still held when
  the lift finishes (the expert does not fail the grasp), else 0;
* ``release``: features at the step the bolt is let go; label 1 if the tip is inside the hole at
  that moment (> 2 mm deep, within the hole radius), else 0.

Features: torque_hist (10 x 15 = tau_ext + tactile summary over 2 s), tactile (2 x 4 x 4, max over
the step), tau_ext (7), wrench history (10 x 6, Cartesian at the grip site).
Writes <out>/events.npz (X, y, kind, seed).
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
        ex.hole_est0 = ex.hole_est.copy()
        site = env.sim.model._model.site(env.grip_site).name
        pending: list = []  # (kind, step, features, part)
        n_fail = 0
        for k in range(3600):
            prev = ex.phase
            if prev == "above" and ex.t_phase == 0:
                ex.grasp_dz = float(rng.uniform(-0.006, 0.012))
            a = ex.act()
            if ex.phase == "enter" and prev != "enter":
                # >= 1 mm: the tip must reach the top-face level, so the outcome (entered freely vs
                # blocked on the chamfer / top face) leaves a contact signature. Releasing above
                # the top (tried: -4 mm) gave labels no force sensor can see (75 % detector).
                ex.release_depth = float(rng.uniform(0.001, 0.010))
                # half the bolt attempts aim 3 mm off and let go where the tip is blocked: the
                # negatives a seated detector must catch (with the default expert ~95 % seat)
                ex.release_on_block = bool(rng.random() < 0.5)
                if ex.release_on_block:
                    ang = rng.uniform(0, 2 * np.pi)
                    ex.hole_est = env.hole_top + [0.003 * np.cos(ang), 0.003 * np.sin(ang), 0.0]
                else:
                    ex.hole_est = ex.hole_est0.copy()
            # + the wrench history: each tau_ext frame mapped through pinv(J^T) at this step (the
            # arm is quasi-static over the 2 s window) - Cartesian, unlike pose-dependent joint
            # torques, which a simple classifier cannot undo
            jt = np.linalg.pinv(env.jts.jacobian(env.sim.data._data, site).T)
            wrench = (jt @ obs["torque_hist"].reshape(10, 15)[:, :7].T).T.ravel()
            feats = np.concatenate([obs["torque_hist"], obs["tactile"], obs["tau_ext"], wrench])
            obs, _, done, info = env.step(a)
            if prev == "close" and ex.phase == "lift":
                pending.append(["grasp", k, feats, ex.part, n_fail])
            if prev == "enter" and ex.phase == "let_go":
                # label at the moment of release: is the tip inside the hole (> 2 mm deep, within
                # the hole radius)? "Seated later" also depended on the chamfer guiding a blocked
                # bolt in after release, which no sensor can know at release (round 3: 78 %)
                g = env.bolt_geometry(ex.part)
                in_hole = g["depth"] > 0.002 and g["tip_lat"] < 0.0085
                X.append(feats), y.append(int(in_hole)), kind.append(1), sd.append(seed)
            # resolve pending events
            for ev in list(pending):
                knd, k0, f, part, nf0 = ev
                if knd == "grasp":
                    failed = len(ex.fail_log) > nf0 and ex.fail_log[-1][1] == part
                    if failed or ex.phase not in ("lift", "close"):
                        label = 0 if failed else 1
                        X.append(f), y.append(label), kind.append(0), sd.append(seed)
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
