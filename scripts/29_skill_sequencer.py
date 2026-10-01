#!/usr/bin/env python
"""V6: full sort with the learned skills inside the scripted sequence (rack presentation).

The scripted expert drives the episode; when it is about to grasp a part the TD3 *pick* policy
takes over until the part is lifted (or the attempt fails), and when a held bolt hovers <= 3 cm
over the hole the TD3 *insert* policy takes over until the tip is 8 mm in. Then the expert
continues (carry / release). Reports full-episode success and per-skill success.

    python scripts/29_skill_sequencer.py --pick outputs/td3_pick/force_s2/best.pt \
        --insert outputs/td3_sort/force_s0/best.pt --seeds 6000-6019 --out outputs/seq_force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_actor(path: str):
    import torch

    from fvb.policy.td3 import Actor

    ck = torch.load(path, map_location="cpu", weights_only=False)
    actor = Actor(ck["obs_dim"], ck["act_dim"], ck["cfg"]["hidden"])
    actor.load_state_dict(ck["actor"])
    actor.eval()
    use_force = bool(ck["meta"]["use_force"])

    def act(o):
        with torch.no_grad():
            return actor(torch.as_tensor(o, dtype=torch.float32)[None])[0].numpy()

    return act, use_force


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pick", required=True)
    ap.add_argument("--insert", required=True)
    ap.add_argument("--seeds", default="6000-6019")
    ap.add_argument("--out", default="outputs/seq")
    ap.add_argument("--noise", action="store_true", help="expert human-like noise in its parts")
    args = ap.parse_args()

    import robosuite as suite

    import fvb.envs  # noqa: F401
    from fvb.envs.sort_bolts_nuts import SortTaskParams
    from fvb.policy.sort_expert import NO_NOISE, ExpertNoise, SortExpert
    from fvb.policy.sort_skill_env import SortInsertEnv, SortPickEnv, attach_insert, attach_pick

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module

    seeds = import_module("23_record_sort_demos").parse_seeds(args.seeds)
    pick_act, pick_force = load_actor(args.pick)
    ins_act, ins_force = load_actor(args.insert)
    pick = SortPickEnv(use_force=pick_force, max_steps=200)
    ins = SortInsertEnv(use_force=ins_force, max_steps=150)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
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
        ex = SortExpert(ExpertNoise() if args.noise else NO_NOISE, seed)
        ex.reset(env)
        mode, o = "expert", None
        stats = {"pick": [0, 0], "insert": [0, 0]}  # [attempts, successes]
        for _ in range(3600):
            if mode == "expert":
                if ex.phase == "descend" and ex.t_phase <= 1:
                    o, mode = attach_pick(pick, env, ex, obs, rng), "pick"
                    stats["pick"][0] += 1
                    continue
                if (
                    ex.phase == "hover"
                    and env.parts[ex.part].kind == "bolt"
                    and ins.depth.__func__(_Shim(env, ex.part)) > -0.03
                ):
                    o, mode = attach_insert(ins, env, ex, obs), "insert"
                    stats["insert"][0] += 1
                    continue
                obs, _, done, _ = env.step(ex.act())
            else:
                skill, act = (pick, pick_act) if mode == "pick" else (ins, ins_act)
                o, _, term, trunc, sinfo = skill.step(act(o))
                obs = skill.obs
                done = env.outcome != "running"
                if term or trunc:
                    ok = sinfo["reason"] == "success"
                    stats[mode][1] += ok
                    if ok and mode == "pick":
                        ex._go("carry" if env.parts[ex.part].kind == "nut" else "reorient")
                    elif ok:
                        ex._go("let_go")
                    else:
                        ex._fail_grasp()
                    mode = "expert"
            if done or ex.phase == "done":
                break
        r = {"seed": seed, "outcome": env.outcome, "n_sorted": env.n_sorted(), **stats}
        rows.append(r)
        print(json.dumps(r), flush=True)
        env.close()
    with open(out / "results.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    n = len(rows)
    summ = {
        "episodes": n,
        "success": sum(r["outcome"] == "success" for r in rows),
        "parts": sum(r["n_sorted"] for r in rows),
        "pick": [sum(r["pick"][0] for r in rows), sum(r["pick"][1] for r in rows)],
        "insert": [sum(r["insert"][0] for r in rows), sum(r["insert"][1] for r in rows)],
        "pick_ckpt": args.pick,
        "insert_ckpt": args.insert,
    }
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ))


class _Shim:
    """Minimal object for SortInsertEnv.depth (tip depth of the expert's held bolt)."""

    def __init__(self, env, part):
        self.env, self.part = env, part

    def _tip(self):
        from fvb.envs.fasteners import DEFAULT_BOLT

        pos, Rp = self.env.part_pose(self.part)
        return pos - Rp[:, 2] * DEFAULT_BOLT.length


if __name__ == "__main__":
    main()
