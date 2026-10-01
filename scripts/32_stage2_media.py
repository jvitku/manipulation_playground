#!/usr/bin/env python
"""V10: media for the Stage 2 report (docs/stage2/media/).

* expert.mp4         - the scripted expert sorting all six parts (rack presentation, noise-free);
* insert_td3.mp4     - TD3+BC insertion, with force | without force, same set-ups side by side;
* curves.json        - TD3+BC insert / pick evaluation curves (mean over seeds) per condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("docs/stage2/media")
CAMS = ("sortview", "robot0_eye_in_hand")


def frame(env, h=256):
    ims = [env.sim.render(camera_name=c, width=h, height=h)[::-1] for c in CAMS]
    return np.concatenate(ims, 1)


def expert_video(seed=7000):
    import imageio.v2 as imageio
    import robosuite as suite

    import fvb.envs  # noqa: F401
    from fvb.envs.sort_bolts_nuts import SortTaskParams
    from fvb.policy.sort_expert import NO_NOISE, SortExpert

    env = suite.make(
        "SortBoltsNuts",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        seed=seed,
        task=SortTaskParams(bolt_presentation="rack"),
    )
    env.reset()
    ex = SortExpert(NO_NOISE, seed)
    ex.reset(env)
    frames = []
    for k in range(3600):
        _, _, done, info = env.step(ex.act())
        if k % 4 == 0:
            frames.append(frame(env, 224))
        if done:
            break
    imageio.mimsave(OUT / "expert.mp4", frames, fps=10, macro_block_size=1)
    return {"outcome": info["outcome"], "steps": k + 1, "sim_s": (k + 1) / 20}


def skill_video(name, acts, seeds):
    """Same set-ups, one row per seed: [policy A | policy B]."""
    import imageio.v2 as imageio

    from fvb.policy.sort_skill_env import SortInsertEnv

    rows, outcomes = [], []
    for s in seeds:
        clips = []
        for act, use_force in acts:
            env = SortInsertEnv(use_force=use_force, grasp_rand=True, offscreen=True)
            o = env.reset(s)
            fr, reason = [frame(env.env, 192)], None
            for _ in range(env.max_steps):
                o, _, term, trunc, info = env.step(act(o))
                fr.append(frame(env.env, 192))
                if term or trunc:
                    reason = info["reason"]
                    break
            clips.append(fr)
            outcomes.append(reason)
        n = max(len(c) for c in clips)
        clips = [c + [c[-1]] * (n - len(c)) for c in clips]
        rows += [np.concatenate([c[i] for c in clips], 1) for i in range(n)]
    imageio.mimsave(OUT / f"{name}.mp4", rows, fps=10, macro_block_size=1)
    return outcomes


def td3_actor(path):
    import torch

    from fvb.policy.td3 import Actor

    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = Actor(ck["obs_dim"], ck["act_dim"], ck["cfg"]["hidden"])
    a.load_state_dict(ck["actor"])
    a.eval()
    return (
        lambda o: a(torch.as_tensor(o, dtype=torch.float32)[None])[0].detach().numpy(),
        bool(ck["meta"]["use_force"]),
    )


def curves():
    out = {}
    for name, pat, n in (
        ("insert", "outputs/td3_sort/{c}_s{s}", 10),
        ("pick", "outputs/td3_pick/{c}_s{s}", 10),
    ):
        for c in ("force", "noforce"):
            series = []
            for s in range(n):
                f = Path(pat.format(c=c, s=s)) / "progress.json"
                if f.exists():
                    ev = json.loads(f.read_text())["evals"]
                    series.append({e["t"]: e["success_rate"] for e in ev})
            ts = sorted({t for d in series for t in d})
            out[f"{name}_{c}"] = {
                "t": ts,
                "mean": [float(np.mean([d[t] for d in series if t in d])) for t in ts],
                "seeds": [[d.get(t) for t in ts] for d in series],
            }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = (
        {"expert": json.loads((OUT / "meta.json").read_text())["expert"]}
        if (OUT / "expert.mp4").exists() and (OUT / "meta.json").exists()
        else {"expert": expert_video()}
    )
    f_act = td3_actor("outputs/td3_sort_grand/force_s0/best.pt")
    n_act = td3_actor("outputs/td3_sort_grand/noforce_s3/best.pt")
    meta["insert_td3"] = skill_video("insert_td3", [f_act, n_act], [50002, 50005])
    (OUT / "curves.json").write_text(json.dumps(curves()))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
