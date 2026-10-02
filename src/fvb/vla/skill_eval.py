"""Closed loop for one skill (V9): a served image policy drives ``SortInsertEnv`` (PLAN §13).

The skill demos (``scripts/33_record_insert_demos.py``) record, per step, the front + wrist
images and the robosuite observation, with action dims 0-2 = pad displacement in the skill's
units. Here the policy receives the same ``policy_obs`` dict and its dims 0-2 (times ``gain``)
go to ``SortInsertEnv.step``; the gripper is held by the skill env.
"""

from __future__ import annotations

import time

import numpy as np

from fvb.vla.sort_io import policy_obs

CAMS = ("sortview", "robot0_eye_in_hand")


def run_insert_episode(
    sk, policy, seed: int, task: str, max_steps: int = 150, size: int = 256, gain: float = 1.0
) -> dict:
    """``sk``: ``SortInsertEnv(..., offscreen=True)``; ``policy``: ``reset()`` + ``act(obs)``."""
    sk.reset(seed)
    policy.reset()
    reason, info, lat, steps = None, {}, [], 0
    for _ in range(max_steps):
        steps += 1
        obs = dict(sk.obs)
        for c in CAMS:
            obs[f"{c}_image"] = sk.env.sim.render(camera_name=c, width=size, height=size)
        t0 = time.perf_counter()
        a = np.asarray(policy.act(policy_obs(obs, task)), float).reshape(-1)
        lat.append(time.perf_counter() - t0)
        _, _, term, trunc, info = sk.step(gain * a[:3])
        reason = info["reason"]
        if term or trunc:
            break
    lat = np.asarray(lat)
    return {
        "seed": seed,
        "outcome": reason or "timeout",
        "steps": steps,
        "depth_mm": float(info.get("depth_mm", float("nan"))),
        "max_force_N": float(info.get("force_N", float("nan"))),
        "latency_p50_s": float(np.median(lat)) if len(lat) else float("nan"),
    }
