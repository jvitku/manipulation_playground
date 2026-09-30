"""SortBoltsNuts <-> policy I/O shared by the recorder, the LeRobot converter and the closed-loop
client, so training and evaluation see identical inputs (PLAN §13, V5/V7).

``policy_obs`` turns a robosuite observation into the flat dict a served policy receives
(numpy only; the VLA side turns it into tensors). ``run_episode`` drives one episode against a
``PolicyClient`` (or anything with ``reset()`` / ``act(obs)``).
"""

from __future__ import annotations

import time

import numpy as np

CAMERAS = {"sortview": "observation.images.front", "robot0_eye_in_hand": "observation.images.wrist"}
# observation.state = joint pos (7), eef pos (3), eef quat (4), gripper qpos (2)
STATE_OBS = ("robot0_joint_pos", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
STATE_KEYS = ("joint_pos", "eef_pos", "eef_quat", "gripper_qpos")  # same order, recorded names


def policy_obs(obs: dict, task: str) -> dict:
    out = {
        "observation.state": np.concatenate([obs[k] for k in STATE_OBS]).astype(np.float32),
        "observation.tau_ext": np.asarray(obs.get("tau_ext", np.zeros(7)), np.float32),
        "observation.torque_hist": np.asarray(obs.get("torque_hist", np.zeros(150)), np.float32),
        "observation.tactile": np.asarray(obs.get("tactile", np.zeros(32)), np.float32),
        "task": task,
    }
    for cam, key in CAMERAS.items():
        img = obs.get(f"{cam}_image")
        if img is not None:
            out[key] = np.ascontiguousarray(img[::-1])  # robosuite renders upside down
    return out


def run_episode(env, policy, max_steps: int = 3600, task: str | None = None) -> dict:
    """Closed loop: obs -> policy -> action, until done. Returns outcome and latency stats."""
    obs = env.reset()
    policy.reset()
    task = task or env.instruction
    info = {"outcome": "running", "n_sorted": 0}
    lat = []
    steps = 0
    for _ in range(max_steps):
        steps += 1
        t0 = time.perf_counter()
        a = np.asarray(policy.act(policy_obs(obs, task)), float).reshape(-1)[:7]
        lat.append(time.perf_counter() - t0)
        obs, _, done, info = env.step(np.clip(a, -1, 1))
        if done:
            break
    lat = np.asarray(lat)
    return {
        "outcome": info["outcome"],
        "n_sorted": info["n_sorted"],
        "parts": info.get("parts"),
        "steps": steps,
        "latency_p50_s": float(np.median(lat)) if len(lat) else float("nan"),
        "latency_p95_s": float(np.percentile(lat, 95)) if len(lat) else float("nan"),
    }
