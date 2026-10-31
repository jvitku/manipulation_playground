"""Record scripted-expert demonstrations to npz shards (one per episode) + index.json.

Layout under ``out_dir``:
    <out_dir>/<task>/ep_<seed>.npz     one episode
    <out_dir>/<task>/index.json        episode list + counts (written last, re-written whole)

Episode shard keys (T = episode length in 10 Hz policy steps; observations are the step's
INPUT, actions its decision — the standard (o_t, a_t) pairing):
    rgb_static     uint8  [T, 96, 96, 3]   static camera
    rgb_wrist      uint8  [T, 96, 96, 3]   wrist camera
    state          f32    [T, 8]           env state vector (env.py legend)
    wrench         f32    [T, 6]           10 Hz wrench (last 200 Hz sample)
    wrench_window  f32    [T, 20, 6]       the full 200 Hz window per step
    dp             f32    [T, 3]           CLEAN expert action — the training label
    dp_exec        f32    [T, 3]           what was executed = dp + DART noise
    gripper        f32    [T]              commanded grip 0..1
    f_target       f32    [T, 6]           wrench the expert held (zeros in free space)
    s_fv2          f32    [T]              ForceVLA2 progress label (experts.py eq.)
    s_lin          f32    [T]              linear phase progress
    subtask        i16    [T]              expert FSM phase mapped into task SUBTASKS
    plus 0-d metadata: seed, task, instruction, outcome, success, peak_force, noise_mm.

DART noise convention (lab/tasks/task4_force_token.py): the EXECUTED dp carries N(0, sigma)
per axis, the stored ``dp`` label stays clean; the images/state see the noisy motion.
``f_target`` is stored as commanded, (6,) zeros when the expert ran pure position mode —
so a later hybrid-head model can regress it directly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from nfv.seeds import demo_seeds
from nfv.sim.env import Env
from nfv.sim.experts import DART_SIGMA_MM, make_expert, noisy


def record_episode(env: Env, seed: int, rng: np.random.Generator, noise_mm: float) -> dict:
    """Roll one expert episode; returns {"arrays": {...}, "meta": {...}}."""
    obs = env.reset(seed)
    expert = make_expert(env)
    rows: dict[str, list] = {k: [] for k in (
        "rgb_static", "rgb_wrist", "state", "wrench", "wrench_window",
        "dp", "dp_exec", "gripper", "f_target", "s_fv2", "s_lin", "subtask",
    )}
    info = {"outcome": "timeout", "peak_force": 0.0}
    while True:
        es = expert.act()
        executed = noisy(es.action, rng, noise_mm)
        rows["rgb_static"].append(obs.rgb_static)
        rows["rgb_wrist"].append(obs.rgb_wrist)
        rows["state"].append(obs.state)
        rows["wrench"].append(obs.wrench)
        rows["wrench_window"].append(obs.wrench_window)
        rows["dp"].append(np.asarray(es.action.dp, dtype=np.float32))
        rows["dp_exec"].append(np.asarray(executed.dp, dtype=np.float32))
        rows["gripper"].append(np.float32(es.action.gripper))
        ft = np.zeros(6, np.float32) if es.action.f_target is None else es.action.f_target
        rows["f_target"].append(np.asarray(ft, dtype=np.float32))
        rows["s_fv2"].append(np.float32(es.s_fv2))
        rows["s_lin"].append(np.float32(es.s_lin))
        rows["subtask"].append(np.int16(es.subtask_id))
        obs, info = env.step(executed)
        if info["done"]:
            break
    arrays = {k: np.stack(v) for k, v in rows.items()}
    meta = {
        "seed": seed,
        "task": env.task_name,
        "instruction": env.instruction,
        "outcome": info["outcome"],
        "success": bool(info["success"]),
        "peak_force": float(info["peak_force"]),
        "steps": int(arrays["dp"].shape[0]),
        "noise_mm": noise_mm,
    }
    return {"arrays": arrays, "meta": meta}


def record_demos(
    task: str,
    n: int,
    out_dir: str | Path,
    noise_mm: float = DART_SIGMA_MM,
    seeds: list[int] | None = None,
    env: Env | None = None,
    keep_failures: bool = False,
) -> dict:
    """Record ``n`` demos of ``task`` to ``out_dir/<task>/`` and return the index dict.

    Seeds default to the task's demo block (nfv.seeds.demo_seeds). Failed episodes are
    recorded through but not stored unless ``keep_failures`` — extra seeds are consumed until
    ``n`` successes are on disk (the index reports how many were tried).
    """
    out = Path(out_dir) / task
    out.mkdir(parents=True, exist_ok=True)
    env = env or Env(task=task, render=True)
    episodes, tried = [], 0
    seed_iter = iter(seeds if seeds is not None else demo_seeds(task, max(2 * n, n + 8)))
    t_start = time.monotonic()
    while len(episodes) < n:
        seed = next(seed_iter)
        tried += 1
        rng = np.random.default_rng(seed + 7_777_777)  # noise stream, decoupled from the env's
        ep = record_episode(env, seed, rng, noise_mm)
        if not ep["meta"]["success"] and not keep_failures:
            continue
        fname = f"ep_{seed:06d}.npz"
        np.savez_compressed(out / fname, **ep["arrays"], **ep["meta"])
        episodes.append({"file": fname, **ep["meta"]})
    index = {
        "task": task,
        "n": len(episodes),
        "tried": tried,
        "noise_mm": noise_mm,
        "wall_s": round(time.monotonic() - t_start, 2),
        "episodes": episodes,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2))
    return index
