"""Task 6 - "RL from scratch: TD3" (id `td3`): the repo's Stage-1 RL experiment in the lab.

Wraps, by import, `scripts/18_train_td3.py`'s default (`--task arm`, no demos) run: `ArmRLEnv` on a
Panda `ArmWorld` with its privileged reward (depth progress + potential shaping toward the true
hole + time cost + force penalty, +50 success / -20 abort), the `fvb.policy.td3` TD3 agent, a
deterministic-actor evaluation every `eval_every` steps on held-out seeds, best-by-eval-success
checkpointing. `scripts/18` itself is untouched and keeps running as before.

Where TD3+BC (the next task) plugs in: script 18's `collect_demos(env, n, seed0, bufs)` fills the
replay buffer with expert transitions, `agent.update(buf, rng, demo_buf, bc_weight)` takes the
demo buffer + a (linearly decaying) BC weight, `agent.update(..., q_weight=...)` does the offline
warm start. Here `_make_env` / `_make_agent` / `_train_loop` are the seams: a TD3+BC task adds a
`demo_buf` and a `bc_weight(t)` to `_train_loop` and keeps everything else.

Metrics (`metrics.jsonl`, step = environment steps):
  - every finished training episode: {ep_return, ep_return_ma (mean of the last 20),
    train_success_ma (success share of the last 50)};
  - every LOSS_EVERY updates (mean over the window): {critic_loss, q1_mean, target_mean,
    actor_loss}  (actor_loss only on windows that held an actor update; the actor updates every
    2nd critic update);
  - every eval: {eval_success, eval_return, eval_s}.
Measured on a 20k-step run: 972 episodes, 77 loss rows, 2 evals (1054 lines, ~120 kB before the
value rounding added after). At 300k steps expect 10-15k episode lines (more if the agent learns
short successful episodes), 1.5k loss lines and 30 eval lines: roughly 1-1.5 MB.

Checkpoints (`ckpt/`): written at every eval, pruned to the newest "best" (eval success, then
return) plus the newest one ("last"); `last.pt` is always the latest. Payload is
`TD3.state(meta)` = {"actor", "critic", "cfg", "obs_dim", "act_dim", "priv_dim", "meta"} with
meta = {algo "td3", task "arm", use_force, history, xy_scale, z_scale, norm (path of the fixed
normalisation, `lab/tasks/arm_norm.json`), args (the form params), task_params, reward, td3, t,
eval}. `evaluate` rebuilds the policy with `fvb.policy.td3.TD3Policy` from the checkpoint alone.

result.json written by `evaluate` (same shape as task 4's so the `video` viewer reads it):
    {"n_episodes", "success", "counts", "wilson95", "force_zeroed", "trained_with_force",
     "model": "td3", "H": history, "K": 1, "trained_steps", "clearance_mm", "offset_sigma_mm",
     "fps", "video", "buckets", "episodes": [rendered episodes with force/act/depth_mm series]}
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from lab.core.runctx import atomic_write_json
from lab.core.video import frames_to_mp4
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.task1_bc_mse import OVERHEAD_S, wilson95
from lab.tasks.task4_force_token import BUCKET_EDGES, FPS, OUTCOMES, VH, VW, _card, _drain, _strip

NORM_PATH = Path(__file__).with_name("arm_norm.json")  # copy of outputs/s1_arm/force/norm.json
LOSS_EVERY = 200  # updates per logged loss row
POLL_EVERY = 10  # steps between ctx.should_continue() polls
EVAL_SEED0, MONITOR_SEED0 = 50000, 70000  # held-out: disjoint from the training seeds (>= 1e6)
HISTORY, MA_RETURN, MA_SUCCESS, KEEP_START = 4, 20, 50, 5000
FALLBACK_STEPS_PER_S, FALLBACK_EVAL_S = 100.0, 8.0
SH = 70


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------- env / agent
def _make_env(P: dict):
    """(env, world, task params, reward params) as scripts/18 builds the arm task."""
    from fvb.policy.arm_task import ArmTaskParams, ArmWorld
    from fvb.policy.data import Norm
    from fvb.policy.rl_env import ArmRLEnv, RewardParams

    p = ArmTaskParams()
    rp = RewardParams(time_cost=float(P["time_cost"]), force_coef=float(P["force_penalty"]),
                      lat_coef=float(P["shaping_weight"]))  # fmt: skip
    norm = Norm.from_json(json.loads(NORM_PATH.read_text()))
    world = ArmWorld(p, seed=int(P["seed"]))
    env = ArmRLEnv(p, world, norm, use_force=bool(P["use_force"]), history=HISTORY, rp=rp)
    return env, world, p, rp


def _make_agent(env, device: str):
    from fvb.policy.td3 import TD3, TD3Config

    cfg = TD3Config()
    return TD3(env.obs_dim, env.act_dim, cfg, device), cfg


def _run_eval(ctx, env, agent, seeds: list[int]) -> dict | None:
    """Deterministic actor on fixed seeds (scripts/18 `run_eval`); None if stopped meanwhile."""
    ok, rets, peaks = [], [], []
    for s in seeds:
        if not ctx.should_continue():
            return None
        o, ret, done, peak, info = env.reset(s), 0.0, False, 0.0, {}
        while not done:
            o, r, term, trunc, info = env.step(agent.act(o))
            ret += r
            peak = max(peak, info["force_N"])
            done = term or trunc
        ok.append(info["reason"] == "success")
        rets.append(ret)
        peaks.append(peak)
    return {"success_rate": float(np.mean(ok)), "mean_return": float(np.mean(rets)),
            "mean_peak_F_N": float(np.mean(peaks))}  # fmt: skip


def _save(ctx, agent, meta: dict, t: int, label: str, metrics: dict | None = None) -> None:
    """Checkpoint, then prune ckpt/ to the newest 'best' and the newest checkpoint."""
    state = agent.state(meta)
    state["actor"] = {k: v.detach().cpu().clone() for k, v in state["actor"].items()}
    state["critic"] = {k: v.detach().cpu().clone() for k, v in state["critic"].items()}
    new = ctx.save_ckpt(t, state, label=label, metrics=metrics)
    idx_path = ctx.dir / "ckpt" / "index.json"
    index = json.loads(idx_path.read_text())
    best = [e for e in index if e["label"] == "best"]
    keep = {new.name} | ({best[-1]["file"]} if best else set())
    for e in index:
        if e["file"] not in keep:
            (ctx.dir / "ckpt" / e["file"]).unlink(missing_ok=True)
    atomic_write_json(idx_path, [e for e in index if e["file"] in keep])


# ------------------------------------------------------------------------------------- training
def train(ctx) -> None:
    _run(ctx, None)


def _run(ctx, hooks) -> None:
    """Build env + agent and run the loop. `hooks` (task 7) adds demos and a BC term: an object
    with `prepare(ctx, env, buf, rng) -> bool` (False = stopped), `update(agent, buf, rng, t) ->
    dict` (replaces `agent.update(buf, rng)`) and `meta() -> dict` (merged into the ckpt meta)."""
    import torch

    P = ctx.params
    torch.set_num_threads(1)
    seed = int(P["seed"])
    torch.manual_seed(seed)
    world = None
    try:
        env, world, p, rp = _make_env(P)
        _train_loop(ctx, P, env, p, rp, hooks)
    finally:
        if world is not None:
            world.close()


def _train_loop(ctx, P: dict, env, p, rp, hooks=None) -> None:
    from fvb.policy.td3 import ReplayBuffer

    steps, eval_every = int(P["env_steps"]), int(P["eval_every"])
    n_eval = int(P.get("eval_episodes", 20))
    seed, expl = int(P["seed"]), float(P["expl_noise"])
    rng = np.random.default_rng(seed)
    dev = _device()
    agent, cfg = _make_agent(env, dev)
    buf = ReplayBuffer(
        env.obs_dim,
        env.act_dim,
        min(1_000_000, steps) if hooks is None else hooks.buffer_size(steps),
    )
    start_steps = min(KEEP_START, steps // 5)  # uniform-random actions first (script: 5000)
    meta = {"algo": "td3", "task": "arm", "use_force": bool(P["use_force"]), "history": HISTORY,
            "xy_scale": float(env.scale[0]), "z_scale": float(env.scale[2]),
            "norm": str(NORM_PATH), "args": dict(P), "task_params": asdict(p),
            "reward": asdict(rp), "td3": asdict(cfg)}  # fmt: skip
    if hooks is not None:
        meta.update(hooks.meta())
    ctx.set_status(step=0, total_steps=steps, message=f"TD3 on the arm task, {dev}")

    eval_seeds = [MONITOR_SEED0 + i for i in range(n_eval)]
    train_seed = 1_000_000 * (seed + 1)
    best = (-1.0, -np.inf)
    rets: deque = deque(maxlen=MA_RETURN)
    succ: deque = deque(maxlen=MA_SUCCESS)
    acc: dict[str, list[float]] = {}
    n_upd = n_ep = 0
    saved_t = -1

    def flush_losses(t):
        nonlocal acc
        if acc:
            ctx.log(t, **{k: round(float(np.mean(v)), 4) for k, v in acc.items()})
            acc = {}

    if hooks is not None and not hooks.prepare(ctx, env, buf, rng):
        ctx.set_status(state="stopped")
        return
    if hooks is not None:
        meta.update(hooks.final_meta())
    o = env.reset(train_seed)
    ep_ret = 0.0
    for t in range(1, steps + 1):
        if t % POLL_EVERY == 1 and not ctx.should_continue():
            flush_losses(t - 1)
            if t > 1 and saved_t != t - 1:
                _save(ctx, agent, {**meta, "t": t - 1}, t - 1, "last")
            ctx.set_status(state="stopped")
            return
        if t <= start_steps:
            a = rng.uniform(-1, 1, env.act_dim).astype(np.float32)
        else:
            a = np.clip(agent.act(o) + rng.normal(0, expl, env.act_dim), -1, 1).astype(np.float32)
        o2, r, term, trunc, info = env.step(a)
        buf.add(o, a, r, o2, term)  # truncation (timeout) still bootstraps
        o = o2
        ep_ret += r
        if t > start_steps:
            upd = agent.update(buf, rng) if hooks is None else hooks.update(agent, buf, rng, t)
            for k, v in upd.items():
                acc.setdefault(k, []).append(v)
            n_upd += 1
            if n_upd % LOSS_EVERY == 0:
                flush_losses(t)
        if term or trunc:
            rets.append(ep_ret)
            succ.append(info["reason"] == "success")
            ctx.log(t, ep_return=round(ep_ret, 3), ep_return_ma=round(float(np.mean(rets)), 3),
                    train_success_ma=round(float(np.mean(succ)), 3))  # fmt: skip
            n_ep += 1
            o = env.reset(train_seed + n_ep)
            ep_ret = 0.0
        if t % 100 == 0:
            ctx.set_status(step=t)
        if t % eval_every == 0 or t == steps:
            flush_losses(t)
            t0 = time.perf_counter()
            ev = _run_eval(ctx, env, agent, eval_seeds)
            if ev is None:  # stopped inside the eval
                _save(ctx, agent, {**meta, "t": t}, t, "last")
                ctx.set_status(state="stopped")
                return
            ctx.log(t, eval_success=ev["success_rate"], eval_return=ev["mean_return"],
                    eval_s=time.perf_counter() - t0)  # fmt: skip
            key = (ev["success_rate"], ev["mean_return"])
            improved = key > best
            best = max(best, key)
            m = {"eval_success": ev["success_rate"], "eval_return": ev["mean_return"]}
            _save(ctx, agent, {**meta, "t": t, "eval": ev}, t, "best" if improved else "last", m)
            saved_t = t
            ctx.set_status(step=t)
            o = env.reset(train_seed + n_ep)  # the eval used the env: restart the training episode
            ep_ret = 0.0
    ctx.done(f"trained {steps} steps ({n_ep} episodes), best eval success {best[0]:.2f}")


# ------------------------------------------------------------------------------------- video
def _episode(policy, p, world, seed: int, render: bool):
    """Closed-loop arm episode with a TD3Policy. A generator: yields video frames when `render`,
    returns the record (series only when rendered) via StopIteration. The frame layout matches
    task 4's, which the `video` viewer depends on: initial frame, one per step, 1 s hold."""
    import cv2
    import mujoco

    from fvb.ft.frames import site_to_world
    from fvb.policy.arm_task import ArmTask

    policy.reset()
    task = ArmTask(replace(p, **policy.task_overrides), seed, world)
    rend = cam = None
    fmag, fz = [], []
    ser = {"force": [], "act": [], "depth_mm": []}
    if render:
        for g in ("hole_w_px", "hole_w_nx", "hole_w_py", "hole_w_ny"):  # see-through, as script 13
            gid = mujoco.mj_name2id(task.rig.m, mujoco.mjtObj.mjOBJ_GEOM, g)
            task.rig.m.geom_rgba[gid] = [0.55, 0.5, 0.45, 0.35]
        rend = mujoco.Renderer(task.rig.m, height=VH, width=VW)
        cam = mujoco.MjvCamera()
        c = task.env.hole_center_world
        cam.lookat[:] = [c[0], c[1], c[2] + 0.06]
        cam.distance, cam.azimuth, cam.elevation = 0.55, 150.0, -22.0

    def frame(status=""):
        rend.update_scene(task.rig.d, camera=cam)
        img = np.vstack([rend.render().copy(), _strip(fmag, fz, -10.0, 60.0, p.max_steps)])
        txt = (f"t {task.k / FPS:4.2f} s  depth {1e3 * task.depth():5.1f} mm  "
               f"|F| {task.force_norm():5.2f} N")  # fmt: skip
        cv2.rectangle(img, (0, 0), (VW, 18), (255, 255, 255), -1)
        cv2.putText(img, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1, cv2.LINE_AA)
        if status:
            col = (30, 140, 60) if status == "success" else (200, 50, 40)
            cv2.putText(img, status.upper(), (10, VH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2,
                        cv2.LINE_AA)  # fmt: skip
        return img

    try:
        if render:
            yield frame()
        done, reason, peak, n_jams, retracting = False, None, 0.0, 0, False
        while not done:
            a = policy.act(task.observe())
            f = task.force_norm()
            if a[2] > 1e-4 and f > 1.0 and not retracting:
                n_jams += 1
                retracting = True
            if a[2] < 0:
                retracting = False
            done, reason = task.step(a)
            peak = max(peak, task.force_norm())
            if render:
                w = site_to_world(task.rig.ft_comp_last(), task.rig.ee_rot())
                ser["force"].append(np.round(w, 3).tolist())
                ser["act"].append(np.round(1e3 * np.asarray(a, float), 3).tolist())
                ser["depth_mm"].append(round(1e3 * task.depth(), 3))
                fmag.append(float(np.linalg.norm(w[:3])))
                fz.append(float(w[2]))
                yield frame()
        if render:
            last = frame(reason)
            for _ in range(FPS):
                yield last
        rec = {"seed": seed, "offset_mm": np.round(1e3 * task.hole_xy, 3).tolist(),
               "outcome": reason, "steps": task.k, "peak_F_N": round(peak, 3),
               "recoveries": n_jams, "final_depth_mm": round(1e3 * task.depth(), 3)}  # fmt: skip
        return {**rec, **ser} if render else rec
    finally:
        if rend is not None:
            rend.close()


def evaluate(ctx) -> None:
    import torch

    from fvb.policy.arm_task import ArmTask, ArmTaskParams, ArmWorld
    from fvb.policy.td3 import TD3Policy

    P = ctx.params
    n = int(P.get("n_episodes", 50))
    render_n = max(0, min(int(P.get("render_n", 4)), n))
    seed = int(P.get("eval_seed", 0))
    sigma = float(P.get("offset_sigma_mm", 1.5))
    torch.set_num_threads(1)
    policy = TD3Policy(P["ckpt"], "cpu")  # network + normalisation come from the ckpt only
    trained_with_force = bool(policy.use_force)
    zeroed = bool(P.get("force_zeroed", False)) or not trained_with_force
    policy.use_force = not zeroed
    p = ArmTaskParams(offset_sigma=sigma * 1e-3)
    seeds = [EVAL_SEED0 + seed * 1000 + i for i in range(n)]
    ctx.set_status(step=0, total_steps=n)
    world = ArmWorld(p, seed=0)
    rows, rendered, stopped = [], [], False
    try:

        def frames():
            nonlocal stopped
            for i in range(render_n):
                if not ctx.should_continue():
                    stopped = True
                    return
                gen = _episode(policy, p, world, seeds[i], True)
                ox, oy = (1e3 * ArmTask(p, seeds[i], world).hole_xy).tolist()
                title = [f"episode {i + 1} of {render_n}", f"seed {seeds[i]}",
                         f"hole offset ({ox:+.2f}, {oy:+.2f}) mm",
                         f"TD3 after {policy.meta.get('t', '?')} steps",
                         "force input: " + ("ZEROED" if zeroed else "on")]  # fmt: skip
                for _ in range(int(0.75 * FPS)):
                    yield _card(title)
                rec = yield from gen
                rendered.append(rec)

        video = None
        if render_n:
            try:
                frames_to_mp4(frames(), ctx.dir / "rollout.mp4", fps=FPS)
                video = "rollout.mp4"
                rows.extend(rendered)
            except Exception as e:  # noqa: BLE001 - a render failure must not lose the statistics
                print(f"video rendering failed ({type(e).__name__}: {e}); re-running unrendered")
                ctx.set_status(message=f"video failed: {type(e).__name__}: {e}")
                (ctx.dir / "rollout.mp4").unlink(missing_ok=True)
                for i in range(render_n):
                    rows.append(_drain(_episode(policy, p, world, seeds[i], False)))
            if stopped:
                ctx.set_status(state="stopped")
                return
            ctx.set_status(step=render_n, total_steps=n)
        for i in range(render_n, n):
            if not ctx.should_continue():
                ctx.set_status(state="stopped")
                return
            rows.append(_drain(_episode(policy, p, world, seeds[i], False)))
            ctx.set_status(step=i + 1, total_steps=n)
    finally:
        world.close()

    k = sum(r["outcome"] == "success" for r in rows)
    cnt = Counter(r["outcome"] for r in rows)
    mags = np.array([np.hypot(*r["offset_mm"]) for r in rows])
    ok = np.array([r["outcome"] == "success" for r in rows])
    buckets = []
    for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:], strict=True):
        sel = (mags >= lo) & (mags < hi) if hi is not None else (mags >= lo)
        buckets.append({"lo": lo, "hi": hi, "n": int(sel.sum()),
                        "success": float(ok[sel].mean()) if sel.any() else None})  # fmt: skip
    atomic_write_json(
        ctx.dir / "result.json",
        {"n_episodes": n, "success": k / max(1, n),
         "counts": {o: int(cnt.get(o, 0)) for o in OUTCOMES}, "wilson95": wilson95(k, n),
         "force_zeroed": zeroed, "trained_with_force": trained_with_force, "model": "td3",
         "H": int(policy.meta["history"]), "K": 1, "trained_steps": policy.meta.get("t"),
         "clearance_mm": 1e3 * p.clearance, "offset_sigma_mm": sigma, "fps": FPS, "video": video,
         "buckets": buckets, "episodes": rows[:render_n] if video else rows[:0]},
    )  # fmt: skip
    ctx.done("evaluated")


# ------------------------------------------------------------------------------------ estimate
def estimate(params: dict) -> float:
    cal = registry.calib("td3")
    steps = int(params["env_steps"])
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    eval_s = cal.get("eval_s") or FALLBACK_EVAL_S
    n_evals = steps // max(1, int(params["eval_every"])) + 1
    return steps / sps + n_evals * eval_s + cal.get("overhead_s", OVERHEAD_S)


SPEC = TaskSpec(
    id="td3",
    order=11,
    title="RL from scratch: TD3",
    subtitle="Learn the hidden-hole insertion from reward alone: twin critics, a delayed actor, "
    "and a seed lottery.",
    params=[
        Param("env_steps", "Environment steps", "int", 100_000, min=20_000, max=300_000,
              step=10_000, long=True,
              help="the repo's runs used 150k; CPU-bound simulation, tens of minutes to hours"),
        Param("seed", "Seed", "int", 0, min=0, max=99,
              help="the whole story: the repo's seeds 0-4 gave very different results"),
        Param("expl_noise", "Exploration noise", "float", 0.2, min=0.05, max=0.5, step=0.05,
              help="Gaussian noise on the actor's action in [-1, 1]"),
        Param("force_penalty", "Force penalty", "float", 0.05, min=0.0, max=0.2, step=0.01,
              help="reward minus this x (|F| - 20 N) above 20 N"),
        Param("time_cost", "Time cost", "float", 0.1, min=0.0, max=0.5, step=0.05,
              help="reward minus this every step (before the 0.1 reward scale)"),
        Param("shaping_weight", "Hole-shaping weight", "float", 2.0, min=0.0, max=5.0, step=0.5,
              help="potential-based shaping toward the true hole (policy never sees it)"),
        Param("use_force", "Feed the force channels", "bool", True),
        Param("eval_every", "Evaluate every (steps)", "int", 10_000, min=1000, max=50_000,
              step=1000),
        Param("eval_episodes", "Episodes per evaluation", "int", 20, min=2, max=50, step=2),
    ],
    eval_params=[
        Param("n_episodes", "Episodes", "int", 50, min=2, max=200, step=10),
        Param("offset_sigma_mm", "Hole offset sigma (mm)", "float", 1.5, min=0.5, max=3.0,
              step=0.5, help="trained at 1.5; the hole position is hidden from the policy"),
        Param("force_zeroed", "Zero the force input (twin)", "bool", False),
        Param("render_n", "Episodes to render", "int", 4, min=0, max=8, step=1),
        Param("eval_seed", "Eval seed", "int", 0, min=0, max=999),
    ],
    charts=[
        {"id": "reward", "title": "Episode reward (rolling mean of 20)",
         "series": [{"key": "ep_return", "label": "episode"},
                    {"key": "ep_return_ma", "label": "rolling mean"}], "log_y": False},
        {"id": "critic", "title": "Critic loss (both Q heads)",
         "series": [{"key": "critic_loss", "label": "critic loss"}], "log_y": True},
        {"id": "q", "title": "Q estimate vs TD target (diverging = unhealthy)",
         "series": [{"key": "q1_mean", "label": "q1 mean"},
                    {"key": "target_mean", "label": "target mean"}], "log_y": False},
        {"id": "actor", "title": "Actor loss (-Q)",
         "series": [{"key": "actor_loss", "label": "actor loss"}], "log_y": False},
        {"id": "success", "title": "Success rate",
         "series": [{"key": "eval_success", "label": "eval (held-out holes)"},
                    {"key": "train_success_ma", "label": "train (last 50 episodes)"}],
         "log_y": False},
    ],
    viewer="video",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="td3.html",
)  # fmt: skip
registry.register(SPEC)
