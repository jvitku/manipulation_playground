"""Lesson 19 - "Runtime RL on the frozen VLA" (id `force-rl`): residual TD3+BC on a frozen
nanoVLA, online in the simulator, several RL seeds per job.

TRAIN takes a frozen nano checkpoint (``base_ckpt``, kind ckpt, from lesson 16's ``nano-vla``
task - a single payload or a suite whose arm is resolved by nfv.rl.loop.load_base: the one
non-blank-text arm trained on ``task``) and runs ``seeds`` independent RL seeds of
``nfv.rl.run_residual_rl`` as PARALLEL one-thread subprocesses (``python -m nfv.rl.worker
train``, one output dir per seed under ``<run>/seeds/s<i>/``). Why parallel processes: a CPU
env step is render/physics/python-bound (66 env steps/s for one process, ~50 each with 2-5 in
parallel on the 24-core host) and torch threads buy almost nothing, so 5 x 20 k steps cost the
wall time of one seed (~10-12 min with the in-training evaluations) instead of 5x that. The
workers are pinned to the CPU (CUDA_VISIBLE_DEVICES="") so they never compete with a GPU job.

The parent polls every worker's metrics.jsonl and re-logs each row with the seed prefix
(``s0_eval_success``, ``s1_eval_res_mm_contact``, ...), plus - once every seed has reported an
evaluation at the same env step - the across-seed ``eval_success_median`` and the mean residual
magnitudes ``res_mm_contact_mean`` / ``res_mm_free_mean``. Stop writes a STOP file into every
seed dir (the workers' should_continue) and waits for them.

Checkpoint (one, at the end): {"kind": "nfv-residual-suite", "task", "seeds": [..],
"residuals": {seed: TD3.state(meta) + "arch" (dp_res_m, f_res_n, residual_force)},
"base": the frozen base's {"model", "ema", "arch"} (stored ONCE), "base_info",
"train_curves": {seed: run_residual_rl curves}, "params"}.

EVAL re-evaluates the frozen base and every saved residual on the same ``episodes`` held-out
seeds (``nfv.seeds.eval_seeds(task, episodes, eval_seed*episodes)``), again one worker process
per policy, then rolls the before/after pair - the base and the MEDIAN seed's residual - on the
same seeds in-process for the fvrollout traces and rollout.mp4 (base first, residual second, on
the seed where they differ most: base failed and residual succeeded, else the first seed).

result.json (viewer `fvrollout`):
    {"task", "n_episodes", "fps", "video", "episodes_traces": [...], "claims": [M4 rows],
     "base": {success, wilson95, outcomes, peak_force_mean, success_cond},
     "seeds": [{"seed", "success", "wilson95", "delta", "outcomes", "peak_force_mean",
                "res_mm_contact", "res_mm_free", "success_cond"}, ...],
     "median": {"success", "delta", "peak_force_mean", "seed"},
     "residual_trace": {"contact_mm", "free_mm", "trace"} (median seed, first eval seed),
     "outcomes": {"success", "counts", "wilson95", "per_task"}  (fvrollout's summary block:
                 per_task = base + one row per seed), "peak_force": {"mean", "max"}}
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from lab.core.runctx import atomic_write_json
from lab.core.video import frames_to_mp4
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.task_nfv0 import CARD_FRAMES, END_FRAMES, FPS, VH, VW, _card, _r3, _strip
from nfv.claims import M4_CLAIMS, NOT_MEASURED, verdict

TASKS = ("t3", "t1")
MAX_SEEDS = 8
POLL_S = 2.0
EVAL_EVERY, N_EVAL = 4_000, 20  # in-training evaluation (the per-seed curves)
N_TRACE = 3  # eval seeds traced for the fvrollout viewer (x 2 policies)
# calib.json fallbacks (measured 2026-11-01 on CPU, 2 parallel 1-thread processes, busy host)
FALLBACK_ENV_STEPS_PER_S, FALLBACK_EVAL_STEPS_PER_S, OVERHEAD_S = 50.0, 80.0, 30.0
STEPS_PER_EP = {"t3": 150, "t1": 30}  # rough mean episode length for the estimate


def _worker_env() -> dict:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
               MUJOCO_GL=env.get("MUJOCO_GL", "osmesa"))  # fmt: skip
    src = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _spawn(args: list[str], log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as f:
        return subprocess.Popen(
            [sys.executable, "-m", "nfv.rl.worker", *args], stdout=f, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=_worker_env(),
            cwd=Path(__file__).resolve().parents[2],
        )  # fmt: skip


def _tail(path: Path, n: int = 1500) -> str:
    try:
        return path.read_text()[-n:]
    except OSError:
        return ""


# ------------------------------------------------------------------------------------- train
def train(ctx) -> None:
    import torch

    from nfv.rl.loop import base_part, load_base

    P = ctx.params
    n_seeds = max(1, min(MAX_SEEDS, int(P["seeds"])))
    seed0, steps, task = int(P.get("seed", 0)), int(P["env_steps"]), str(P["task"])
    base_ckpt = str(P["base_ckpt"])
    ctx.set_status(message=f"resolving the frozen base for {task}")
    base, base_info = load_base(base_ckpt, task)  # fail fast, before any worker starts
    root = ctx.dir / "seeds"
    seeds = [seed0 + i for i in range(n_seeds)]
    dirs = {s: root / f"s{s}" for s in seeds}
    common = [
        "--base", base_ckpt, "--task", task, "--env-steps", str(steps),
        "--eval-every", str(int(P.get("eval_every", EVAL_EVERY))),
        "--n-eval", str(int(P.get("n_eval", N_EVAL))),
        "--bc-floor", str(float(P["bc_floor"])),
        "--peak-force-penalty", str(float(P["peak_force_penalty"])),
        "--hole-shaping", str(float(P.get("hole_shaping", 0.0))),
        "--dp-res", str(float(P.get("dp_res_mm", 2.0)) * 1e-3),
    ]  # fmt: skip
    if base_info.get("arm"):
        common += ["--arm", str(base_info["arm"])]
    if bool(P.get("residual_force")):
        common.append("--residual-force")
    procs = {
        s: _spawn(["train", *common, "--seed", str(s), "--out", str(dirs[s])],
                  dirs[s] / "worker.log")
        for s in seeds
    }  # fmt: skip
    offsets = dict.fromkeys(seeds, 0)
    evals: dict[int, dict[int, dict]] = {}  # env step -> seed -> eval row
    logged_eval_steps: set[int] = set()
    stopped = False

    def drain() -> None:
        for s in seeds:
            path = dirs[s] / "metrics.jsonl"
            if not path.exists():
                continue
            with path.open() as f:
                f.seek(offsets[s])
                chunk = f.read()
            end = chunk.rfind("\n")
            if end < 0:
                continue
            offsets[s] += len(chunk[: end + 1].encode())
            for line in chunk[: end + 1].splitlines():
                row = json.loads(line)
                step = int(row.pop("step"))
                ctx.log(step, **{f"s{s}_{k}": v for k, v in row.items()})
                if "eval_success" in row:
                    evals.setdefault(step, {})[s] = row
        for step, by_seed in sorted(evals.items()):
            if step in logged_eval_steps or len(by_seed) < len(seeds):
                continue
            logged_eval_steps.add(step)
            rows = list(by_seed.values())
            agg = {"eval_success_median": float(np.median([r["eval_success"] for r in rows]))}
            for k in ("res_mm_contact", "res_mm_free"):
                vals = [r[f"eval_{k}"] for r in rows if f"eval_{k}" in r]
                if vals:
                    agg[f"{k}_mean"] = float(np.mean(vals))
            ctx.log(step, **agg)

    try:
        while any(p.poll() is None for p in procs.values()):
            time.sleep(POLL_S)
            drain()
            done_steps = 0
            for s in seeds:
                try:
                    done_steps += int(json.loads((dirs[s] / "progress.json").read_text())
                                      .get("step", 0))  # fmt: skip
                except (OSError, ValueError):
                    pass
            ctx.set_status(
                step=done_steps, total_steps=steps * len(seeds),
                message=f"{len(seeds)} RL seeds in parallel on {task}: "
                        f"{done_steps}/{steps * len(seeds)} env steps",
            )  # fmt: skip
            if not stopped and not ctx.should_continue():
                stopped = True
                for s in seeds:
                    (dirs[s] / "STOP").touch()
    except BaseException:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        raise
    drain()
    failed = [s for s, p in procs.items() if p.returncode != 0]
    if failed:
        raise RuntimeError(f"RL seeds {failed} failed:\n{_tail(dirs[failed[0]] / 'worker.log')}")
    residuals, curves, finals = {}, {}, {}
    for s in seeds:
        residuals[str(s)] = torch.load(dirs[s] / "residual.pt", map_location="cpu",
                                       weights_only=False)  # fmt: skip
        res = json.loads((dirs[s] / "result.json").read_text())
        curves[str(s)] = res["curves"]
        finals[s] = res["final_eval"]["success"]
    base_eval = json.loads((dirs[seeds[0]] / "result.json").read_text())["base_eval"]
    payload = {
        "kind": "nfv-residual-suite", "task": task, "seeds": seeds, "residuals": residuals,
        "base": base_part(base), "base_info": base_info, "train_curves": curves,
        "params": dict(P),
    }  # fmt: skip
    med = float(np.median(list(finals.values())))
    ctx.save_ckpt(steps, payload, label="stopped" if stopped else "final",
                  metrics={"final_success_median": round(med, 4),
                           "base_success": round(base_eval["success"], 4)})  # fmt: skip
    if stopped:
        ctx.set_status(state="stopped")
        return
    per = ", ".join(f"s{s} {v:.0%}" for s, v in finals.items())
    ctx.done(f"{len(seeds)} RL seeds on {task}: final success {per} (median {med:.0%}) vs "
             f"frozen base {base_eval['success']:.0%} on {base_eval['n']} in-training eval "
             f"seeds")  # fmt: skip


# ---------------------------------------------------------------------- traces and video
def _episode(actor, env, task: str, seed: int, off: bool, render: bool, group: str):
    """One traced (optionally rendered) episode; yields frames, returns the record."""
    import cv2
    import mujoco

    from nfv.rl.loop import _residual_mm, _Slot

    slot = _Slot(env=env, task=task)
    actor.start(slot, seed)
    tmax = int(env.task.TIME_LIMIT_S * FPS)
    tr: dict = {"t": [], "fz": [], "mode": [], "subtask": [], "events": [], "res_mm": []}
    fz_v: list[float] = []
    rend = mujoco.Renderer(env.model, height=VH, width=VW) if render else None

    def frame(status=""):
        rend.update_scene(env.data, camera="static")
        img = np.vstack([rend.render().copy(), _strip(fz_v, [0.0] * len(fz_v), tmax)])
        txt = f"{group}  t {env.t:4.1f} s  Fz {float(env.wrench[2]):+6.2f} N"
        cv2.rectangle(img, (0, 0), (VW, 18), (255, 255, 255), -1)
        cv2.putText(img, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1,
                    cv2.LINE_AA)  # fmt: skip
        if status:
            col = (30, 140, 60) if status == "success" else (200, 50, 40)
            cv2.putText(img, status.upper(), (10, VH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col,
                        2, cv2.LINE_AA)  # fmt: skip
        return img

    try:
        if render:
            yield frame()
        info: dict = {"done": False}
        aborted = False
        while not info["done"]:
            r = actor.residual(slot.feat, 0.0, None, off=off)
            obs, info, _ = actor.step(slot, r)
            tr["t"].append(round(float(info["t"]), 2))
            tr["fz"].append(_r3(obs.wrench[2]))
            tr["mode"].append("position")
            tr["subtask"].append(env.task.SUBTASKS[obs.subtask_id])
            tr["res_mm"].append(round(_residual_mm(r, actor.dp_res), 4))
            fz_v.append(float(obs.wrench[2]))
            if info["force_abort"] and not aborted:
                aborted = True
                tr["events"].append({"t": tr["t"][-1], "label": "abort"})
            if info["outcome"] == "success":
                tr["events"].append({"t": tr["t"][-1], "label": "seated" if task == "t3"
                                     else "click"})  # fmt: skip
            if render:
                yield frame()
            if not info["done"]:
                slot.feat = actor._features(slot)
        rec = {"task": task, "seed": seed, "group": group, "outcome": info["outcome"],
               "success": bool(info["success"]), "peak_force": _r3(env.peak_force),
               "instruction": env.instruction, **tr}  # fmt: skip
        if render:
            last = frame(rec["outcome"])
            for _ in range(END_FRAMES):
                yield last
        return rec
    finally:
        if rend is not None:
            rend.close()


def _drain(gen):
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


# ------------------------------------------------------------------------------------- eval
def _summary(ev: dict) -> dict:
    return {
        "n": ev["n"], "success": round(ev["success"], 4),
        "wilson95": [round(x, 4) for x in ev["success_ci95"]],
        "outcomes": ev["outcomes"], "peak_force_mean": _r3(ev["peak_force_mean"]),
        "success_cond": round(ev["success_cond"], 4),
        "res_mm_contact": ev["res_mm_contact"], "res_mm_free": ev["res_mm_free"],
    }  # fmt: skip


def _claims(base: dict, seeds: list[dict], med: dict) -> list[dict]:
    filled: dict[str, dict] = {}
    if seeds:
        ms = next(r for r in seeds if r["seed"] == med["seed"])
        ci = (ms["wilson95"][0] - base["wilson95"][1], ms["wilson95"][1] - base["wilson95"][0])
        filled["m4.residual_improves_t3"] = {
            "measured": f"median seed s{ms['seed']} {ms['success']:.0%} vs frozen base "
            f"{base['success']:.0%} (delta {med['delta']:+.0%}; median-seed vs base CI "
            f"{ci[0]:+.1%}..{ci[1]:+.1%}); per seed: "
            + ", ".join(f"s{r['seed']} {r['delta']:+.0%}" for r in seeds),
            "verdict": verdict(
                "a-b >= 0.10", {"a": base["success"] + med["delta"], "b": base["success"]}, ci=ci
            ),  # fmt: skip
        }
        drop = 1.0 - med["peak_force_mean"] / max(base["peak_force_mean"], 1e-9)
        filled["m4.peak_force_down"] = {
            "measured": f"median over seeds of mean peak |F| {med['peak_force_mean']:.2f} N vs "
            f"base {base['peak_force_mean']:.2f} N ({drop:+.0%} reduction)",
            "verdict": verdict("a >= 0.20", {"a": drop}),
        }
        con = [r["res_mm_contact"] for r in seeds if r["res_mm_contact"] is not None]
        fre = [r["res_mm_free"] for r in seeds if r["res_mm_free"] is not None]
        if con and fre and np.mean(con) > 0:
            ratio = float(np.mean(fre) / np.mean(con))
            filled["m4.residual_quiet_in_free_space"] = {
                "measured": f"mean |residual| free {np.mean(fre):.3f} mm/step vs contact "
                f"{np.mean(con):.3f} mm/step (ratio {ratio:.2f}) over {len(seeds)} seeds",
                "verdict": verdict("a <= 0.3333", {"a": ratio}),
            }
        filled["m4.seed_spread_reported"] = {
            "measured": f"{len(seeds)} RL seeds, final success "
            + ", ".join(f"s{r['seed']} {r['success']:.0%}" for r in seeds)
            + f" (spread {min(r['success'] for r in seeds):.0%}-"
            f"{max(r['success'] for r in seeds):.0%})",
            "verdict": verdict("a >= 5", {"a": float(len(seeds))}),
        }
    rows = []
    for c in M4_CLAIMS:
        f = filled.get(c.id)
        rows.append({
            "id": c.id, "statement": c.statement, "metric": c.metric, "expected": c.expect,
            "measured": f["measured"] if f else "not measured in this run",
            "verdict": f["verdict"] if f else NOT_MEASURED,
            "source": c.source, "url": c.url,
        })  # fmt: skip
    return rows


def evaluate(ctx) -> None:
    import torch

    from nfv.rl.loop import ResidualActor, policy_from_payload
    from nfv.rl.residual import make_agent
    from nfv.rl.worker import actor_from_payload
    from nfv.seeds import eval_seeds
    from nfv.sim.env import Env
    from nfv.sim.gantry_xml import GantryParams

    P = ctx.params
    episodes, eseed, video = int(P["episodes"]), int(P["eval_seed"]), bool(P["video"])
    ctx.set_status(message="loading the residual suite checkpoint")
    suite = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    if suite.get("kind") != "nfv-residual-suite":
        raise ValueError("not a force-rl checkpoint (kind nfv-residual-suite)")
    task, seeds = str(suite["task"]), [int(s) for s in suite["seeds"]]
    offset = eseed * episodes
    wd = ctx.dir / "eval_workers"
    procs = {
        w: _spawn(["eval", "--suite", str(P["ckpt"]), "--which", w, "--n", str(episodes),
                   "--offset", str(offset), "--out", str(wd / f"{w}.json")],
                  wd / f"{w}.log")
        for w in ["base", *map(str, seeds)]
    }  # fmt: skip
    while any(p.poll() is None for p in procs.values()):
        time.sleep(POLL_S)
        n_done = sum(p.poll() is not None for p in procs.values())
        ctx.set_status(step=n_done, total_steps=len(procs),
                       message=f"evaluating base + {len(seeds)} residuals on {episodes} "
                               f"shared {task} seeds ({n_done}/{len(procs)} done)")  # fmt: skip
        if not ctx.should_continue():
            for p in procs.values():
                p.terminate()
            ctx.set_status(state="stopped")
            return
    bad = [w for w, p in procs.items() if p.returncode != 0]
    if bad:
        raise RuntimeError(f"eval workers {bad} failed:\n{_tail(wd / f'{bad[0]}.log')}")
    evs = {w: json.loads((wd / f"{w}.json").read_text()) for w in procs}
    base = _summary(evs["base"])
    rows = []
    for s in seeds:
        r = {"seed": s, **_summary(evs[str(s)])}
        r["delta"] = round(r["success"] - base["success"], 4)
        rows.append(r)
    order = sorted(rows, key=lambda r: (r["success"], r["seed"]))
    mrow = order[(len(order) - 1) // 2]  # the (lower) median seed: an actual seed's policy
    med = {"success": float(np.median([r["success"] for r in rows])),
           "delta": float(np.median([r["delta"] for r in rows])),
           "peak_force_mean": float(np.median([r["peak_force_mean"] for r in rows])),
           "seed": mrow["seed"]}  # fmt: skip

    # ---- before/after: traces + video on shared seeds (base, then the median seed) ----
    ctx.set_status(message=f"tracing base vs residual s{mrow['seed']} for the viewer")
    torch.set_num_threads(1)
    base_pol = policy_from_payload(suite["base"])
    a_base = ResidualActor(base_pol, make_agent(hidden=32))
    a_res = actor_from_payload(suite["residuals"][str(mrow["seed"])], base_pol)
    env = Env(task=task, render=True, gantry=GantryParams(offwidth=VW, offheight=VH))
    eps_seeds = eval_seeds(task, episodes, offset)
    b_ok = {r["seed"]: r["success"] for r in evs["base"]["episodes"]}
    r_ok = {r["seed"]: r["success"] for r in evs[str(mrow["seed"])]["episodes"]}
    pick = next((s for s in eps_seeds if r_ok.get(s) and not b_ok.get(s)), eps_seeds[0])
    trace_seeds = [pick] + [s for s in eps_seeds if s != pick][: N_TRACE - 1]
    traces: list[dict] = []
    video_name = None
    try:
        for s in trace_seeds[1:]:
            traces.append(_drain(_episode(a_base, env, task, s, True, False, "frozen base")))
            traces.append(_drain(_episode(a_res, env, task, s, False, False,
                                          f"residual s{mrow['seed']}")))  # fmt: skip
        pair: list[dict] = []

        def frames():
            for actor, off, grp in ((a_base, True, "frozen base"),
                                    (a_res, False, f"residual s{mrow['seed']}")):  # fmt: skip
                for _ in range(CARD_FRAMES):
                    yield _card(
                        [
                            "Runtime RL on the frozen VLA (lesson 19)",
                            f"{grp} on {task}",
                            f"seed {pick} (same physics both)",
                        ]
                    )
                rec = yield from _episode(actor, env, task, pick, off, True, grp)
                pair.append(rec)

        if video:
            try:
                frames_to_mp4(frames(), ctx.dir / "rollout.mp4", fps=FPS)
                video_name = "rollout.mp4"
            except Exception as e:  # noqa: BLE001 - a render failure must not lose the stats
                ctx.set_status(message=f"video failed: {type(e).__name__}: {e}")
                (ctx.dir / "rollout.mp4").unlink(missing_ok=True)
                pair.clear()
        if not pair:
            pair = [_drain(_episode(a_base, env, task, pick, True, False, "frozen base")),
                    _drain(_episode(a_res, env, task, pick, False, False,
                                    f"residual s{mrow['seed']}"))]  # fmt: skip
        traces = pair + traces
    finally:
        env.close()

    all_eps = [e for w in procs for e in evs[w]["episodes"]]
    k = sum(r["success"] for r in evs["base"]["episodes"])
    result = {
        "task": task, "episodes": episodes, "n_episodes": len(all_eps), "fps": FPS,
        "video": video_name, "video_seed": pick, "base": base, "seeds": rows, "median": med,
        "residual_trace": {"seed": mrow["seed"], "contact_mm": mrow["res_mm_contact"],
                           "free_mm": mrow["res_mm_free"],
                           "trace": evs[str(mrow["seed"])].get("trace")},
        "outcomes": {
            "success": base["success"], "counts": dict(Counter(
                r["outcome"] for r in evs["base"]["episodes"])),
            "wilson95": base["wilson95"],
            "per_task": {
                "frozen base": {"n": base["n"], "success": base["success"],
                                "wilson95": base["wilson95"],
                                "aborts": base["outcomes"].get("force_abort", 0),
                                "peak_force_mean": base["peak_force_mean"]},
                **{f"residual s{r['seed']}": {
                    "n": r["n"], "success": r["success"], "wilson95": r["wilson95"],
                    "aborts": r["outcomes"].get("force_abort", 0),
                    "peak_force_mean": r["peak_force_mean"]} for r in rows},
            },
        },
        "peak_force": {"mean": _r3(np.mean([e["peak_force"] for e in all_eps])),
                       "max": _r3(np.max([e["peak_force"] for e in all_eps]))},
        "episodes_traces": traces,
        "params": dict(P), "base_successes_n": k,
    }  # fmt: skip
    result["claims"] = _claims(base, rows, med)
    atomic_write_json(ctx.dir / "result.json", result)
    verdicts = "; ".join(f"{r['id'].split('.')[-1]}: {r['verdict']}" for r in result["claims"]
                         if r["verdict"] != NOT_MEASURED)  # fmt: skip
    ctx.done(
        f"{task}: base {base['success']:.0%}, residual median {med['success']:.0%} "
        f"(delta {med['delta']:+.0%}) over {episodes} shared seeds - {verdicts}"
    )


# ---------------------------------------------------------------------------------- estimate
def estimate(params: dict) -> float:
    cal = registry.calib("force-rl")
    sps = cal.get("env_steps_per_s") or FALLBACK_ENV_STEPS_PER_S
    eps = cal.get("eval_steps_per_s") or FALLBACK_EVAL_STEPS_PER_S
    steps = int(params["env_steps"])
    n_evals = steps // int(params.get("eval_every", EVAL_EVERY)) + 2  # + base + final
    ep_len = STEPS_PER_EP.get(str(params["task"]), 100)
    eval_steps = n_evals * int(params.get("n_eval", N_EVAL)) * ep_len
    # the seeds run in parallel: wall time ~ one seed (x1.2 contention allowance above 4)
    par = 1.0 if int(params["seeds"]) <= 4 else 1.2
    return par * (steps / sps + eval_steps / eps) + cal.get("overhead_s", OVERHEAD_S)


def check(params: dict) -> None:
    if float(params["bc_floor"]) < 1.0:
        raise ValueError("bc_floor must be >= 1.0: the BC term toward the base must never "
                         "anneal away (lesson 12)")  # fmt: skip


SPEC = TaskSpec(
    id="force-rl",
    order=19,
    title="Runtime RL on the frozen VLA",
    subtitle="A bounded residual learns online on top of a frozen nanoVLA: TD3 with twin "
             "critics, BC toward zero with a floor, the wrench as input - per-seed curves, "
             "before/after on the same seed.",  # fmt: skip
    params=[
        Param("base_ckpt", "Frozen base (nano-vla checkpoint)", "ckpt", "", ckpt_task="nano-vla",
              help="a lesson-16 checkpoint trained on the chosen task; a suite checkpoint "
                   "resolves its single non-blank arm on that task"),
        Param("task", "Task", "choice", "t3", choices=list(TASKS),
              help="t3 = peg insertion with the hole hidden +-4 mm from the marker (the "
                   "residual's headroom); t1 = the button press (measured: little headroom)"),
        Param("seeds", "RL seeds (parallel processes)", "int", 5, min=1, max=MAX_SEEDS,
              help="independent RL seeds, one 1-thread CPU process each, run in parallel"),
        Param("seed", "First seed", "int", 0, min=0, max=999),
        Param("env_steps", "Env steps per seed", "int", 20000, min=300, max=100000, step=100,
              long=True),
        Param("bc_floor", "BC weight floor", "float", 1.0, min=1.0, max=2.5, step=0.1,
              help="BC toward residual = 0 anneals 2.5 -> this floor; never below 1.0"),
        Param("peak_force_penalty", "Peak-force penalty", "float", 0.0, min=0.0, max=2.0,
              step=0.1, help="k x the episode's peak |F| / 25 N, paid as it rises"),
        Param("hole_shaping", "Hole shaping (privileged, T3)", "float", 0.0, min=0.0,
              max=2.0, step=0.1,
              help="lesson 13's potential shaping toward the TRUE hole, reward only: "
                   "k x (lateral distance decrease) / 4 mm; 0 = the plan's sparse reward"),
        Param("dp_res_mm", "Residual bound (mm/step)", "float", 2.0, min=0.5, max=5.0, step=0.5,
              help="per-axis bound on the residual dp; 2 mm = step x stiffness scaled, 5 mm = "
                   "v1's value (the wrong exercise)"),
        Param("residual_force", "Residual on f_target (hybrid base only)", "bool", False,
              help="+-2 N residual on the base's f_target; refuses a position-head base"),
        Param("eval_every", "Eval every (env steps)", "int", EVAL_EVERY, min=100, max=20000,
              step=100),
        Param("n_eval", "Eval episodes per point", "int", N_EVAL, min=1, max=100),
    ],  # fmt: skip
    eval_params=[
        Param("episodes", "Shared held-out seeds", "int", 100, min=2, max=500,
              help="base and every residual roll on the same seeds"),
        Param("eval_seed", "Eval seed", "int", 0, min=0, max=99,
              help="shifts the window inside the task's eval-seed block"),
        Param("video", "Render the before/after video", "bool", True),
    ],  # fmt: skip
    charts=[
        {"id": "eval", "title": "Held-out success per RL seed (point 0 = frozen base)",
         "series": [{"key": "eval_success_median", "label": "median"}]
         + [{"key": f"s{i}_eval_success", "label": f"seed {i}"} for i in range(MAX_SEEDS)],
         "log_y": False},
        {"id": "res", "title": "Residual magnitude (mm/step): contact vs free space",
         "series": [{"key": "res_mm_contact_mean", "label": "contact (mean)"},
                    {"key": "res_mm_free_mean", "label": "free space (mean)"}],
         "log_y": False},
        {"id": "train", "title": "Training success (moving average) per seed",
         "series": [{"key": f"s{i}_train_success_ma", "label": f"seed {i}"}
                    for i in range(MAX_SEEDS)], "log_y": False},
        {"id": "ffail", "title": "Training episodes lost to force (moving average) per seed",
         "series": [{"key": f"s{i}_train_force_fail_ma", "label": f"seed {i}"}
                    for i in range(MAX_SEEDS)], "log_y": False},
        {"id": "critic", "title": "Critic loss per seed",
         "series": [{"key": f"s{i}_critic_loss", "label": f"seed {i}"}
                    for i in range(MAX_SEEDS)], "log_y": True},
        {"id": "speed", "title": "Env steps/s per seed",
         "series": [{"key": f"s{i}_env_steps_per_s", "label": f"seed {i}"}
                    for i in range(MAX_SEEDS)], "log_y": False},
    ],  # fmt: skip
    viewer="fvrollout",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="force-rl.html",
    check=check,
)  # fmt: skip
registry.register(SPEC)
