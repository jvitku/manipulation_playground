"""Lesson 15 - "The contact testbed" (id `force-sim`): the M0 gantry testbed in the lab.

TRAIN = record the canonical demo datasets. `nfv.data.record.record_demos` writes npz shards
(one per episode) + `index.json` under ``data_root()/nfv/demos/<task>/`` - the canonical
location the later force-VLA modules (lessons 16-19) read their training data from. Recording
runs in chunks so the charts show progress and stop/pause stay responsive; when a task
finishes, its `index.json` is rewritten whole with provenance on top of the record.py fields:
``run_id`` (the lab run that recorded it), ``params`` (the typed form values) and ``git``
(`git describe --always --dirty`, when available). Demo seeds come from the task's
`nfv.seeds.demo_seeds` block; the form seed shifts the window inside the block.

Checkpoint: the testbed trains no model, but the house eval contract requires a checkpoint
(`start_eval` resolves a mandatory top-level ``ckpt``), so the run saves ONE tiny marker
checkpoint, label "demo set", whose payload describes what was recorded:
    {"kind": "nfv-demo-set", "demo_root", "tasks": {task: {n, tried, noise_mm}},
     "params": <train params>, "run_id"}
`evaluate` reads it only for provenance (``result.demo_set``) and tolerates any payload - the
eval scenarios roll the scripted experts and controllers live and consume no model.

EVAL = testbed showcase. ``scenario`` picks one (`episodes` counts per task / height / arm;
seeds from the task's `nfv.seeds.eval_seeds` block, shifted by ``eval_seed * episodes``):
  expert_rollouts        the scripted experts on held-out seeds (DART noise 0.5 mm), per task
                         (``eval_task``; "all" = T0-T3) -> claim c1 (the M0 expert gate)
  kp_wipe_150_vs_800     a position-only wipe at K_p 150 vs 800 on the hidden-height plate:
                         the contact force is ~K_p x height error (c3's mechanism; the claim
                         itself needs the hybrid arm, so this scenario fills no claim)
  hybrid_hold_5N         the hybrid controller commanded F_n = 5 N at hidden heights
                         -8 / 0 / +8 mm (round-robin over episodes) -> claim c2
  position_vs_hybrid_t2  the T2 wipe, position-only baseline vs the hybrid expert on the SAME
                         seeds -> claim c3 (the course's motivating claim)
The controller-measurement scenarios (hold, kp, position_vs_hybrid) run noise-free; only
expert_rollouts keeps the demos' 0.5 mm DART noise.

result.json (the contract with the `fvrollout` viewer; every field optional there):
    {"scenario", "episodes" (per group), "n_episodes", "fps", "video": "rollout.mp4"|null,
     "demo_set": {...}|null,
     "outcomes": {"success", "counts": {outcome: n}, "wilson95": [lo, hi],
                  "per_task": {task: {...}}  (expert_rollouts only)},
     "peak_force": {"mean", "max"},                              # N, over all episodes
     "tracking": {"target_n", "per_height": [{dz_mm, n, mean_f_n, max_abs_err_n}]}  (hold)
               | {"target_n", "arms": {arm: {n, success, wilson95, aborts, peak_force_mean,
                  in_band_frac, held_n, held_frac, held_wilson95, coverage_mean,
                  contact_f_n_med}}},
     "kp": {"150": {...arm stats...}, "800": {...}}              # kp scenario only
     "claims": [{"id", "statement", "metric", "expected", "measured", "verdict", "source",
                 "url"}],                                        # all of nfv.claims.M0_CLAIMS;
                                                                 # unmeasured rows say so
     "episodes_traces": [<= 10 of {"task", "seed", "group", "outcome", "peak_force",
                         "t": [s], "fz": [N], "f_target": [N], "mode": ["position"|"hybrid"],
                         "subtask": [str], "events": [{"t", "label"}]}]}  # per 10 Hz step
Traces and the video are spread round-robin over the scenario's groups; rollout.mp4 renders
up to 4 episodes (static camera + a force strip, task 5 (force-token)'s video idiom) at the
10 Hz policy rate with a title card per episode.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lab.core.runctx import atomic_write_json
from lab.core.video import frames_to_mp4
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.task1_bc_mse import wilson95
from nfv.claims import M0_CLAIMS, NOT_MEASURED, verdict
from nfv.data.record import record_demos
from nfv.seeds import TASK_NAMES, demo_seeds, eval_seeds
from nfv.sim.env import Action, Env
from nfv.sim.experts import DART_SIGMA_MM, WipeExpert, make_expert, noisy
from nfv.sim.gantry_xml import START_HEIGHT, GantryParams
from nfv.sim.tasks import TIP_TIP_Z, Wipe

TASK_CHOICES = ["all", *TASK_NAMES]
SCENARIOS = ("expert_rollouts", "kp_wipe_150_vs_800", "hybrid_hold_5N", "position_vs_hybrid_t2")
FPS = 10  # video frame per 10 Hz policy step = real time
VW, VH, SH = 320, 240, 70  # video close-up size and force-strip height (task 5's layout)
CARD_FRAMES, END_FRAMES = 8, FPS  # 0.8 s title card, 1 s result hold
CHUNK = 20  # demos recorded per progress row / stop check
MAX_TRACES, MAX_VIDEO = 10, 4  # per eval, spread round-robin over the scenario's groups
FT_5N = np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0])  # "press down with 5 N" (course sign)
HOLD_DZ_MM = (-8.0, 0.0, 8.0)  # the three hidden heights of the M0 gate
HOLD_STEPS, HOLD_MEASURE_FROM = 30, 20  # 3 s hold; the final second is measured (200 Hz)
HOLD_TOL_N = 1.0  # |mean F_n - 5| below this is an in-band hold (the gate's bound)
KP_ARMS = (150.0, 800.0)  # N/m; 800 is the deployed default (GantryParams.kp)
WIPE_TARGET_N = 3.0  # the "gently" instruction's force target (tasks.Wipe.FORCE_TARGETS[0])
# calib.json fallbacks (measured 2026-10-31 in the fvb-lab image: 0.83 s/demo rendered)
FALLBACK_S_PER_DEMO, TASK_OVERHEAD_S, OVERHEAD_S = 0.85, 2.0, 5.0


def _demo_root() -> Path:
    """``data_root()/nfv/demos`` - the canonical demo location (plan §5); LAB_DATA-aware."""
    from lab.server.app import data_root  # lazy: lab.server.app imports lab.tasks at startup

    return data_root() / "nfv" / "demos"


def _git_describe() -> str | None:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=5,
        )  # fmt: skip
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fails a run
        return None


def _r3(x) -> float:
    return round(float(x), 3)


# ------------------------------------------------------------------------------------- train
def train(ctx) -> None:
    P = ctx.params
    tasks = list(TASK_NAMES) if P["task"] == "all" else [str(P["task"])]
    n, noise, seed = int(P["demos_per_task"]), float(P["noise"]), int(P.get("seed", 0))
    root = _demo_root()
    total = n * len(tasks)
    env = Env(task=tasks[0], render=True)
    recorded: dict[str, dict] = {}
    done_total = 0
    try:
        for ti, task in enumerate(tasks):
            ctx.set_status(
                step=done_total, total_steps=total,
                message=f"recording {task} demos ({ti + 1}/{len(tasks)}) -> {root / task}",
            )  # fmt: skip
            pool = demo_seeds(task, min(2 * n + 8, 10_000 - seed), offset=seed)
            env.reset(pool[0], task=task)  # switch the shared env to this task's model
            pos = got = tried = 0
            episodes: list[dict] = []
            t_task = time.monotonic()
            while got < n:
                if not ctx.should_continue():
                    ctx.set_status(state="stopped")
                    return
                want = min(CHUNK, n - got)
                t0 = time.monotonic()
                idx = record_demos(task, want, root, noise_mm=noise, seeds=pool[pos:], env=env)
                pos += idx["tried"]
                tried += idx["tried"]
                got += idx["n"]
                episodes.extend(idx["episodes"])
                done_total += idx["n"]
                ctx.log(
                    done_total,
                    **{f"{task}_demos": got, f"{task}_ok_of_tries": round(got / tried, 4),
                       f"{task}_s_per_demo": _r3((time.monotonic() - t0) / max(1, idx["n"]))},
                )  # fmt: skip
                ctx.set_status(
                    step=done_total, total_steps=total,
                    message=f"{task}: {got}/{n} demos ({tried} tried)",
                )  # fmt: skip
            # the whole-task index, with provenance on top of record.py's fields
            index = {
                "task": task, "n": got, "tried": tried, "noise_mm": noise,
                "wall_s": round(time.monotonic() - t_task, 2),
                "run_id": ctx.dir.name, "params": dict(P), "git": _git_describe(),
                "episodes": episodes,
            }  # fmt: skip
            (root / task / "index.json").write_text(json.dumps(index, indent=2))
            recorded[task] = {"n": got, "tried": tried, "noise_mm": noise}
    finally:
        env.close()
    payload = {
        "kind": "nfv-demo-set", "demo_root": str(root), "tasks": recorded,
        "params": dict(P), "run_id": ctx.dir.name,
    }  # fmt: skip
    ctx.save_ckpt(total, payload, label="demo set", metrics={"demos": total})
    parts = ", ".join(f"{t} {d['n']}/{d['tried']}" for t, d in recorded.items())
    ctx.done(f"recorded {total} demos ({parts}) to {root}")


# ---------------------------------------------------------------------------- eval policies
class _ExpertPolicy:
    """The task's privileged scripted expert (nfv.sim.experts), adapted to (Action, label)."""

    def __init__(self, env: Env):
        self.expert = make_expert(env)
        self.task = env.task

    def act(self) -> tuple[Action, str]:
        es = self.expert.act()
        return es.action, self.task.SUBTASKS[es.subtask_id]


class _Hold5N:
    """Hold still and command F_n = 5 N on the z axis - the controller measurement of c2."""

    def __init__(self, env: Env):
        del env

    def act(self) -> tuple[Action, str]:
        return Action(dp=np.zeros(3), f_target=FT_5N.copy()), "hold"


class _PositionWipe:
    """Position-only T2 wipe baseline: f_target is NEVER set.

    It aims the tip ``PRESS_MM`` below the NOMINAL plate top and sweeps - it cannot read the
    hidden height, so on a raised surface it presses its full height error times the
    K_p-with-pad series stiffness, and on a lowered one it never touches the plate at all.
    That IS the c3 lesson: a position policy commands a penetration depth, not a force.

    The z step is computed against the policy's OWN COMMAND, never against the measured joint
    (which stalls in contact): stepping toward ``measured - target`` would silently integrate
    the command to the joint-range floor - command windup, abort at any K_p. Measured with
    the windup bug in place: every arm ramped to the 25 N abort, K_p 150 and 800 alike.
    """

    PRESS_MM = 5.0  # commanded tip depth below the nominal (dz = 0) plate top

    def __init__(self, env: Env):
        self.env = env
        if not isinstance(env.task, Wipe):
            raise TypeError("_PositionWipe runs on task t2 only")
        nominal_top = Wipe.PLATE_Z0 + 0.010  # plate half-height; the hidden dz is not read
        self._jz = (nominal_top - self.PRESS_MM * 1e-3) - TIP_TIP_Z - START_HEIGHT
        self._zcmd: float | None = None  # the policy's own z command (anti-windup, see above)
        self._dir = 1.0
        self._phase = 0

    def _dz(self, limit: float = 0.01) -> float:
        """One z step of the own-command descent toward the fixed target depth."""
        if self._zcmd is None:
            self._zcmd = float(self.env.qpos_xyz[2])
        step = float(np.clip(self._jz - self._zcmd, -limit, limit))
        self._zcmd += step
        return step

    def act(self) -> tuple[Action, str]:
        cur = self.env.qpos_xyz
        if self._phase == 0:  # approach: xy above the strip start, keep height
            tgt = np.array([WipeExpert.X_START, 0.0, cur[2]])
            if np.linalg.norm(tgt[:2] - cur[:2]) < 0.004:
                self._phase = 1
            else:
                return Action(dp=np.clip(tgt - cur, -0.03, 0.03)), "approach"
        if self._phase == 1:  # descend: walk the COMMAND to the fixed depth, then sweep
            dz = self._dz()
            if abs(self._jz - self._zcmd) < 1e-9:
                self._phase = 2
            else:
                return Action(dp=np.array([0.0, 0.0, dz])), "approach"
        end = WipeExpert.X_END if self._dir > 0 else -WipeExpert.X_END
        if (end - cur[0]) * self._dir < 0.002 and self.env.task.coverage < Wipe.COVER_FRAC:
            self._dir = -self._dir
            end = -end
        dp = np.array([
            float(np.clip(end - cur[0], -WipeExpert.SWEEP_DP, WipeExpert.SWEEP_DP)),
            -cur[1] * 0.3,
            self._dz(),
        ])  # fmt: skip
        return Action(dp=dp), "wipe"


_POLICIES = {"expert": _ExpertPolicy, "hold_5n": _Hold5N, "position_wipe": _PositionWipe}


# ------------------------------------------------------------------------------- eval video
def _card(lines: list[str]) -> np.ndarray:
    import cv2

    img = np.full((VH + SH, VW, 3), 250, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (14, 70 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1,
                    cv2.LINE_AA)  # fmt: skip
    return img


def _strip(fz: list, ft: list, tmax: int, lo: float = -12.0, hi: float = 4.0) -> np.ndarray:
    """fz (dark) and the commanded f_target_z (orange dashes) over the episode, course sign."""
    import cv2

    img = np.full((SH, VW, 3), 255, np.uint8)

    def pt(i, v):
        return int(i / max(1, tmax) * (VW - 1)), int(
            SH - 2 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (SH - 4)
        )

    y0 = pt(0, 0.0)[1]
    cv2.line(img, (0, y0), (VW - 1, y0), (200, 200, 200), 1)
    for i, v in enumerate(ft):
        if i % 3 == 0 and v != 0.0:  # dashed command line
            cv2.circle(img, pt(i, v), 1, (30, 110, 220), -1)
    if len(fz) > 1:
        pts = np.array([pt(i, v) for i, v in enumerate(fz)], np.int32)
        cv2.polylines(img, [pts], False, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, "Fz (dark), command (orange), N", (4, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                (90, 90, 90), 1, cv2.LINE_AA)  # fmt: skip
    return img


# ------------------------------------------------------------------------------ eval episodes
@dataclass
class _Job:
    """One eval episode: where it runs, which policy drives it, what gets recorded."""

    task: str
    seed: int
    group: str  # aggregation key: task name / "dz -8 mm" / "kp 150" / "position" ...
    policy: str  # key into _POLICIES
    env: Env
    instruction: str | None = None
    overrides: dict | None = None
    max_steps: int | None = None  # None = run to the env's own episode end
    noise_mm: float = DART_SIGMA_MM
    render: bool = False
    trace: bool = False
    card: list = field(default_factory=list)  # extra title-card lines


def _episode(job: _Job):
    """Run one episode; a generator that yields video frames when ``job.render`` and returns
    the record (outcome + optional trace) via StopIteration - task 5's video pattern."""
    import mujoco

    env = job.env
    env.reset(job.seed, task=job.task, instruction=job.instruction,
              hidden_overrides=job.overrides)  # fmt: skip
    pol = _POLICIES[job.policy](env)
    rng = np.random.default_rng(job.seed + 7_777_777)  # the record.py noise convention
    tmax = job.max_steps or int(env.task.TIME_LIMIT_S * FPS)
    tr: dict = {"t": [], "fz": [], "f_target": [], "mode": [], "subtask": [], "events": []}
    fz_v: list[float] = []
    ft_v: list[float] = []
    fn_hold: list[float] = []  # 200 Hz F_n samples of the measured hold window
    rend = cam = None
    if job.render:
        rend = mujoco.Renderer(env.model, height=VH, width=VW)
        cam = "static"

    def frame(status=""):
        import cv2

        rend.update_scene(env.data, camera=cam)
        img = np.vstack([rend.render().copy(), _strip(fz_v, ft_v, tmax)])
        mode = tr["mode"][-1] if tr["mode"] else "position"
        txt = f"t {env.t:4.1f} s  Fz {float(env.wrench[2]):+6.2f} N  z-axis: {mode}"
        cv2.rectangle(img, (0, 0), (VW, 18), (255, 255, 255), -1)
        cv2.putText(img, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1,
                    cv2.LINE_AA)  # fmt: skip
        if status:
            ok = status in ("success", "held")
            col = (30, 140, 60) if ok else (200, 50, 40)
            cv2.putText(img, status.upper(), (10, VH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col,
                        2, cv2.LINE_AA)  # fmt: skip
        return img

    try:
        if job.render:
            yield frame()
        outcome, clicked, aborted = "held", False, False
        k = 0
        while True:
            act, sub = pol.act()
            obs, info = env.step(noisy(act, rng, job.noise_mm))
            k += 1
            ft_z = 0.0 if act.f_target is None else float(np.asarray(act.f_target)[2])
            mode = "hybrid" if act.f_target is not None and env.controller.selected[2] else \
                "position"  # fmt: skip
            tr["t"].append(round(info["t"], 2))
            tr["fz"].append(_r3(obs.wrench[2]))
            tr["f_target"].append(_r3(ft_z))
            tr["mode"].append(mode)
            tr["subtask"].append(sub)
            fz_v.append(float(obs.wrench[2]))
            ft_v.append(ft_z)
            if job.task == "t1" and info.get("click") and not clicked:
                clicked = True
                tr["events"].append({"t": round(info["t"], 2), "label": "click"})
            if info["force_abort"] and not aborted:
                aborted = True
                tr["events"].append({"t": round(info["t"], 2), "label": "abort"})
            if job.task == "t3" and info["outcome"] == "success":
                tr["events"].append({"t": round(info["t"], 2), "label": "seated"})
            if job.policy == "hold_5n" and k > HOLD_MEASURE_FROM:
                fn_hold.extend((-obs.wrench_window[:, 2]).tolist())
            if job.render:
                yield frame()
            if info["done"]:
                outcome = info["outcome"]
                break
            if job.max_steps is not None and k >= job.max_steps:
                break  # a fixed-window measurement episode (the hold): outcome stays "held"
        rec = {
            "task": job.task, "seed": job.seed, "group": job.group, "outcome": outcome,
            "peak_force": _r3(env.peak_force),
        }  # fmt: skip
        if job.policy == "hold_5n":
            mean_fn = float(np.mean(fn_hold)) if fn_hold else float("nan")
            in_band = bool(fn_hold) and abs(mean_fn - 5.0) < HOLD_TOL_N
            rec["mean_f_n"] = _r3(mean_fn) if fn_hold else None
            rec["outcome"] = outcome if outcome != "held" else \
                ("success" if in_band else "off_target")  # fmt: skip
        if isinstance(env.task, Wipe):
            rec["in_band_frac"] = 0.0 if rec["outcome"] == "force_abort" else \
                _r3(env.task.track_frac)  # fmt: skip
            rec["coverage"] = _r3(env.task.coverage)
            fn = [-v for v in fz_v if -v > Wipe.CONTACT_N]  # 10 Hz F_n samples in contact
            rec["contact_f_n_med"] = _r3(np.median(fn)) if fn else None  # impact-robust level
        rec["success"] = rec["outcome"] == "success"
        if job.render:
            last = frame(rec["outcome"])
            for _ in range(END_FRAMES):
                yield last
        if job.trace:
            rec["trace"] = tr
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


def _spread(jobs: list[_Job], budget: int, flag: str) -> None:
    """Flag up to ``budget`` jobs round-robin over the groups, earliest of each group first."""
    by_group: dict[str, list[_Job]] = {}
    for j in jobs:
        by_group.setdefault(j.group, []).append(j)
    order = list(by_group)
    gi = taken = 0
    while taken < budget and any(by_group.values()):
        g = order[gi % len(order)]
        gi += 1
        if by_group[g]:
            setattr(by_group[g].pop(0), flag, True)
            taken += 1


# ---------------------------------------------------------------------------------- evaluate
def _arm_stats(rows: list[dict]) -> dict:
    k = sum(r["success"] for r in rows)
    out = {
        "n": len(rows), "success": round(k / max(1, len(rows)), 4),
        "wilson95": wilson95(k, len(rows)),
        "aborts": sum(r["outcome"] == "force_abort" for r in rows),
        "peak_force_mean": _r3(np.mean([r["peak_force"] for r in rows])),
    }  # fmt: skip
    bands = [r["in_band_frac"] for r in rows if "in_band_frac" in r]
    if bands:
        out["in_band_frac"] = _r3(np.mean(bands))
        held = sum(b >= Wipe.TRACK_FRAC for b in bands)  # episodes that HELD the band
        out["held_n"], out["held_frac"] = held, round(held / len(bands), 4)
        out["held_wilson95"] = wilson95(held, len(bands))
        out["coverage_mean"] = _r3(np.mean([r.get("coverage", 0.0) for r in rows]))
        meds = [r["contact_f_n_med"] for r in rows if r.get("contact_f_n_med") is not None]
        out["contact_f_n_med"] = _r3(np.mean(meds)) if meds else None  # mean of ep medians
    return out


def _claim_rows(filled: dict[str, dict]) -> list[dict]:
    rows = []
    for c in M0_CLAIMS:
        f = filled.get(c.id)
        rows.append({
            "id": c.id, "statement": c.statement, "metric": c.metric, "expected": c.expect,
            "measured": f["measured"] if f else "not measured in this run",
            "verdict": f["verdict"] if f else NOT_MEASURED,
            "source": c.source, "url": c.url,
        })  # fmt: skip
    return rows


def _demo_set_note(ckpt_path: str) -> dict | None:
    try:
        import torch

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and ck.get("kind") == "nfv-demo-set":
            tasks = {t: d.get("n") for t, d in (ck.get("tasks") or {}).items()}
            return {"demo_root": ck.get("demo_root"), "run_id": ck.get("run_id"),
                    "tasks": tasks}  # fmt: skip
    except Exception:  # noqa: BLE001 - the eval consumes no model; any payload is tolerated
        return None
    return None


def evaluate(ctx) -> None:
    P = ctx.params
    scenario = str(P["scenario"])
    episodes, eseed = int(P["episodes"]), int(P["eval_seed"])
    video = bool(P["video"])
    offset = eseed * episodes
    gp = GantryParams(offwidth=VW, offheight=VH)  # room for the 320x240 video renderer

    def seeds(task):
        return eval_seeds(task, episodes, offset=offset)

    jobs: list[_Job] = []
    filled: dict[str, dict] = {}
    result: dict = {"scenario": scenario, "episodes": episodes, "fps": FPS}
    if scenario == "expert_rollouts":
        tasks = list(TASK_NAMES) if P["eval_task"] == "all" else [str(P["eval_task"])]
        env = Env(task=tasks[0], render=False, gantry=gp)
        for task in tasks:
            jobs += [_Job(task, s, task, "expert", env,
                          card=[f"task {task}: scripted expert"]) for s in seeds(task)]  # fmt: skip
    elif scenario == "hybrid_hold_5N":
        env = Env(task="t2", render=False, gantry=gp)
        for i, s in enumerate(seeds("t2")):
            dz = HOLD_DZ_MM[i % len(HOLD_DZ_MM)]
            jobs.append(_Job(
                "t2", s, f"dz {dz:+.0f} mm", "hold_5n", env, max_steps=HOLD_STEPS,
                noise_mm=0.0, overrides={"dz": dz * 1e-3, "tilt_x": 0.0, "tilt_y": 0.0},
                card=["hybrid controller, command 5 N", f"hidden height dz = {dz:+.0f} mm"],
            ))  # fmt: skip
    elif scenario == "kp_wipe_150_vs_800":
        for kp in KP_ARMS:
            env = Env(task="t2", render=False, gantry=GantryParams(kp=kp, offwidth=VW,
                                                                   offheight=VH))  # fmt: skip
            jobs += [_Job("t2", s, f"kp {kp:g}", "position_wipe", env, noise_mm=0.0,
                          instruction="wipe the plate gently",
                          card=[f"position-only wipe, K_p = {kp:g} N/m", "target 3 N (gently)"])
                     for s in seeds("t2")]  # fmt: skip
    elif scenario == "position_vs_hybrid_t2":
        env = Env(task="t2", render=False, gantry=gp)
        for arm, pol in (("position", "position_wipe"), ("hybrid", "expert")):
            jobs += [_Job("t2", s, arm, pol, env, noise_mm=0.0,
                          instruction="wipe the plate gently",
                          card=[f"T2 wipe, {arm} controller", "target 3 N (gently)"])
                     for s in seeds("t2")]  # fmt: skip
    else:  # pragma: no cover - the choice Param guarantees a known scenario
        raise ValueError(f"unknown scenario {scenario!r}")

    _spread(jobs, min(MAX_TRACES, len(jobs)), "trace")
    if video:
        _spread(jobs, min(MAX_VIDEO, len(jobs)), "render")
    result["n_episodes"] = len(jobs)
    result["demo_set"] = _demo_set_note(P["ckpt"])

    rows: list[dict] = []
    done_n = 0
    stopped = False

    def run_one(job):
        nonlocal done_n
        done_n += 1
        msg = f"{scenario}: {job.group}, episode {done_n}/{len(jobs)}"
        ctx.set_status(step=done_n, total_steps=len(jobs), message=msg)

    rendered = [j for j in jobs if j.render]
    plain = [j for j in jobs if not j.render]

    def frames():
        nonlocal stopped
        for i, job in enumerate(rendered):
            if not ctx.should_continue():
                stopped = True
                return
            run_one(job)
            card = [f"{scenario}", f"episode {i + 1} of {len(rendered)} on video",
                    *job.card, f"seed {job.seed}"]  # fmt: skip
            for _ in range(CARD_FRAMES):
                yield _card(card)
            rec = yield from _episode(job)
            rows.append(rec)

    video_name = None
    if rendered:
        try:
            frames_to_mp4(frames(), ctx.dir / "rollout.mp4", fps=FPS)
            video_name = "rollout.mp4"
        except Exception as e:  # noqa: BLE001 - a render failure must not lose the statistics
            ctx.set_status(message=f"video failed: {type(e).__name__}: {e}")
            (ctx.dir / "rollout.mp4").unlink(missing_ok=True)
            rows.clear()  # drop the partial rendered rows; same seeds re-run unrendered
            for job in rendered:
                job.render = False
                run_one(job)
                rows.append(_drain(_episode(job)))
        if stopped:
            ctx.set_status(state="stopped")
            return
    for job in plain:
        if not ctx.should_continue():
            ctx.set_status(state="stopped")
            return
        run_one(job)
        rows.append(_drain(_episode(job)))
    result["video"] = video_name

    # ---- aggregate ----
    k = sum(r["success"] for r in rows)
    cnt = Counter(r["outcome"] for r in rows)
    result["outcomes"] = {"success": round(k / max(1, len(rows)), 4),
                          "counts": dict(cnt), "wilson95": wilson95(k, len(rows))}  # fmt: skip
    result["peak_force"] = {"mean": _r3(np.mean([r["peak_force"] for r in rows])),
                            "max": _r3(np.max([r["peak_force"] for r in rows]))}  # fmt: skip
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)

    if scenario == "expert_rollouts":
        per_task = {g: _arm_stats(rs) for g, rs in by_group.items()}
        result["outcomes"]["per_task"] = per_task
        worst = min(per_task, key=lambda t: per_task[t]["success"])
        lo, hi = per_task[worst]["wilson95"]
        filled["m0.expert_solves_every_task"] = {
            "measured": ", ".join(
                f"{t} {int(round(d['success'] * d['n']))}/{d['n']}" for t, d in per_task.items()
            )
            + f" (min {per_task[worst]['success']:.1%}, CI {lo:.1%}-{hi:.1%})"
            + ("" if len(per_task) == 4 else " - not every task was evaluated"),
            "verdict": verdict("a >= 0.95", {"a": per_task[worst]["success"]}, ci=(lo, hi)),
        }
    elif scenario == "hybrid_hold_5N":
        per_height = []
        errs = []
        for dz in HOLD_DZ_MM:
            rs = by_group.get(f"dz {dz:+.0f} mm", [])
            means = [r["mean_f_n"] for r in rs if r.get("mean_f_n") is not None]
            mean_fn = float(np.mean(means)) if means else float("nan")
            per_height.append({
                "dz_mm": dz, "n": len(rs), "mean_f_n": _r3(mean_fn) if means else None,
                "max_abs_err_n": _r3(max(abs(m - 5.0) for m in means)) if means else None,
            })  # fmt: skip
            if means:
                errs.append(abs(mean_fn - 5.0))
        result["tracking"] = {"target_n": 5.0, "per_height": per_height}
        if errs:
            a = max(errs)
            filled["m0.hybrid_holds_5n"] = {
                "measured": ", ".join(
                    f"dz {h['dz_mm']:+.0f}: {h['mean_f_n']} N"
                    for h in per_height
                    if h["mean_f_n"] is not None
                )
                + f" (max |mean err| {a:.3f} N)",
                "verdict": verdict("a <= 1.0", {"a": a}),
            }
    else:  # the two wipe scenarios share the arm aggregation; pos-vs-hybrid fills c3
        arms = {g: _arm_stats(rs) for g, rs in by_group.items()}
        if scenario == "kp_wipe_150_vs_800":
            result["kp"] = {f"{kp:g}": arms.get(f"kp {kp:g}", {}) for kp in KP_ARMS}
        else:
            result["tracking"] = {"target_n": WIPE_TARGET_N, "arms": arms}
        pos, hyb = arms.get("position"), arms.get("hybrid")
        if pos and hyb and "held_frac" in pos and "held_frac" in hyb:
            # conservative CI of the gap: (lo_a - hi_b, hi_a - lo_b) from the arms' Wilsons
            ci = (
                hyb["held_wilson95"][0] - pos["held_wilson95"][1],
                hyb["held_wilson95"][1] - pos["held_wilson95"][0],
            )
            filled["m0.position_cannot_hold_force"] = {
                "measured": f"held the band (>=80% of contact): hybrid {hyb['held_n']}/"
                f"{hyb['n']}, position {pos['held_n']}/{pos['n']} (gap "
                f"{hyb['held_frac'] - pos['held_frac']:.0%}, CI {ci[0]:.1%}-{ci[1]:.1%}); "
                f"mean in-band time {hyb['in_band_frac']:.1%} vs {pos['in_band_frac']:.1%}, "
                f"position aborts {pos['aborts']}",
                "verdict": verdict(
                    "a-b >= 0.5", {"a": hyb["held_frac"], "b": pos["held_frac"]}, ci=ci
                ),
            }

    result["claims"] = _claim_rows(filled)
    result["episodes_traces"] = [
        {k2: v for k2, v in {**r, **r["trace"]}.items() if k2 != "trace"}
        for r in rows
        if "trace" in r
    ][:MAX_TRACES]
    atomic_write_json(ctx.dir / "result.json", result)
    verdicts = "; ".join(f"{r['id'].split('.')[-1]}: {r['verdict']}" for r in result["claims"]
                         if r["verdict"] != NOT_MEASURED)  # fmt: skip
    ctx.done(
        f"{scenario}: success {result['outcomes']['success']:.0%} over {len(rows)} episodes"
        + (f" - {verdicts}" if verdicts else "")
    )


# ---------------------------------------------------------------------------------- estimate
def estimate(params: dict) -> float:
    cal = registry.calib("force-sim")
    n_tasks = 4 if params["task"] == "all" else 1
    spd = cal.get("s_per_demo") or FALLBACK_S_PER_DEMO
    per_task = int(params["demos_per_task"]) * spd + cal.get("task_overhead_s", TASK_OVERHEAD_S)
    return n_tasks * per_task + cal.get("overhead_s", OVERHEAD_S)


SPEC = TaskSpec(
    id="force-sim",
    order=15,
    title="The contact testbed",
    subtitle="A gantry that can feel: wrench sensing, position vs hybrid force control, and "
             "scripted experts whose demos the force-VLA lessons will learn from.",  # fmt: skip
    params=[
        Param("task", "Tasks to record", "choice", "all", choices=TASK_CHOICES,
              help="which demo datasets to record (all = T0-T3)"),
        Param("demos_per_task", "Demos per task", "int", 200, min=20, max=400, step=20,
              help="successful episodes stored per task (~0.8 s each, rendered 96x96)"),
        Param("noise", "DART action noise (mm)", "float", 0.5, min=0.0, max=2.0, step=0.1,
              help="std of the EXECUTED action noise; the stored label stays the clean "
                   "expert action"),
        Param("seed", "Seed", "int", 0, min=0, max=9000,
              help="shifts the window inside the task's demo-seed block"),
    ],  # fmt: skip
    eval_params=[
        Param("scenario", "Scenario", "choice", "expert_rollouts", choices=list(SCENARIOS),
              help="what to showcase; each fills the claims it can measure"),
        Param("eval_task", "Task (expert_rollouts)", "choice", "all", choices=TASK_CHOICES),
        Param("episodes", "Episodes (per task / height / arm)", "int", 20, min=2, max=100,
              step=1, help="claim c1 needs 100 for a decisive Wilson interval"),
        Param("eval_seed", "Eval seed", "int", 0, min=0, max=99),
        Param("video", "Render a rollout video", "bool", True),
    ],  # fmt: skip
    charts=[
        {"id": "demos", "title": "Demos recorded (cumulative)",
         "series": [{"key": f"{t}_demos", "label": t} for t in TASK_NAMES], "log_y": False},
        {"id": "ok", "title": "Expert success of tried episodes",
         "series": [{"key": f"{t}_ok_of_tries", "label": t} for t in TASK_NAMES],
         "log_y": False},
        {"id": "speed", "title": "Seconds per stored demo",
         "series": [{"key": f"{t}_s_per_demo", "label": t} for t in TASK_NAMES],
         "log_y": False},
    ],  # fmt: skip
    viewer="fvrollout",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="force-sim.html",
)  # fmt: skip
registry.register(SPEC)
