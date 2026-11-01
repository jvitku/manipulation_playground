"""Lesson 16 - "nanoVLA: a tiny VLA from scratch" (id `nano-vla`): the M1 model in the lab.

TRAIN builds nanoVLA (`nfv.model`, force_in="none") on the canonical demos lesson 15 recorded
under ``data_root()/nfv/demos/<task>/`` and trains it with `nfv.train.fit` on the real
RunContext. ``preset`` picks the shape of the run:

  single        ONE model on ``task``'s demos. Metrics: loss / lr / steps_per_s from fit, plus
                a periodic closed-loop evaluation (CB_SEEDS held-out seeds, every EVAL_EVERY
                steps and at the end) logged as ``eval_success`` - the house "evaluation
                points on the loss axis" idea. Checkpoint = fit's house payload
                {"model", "ema", "norm", "arch", "params", "gen"}.
  claims_suite  THREE arms sequentially in one job (the task_token / task_embed pattern),
                fresh seeds per arm (seed + 101*i):
                  A t0 default      nanoVLA on T0 (the visible-task control)
                  B t1 default      nanoVLA on T1 (the hidden-threshold task)
                  C t1 no-language  identical config on T1, but the instruction is replaced
                                    by BLANK_INSTRUCTION at BOTH train and eval time
                Suite metrics are prefixed per arm (a_loss / b_loss / c_loss); ONE checkpoint
                payload holds all arms: {"kind": "nfv-suite", "arms": [labels...],
                <label>: {fit payload + "task" + "blank_text"}, "params"} - saved after every
                arm (mid-run checkpoints hold the arms trained so far, label "<arm> trained").

``demos`` subsets the canonical set by episode order (first n of index.json - the data-curve
knob); ``chunk``/``n_exec``/``euler_steps`` go straight into ModelConfig. BLANK_INSTRUCTION is
a single constant ``<unk>`` token rather than the all-``<pad>`` text: with every text key
masked, the text encoder's self-attention softmax is NaN (all keys -inf), so "no language" is
implemented as text that exists but says nothing. The constant lives in the MODEL since
lesson 17 round 2 (nfv.model.text.BLANK_INSTRUCTION - instruction dropout and CFG condition
on the same blank text); arm C stays a flag in this wrapper that swaps the instruction for
it. The env still samples its real instruction (it defines which button is correct); only
what the POLICY reads is blanked.

EVAL rolls the checkpoint closed-loop on held-out seeds (`nfv.seeds.eval_seeds`, shifted by
``eval_seed * episodes``), executing the first n_exec steps of each predicted chunk (arch
values, not the form's). Two pure act-time knobs override the checkpoint's recipe - the
levers lesson 17 round 2 measured: ``cfg_w`` (classifier-free guidance on the instruction,
v = v_blank + w·(v_cond − v_blank) per Euler step; 1.0 = the plain conditioned model, and on
arm C both branches read the same blank text so CFG is a no-op by construction) and
``n_exec_override`` (0 = the checkpoint's trained value, else execute 1..16 steps of each
chunk before re-planning, clamped to the chunk length). Both land in the rebuilt ModelConfig
only - the checkpoint weights are untouched - and are echoed in result.json ("cfg_w",
"n_exec_override"). A suite checkpoint evaluates every arm on its own task - arm C on T1
with BOTH buttons pooled (the env's 50/50 target draw), reported per instruction plus which
buttons it actually pressed; a single checkpoint evaluates its ``params["task"]``. Claims
(nfv.claims.M1_CLAIMS) are filled from whatever arms the checkpoint provides: m1 c4 needs a
default t0 arm, c5 a default t1 arm, c6 both t1 arms. A c5 point value above 0.70 appends the
re-randomise note from the plan's M1 gate to the measured text.

result.json (viewer `fvrollout`; every field optional there):
    {"scenario": "single"|"claims_suite", "episodes" (per arm), "n_episodes", "fps",
     "video": "rollout.mp4"|null,
     "arms": {label: {"task", "blank_text", "n", "success", "wilson95", "outcomes": {o: n},
                      "peak_force": {"mean", "max"},
                      "per_instruction": {instr: {n, success}},   # t1 arms only
                      "pressed_counts": {"red": n, "blue": n}}},  # t1 arms only
     "outcomes": {"success", "counts", "wilson95", "per_task": {label: arm-stats row}},
     "peak_force": {"mean", "max"},
     "claims": [rows for all of M1_CLAIMS; unmeasured rows say so],
     "episodes_traces": [<= 10 of the task-15 trace schema: {"task", "seed", "group",
                         "outcome", "instruction", "peak_force", "t", "fz", "mode"
                         (all "position" - nanoVLA commands no wrench), "subtask", "events"};
                         "f_target" is omitted - the viewer draws no command line]}
rollout.mp4: up to 4 episodes spread round-robin over the arms, static camera + force strip
(task 15's video idiom, imported from task_nfv0).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch

from lab.core.runctx import atomic_write_json
from lab.core.video import frames_to_mp4
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.task1_bc_mse import wilson95
from lab.tasks.task_nfv0 import (
    CARD_FRAMES,
    END_FRAMES,
    FPS,
    MAX_TRACES,
    MAX_VIDEO,
    VH,
    VW,
    _card,
    _demo_root,
    _drain,
    _r3,
    _spread,
    _strip,
)
from nfv.claims import M1_CLAIMS, NOT_MEASURED, verdict
from nfv.data.dataset import DemoDataset
from nfv.model import ModelConfig, build
from nfv.model.text import BLANK_INSTRUCTION
from nfv.seeds import TASK_NAMES, eval_seeds
from nfv.sim.env import Action, Env
from nfv.sim.gantry_xml import GantryParams
from nfv.train import fit

PRESETS = ("single", "claims_suite")
# (metric prefix, label, task, blank_text) - the suite's three arms, in training order
SUITE = (
    ("a", "A t0 default", "t0", False),
    ("b", "B t1 default", "t1", False),
    ("c", "C t1 no-language", "t1", True),
)
# The no-language text (ONE constant <unk> token) is the model's own constant now:
# nfv.model.text.BLANK_INSTRUCTION, imported above - round 2 of lesson 17 promoted it there
# so instruction dropout and CFG condition on the same blank text arm C uses.
CB_SEEDS = 20  # held-out seeds of the single preset's periodic closed-loop evaluation
CB_SEED_OFFSET = 9_000  # top of the eval block - disjoint from the eval job's offsets
EVAL_EVERY = 5_000  # steps between evaluation points (fit also fires at the final step)
T1_INSTRUCTIONS = ("press the red button", "press the blue button")
# calib.json fallbacks (steps_per_s measured 2026-10-31 on the RTX 3080, batch 64)
FALLBACK_STEPS_PER_S, FALLBACK_LOAD_S_PER_DEMO = 20.0, 0.06
FALLBACK_EP_S, OVERHEAD_S = 1.2, 10.0


@dataclass(frozen=True)
class _Arm:
    """One trained model: its metric prefix, checkpoint label, task and the text flag."""

    key: str  # metric prefix ("a"/"b"/"c"; "" = unprefixed, the single preset)
    label: str  # checkpoint payload / result.json label
    task: str
    blank_text: bool


def _train_arms(preset: str, task: str) -> list[_Arm]:
    if preset == "single":
        return [_Arm("", "single", task, False)]
    return [_Arm(*row) for row in SUITE]


class _DemoView:
    """First-``n``-episodes slice of a DemoDataset, optionally with the instructions blanked.

    Serves exactly the attributes nfv.train reads (episodes, lengths, starts, index): the
    ``demos`` knob subsets the canonical set by episode order, and arm C trains against
    BLANK_INSTRUCTION. The npz arrays are shared with the source dataset, never copied.
    """

    def __init__(self, ds: DemoDataset, n: int, blank: bool = False):
        n = max(1, min(int(n), len(ds.episodes)))
        self.episodes = ds.episodes[:n]
        self.lengths = ds.lengths[:n]
        self.starts = np.array(
            [(i, t) for i, ln in enumerate(self.lengths) for t in range(int(ln))],
            dtype=np.int64,
        )
        eps = ds.index["episodes"][:n]
        if blank:
            eps = [{**ep, "instruction": BLANK_INSTRUCTION} for ep in eps]
        self.index = {**ds.index, "n": n, "episodes": eps}


class _ArmCtx:
    """fit()'s ctx duck-type over the real RunContext, scoped to one arm of the run.

    Prefixes the arm's loss metrics (lr / steps_per_s stay shared - same schedule per arm),
    offsets steps into the whole run, injects the arm's status message, and - for the suite -
    captures fit's final checkpoint payload instead of writing it, so train() can save the
    combined multi-arm payload (the task_token house pattern).
    """

    def __init__(self, ctx, arm: _Arm, i: int, n_arms: int, steps: int, capture: bool):
        self._ctx, self._arm = ctx, arm
        self._i, self._n, self._steps = i, n_arms, steps
        self._off = i * steps
        self._capture = capture
        self.captured: dict | None = None
        self.params = ctx.params

    def should_continue(self) -> bool:
        return self._ctx.should_continue()

    def log(self, step: int, **series) -> None:
        if self._arm.key:
            series = {
                (f"{self._arm.key}_{k}" if k.startswith("loss") else k): v
                for k, v in series.items()
            }
        self._ctx.log(self._off + step, **series)

    def set_status(self, **kw) -> None:
        local = int(kw.get("step", 0))
        if "step" in kw:
            kw["step"] = self._off + local
        kw["total_steps"] = self._n * self._steps
        kw.setdefault(
            "message",
            f"arm {self._i + 1}/{self._n} ({self._arm.label}): step {local}/{self._steps}",
        )
        self._ctx.set_status(**kw)

    def save_ckpt(self, step: int, payload: dict, label=None, metrics=None):
        if self._capture:
            self.captured = payload  # train() saves the combined payload instead
            return None
        return self._ctx.save_ckpt(self._off + step, payload, label=label, metrics=metrics)


def _make_cb(env: Env, arm: _Arm, device: str):
    """The single preset's evaluation callback: closed-loop success on held-out seeds."""

    def cb(policy, step: int) -> dict:
        seeds = eval_seeds(arm.task, CB_SEEDS, offset=CB_SEED_OFFSET)  # global read: patchable
        ok = 0
        for s in seeds:
            rec = _drain(_episode(_Job(arm=arm, seed=s, policy=policy, env=env, device=device)))
            ok += rec["success"]
        return {"eval_success": ok / max(1, len(seeds))}

    return cb


# ------------------------------------------------------------------------------------- train
def train(ctx) -> None:
    P = ctx.params
    preset, steps = str(P["preset"]), int(P["steps"])
    lr, batch, n_demos = float(P["lr"]), int(P["batch"]), int(P["demos"])
    seed = int(P.get("seed", 0))
    cfg_kw = {
        "chunk": int(P["chunk"]),
        "n_exec": int(P["n_exec"]),
        "euler_steps": int(P["euler_steps"]),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arms = _train_arms(preset, str(P["task"]))
    root = _demo_root()
    cache: dict[str, DemoDataset] = {}  # the suite reuses t1 between arms B and C
    captured: dict[str, dict] = {}
    arm_losses: dict[str, float] = {}
    env: Env | None = None
    last_hist: list[dict] = []
    try:
        for i, arm in enumerate(arms):
            ctx.set_status(
                step=i * steps, total_steps=len(arms) * steps,
                message=f"arm {i + 1}/{len(arms)} ({arm.label}): loading {arm.task} demos "
                        f"from {root / arm.task}",
            )  # fmt: skip
            if arm.task not in cache:
                cache[arm.task] = DemoDataset(root, arm.task)
            view = _DemoView(cache[arm.task], n_demos, blank=arm.blank_text)
            callbacks, eval_every = (), 0
            if preset == "single":
                env = env or Env(task=arm.task, render=True)
                callbacks, eval_every = (_make_cb(env, arm, device),), EVAL_EVERY
            actx = _ArmCtx(ctx, arm, i, len(arms), steps, capture=preset != "single")
            _, hist = fit(
                ModelConfig(**cfg_kw), view, actx,
                steps=steps, lr=lr, batch=batch,
                eval_every=eval_every, callbacks=callbacks,
                seed=seed + 101 * i,
                log_every=max(10, steps // 200),
                ckpt_every=max(2000, steps // 5) if preset == "single" else 0,
                device=device,
            )  # fmt: skip
            last_hist = hist
            arm_losses[arm.label] = next(
                (r["loss"] for r in reversed(hist) if "loss" in r), float("nan")
            )
            if preset != "single":
                captured[arm.label] = {
                    **(actx.captured or {}), "task": arm.task, "blank_text": arm.blank_text,
                }  # fmt: skip
                payload = {
                    "kind": "nfv-suite", "arms": [a.label for a in arms[: i + 1]],
                    **captured, "params": dict(P),
                }  # fmt: skip
                label = "final" if i == len(arms) - 1 else f"{arm.label} trained"
                metrics = {f"{a.key}_loss": round(arm_losses[a.label], 4) for a in arms[: i + 1]}
                ctx.save_ckpt((i + 1) * steps, payload, label=label, metrics=metrics)
            if not ctx.should_continue():
                ctx.set_status(state="stopped")
                return
    finally:
        if env is not None:
            env.close()
    if preset == "single":
        last_loss = arm_losses.get(arms[0].label, float("nan"))
        ev = next((r["eval_success"] for r in reversed(last_hist) if "eval_success" in r), None)
        ctx.done(
            f"trained nanoVLA on {arms[0].task}: loss {last_loss:.4f}"
            + (f", closed-loop success {ev:.0%} on {CB_SEEDS} held-out seeds" if ev is not None
               else "")
        )  # fmt: skip
    else:
        parts = ", ".join(f"{a.label} loss {arm_losses[a.label]:.4f}" for a in arms)
        ctx.done(f"trained 3 arms in one checkpoint: {parts}")


# ---------------------------------------------------------------------------- eval episodes
@dataclass
class _Job:
    """One closed-loop episode: which arm's policy rolls where, what gets recorded."""

    arm: _Arm
    seed: int
    policy: object  # nfv.model.build.Policy (acts through its EMA)
    env: Env
    device: str
    render: bool = False
    trace: bool = False
    card: list = field(default_factory=list)  # extra title-card lines

    @property
    def group(self) -> str:
        """Aggregation key (task_nfv0._spread spreads traces/video round-robin over it)."""
        return self.arm.label


def _episode(job: _Job):
    """Roll the policy for one episode; yields video frames when ``job.render`` and returns
    the record via StopIteration — task 15's video pattern (task_nfv0._episode)."""
    import mujoco

    env, arm = job.env, job.arm
    obs = env.reset(job.seed, task=arm.task)
    instr = BLANK_INSTRUCTION if arm.blank_text else env.instruction
    gen = torch.Generator(device=job.device)
    gen.manual_seed(job.seed)  # flow noise per episode: the rollout is seed-deterministic
    tmax = int(env.task.TIME_LIMIT_S * FPS)
    n_exec = max(1, int(job.policy.cfg.n_exec))
    tr: dict = {"t": [], "fz": [], "mode": [], "subtask": [], "events": []}
    fz_v: list[float] = []
    ft_v: list[float] = []  # all-zero: nanoVLA commands no wrench (the strip's command line)
    rend = None
    if job.render:
        rend = mujoco.Renderer(env.model, height=VH, width=VW)

    def frame(status=""):
        import cv2

        rend.update_scene(env.data, camera="static")
        img = np.vstack([rend.render().copy(), _strip(fz_v, ft_v, tmax)])
        txt = f't {env.t:4.1f} s  Fz {float(env.wrench[2]):+6.2f} N  "{instr}"'
        cv2.rectangle(img, (0, 0), (VW, 18), (255, 255, 255), -1)
        cv2.putText(img, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1,
                    cv2.LINE_AA)  # fmt: skip
        if status:
            col = (30, 140, 60) if status == "success" else (200, 50, 40)
            cv2.putText(img, status.upper(), (10, VH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col,
                        2, cv2.LINE_AA)  # fmt: skip
        return img

    try:
        if job.render:
            yield frame()
        info: dict = {"done": False, "outcome": "timeout", "t": 0.0}
        k, clicked, aborted = 0, False, False
        while not info["done"] and k < tmax:
            chunk = job.policy.act(obs, instr, generator=gen)  # {"dp" [H, 3], "gripper" [H]}
            for j in range(min(n_exec, tmax - k)):
                act = Action(
                    dp=chunk["dp"][j].astype(np.float64), gripper=float(chunk["gripper"][j])
                )
                obs, info = env.step(act)
                k += 1
                tr["t"].append(round(info["t"], 2))
                tr["fz"].append(_r3(obs.wrench[2]))
                tr["mode"].append("position")  # the position head commands no wrench
                tr["subtask"].append(env.task.SUBTASKS[obs.subtask_id])
                fz_v.append(float(obs.wrench[2]))
                ft_v.append(0.0)
                if arm.task == "t1" and info.get("click") and not clicked:
                    clicked = True
                    tr["events"].append({"t": round(info["t"], 2), "label": "click"})
                if info["force_abort"] and not aborted:
                    aborted = True
                    tr["events"].append({"t": round(info["t"], 2), "label": "abort"})
                if arm.task == "t3" and info["outcome"] == "success":
                    tr["events"].append({"t": round(info["t"], 2), "label": "seated"})
                if job.render:
                    yield frame()
                if info["done"]:
                    break
        rec = {
            "task": arm.task, "seed": job.seed, "group": arm.label,
            "outcome": info["outcome"] or "timeout", "peak_force": _r3(env.peak_force),
            "instruction": env.instruction,
        }  # fmt: skip
        rec["success"] = rec["outcome"] == "success"
        if arm.task == "t1":
            rec["target"] = env.task.target
            rec["pressed"] = [c for c in env.task.COLORS if env.task.clicked[c]]
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


# ---------------------------------------------------------------------------------- evaluate
def _policy_from(payload: dict, device: str, cfg_w: float = 1.0, n_exec: int = 0):
    """Rebuild a Policy from one fit payload (arch + model + ema), ready to act on device.

    ``cfg_w`` / ``n_exec`` (0 = keep the trained value) are the act-time overrides: they go
    into the rebuilt ModelConfig only — the checkpoint weights are untouched.
    """
    arch = dict(payload["arch"])
    arch["cfg_w"] = float(cfg_w)
    if n_exec:
        arch["n_exec"] = min(int(n_exec), int(arch["chunk"]))
    cfg = ModelConfig(**arch)
    policy = build(cfg)
    policy.net.load_state_dict(payload["model"])
    policy.ema.load_state_dict(payload["ema"])
    return policy.to(device).eval()


def _eval_arms(
    ck: dict, device: str, cfg_w: float = 1.0, n_exec: int = 0
) -> list[tuple[_Arm, object]]:
    """(arm, policy) pairs of a checkpoint: every suite arm, or the one single model."""
    if "arms" in ck:  # the claims_suite payload
        out = []
        for label in ck["arms"]:
            pl = ck[label]
            arm = _Arm("", str(label), str(pl["task"]), bool(pl.get("blank_text")))
            out.append((arm, _policy_from(pl, device, cfg_w, n_exec)))
        return out
    task = str(ck.get("params", {}).get("task", "t1"))
    return [(_Arm("", "single", task, False), _policy_from(ck, device, cfg_w, n_exec))]


def _arm_summary(arm: _Arm, rows: list[dict]) -> dict:
    k = sum(r["success"] for r in rows)
    out = {
        "task": arm.task, "blank_text": arm.blank_text, "n": len(rows),
        "success": round(k / max(1, len(rows)), 4), "wilson95": wilson95(k, len(rows)),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "peak_force": {"mean": _r3(np.mean([r["peak_force"] for r in rows])),
                       "max": _r3(np.max([r["peak_force"] for r in rows]))},
    }  # fmt: skip
    if arm.task == "t1":
        per = {}
        for instr in T1_INSTRUCTIONS:
            sub = [r for r in rows if r.get("instruction") == instr]
            if sub:
                kk = sum(r["success"] for r in sub)
                per[instr] = {"n": len(sub), "success": round(kk / len(sub), 4)}
        out["per_instruction"] = per  # arm C's cross-test: both buttons pooled, split here
        out["pressed_counts"] = dict(Counter(c for r in rows for c in r.get("pressed", ())))
    return out


def _claim_rows(filled: dict[str, dict]) -> list[dict]:
    rows = []
    for c in M1_CLAIMS:
        f = filled.get(c.id)
        rows.append({
            "id": c.id, "statement": c.statement, "metric": c.metric, "expected": c.expect,
            "measured": f["measured"] if f else "not measured in this run",
            "verdict": f["verdict"] if f else NOT_MEASURED,
            "source": c.source, "url": c.url,
        })  # fmt: skip
    return rows


def _fill_claims(arms: list[tuple[_Arm, object]], summaries: dict[str, dict]) -> dict:
    def find(task: str, blank: bool):
        for arm, _ in arms:
            s = summaries.get(arm.label)
            if s and arm.task == task and arm.blank_text == blank and s["n"]:
                return s
        return None

    def kn(s):
        return f"{int(round(s['success'] * s['n']))}/{s['n']}"

    filled: dict[str, dict] = {}
    t0 = find("t0", False)
    if t0:
        lo, hi = t0["wilson95"]
        filled["m1.nanovla_solves_visible"] = {
            "measured": f"T0 {kn(t0)} ({t0['success']:.0%}, CI {lo:.1%}-{hi:.1%})",
            "verdict": verdict("a >= 0.95", {"a": t0["success"]}, ci=(lo, hi)),
        }
    t1d = find("t1", False)
    if t1d:
        lo, hi = t1d["wilson95"]
        note = (
            " - vision solves T1: the task premise broke, re-randomise T1 (plan §8, M1 gate)"
            if t1d["success"] > 0.70
            else ""
        )
        filled["m1.hidden_threshold_blinds_vla"] = {
            "measured": f"T1 {kn(t1d)} ({t1d['success']:.0%}, CI {lo:.1%}-{hi:.1%}){note}",
            "verdict": verdict("0.30 <= a <= 0.70", {"a": t1d["success"]}, ci=(lo, hi)),
        }
    t1b = find("t1", True)
    if t1d and t1b:
        # conservative CI of the gap from the two arms' Wilson intervals (the task-15 idiom)
        ci = (
            t1d["wilson95"][0] - t1b["wilson95"][1],
            t1d["wilson95"][1] - t1b["wilson95"][0],
        )
        gap = t1d["success"] - t1b["success"]
        filled["m1.language_is_load_bearing"] = {
            "measured": f"T1 default {kn(t1d)} vs no-language {kn(t1b)} "
            f"(gap {gap:.0%}, CI {ci[0]:.1%}-{ci[1]:.1%})",
            "verdict": verdict("a-b >= 0.2", {"a": t1d["success"], "b": t1b["success"]}, ci=ci),
        }
    return filled


def evaluate(ctx) -> None:
    P = ctx.params
    episodes, eseed, video = int(P["episodes"]), int(P["eval_seed"]), bool(P["video"])
    cfg_w = float(P.get("cfg_w", 1.0))
    n_exec_override = int(P.get("n_exec_override", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.set_status(message="loading the checkpoint and rebuilding the policies")
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    arms = _eval_arms(ck, device, cfg_w, n_exec_override)
    scenario = "claims_suite" if "arms" in ck else "single"
    offset = eseed * episodes
    env = Env(task=arms[0][0].task, render=True,
              gantry=GantryParams(offwidth=VW, offheight=VH))  # fmt: skip

    jobs: list[_Job] = []
    for arm, policy in arms:
        for s in eval_seeds(arm.task, episodes, offset=offset):
            jobs.append(_Job(
                arm=arm, seed=s, policy=policy, env=env, device=device,
                card=[f"arm {arm.label} on {arm.task}",
                      "policy text: blanked" if arm.blank_text else "policy text: instruction"],
            ))  # fmt: skip
    _spread(jobs, min(MAX_TRACES, len(jobs)), "trace")
    if video:
        _spread(jobs, min(MAX_VIDEO, len(jobs)), "render")

    rows: list[dict] = []
    done_n = 0
    stopped = False

    def run_one(job: _Job) -> None:
        nonlocal done_n
        done_n += 1
        ctx.set_status(
            step=done_n, total_steps=len(jobs),
            message=f"{scenario}: {job.arm.label}, episode {done_n}/{len(jobs)}",
        )  # fmt: skip

    rendered = [j for j in jobs if j.render]
    plain = [j for j in jobs if not j.render]

    def frames():
        nonlocal stopped
        for i, job in enumerate(rendered):
            if not ctx.should_continue():
                stopped = True
                return
            run_one(job)
            card = [f"nanoVLA ({scenario})", f"episode {i + 1} of {len(rendered)} on video",
                    *job.card, f"seed {job.seed}"]  # fmt: skip
            for _ in range(CARD_FRAMES):
                yield _card(card)
            rec = yield from _episode(job)
            rows.append(rec)

    video_name = None
    try:
        if rendered:
            try:
                frames_to_mp4(frames(), ctx.dir / "rollout.mp4", fps=FPS)
                video_name = "rollout.mp4"
            except Exception as e:  # noqa: BLE001 - a render failure must not lose the stats
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
    finally:
        env.close()

    # ---- aggregate ----
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["group"], []).append(r)
    summaries = {
        arm.label: _arm_summary(arm, by_arm.get(arm.label, []))
        for arm, _ in arms
        if by_arm.get(arm.label)
    }
    k = sum(r["success"] for r in rows)
    result: dict = {
        "scenario": scenario, "episodes": episodes, "n_episodes": len(jobs), "fps": FPS,
        "video": video_name,
        "cfg_w": cfg_w, "n_exec_override": n_exec_override,  # the act-time recipe knobs
        "arms": summaries,
        "outcomes": {
            "success": round(k / max(1, len(rows)), 4),
            "counts": dict(Counter(r["outcome"] for r in rows)),
            "wilson95": wilson95(k, len(rows)),
            "per_task": {
                lbl: {"n": s["n"], "success": s["success"], "wilson95": s["wilson95"],
                      "aborts": s["outcomes"].get("force_abort", 0),
                      "peak_force_mean": s["peak_force"]["mean"]}
                for lbl, s in summaries.items()
            },
        },
        "peak_force": {"mean": _r3(np.mean([r["peak_force"] for r in rows])),
                       "max": _r3(np.max([r["peak_force"] for r in rows]))},
    }  # fmt: skip
    result["claims"] = _claim_rows(_fill_claims(arms, summaries))
    result["episodes_traces"] = [
        {k2: v for k2, v in {**r, **r["trace"]}.items() if k2 != "trace"}
        for r in rows
        if "trace" in r
    ][:MAX_TRACES]
    atomic_write_json(ctx.dir / "result.json", result)
    arm_txt = ", ".join(f"{lbl} {s['success']:.0%}" for lbl, s in summaries.items())
    verdicts = "; ".join(f"{r['id'].split('.')[-1]}: {r['verdict']}" for r in result["claims"]
                         if r["verdict"] != NOT_MEASURED)  # fmt: skip
    ctx.done(
        f"{scenario}: {arm_txt} over {len(rows)} episodes" + (f" - {verdicts}" if verdicts else "")
    )


# ---------------------------------------------------------------------------------- estimate
def estimate(params: dict) -> float:
    cal = registry.calib("nano-vla")
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    load = cal.get("load_s_per_demo", FALLBACK_LOAD_S_PER_DEMO)
    arms = 1 if params["preset"] == "single" else 3
    loads = 1 if params["preset"] == "single" else 2  # the suite loads t0 and t1 once each
    t = arms * params["steps"] / sps + loads * params["demos"] * load
    if params["preset"] == "single":  # periodic closed-loop evaluation points
        points = max(1, params["steps"] // EVAL_EVERY)
        t += points * CB_SEEDS * cal.get("ep_s", FALLBACK_EP_S)
    return t + cal.get("overhead_s", OVERHEAD_S)


def check(params: dict) -> None:
    if params["n_exec"] > params["chunk"]:
        raise ValueError("n_exec must be <= chunk: a chunk cannot execute more steps than "
                         "it predicts")  # fmt: skip


SPEC = TaskSpec(
    id="nano-vla",
    order=16,
    title="nanoVLA: a tiny VLA from scratch",
    subtitle="Pixels and words to action chunks - vision stem, tiny language path, trunk and "
             "a flow-matching action expert on T0/T1; the claims suite shows what the camera "
             "cannot see and what the words carry.",  # fmt: skip
    params=[
        Param("preset", "Preset", "choice", "single", choices=list(PRESETS),
              help="single = one model on the chosen task; claims_suite = the three M1 arms "
                   "(A t0, B t1, C t1 no-language) in one job, one checkpoint"),
        Param("task", "Task (single preset)", "choice", "t1", choices=list(TASK_NAMES),
              help="which canonical demo set the single preset trains on"),
        Param("steps", "Training steps per arm", "int", 20000, min=50, max=60000, step=50),
        Param("lr", "Learning rate", "float", 3e-4, min=1e-5, max=3e-3,
              help="AdamW peak LR (linear warmup, cosine to 0 - nfv.train.fit)"),
        Param("batch", "Batch size", "int", 64, min=8, max=128, step=8),
        Param("demos", "Demos (first n episodes)", "int", 200, min=8, max=400, step=4,
              help="subset of the canonical set by episode order (the data-curve knob); "
                   "clamped to what lesson 15 recorded"),
        Param("chunk", "Chunk H (actions per prediction)", "int", 16, min=2, max=32),
        Param("n_exec", "Executed steps per chunk", "int", 8, min=1, max=32,
              help="re-plan after this many of the H steps (<= chunk)"),
        Param("euler_steps", "Euler steps at inference", "int", 10, min=1, max=50),
        Param("seed", "Seed", "int", 0, min=0, max=9999,
              help="init + batch sampling + flow noise; suite arms use seed + 101*i"),
    ],  # fmt: skip
    eval_params=[
        Param("episodes", "Episodes per arm", "int", 100, min=2, max=1000, step=1,
              help="held-out seeds per arm; the claims want 100 for decisive intervals"),
        Param("eval_seed", "Eval seed", "int", 0, min=0, max=99,
              help="shifts the window inside the task's eval-seed block"),
        Param("video", "Render a rollout video", "bool", True),
        Param("cfg_w", "CFG weight on the instruction", "float", 1.0, min=0.5, max=5.0,
              help="act-time classifier-free guidance: each Euler step combines "
                   "v_blank + w*(v_cond - v_blank); 1.0 = the plain conditioned model. "
                   "Lesson 17 round 2 measures the cure; on arm C it is a no-op (both "
                   "branches read the blank text)"),
        Param("n_exec_override", "Executed steps override", "int", 0, min=0, max=16,
              help="act-time only: 0 = the checkpoint's trained n_exec; 1..16 = execute this "
                   "many steps of each chunk before re-planning (clamped to the chunk "
                   "length) - the reactivity lever from lesson 4"),
    ],  # fmt: skip
    charts=[
        {"id": "loss", "title": "Flow-matching loss",
         "series": [{"key": "loss", "label": "single"}, {"key": "a_loss", "label": "A t0"},
                    {"key": "b_loss", "label": "B t1"},
                    {"key": "c_loss", "label": "C t1 no-lang"}],
         "log_y": True},
        {"id": "eval", "title": "Closed-loop success during training (single preset, "
                                "20 held-out seeds)",
         "series": [{"key": "eval_success", "label": "success"}], "log_y": False},
        {"id": "speed", "title": "Training speed (steps/s)",
         "series": [{"key": "steps_per_s", "label": "steps/s"}], "log_y": False},
        {"id": "lr", "title": "Learning rate", "series": [{"key": "lr", "label": "lr"}],
         "log_y": False},
    ],  # fmt: skip
    viewer="fvrollout",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="nano-vla.html",
    check=check,
)  # fmt: skip
registry.register(SPEC)
