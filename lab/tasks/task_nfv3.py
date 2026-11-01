"""Lesson 18 - "ForceVLA2: force as an output" (id `nano-force-vla2`): the M3 model in the lab.

TRAIN builds the nano policy with ForceVLA2's modules (nfv.model.forcevla2: the cross-attention
encoder with the force bypass, the dense 3-expert MoE, the hybrid head; nfv.sim.subtasks: the
force prompt) on lesson 15's canonical demos and trains it with `nfv.train.fit` on the real
RunContext. ``preset`` picks the shape of the run:

  single        ONE model from the form: force_in (default forcevla2), head (hybrid |
                position), force_prompt, expert_inputs, on ``task`` (default t2) - with the
                periodic closed-loop evaluation (CB_SEEDS held-out seeds every EVAL_EVERY
                steps and at the end, logged as ``eval_success``).
  claims_suite  THREE arms sequentially in one job, all on T2 - the commanded-force wipe, the
                one task where a force OUTPUT is the lever (fresh seeds per arm, seed + 101*i):
                  a  fvlmoe,    position head, no prompt  (lesson 17's model on T2)
                  b  forcevla2, hybrid head,   prompt     (full ForceVLA2)
                  c  forcevla2, position head, prompt     (ForceVLA2 INPUTS, no force output -
                                                           the row the paper never ran)
                Budget-reduced from the plan's build-up chain (+prompt / +encoder / +MoE /
                expert ablations): three arms answer the two questions lesson 18 asks. ONE
                checkpoint payload holds all arms ({"kind": "nfv-suite", "arms": [...],
                <label>: {fit payload + "task"}, "params"}), saved after every arm.

EVAL rolls every arm closed-loop on SHARED held-out seeds (`nfv.seeds.eval_seeds`, shifted by
``eval_seed * episodes``). The executor (one per episode):

    sm = SubtaskMachine.for_task(task)                     # only if the arm has force_prompt
    every n_exec steps: chunk = policy.act(obs, sm.prompt(instr) or instr)
    for each executed step j:  env.step(Action(dp_j, gripper_j, f_target_j or None, s_j))
                               sm.update(s_j)              # hybrid head: the model's own s

A position-head arm with the prompt (arm c) predicts no progress, so its prompt follows the
env's state heuristic ``Obs.subtask_id`` (clamped to the subtasks the demos label) - a prompt
at least as informed as arm b's self-driven one, so c's gap to b is not a prompt artefact.

Per group the eval reports success + Wilson, outcomes/abort rate, peak force, the T2 force
tracking in contact (RMSE of F_n - F_target over policy steps with F_n > CONTACT_N, against the
instruction's commanded target: 3 N gently / 8 N firmly), the task's own track_frac/coverage,
and for hybrid arms the free-space force-axis selection: ``xy_select_free`` = fraction of
free-space steps with |f_target_x| or |f_target_y| > SELECT_ON_N (the wipe expert never
commands a lateral force - every one is spurious) and ``z_select_free`` = the same for z
(includes the legitimate force-controlled descent of "engage"). ForceVLA2 arms log the gate:
per act() decision the mean [visual, state, force] expert weights per token group (vision /
text / state / force tokens), filed under contact / free by F_n at the decision.

Claims (nfv.claims.M3_CLAIMS) are filled from the t2 groups: b alone -> m3.hybrid_tracks_force;
c -> m3.position_head_cannot; b + c -> m3.force_output_is_the_lever; b + a ->
m3.beats_forcevla; b's gate -> m3.gate_tilts_to_force. Gap CIs are the conservative interval
from the two arms' Wilsons (the house idiom).

result.json (viewer `fvrollout`; every field optional there):
    {"scenario", "episodes", "n_episodes", "fps", "video": "rollout.mp4"|null,
     "arms": {group: {"task", "force_in", "head", "force_prompt", "n", "success", "wilson95",
                      "outcomes", "abort_rate", "peak_force": {"mean", "max"},
                      "force_rmse_contact", "track_frac", "coverage",
                      "per_instruction": {instr: {n, success, force_rmse_contact}},
                      "xy_select_free", "z_select_free"}},     # last two: hybrid arms only
     "outcomes": {...}, "peak_force": {...},
     "gate": {group: {"contact"|"free": {"decisions", <token group>: [w_vis, w_state,
              w_force]}}},                                       # ForceVLA2 arms only
     "claims": [rows for all of M3_CLAIMS],
     "episodes_traces": [<= 10 lesson-16 traces: t, fz, f_target (commanded F_z), mode
                         ("hybrid" when |f_target_z| selects z), subtask (the PROMPT's word),
                         events]}
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
from lab.tasks.task_nfv1 import _ArmCtx, _DemoView
from nfv.claims import M3_CLAIMS, NOT_MEASURED, verdict
from nfv.data.dataset import DemoDataset
from nfv.model import ModelConfig, build
from nfv.model.config import HYBRID_ACT_DIM
from nfv.seeds import TASK_NAMES, eval_seeds
from nfv.sim.controllers import SELECT_ON_N
from nfv.sim.env import Action, Env
from nfv.sim.gantry_xml import GantryParams
from nfv.sim.subtasks import SubtaskMachine
from nfv.train import fit

PRESETS = ("single", "claims_suite")
FORCE_IN_CHOICES = ["forcevla2", "fvlmoe", "none"]
HEAD_CHOICES = ["hybrid", "position"]
EXPERT_INPUT_CHOICES = ["all", "vision_only", "none"]
SUITE_TASK = "t2"  # the commanded-force wipe: the task where the force OUTPUT matters
CONTACT_N = 0.5  # N: F_n (or |F| on non-T2 tasks) above this = contact
CB_SEEDS = 20  # held-out seeds of the single preset's periodic closed-loop evaluation
CB_SEED_OFFSET = 9_000  # top of the eval block - disjoint from the eval job's offsets
EVAL_EVERY = 4_000  # steps between evaluation points (fit also fires at the final step)
TOKEN_GROUPS = ("vision", "text", "state", "force")
# who advances the force prompt at eval: the model's own s (hybrid; position-head arms follow
# the env heuristic) | the env's state heuristic for every prompt arm | never (first word)
PROMPT_MODES = ("machine", "heuristic", "frozen")
# Test hook: extra ModelConfig fields (tests shrink chunk/layers; the form exposes no
# architecture knobs - the suite compares heads/fusions on ONE canonical architecture).
ARCH_OVERRIDES: dict = {}
# calib.json fallbacks (lesson 17's measured rate; ForceVLA2 adds ~1.1 M params)
FALLBACK_STEPS_PER_S, FALLBACK_LOAD_S_PER_DEMO = 25.0, 0.06
FALLBACK_EP_S, OVERHEAD_S = 4.0, 15.0


@dataclass(frozen=True)
class _Arm:
    """One trained model: its metric prefix, label, and the four lesson-18 switches."""

    key: str  # metric prefix ("a".."c"; "" = unprefixed, the single preset)
    label: str
    force_in: str
    head: str
    force_prompt: bool
    task: str
    expert_inputs: str = "all"


SUITE = (
    _Arm("a", "A fvlmoe+position", "fvlmoe", "position", False, SUITE_TASK),
    _Arm("b", "B forcevla2+hybrid", "forcevla2", "hybrid", True, SUITE_TASK),
    _Arm("c", "C forcevla2+position", "forcevla2", "position", True, SUITE_TASK),
)


def _train_arms(P: dict) -> list[_Arm]:
    if str(P["preset"]) == "single":
        fi = str(P["force_in"])
        ei = str(P["expert_inputs"]) if fi == "forcevla2" else "all"
        return [_Arm("", "single", fi, str(P["head"]), bool(P["force_prompt"]), str(P["task"]),
                     ei)]  # fmt: skip
    return list(SUITE)


def _model_cfg(arm: _Arm) -> ModelConfig:
    return ModelConfig(
        force_in=arm.force_in,
        head=arm.head,
        act_dim=HYBRID_ACT_DIM if arm.head == "hybrid" else 4,
        force_prompt=arm.force_prompt,
        expert_inputs=arm.expert_inputs,
        **ARCH_OVERRIDES,
    )


class _Fv2ArmCtx(_ArmCtx):
    """lesson 16/17's arm ctx, also prefixing the gate series (gate_force_*) per arm."""

    def log(self, step: int, **series) -> None:
        if self._arm.key:
            series = {
                (f"{self._arm.key}_{k}" if k.startswith(("loss", "gate")) else k): v
                for k, v in series.items()
            }
        self._ctx.log(self._off + step, **series)


def _make_cb(env: Env, arm: _Arm, device: str):
    """The single preset's evaluation callback: closed-loop success on held-out seeds."""

    def cb(policy, step: int) -> dict:
        seeds = eval_seeds(arm.task, CB_SEEDS, offset=CB_SEED_OFFSET)  # global read: patchable
        ok = 0
        for s in seeds:
            rec = _drain(_episode(_Job(arm=arm, group=arm.label, seed=s, policy=policy,
                                       env=env, device=device)))  # fmt: skip
            ok += rec["success"]
        return {"eval_success": ok / max(1, len(seeds))}

    return cb


# ------------------------------------------------------------------------------------- train
def train(ctx) -> None:
    P = ctx.params
    preset, steps = str(P["preset"]), int(P["steps"])
    lr, batch, n_demos = float(P["lr"]), int(P["batch"]), int(P["demos"])
    seed = int(P.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arms = _train_arms(P)
    root = _demo_root()
    cache: dict[str, DemoDataset] = {}  # the suite's three arms share the one t2 set
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
            view = _DemoView(cache[arm.task], n_demos)
            callbacks, eval_every = (), 0
            if preset == "single":
                env = env or Env(task=arm.task, render=True)
                callbacks, eval_every = (_make_cb(env, arm, device),), EVAL_EVERY
            actx = _Fv2ArmCtx(ctx, arm, i, len(arms), steps, capture=preset != "single")
            policy, hist = fit(
                _model_cfg(arm), view, actx,
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
                captured[arm.label] = {**(actx.captured or {}), "task": arm.task}
                payload = {
                    "kind": "nfv-suite", "arms": [a.label for a in arms[: i + 1]],
                    **captured, "params": dict(P),
                }  # fmt: skip
                label = "final" if i == len(arms) - 1 else f"{arm.key} trained"
                metrics = {f"{a.key}_loss": round(arm_losses[a.label], 4) for a in arms[: i + 1]}
                ctx.save_ckpt((i + 1) * steps, payload, label=label, metrics=metrics)
            if not ctx.should_continue():
                ctx.set_status(state="stopped")
                return
    finally:
        if env is not None:
            env.close()
    if preset == "single":
        a = arms[0]
        last_loss = arm_losses.get(a.label, float("nan"))
        ev = next((r["eval_success"] for r in reversed(last_hist) if "eval_success" in r), None)
        ctx.done(
            f"trained {a.force_in}+{a.head} on {a.task}: loss {last_loss:.4f}"
            + (f", closed-loop success {ev:.0%} on {CB_SEEDS} held-out seeds" if ev is not None
               else "")
        )  # fmt: skip
    else:
        parts = ", ".join(f"{a.label} loss {arm_losses[a.label]:.4f}" for a in arms)
        ctx.done(f"trained the 3 lesson-18 arms in one checkpoint: {parts}")


# ---------------------------------------------------------------------------- eval episodes
class _GateAcc:
    """Per-group gate accumulator: one act() decision = one sample of the per-token-group mean
    expert weights [visual, state, force], filed under contact / free (F_n at the decision)."""

    def __init__(self):
        self.b = {k: {"decisions": 0, **{g: np.zeros(3) for g in TOKEN_GROUPS}}
                  for k in ("contact", "free")}  # fmt: skip

    def add(self, gate_groups: dict, contact: bool) -> None:
        b = self.b["contact" if contact else "free"]
        b["decisions"] += 1
        for g in TOKEN_GROUPS:
            b[g] += np.asarray(gate_groups[g], dtype=np.float64)

    def summary(self) -> dict | None:
        if not any(b["decisions"] for b in self.b.values()):
            return None
        out = {}
        for k, b in self.b.items():
            n = b["decisions"]
            out[k] = {"decisions": n,
                      **{g: [round(float(x) / n, 4) for x in b[g]] if n else [0.0] * 3
                         for g in TOKEN_GROUPS}}  # fmt: skip
        return out


@dataclass
class _Job:
    """One closed-loop episode of one arm on one seed."""

    arm: _Arm
    group: str
    seed: int
    policy: object  # nfv.model.build.Policy (acts through its EMA)
    env: Env
    device: str
    gate: _GateAcc | None = None  # shared per group (ForceVLA2 arms only)
    prompt_mode: str = "machine"  # machine | heuristic | frozen (PROMPT_MODES)
    render: bool = False
    trace: bool = False
    card: list = field(default_factory=list)


def _normal_force(env: Env, wrench) -> float:
    """T2's F_n = -F_z (positive pressing); elsewhere the force magnitude."""
    if hasattr(env.task, "normal_force"):
        return float(env.task.normal_force(wrench))
    return float(np.linalg.norm(np.asarray(wrench)[:3]))


def _episode(job: _Job):
    """Roll the policy for one episode with the hybrid executor (module docstring); yields
    video frames when ``job.render`` and returns the record via StopIteration."""
    import mujoco

    env, arm = job.env, job.arm
    cfg = job.policy.cfg
    obs = env.reset(job.seed, task=arm.task)
    instr = env.instruction
    gen = torch.Generator(device=job.device)
    gen.manual_seed(job.seed)  # flow noise per episode: seed-deterministic
    tmax = int(env.task.TIME_LIMIT_S * FPS)
    n_exec = max(1, int(cfg.n_exec))
    sm = SubtaskMachine.for_task(arm.task) if cfg.force_prompt else None
    hybrid = cfg.head == "hybrid"
    f_cmd = float(getattr(env.task, "f_target_n", 0.0))  # T2: the instruction's target
    tr: dict = {"t": [], "fz": [], "f_target": [], "mode": [], "subtask": [], "events": []}
    fz_v: list[float] = []
    ft_v: list[float] = []
    err2: list[float] = []  # (F_n - F_target)^2 over contact policy steps
    free_n = xy_sel = z_sel = 0
    rend = None
    if job.render:
        rend = mujoco.Renderer(env.model, height=VH, width=VW)

    def frame(status=""):
        import cv2

        rend.update_scene(env.data, camera="static")
        img = np.vstack([rend.render().copy(), _strip(fz_v, ft_v, tmax)])
        word = sm.current if sm is not None else "-"
        txt = f"t {env.t:4.1f} s  Fz {float(env.wrench[2]):+6.2f} N  [{word}]  {job.group}"
        cv2.rectangle(img, (0, 0), (VW, 18), (255, 255, 255), -1)
        cv2.putText(img, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (20, 20, 20), 1,
                    cv2.LINE_AA)  # fmt: skip
        if status:
            col = (30, 140, 60) if status == "success" else (200, 50, 40)
            cv2.putText(img, status.upper(), (10, VH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col,
                        2, cv2.LINE_AA)  # fmt: skip
        return img

    def prompt() -> str:
        if sm is None:
            return instr
        if job.prompt_mode == "frozen":  # the first subtask's word, forever
            return sm.prompt(instr)
        if not hybrid or job.prompt_mode == "heuristic":  # follow the env's state heuristic
            word = env.task.SUBTASKS[obs.subtask_id]
            if word in sm.subtasks:
                sm.index = sm.subtasks.index(word)
        return sm.prompt(instr)

    try:
        if job.render:
            yield frame()
        info: dict = {"done": False, "outcome": "timeout", "t": 0.0}
        k, aborted = 0, False
        while not info["done"] and k < tmax:
            chunk = job.policy.act(obs, prompt(), generator=gen)
            if job.gate is not None and "gate_groups" in chunk["aux"]:
                job.gate.add(chunk["aux"]["gate_groups"], _normal_force(env, obs.wrench)
                             > CONTACT_N)  # fmt: skip
            for j in range(min(n_exec, tmax - k)):
                ft = chunk["f_target"][j].astype(np.float64) if hybrid else None
                s = float(chunk["s"][j]) if hybrid else None
                act = Action(dp=chunk["dp"][j].astype(np.float64),
                             gripper=float(chunk["gripper"][j]), f_target=ft, s=s)  # fmt: skip
                word = sm.current if sm is not None else env.task.SUBTASKS[obs.subtask_id]
                obs, info = env.step(act)
                k += 1
                if sm is not None and s is not None and job.prompt_mode == "machine":
                    sm.update(s)
                f_n = _normal_force(env, obs.wrench)
                if f_n > CONTACT_N:
                    err2.append((f_n - f_cmd) ** 2)
                elif ft is not None:
                    free_n += 1
                    xy_sel += bool(np.abs(ft[:2]).max() > SELECT_ON_N)
                    z_sel += bool(abs(ft[2]) > SELECT_ON_N)
                ft_z = float(ft[2]) if ft is not None else 0.0
                tr["t"].append(round(info["t"], 2))
                tr["fz"].append(_r3(obs.wrench[2]))
                tr["f_target"].append(_r3(ft_z))
                tr["mode"].append("hybrid" if abs(ft_z) > SELECT_ON_N else "position")
                tr["subtask"].append(word)
                fz_v.append(float(obs.wrench[2]))
                ft_v.append(ft_z)
                if info["force_abort"] and not aborted:
                    aborted = True
                    tr["events"].append({"t": round(info["t"], 2), "label": "abort"})
                if job.render:
                    yield frame()
                if info["done"]:
                    break
        rec = {
            "task": arm.task, "seed": job.seed, "group": job.group,
            "outcome": info["outcome"] or "timeout", "peak_force": _r3(env.peak_force),
            "instruction": env.instruction, "f_target_n": f_cmd,
            "contact_steps": len(err2),
            "err2_sum": float(np.sum(err2)) if err2 else 0.0,
            "free_steps": free_n, "xy_sel": xy_sel, "z_sel": z_sel,
            "track_frac": _r3(info.get("track_frac", 0.0)),
            "coverage": _r3(info.get("coverage", 0.0)),
        }  # fmt: skip
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


# ---------------------------------------------------------------------------------- evaluate
def _policy_from(payload: dict, device: str):
    policy = build(ModelConfig(**payload["arch"]))
    policy.net.load_state_dict(payload["model"])
    policy.ema.load_state_dict(payload["ema"])
    return policy.to(device).eval()


def _arm_of(label: str, pl: dict, task: str) -> _Arm:
    a = pl["arch"]
    return _Arm("", label, str(a["force_in"]), str(a["head"]), bool(a["force_prompt"]), task,
                str(a.get("expert_inputs", "all")))  # fmt: skip


def _eval_arms(ck: dict) -> list[tuple[_Arm, dict]]:
    if "arms" in ck:  # the claims_suite payload
        return [(_arm_of(str(lbl), ck[lbl], str(ck[lbl]["task"])), ck[lbl]) for lbl in ck["arms"]]
    task = str(ck.get("params", {}).get("task", SUITE_TASK))
    return [(_arm_of("single", ck, task), ck)]


def _rmse(rows: list[dict]) -> float | None:
    n = sum(r["contact_steps"] for r in rows)
    return _r3(np.sqrt(sum(r["err2_sum"] for r in rows) / n)) if n else None


def _group_summary(arm: _Arm, rows: list[dict]) -> dict:
    k = sum(r["success"] for r in rows)
    oc = Counter(r["outcome"] for r in rows)
    out = {
        "task": arm.task, "force_in": arm.force_in, "head": arm.head,
        "force_prompt": arm.force_prompt, "expert_inputs": arm.expert_inputs, "n": len(rows),
        "success": round(k / max(1, len(rows)), 4), "wilson95": wilson95(k, len(rows)),
        "outcomes": dict(oc), "abort_rate": round(oc.get("force_abort", 0) / max(1, len(rows)), 4),
        "peak_force": {"mean": _r3(np.mean([r["peak_force"] for r in rows])),
                       "max": _r3(np.max([r["peak_force"] for r in rows]))},
        "force_rmse_contact": _rmse(rows),
        "contact_steps": int(sum(r["contact_steps"] for r in rows)),
        "track_frac": _r3(np.mean([r["track_frac"] for r in rows])),
        "coverage": _r3(np.mean([r["coverage"] for r in rows])),
    }  # fmt: skip
    per = {}
    for instr in sorted({r["instruction"] for r in rows}):
        sub = [r for r in rows if r["instruction"] == instr]
        kk = sum(r["success"] for r in sub)
        per[instr] = {"n": len(sub), "success": round(kk / len(sub), 4),
                      "force_rmse_contact": _rmse(sub)}  # fmt: skip
    out["per_instruction"] = per
    if arm.head == "hybrid":
        fn = sum(r["free_steps"] for r in rows)
        out["free_steps"] = int(fn)
        out["xy_select_free"] = round(sum(r["xy_sel"] for r in rows) / fn, 4) if fn else 0.0
        out["z_select_free"] = round(sum(r["z_sel"] for r in rows) / fn, 4) if fn else 0.0
    return out


def _both(v1: str, v2: str) -> str:
    """AND of two verdicts: any NOT CONFIRMED wins, then INCONCLUSIVE, else CONFIRMED."""
    if "NOT CONFIRMED" in (v1, v2):
        return "NOT CONFIRMED"
    return "INCONCLUSIVE" if "INCONCLUSIVE" in (v1, v2) else "CONFIRMED"


def _claim_rows(filled: dict[str, dict]) -> list[dict]:
    rows = []
    for c in M3_CLAIMS:
        f = filled.get(c.id)
        rows.append({
            "id": c.id, "statement": c.statement, "metric": c.metric, "expected": c.expect,
            "measured": f["measured"] if f else "not measured in this run",
            "verdict": f["verdict"] if f else NOT_MEASURED,
            "source": c.source, "url": c.url,
        })  # fmt: skip
    return rows


def _fill_claims(arms: list[_Arm], summaries: dict[str, dict], gate: dict) -> dict:
    """Fill every M3 claim the evaluated t2 groups can measure."""

    def find(force_in: str, head: str, prompt: bool | None = None):
        for a in arms:
            s = summaries.get(a.label)
            if (s and s["n"] and a.task == "t2" and a.force_in == force_in and a.head == head
                    and a.expert_inputs == "all"
                    and (prompt is None or a.force_prompt == prompt)):  # fmt: skip
                return a, s
        return None, None

    def kn(s):
        return f"{int(round(s['success'] * s['n']))}/{s['n']}"

    def gap_ci(a, b):
        return (a["wilson95"][0] - b["wilson95"][1], a["wilson95"][1] - b["wilson95"][0])

    filled: dict[str, dict] = {}
    ab, b = find("forcevla2", "hybrid", True)
    _, c = find("forcevla2", "position", True)
    _, a = find("fvlmoe", "position")
    if b:
        lo, hi = b["wilson95"]
        rmse = b["force_rmse_contact"]
        filled["m3.hybrid_tracks_force"] = {
            "measured": f"B T2 {kn(b)} ({b['success']:.0%}, CI {lo:.1%}-{hi:.1%}), contact "
            f"force RMSE {rmse if rmse is not None else 'n/a'} N",
            "verdict": _both(
                verdict("a >= 0.75", {"a": b["success"]}, ci=(lo, hi)),
                verdict("a <= 1.0", {"a": rmse if rmse is not None else float("inf")}),
            ),
        }
    if c:
        lo, hi = c["wilson95"]
        filled["m3.position_head_cannot"] = {
            "measured": f"C T2 {kn(c)} ({c['success']:.0%}, CI {lo:.1%}-{hi:.1%}), contact "
            f"force RMSE {c['force_rmse_contact']} N",
            "verdict": verdict("a <= 0.20", {"a": c["success"]}, ci=(lo, hi)),
        }
    for cid, other, name in (("m3.force_output_is_the_lever", c, "C"),
                             ("m3.beats_forcevla", a, "A")):  # fmt: skip
        if b and other:
            ci = gap_ci(b, other)
            filled[cid] = {
                "measured": f"B {kn(b)} vs {name} {kn(other)} "
                f"(gap {b['success'] - other['success']:+.0%}, CI {ci[0]:.1%}-{ci[1]:.1%})",
                "verdict": verdict(
                    "a-b >= 0.30", {"a": b["success"], "b": other["success"]}, ci=ci
                ),
            }
    g = gate.get(ab.label) if ab else None
    if g and g["contact"]["decisions"] and g["free"]["decisions"]:
        # the force expert's weight (index 2) averaged over ALL token groups' means
        def wf(bucket):
            return float(np.mean([g[bucket][t][2] for t in TOKEN_GROUPS]))

        wc, wfr = wf("contact"), wf("free")
        filled["m3.gate_tilts_to_force"] = {
            "measured": f"B force-expert gate weight: contact {wc:.4f} vs free {wfr:.4f} "
            f"(force token: {g['contact']['force'][2]:.4f} vs {g['free']['force'][2]:.4f}; "
            f"{g['contact']['decisions']} / {g['free']['decisions']} decisions)",
            "verdict": verdict("a-b >= 0", {"a": wc, "b": wfr}),
        }
    return filled


def evaluate(ctx) -> None:
    P = ctx.params
    episodes, eseed, video = int(P["episodes"]), int(P["eval_seed"]), bool(P["video"])
    prompt_mode = str(P.get("prompt", "machine"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.set_status(message="loading the checkpoint and rebuilding the policies")
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    pairs = _eval_arms(ck)
    arms = [a for a, _ in pairs]
    policies = {a.label: _policy_from(pl, device) for a, pl in pairs}
    gates = {a.label: _GateAcc() for a in arms if a.force_in == "forcevla2"}
    scenario = "claims_suite" if "arms" in ck else "single"
    offset = eseed * episodes
    env = Env(task=arms[0].task, render=True, gantry=GantryParams(offwidth=VW, offheight=VH))

    jobs: list[_Job] = []
    for a in arms:
        for s in eval_seeds(a.task, episodes, offset=offset):  # shared seeds per task
            jobs.append(_Job(
                arm=a, group=a.label, seed=s, policy=policies[a.label], env=env, device=device,
                gate=gates.get(a.label), prompt_mode=prompt_mode,
                card=[f"arm {a.label} on {a.task}",
                      f"{a.force_in} + {a.head} head, prompt {'on' if a.force_prompt else 'off'}"],
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
            step=done_n,
            total_steps=len(jobs),
            message=f"{scenario}: {job.group}, episode {done_n}/{len(jobs)}",
        )

    rendered = [j for j in jobs if j.render]
    plain = [j for j in jobs if not j.render]

    def frames():
        nonlocal stopped
        for i, job in enumerate(rendered):
            if not ctx.should_continue():
                stopped = True
                return
            run_one(job)
            card = [f"nanoForceVLA2 ({scenario})",
                    f"episode {i + 1} of {len(rendered)} on video", *job.card,
                    f"seed {job.seed}"]  # fmt: skip
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
                rows.clear()
                for lbl in gates:
                    gates[lbl] = _GateAcc()
                for job in jobs:
                    job.gate = gates.get(job.group)
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

    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    summaries = {a.label: _group_summary(a, by_group[a.label]) for a in arms
                 if by_group.get(a.label)}  # fmt: skip
    gate = {lbl: g.summary() for lbl, g in gates.items() if g.summary() is not None}
    k = sum(r["success"] for r in rows)
    result: dict = {
        "scenario": scenario, "episodes": episodes, "n_episodes": len(jobs), "fps": FPS,
        "video": video_name, "prompt": prompt_mode,
        "arms": summaries,
        "outcomes": {
            "success": round(k / max(1, len(rows)), 4),
            "counts": dict(Counter(r["outcome"] for r in rows)),
            "wilson95": wilson95(k, len(rows)),
            "per_task": {
                lbl: {"n": s["n"], "success": s["success"], "wilson95": s["wilson95"],
                      "aborts": s["outcomes"].get("force_abort", 0),
                      "peak_force_mean": s["peak_force"]["mean"],
                      "force_rmse_contact": s["force_rmse_contact"]}
                for lbl, s in summaries.items()
            },
        },
        "peak_force": {"mean": _r3(np.mean([r["peak_force"] for r in rows])),
                       "max": _r3(np.max([r["peak_force"] for r in rows]))},
    }  # fmt: skip
    if gate:
        result["gate"] = gate
    result["claims"] = _claim_rows(_fill_claims(arms, summaries, gate))
    result["episodes_traces"] = [
        {k2: v for k2, v in {**r, **r["trace"]}.items() if k2 != "trace"}
        for r in rows
        if "trace" in r
    ][:MAX_TRACES]
    atomic_write_json(ctx.dir / "result.json", result)
    arm_txt = ", ".join(
        f"{lbl} {s['success']:.0%} (RMSE {s['force_rmse_contact']} N)"
        for lbl, s in summaries.items()
    )
    verdicts = "; ".join(f"{r['id'].split('.')[-1]}: {r['verdict']}" for r in result["claims"]
                         if r["verdict"] != NOT_MEASURED)  # fmt: skip
    ctx.done(f"{scenario}: {arm_txt} over {len(rows)} episodes"
             + (f" - {verdicts}" if verdicts else ""))  # fmt: skip


# ---------------------------------------------------------------------------------- estimate
def estimate(params: dict) -> float:
    cal = registry.calib("nano-force-vla2")
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    load = cal.get("load_s_per_demo", FALLBACK_LOAD_S_PER_DEMO)
    arms = 1 if params["preset"] == "single" else len(SUITE)
    t = arms * params["steps"] / sps + params["demos"] * load
    if params["preset"] == "single":
        points = max(1, params["steps"] // EVAL_EVERY)
        t += points * CB_SEEDS * cal.get("ep_s", FALLBACK_EP_S)
    return t + cal.get("overhead_s", OVERHEAD_S)


SPEC = TaskSpec(
    id="nano-force-vla2",
    order=18,
    title="ForceVLA2: force as an output",
    subtitle="Force prompt + subtask machine, a cross-attention encoder with a force bypass, "
             "a dense 3-expert MoE and a hybrid head that commands a contact force - on the "
             "T2 wipe against lesson 17's model and the position-head row the paper never ran.",
    params=[
        Param("preset", "Preset", "choice", "single", choices=list(PRESETS),
              help="single = one model from the switches below; claims_suite = three arms on "
                   "t2 in one job (A fvlmoe+position, B forcevla2+hybrid+prompt, C "
                   "forcevla2+position+prompt), one checkpoint"),
        Param("force_in", "Force route (single preset)", "choice", "forcevla2",
              choices=FORCE_IN_CHOICES,
              help="forcevla2 = cross-attention + bypass + 3-expert MoE (nfv.model.forcevla2); "
                   "fvlmoe / none = lesson 17's routes for comparison"),
        Param("head", "Action head (single preset)", "choice", "hybrid", choices=HEAD_CHOICES,
              help="hybrid = [dp ; gripper ; f_target 6 ; progress s] (11-D, the controller "
                   "selects force axes from f_target); position = dp + gripper only"),
        Param("force_prompt", "Force prompt (single preset)", "bool", True,
              help="append the running subtask's word to the instruction; the subtask "
                   "machine advances on predicted s >= 0.8, re-arms below 0.3"),
        Param("expert_inputs", "Expert inputs (single, forcevla2)", "choice", "all",
              choices=EXPERT_INPUT_CHOICES,
              help="the paper's expert ablation: all | vision_only (state+force tokens zeroed) "
                   "| none (only the two cross-attended state tokens); the gate sees all"),
        Param("task", "Task (single preset)", "choice", SUITE_TASK, choices=list(TASK_NAMES),
              help="the claims suite always runs t2"),
        Param("steps", "Training steps per arm", "int", 12000, min=50, max=60000, step=50),
        Param("lr", "Learning rate", "float", 3e-4, min=1e-5, max=3e-3,
              help="AdamW peak LR (linear warmup, cosine to 0 - nfv.train.fit)"),
        Param("batch", "Batch size", "int", 64, min=8, max=128, step=8),
        Param("demos", "Demos (first n episodes)", "int", 200, min=8, max=400, step=4,
              help="subset of the canonical set by episode order"),
        Param("seed", "Seed", "int", 0, min=0, max=9999,
              help="init + batch sampling + flow noise; suite arms use seed + 101*i"),
    ],  # fmt: skip
    eval_params=[
        Param("episodes", "Episodes per arm", "int", 100, min=2, max=200, step=1,
              help="held-out seeds per arm, SHARED across arms; the claims want 100"),
        Param("eval_seed", "Eval seed", "int", 0, min=0, max=99,
              help="shifts the window inside the task's eval-seed block"),
        Param("video", "Render a rollout video", "bool", True),
        Param("prompt", "Force-prompt driver", "choice", "machine", choices=list(PROMPT_MODES),
              help="machine = the subtask machine on the model's own predicted s (position-head "
                   "arms follow the env heuristic); heuristic = every prompt arm follows the "
                   "env's state heuristic; frozen = the first subtask's word for the whole "
                   "episode (exercise 3)"),
    ],  # fmt: skip
    charts=[
        {"id": "loss", "title": "Flow-matching loss",
         "series": [{"key": "loss", "label": "single"}]
         + [{"key": f"{a.key}_loss", "label": a.label} for a in SUITE],
         "log_y": True},
        {"id": "loss_parts", "title": "Hybrid head: f_target and progress losses",
         "series": [{"key": "loss_f_target", "label": "f_target (single)"},
                    {"key": "loss_s", "label": "s (single)"},
                    {"key": "b_loss_f_target", "label": "f_target (B)"},
                    {"key": "b_loss_s", "label": "s (B)"}],
         "log_y": True},
        {"id": "gate", "title": "Force-expert gate weight on the force token (training batch)",
         "series": [{"key": "gate_force_force", "label": "single"},
                    {"key": "b_gate_force_force", "label": "B"},
                    {"key": "c_gate_force_force", "label": "C"}],
         "log_y": False},
        {"id": "eval", "title": "Closed-loop success during training (single preset, "
                                "20 held-out seeds)",
         "series": [{"key": "eval_success", "label": "success"}], "log_y": False},
        {"id": "speed", "title": "Training speed (steps/s)",
         "series": [{"key": "steps_per_s", "label": "steps/s"}], "log_y": False},
    ],  # fmt: skip
    viewer="fvrollout",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="nano-force-vla2.html",
)  # fmt: skip
registry.register(SPEC)
