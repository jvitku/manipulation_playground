"""Lesson 17 - "ForceVLA: force as an input" (id `nano-force-vla`): the M2 model in the lab.

TRAIN builds the nano policy with a ``force_in`` route (nfv.model.force_in: the five ForceVLA
Table-3 rows + ForceBench's prompt_bins) on lesson 15's canonical demos and trains it with
`nfv.train.fit` on the real RunContext. ``preset`` picks the shape of the run:

  single        ONE model: ``force_in``/``moe``/``n_experts``/``balance_loss`` from the form,
                on ``task``'s demos, with the periodic closed-loop evaluation (CB_SEEDS
                held-out seeds every EVAL_EVERY steps and at the end, logged as
                ``eval_success``) - exactly lesson 16's single preset plus the force knobs.
  claims_suite  the Table-3 mini: FIVE arms sequentially in one job, all on T1 (fresh seeds
                per arm, seed + 101*i), in the paper's row order:
                  a none / b prefix_linear / c prefix_moe / d concat_after / e fvlmoe
                Suite metrics are prefixed per arm (a_loss .. e_loss); ONE checkpoint payload
                holds all arms (the lesson-16 pattern): {"kind": "nfv-suite", "arms":
                [labels...], <label>: {fit payload + "task" + "force_in" (+ the last training
                batch's "train_fraction_dropped"/"train_expert_counts" for the LIMoE routes -
                the capacity cap binds at batch size, see below)}, "params"} - saved after
                every arm, label "<arm> trained" / "final".

The architecture knobs (chunk 16, n_exec 8, 10 Euler steps, 4+4 layers) are deliberately NOT
on this form: the Table-3 suite compares WHERE force enters on one canonical architecture.
``ARCH_OVERRIDES`` is the test hook that shrinks it (tests/lab/test_task_nfv2.py). The ONE
training knob round 2 added is ``text_dropout`` (ModelConfig.text_dropout): per sample,
replace the instruction with the model's blank text (nfv.model.text.BLANK_INSTRUCTION) with
this probability - the conditioning dropout classifier-free guidance needs. 0.0 (the default)
trains exactly the round-1 model.

EVAL rolls checkpoint arms closed-loop on SHARED held-out seeds (`nfv.seeds.eval_seeds`,
shifted by ``eval_seed * episodes`` - every arm sees the same episodes, so gaps are paired).
``force_noise`` corrupts the wrench AT ACT TIME (nfv.model.force_in.apply_force_noise, wired
through ModelConfig.force_noise - the checkpoint is NOT retrained): off / zero (the
force-blind twin) / shuffle (a batch-of-1 no-op at act time - the lesson's deliberately-wrong
exercise) / gaussian (FACT's force-to-noise). It applies to every evaluated arm; while it is
"off", any fvlmoe arm is ADDITIONALLY evaluated at zero and gaussian (groups "fvlmoe [zero]"
/ "fvlmoe [gaussian]") - the twin/FACT claim rows reuse the one checkpoint, no retrain.
``ckpt_b`` (optional, kind "ckpt") rolls a second nano-force-vla checkpoint on the same seeds;
its arms appear as "B: <label>" groups (rendered, never in claims).

Two more pure act-time knobs (round 2's levers; both echoed in result.json and applied to
every evaluated group, the checkpoint weights untouched): ``cfg_w`` - classifier-free
guidance on the instruction, v = v_blank + w·(v_cond − v_blank) per Euler step (1.0 = the
plain conditioned model, the blank branch skipped) - and ``n_exec_override`` - 0 = the
checkpoint's trained n_exec, else execute 1..16 steps of each predicted chunk before
re-planning (clamped to the chunk length): the reactivity lever that decides how stale the
one wrench sample each decision reads can get.

Routing instrumentation (the plan's "which expert fires in contact vs free space"): for the
LIMoE routes (fvlmoe, prefix_moe) every act() decision reads the port's aux (expert counts,
fraction dropped, the force token's expert) and files it under "contact" when |F| at the
decision's observation exceeds CONTACT_N, else "free". NOTE the capacity cap is INERT at act
time: 85 tokens at batch 1 have no viable group-of-16 split, so the reference `_num_groups`
degenerates to groups of 1 where the capacity floor (4) never binds - fraction_dropped is 0
by construction at act time and real at training batch size (the train-time number is stored
in the suite payload per arm).

Claims (nfv.claims.M2_CLAIMS) are filled from whatever t1 groups the eval produced (never
from "B:" groups): fvlmoe@off + none@off -> m2.force_in_beats_none; fvlmoe@off + @zero ->
m2.twin_gap; + @gaussian -> m2.noise_drop; prefix_moe@off + concat_after@off ->
m2.moe_before_vlm_collapses; fvlmoe@off alone -> m2.fvlmoe_gate. Gap CIs are the conservative
interval from the two arms' Wilsons (the house idiom).

result.json (viewer `fvrollout`; every field optional there):
    {"scenario": "single"|"claims_suite", "episodes" (per group), "n_episodes", "fps",
     "video": "rollout.mp4"|null,
     "cfg_w": w, "n_exec_override": n,  # the act-time recipe the whole eval ran under
     "arms": {group: {"task", "force_in", "force_noise", "n", "success", "wilson95",
                      "outcomes": {o: n}, "peak_force": {"mean", "max"},
                      "per_instruction": {instr: {n, success}},   # t1 groups only
                      "pressed_counts": {"red": n, "blue": n},    # t1 groups only
                      "button_match": {"pressed", "matched", "rate"}},  # t1: of the episodes
                      # that pressed any button, how many pressed the instructed one
     "outcomes": {"success", "counts", "wilson95", "per_task": {group: row}},
     "peak_force": {"mean", "max"},
     "routing": {group: {"n_experts", "contact"/"free": {"decisions", "expert_frac": [E],
                 "force_token_expert_frac": [E], "fraction_dropped",
                 "force_token_kept_frac"}}},                      # LIMoE groups only
     "claims": [rows for all of M2_CLAIMS; unmeasured rows say so],
     "episodes_traces": [<= 10 of the lesson-16 trace schema + "force_noise"; "mode" all
                         "position" - the force INPUT commands no wrench, "f_target" omitted]}
rollout.mp4: up to 4 episodes spread round-robin over the groups (task 15's video idiom).
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
from lab.tasks.task_nfv1 import T1_INSTRUCTIONS, _ArmCtx, _DemoView
from nfv.claims import M2_CLAIMS, NOT_MEASURED, verdict
from nfv.data.dataset import DemoDataset
from nfv.model import ModelConfig, build
from nfv.seeds import TASK_NAMES, eval_seeds
from nfv.sim.env import Action, Env
from nfv.sim.gantry_xml import GantryParams
from nfv.train import fit

PRESETS = ("single", "claims_suite")
FORCE_IN_CHOICES = ["fvlmoe", "prefix_linear", "prefix_moe", "concat_after", "prompt_bins",
                    "none"]  # fmt: skip
MOE_CHOICES = ["top1_capacity", "top1", "soft"]
NOISE_CHOICES = ["off", "zero", "shuffle", "gaussian"]
# (metric prefix, route = arm label) - the Table-3 mini, in the paper's row order, all on T1
SUITE = (
    ("a", "none"),
    ("b", "prefix_linear"),
    ("c", "prefix_moe"),
    ("d", "concat_after"),
    ("e", "fvlmoe"),
)
SUITE_TASK = "t1"  # the claims are Table-3 rows; the task knob steers the single preset only
NOISE_ROWS = ("zero", "gaussian")  # extra act-time rows of any fvlmoe arm (form noise "off")
LIMOE_ROUTES = ("prefix_moe", "fvlmoe")  # the routes whose act() aux feeds result.routing
CONTACT_N = 0.5  # |F| (N) at the decision's observation above this = a contact decision
CB_SEEDS = 20  # held-out seeds of the single preset's periodic closed-loop evaluation
CB_SEED_OFFSET = 9_000  # top of the eval block - disjoint from the eval job's offsets
EVAL_EVERY = 5_000  # steps between evaluation points (fit also fires at the final step)
# Test hook: extra ModelConfig fields (tests shrink chunk/layers; the form exposes no
# architecture knobs on purpose - the suite compares routes on ONE canonical architecture).
ARCH_OVERRIDES: dict = {}
# calib.json fallbacks (steps_per_s measured 2026-11-01 on the RTX 3080, batch 64, fvlmoe)
FALLBACK_STEPS_PER_S, FALLBACK_LOAD_S_PER_DEMO = 20.0, 0.06
FALLBACK_EP_S, OVERHEAD_S = 2.5, 15.0


@dataclass(frozen=True)
class _Arm:
    """One trained model: its metric prefix, checkpoint label, force route and task."""

    key: str  # metric prefix ("a".."e"; "" = unprefixed, the single preset)
    label: str  # checkpoint payload / result.json label (= the route, for the suite)
    force_in: str
    task: str


def _train_arms(P: dict) -> list[_Arm]:
    if str(P["preset"]) == "single":
        return [_Arm("", "single", str(P["force_in"]), str(P["task"]))]
    return [_Arm(k, route, route, SUITE_TASK) for k, route in SUITE]


def _model_cfg(P: dict, force_in: str) -> ModelConfig:
    return ModelConfig(
        force_in=force_in,
        moe=str(P["moe"]),
        n_experts=int(P["n_experts"]),
        balance_loss=bool(P["balance_loss"]),
        text_dropout=float(P.get("text_dropout", 0.0)),
        **ARCH_OVERRIDES,
    )


def _make_cb(env: Env, arm: _Arm, device: str):
    """The single preset's evaluation callback: closed-loop success on held-out seeds."""

    def cb(policy, step: int) -> dict:
        seeds = eval_seeds(arm.task, CB_SEEDS, offset=CB_SEED_OFFSET)  # global read: patchable
        ok = 0
        for s in seeds:
            rec = _drain(_episode(_Job(arm=arm, noise="off", group=arm.label, seed=s,
                                       policy=policy, env=env, device=device)))  # fmt: skip
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
    cache: dict[str, DemoDataset] = {}  # the suite's five arms share the one t1 set
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
            actx = _ArmCtx(ctx, arm, i, len(arms), steps, capture=preset != "single")
            policy, hist = fit(
                _model_cfg(P, arm.force_in), view, actx,
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
            # The capacity cap binds at training batch size (groups of 16), not at act time
            # (batch 1 -> groups of 1, floor 4 never binds): keep the LAST training batch's
            # drop fraction + expert counts - "capacity-drop fraction logged" (plan §8, M2).
            moe_aux = policy.net.force.last_aux
            train_moe = None
            if moe_aux is not None:
                train_moe = {
                    "train_fraction_dropped": round(float(moe_aux["fraction_dropped"]), 4),
                    "train_expert_counts": [int(x) for x in moe_aux["expert_counts"]],
                }
            if preset != "single":
                captured[arm.label] = {
                    **(actx.captured or {}), "task": arm.task, "force_in": arm.force_in,
                    **(train_moe or {}),
                }  # fmt: skip
                payload = {
                    "kind": "nfv-suite", "arms": [a.label for a in arms[: i + 1]],
                    **captured, "params": dict(P),
                }  # fmt: skip
                label = "final" if i == len(arms) - 1 else f"{arm.label} trained"
                metrics = {f"{a.key}_loss": round(arm_losses[a.label], 4) for a in arms[: i + 1]}
                if train_moe is not None:
                    metrics[f"{arm.key}_drop"] = train_moe["train_fraction_dropped"]
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
            f"trained {arms[0].force_in} on {arms[0].task}: loss {last_loss:.4f}"
            + (f", closed-loop success {ev:.0%} on {CB_SEEDS} held-out seeds" if ev is not None
               else "")
        )  # fmt: skip
    else:
        parts = ", ".join(f"{a.label} loss {arm_losses[a.label]:.4f}" for a in arms)
        ctx.done(f"trained the 5 Table-3 arms in one checkpoint: {parts}")


# ---------------------------------------------------------------------------- eval episodes
class _RoutingAcc:
    """Per-group expert-usage accumulator: one act() decision = one sample, filed under
    contact/free by the wrench magnitude at the decision's observation (CONTACT_N)."""

    def __init__(self, n_experts: int):
        self.n_experts = n_experts
        self.b = {
            k: {"decisions": 0, "expert": [0] * n_experts, "force_expert": [0] * n_experts,
                "dropped_sum": 0.0, "force_kept": 0}
            for k in ("contact", "free")
        }  # fmt: skip

    def add(self, aux: dict, contact: bool) -> None:
        b = self.b["contact" if contact else "free"]
        b["decisions"] += 1
        for e, c in enumerate(aux["expert_counts"].tolist()):
            b["expert"][e] += int(c)
        fe = int(aux["expert_index"][0, -1])  # the force token is LAST ([E ; E_F] / prefix)
        b["force_expert"][fe] += 1
        b["force_kept"] += bool(aux["kept"][0, -1])
        b["dropped_sum"] += float(aux["fraction_dropped"])

    def summary(self) -> dict | None:
        if not any(b["decisions"] for b in self.b.values()):
            return None
        out: dict = {"n_experts": self.n_experts}
        for k, b in self.b.items():
            n, tot = b["decisions"], sum(b["expert"])
            out[k] = {
                "decisions": n,
                "expert_frac": [round(c / tot, 4) if tot else 0.0 for c in b["expert"]],
                "force_token_expert_frac": [
                    round(c / n, 4) if n else 0.0 for c in b["force_expert"]
                ],  # fmt: skip
                "fraction_dropped": round(b["dropped_sum"] / n, 4) if n else 0.0,
                "force_token_kept_frac": round(b["force_kept"] / n, 4) if n else 0.0,
            }
        return out


@dataclass
class _Job:
    """One closed-loop episode: which policy rolls where, under which wrench corruption."""

    arm: _Arm
    noise: str  # the act-time force_noise this row runs under
    group: str  # aggregation key ("fvlmoe", "fvlmoe [zero]", "B: single", ...)
    seed: int
    policy: object  # nfv.model.build.Policy (acts through its EMA; cfg.force_noise = noise)
    env: Env
    device: str
    routing: _RoutingAcc | None = None  # shared per group (LIMoE routes only)
    render: bool = False
    trace: bool = False
    card: list = field(default_factory=list)  # extra title-card lines


def _episode(job: _Job):
    """Roll the policy for one episode; yields video frames when ``job.render`` and returns
    the record via StopIteration - the house video pattern (task_nfv0._episode)."""
    import mujoco

    env, arm = job.env, job.arm
    obs = env.reset(job.seed, task=arm.task)
    instr = env.instruction
    gen = torch.Generator(device=job.device)
    gen.manual_seed(job.seed)  # flow (and force) noise per episode: seed-deterministic
    tmax = int(env.task.TIME_LIMIT_S * FPS)
    n_exec = max(1, int(job.policy.cfg.n_exec))
    tr: dict = {"t": [], "fz": [], "mode": [], "subtask": [], "events": []}
    fz_v: list[float] = []
    ft_v: list[float] = []  # all-zero: the force INPUT commands no wrench (strip command line)
    rend = None
    if job.render:
        rend = mujoco.Renderer(env.model, height=VH, width=VW)

    def frame(status=""):
        import cv2

        rend.update_scene(env.data, camera="static")
        img = np.vstack([rend.render().copy(), _strip(fz_v, ft_v, tmax)])
        txt = f"t {env.t:4.1f} s  Fz {float(env.wrench[2]):+6.2f} N  {job.group}"
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
            if job.routing is not None:
                aux = job.policy.ema.force.last_aux  # the LIMoE pass of THIS decision
                if aux is not None:
                    contact = float(np.linalg.norm(obs.wrench[:3])) > CONTACT_N
                    job.routing.add(aux, contact)
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
            "task": arm.task, "seed": job.seed, "group": job.group,
            "force_noise": job.noise, "outcome": info["outcome"] or "timeout",
            "peak_force": _r3(env.peak_force), "instruction": env.instruction,
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
def _policy_from(
    payload: dict, device: str, force_noise: str = "off", cfg_w: float = 1.0, n_exec: int = 0
):
    """Rebuild a Policy from one fit payload with the act-time knobs dialled in.

    ``force_noise``, ``cfg_w`` and ``n_exec`` (0 = keep the trained value) go into the
    rebuilt ModelConfig, so the corruption, the CFG combination and the re-plan cadence run
    inside act() exactly where they belong - the checkpoint itself is untouched.
    """
    arch = {**payload["arch"], "force_noise": force_noise, "cfg_w": float(cfg_w)}
    if n_exec:
        arch["n_exec"] = min(int(n_exec), int(arch["chunk"]))
    cfg = ModelConfig(**arch)
    policy = build(cfg)
    policy.net.load_state_dict(payload["model"])
    policy.ema.load_state_dict(payload["ema"])
    return policy.to(device).eval()


def _eval_arms(ck: dict) -> list[tuple[_Arm, dict]]:
    """(arm, fit payload) pairs of a checkpoint: every suite arm, or the one single model."""
    if "arms" in ck:  # the claims_suite payload
        out = []
        for label in ck["arms"]:
            pl = ck[label]
            fi = str(pl.get("force_in") or pl["arch"]["force_in"])
            out.append((_Arm("", str(label), fi, str(pl["task"])), pl))
        return out
    task = str(ck.get("params", {}).get("task", "t1"))
    fi = str(ck.get("arch", {}).get("force_in", "none"))
    return [(_Arm("", "single", fi, task), ck)]


@dataclass
class _Group:
    """One evaluated row: an arm under one act-time noise, with its policy + routing acc."""

    arm: _Arm
    noise: str
    label: str
    policy: object
    routing: _RoutingAcc | None
    claims: bool  # False for ckpt_b ("B:") groups - rendered, never measured


def _make_groups(
    ck: dict, ckb: dict | None, form_noise: str, device: str,
    cfg_w: float = 1.0, n_exec: int = 0,
) -> list[_Group]:  # fmt: skip
    def grp(arm: _Arm, pl: dict, noise: str, prefix: str = "", claims: bool = True) -> _Group:
        label = prefix + arm.label + (f" [{noise}]" if noise != "off" else "")
        policy = _policy_from(pl, device, noise, cfg_w, n_exec)
        routing = _RoutingAcc(int(policy.cfg.n_experts)) if arm.force_in in LIMOE_ROUTES else None
        return _Group(arm, noise, label, policy, routing, claims)

    groups = [grp(arm, pl, form_noise) for arm, pl in _eval_arms(ck)]
    if form_noise == "off":  # the twin/FACT rows: the SAME fvlmoe checkpoint, corrupted
        for arm, pl in _eval_arms(ck):
            if arm.force_in == "fvlmoe":
                groups += [grp(arm, pl, nz) for nz in NOISE_ROWS]
    if ckb is not None:
        groups += [grp(arm, pl, form_noise, prefix="B: ", claims=False)
                   for arm, pl in _eval_arms(ckb)]  # fmt: skip
    return groups


def _group_summary(g: _Group, rows: list[dict]) -> dict:
    k = sum(r["success"] for r in rows)
    out = {
        "task": g.arm.task, "force_in": g.arm.force_in, "force_noise": g.noise,
        "n": len(rows),
        "success": round(k / max(1, len(rows)), 4), "wilson95": wilson95(k, len(rows)),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "peak_force": {"mean": _r3(np.mean([r["peak_force"] for r in rows])),
                       "max": _r3(np.max([r["peak_force"] for r in rows]))},
    }  # fmt: skip
    if g.arm.task == "t1":
        per = {}
        for instr in T1_INSTRUCTIONS:
            sub = [r for r in rows if r.get("instruction") == instr]
            if sub:
                kk = sum(r["success"] for r in sub)
                per[instr] = {"n": len(sub), "success": round(kk / len(sub), 4)}
        out["per_instruction"] = per
        out["pressed_counts"] = dict(Counter(c for r in rows for c in r.get("pressed", ())))
        # Button match: of the episodes that pressed ANY button, how many pressed the
        # instructed one - the instruction-following rate, separated from press control
        # (round 2's recipe is chosen on the none arm's rate, never on the gated arm).
        pressed = [r for r in rows if r.get("pressed")]
        matched = sum(1 for r in pressed if r.get("target") in r["pressed"])
        out["button_match"] = {
            "pressed": len(pressed), "matched": matched,
            "rate": round(matched / len(pressed), 4) if pressed else None,
        }  # fmt: skip
    return out


def _claim_rows(filled: dict[str, dict]) -> list[dict]:
    rows = []
    for c in M2_CLAIMS:
        f = filled.get(c.id)
        rows.append({
            "id": c.id, "statement": c.statement, "metric": c.metric, "expected": c.expect,
            "measured": f["measured"] if f else "not measured in this run",
            "verdict": f["verdict"] if f else NOT_MEASURED,
            "source": c.source, "url": c.url,
        })  # fmt: skip
    return rows


def _fill_claims(groups: list[_Group], summaries: dict[str, dict]) -> dict:
    """Fill every M2 claim the evaluated t1 groups can measure (ckpt_b groups never count)."""

    def find(force_in: str, noise: str):
        for g in groups:
            s = summaries.get(g.label)
            if (g.claims and s and g.arm.task == "t1" and g.arm.force_in == force_in
                    and g.noise == noise and s["n"]):  # fmt: skip
                return s
        return None

    def kn(s):
        return f"{int(round(s['success'] * s['n']))}/{s['n']}"

    def gap_ci(a, b):  # conservative CI of a-b from the two arms' Wilsons (house idiom)
        return (a["wilson95"][0] - b["wilson95"][1], a["wilson95"][1] - b["wilson95"][0])

    filled: dict[str, dict] = {}
    fv = find("fvlmoe", "off")
    if fv:
        lo, hi = fv["wilson95"]
        filled["m2.fvlmoe_gate"] = {
            "measured": f"fvlmoe T1 {kn(fv)} ({fv['success']:.0%}, CI {lo:.1%}-{hi:.1%})",
            "verdict": verdict("a >= 0.80", {"a": fv["success"]}, ci=(lo, hi)),
        }
    none = find("none", "off")
    if fv and none:
        ci = gap_ci(fv, none)
        filled["m2.force_in_beats_none"] = {
            "measured": f"fvlmoe {kn(fv)} vs none {kn(none)} "
            f"(gap {fv['success'] - none['success']:+.0%}, CI {ci[0]:.1%}-{ci[1]:.1%})",
            "verdict": verdict("a-b >= 0.15", {"a": fv["success"], "b": none["success"]}, ci=ci),
        }
    for cid, noise, expect in (
        ("m2.twin_gap", "zero", "a-b >= 0.20"),
        ("m2.noise_drop", "gaussian", "a-b >= 0.10"),
    ):
        cor = find("fvlmoe", noise)
        if fv and cor:
            ci = gap_ci(fv, cor)
            filled[cid] = {
                "measured": f"fvlmoe off {kn(fv)} vs {noise} {kn(cor)} "
                f"(drop {fv['success'] - cor['success']:+.0%}, CI {ci[0]:.1%}-{ci[1]:.1%})",
                "verdict": verdict(expect, {"a": fv["success"], "b": cor["success"]}, ci=ci),
            }
    pm, ca = find("prefix_moe", "off"), find("concat_after", "off")
    if pm and ca:
        ci = gap_ci(pm, ca)
        filled["m2.moe_before_vlm_collapses"] = {
            "measured": f"prefix_moe {kn(pm)} vs concat_after {kn(ca)} "
            f"(gap {pm['success'] - ca['success']:+.0%}, CI {ci[0]:.1%}-{ci[1]:.1%})",
            "verdict": verdict("a-b <= 0", {"a": pm["success"], "b": ca["success"]}, ci=ci),
        }
    return filled


def evaluate(ctx) -> None:
    P = ctx.params
    episodes, eseed, video = int(P["episodes"]), int(P["eval_seed"]), bool(P["video"])
    form_noise = str(P["force_noise"])
    cfg_w = float(P.get("cfg_w", 1.0))
    n_exec_override = int(P.get("n_exec_override", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.set_status(message="loading the checkpoint(s) and rebuilding the policies")
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    ckb = None
    if str(P.get("ckpt_b") or ""):
        ckb = torch.load(P["ckpt_b"], map_location="cpu", weights_only=False)
    groups = _make_groups(ck, ckb, form_noise, device, cfg_w, n_exec_override)
    scenario = "claims_suite" if "arms" in ck else "single"
    offset = eseed * episodes
    env = Env(task=groups[0].arm.task, render=True,
              gantry=GantryParams(offwidth=VW, offheight=VH))  # fmt: skip

    jobs: list[_Job] = []
    for g in groups:
        for s in eval_seeds(g.arm.task, episodes, offset=offset):  # shared seeds per task
            jobs.append(_Job(
                arm=g.arm, noise=g.noise, group=g.label, seed=s, policy=g.policy, env=env,
                device=device, routing=g.routing,
                card=[f"arm {g.label} on {g.arm.task}", f"wrench at act time: {g.noise}"],
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
            message=f"{scenario}: {job.group}, episode {done_n}/{len(jobs)}",
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
            card = [f"nanoForceVLA ({scenario})",
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
                rows.clear()  # drop the partial rendered rows; same seeds re-run unrendered
                for g in groups:  # routing stats of the partial rows go too
                    if g.routing is not None:
                        g.routing = _RoutingAcc(g.routing.n_experts)
                for job in jobs:
                    job.routing = next((g.routing for g in groups if g.label == job.group), None)
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
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    summaries = {
        g.label: _group_summary(g, by_group[g.label]) for g in groups if by_group.get(g.label)
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
    routing = {
        g.label: g.routing.summary() for g in groups
        if g.routing is not None and g.routing.summary() is not None
    }  # fmt: skip
    if routing:
        result["routing"] = routing
    result["claims"] = _claim_rows(_fill_claims(groups, summaries))
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
    cal = registry.calib("nano-force-vla")
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    load = cal.get("load_s_per_demo", FALLBACK_LOAD_S_PER_DEMO)
    arms = 1 if params["preset"] == "single" else len(SUITE)
    t = arms * params["steps"] / sps + params["demos"] * load  # both presets load ONE set
    if params["preset"] == "single":  # periodic closed-loop evaluation points
        points = max(1, params["steps"] // EVAL_EVERY)
        t += points * CB_SEEDS * cal.get("ep_s", FALLBACK_EP_S)
    return t + cal.get("overhead_s", OVERHEAD_S)


SPEC = TaskSpec(
    id="nano-force-vla",
    order=17,
    title="ForceVLA: force as an input",
    subtitle="A wrench token through the ported FVLMoE into nanoVLA - the Table-3 mini "
             "(five entry routes on T1), the force-blind twin and FACT's force-to-noise as "
             "claims, and which expert fires in contact vs free space.",  # fmt: skip
    params=[
        Param("preset", "Preset", "choice", "single", choices=list(PRESETS),
              help="single = one model, one force route; claims_suite = the Table-3 mini "
                   "(none / prefix_linear / prefix_moe / concat_after / fvlmoe on t1) in one "
                   "job, one checkpoint"),
        Param("force_in", "Force route (single preset)", "choice", "fvlmoe",
              choices=FORCE_IN_CHOICES,
              help="where the wrench enters (nfv.model.force_in); the suite always trains "
                   "the five Table-3 routes"),
        Param("moe", "MoE router", "choice", "top1_capacity", choices=MOE_CHOICES,
              help="router of the LIMoE routes (prefix_moe / fvlmoe): the reference top-1 "
                   "with the group capacity cap, top-1 without the cap, or a dense soft mix"),
        Param("n_experts", "Experts", "choice", 4, choices=[2, 4, 8],
              help="experts in the LIMoE MoE (the reference uses 4)"),
        Param("balance_loss", "Add the balance loss", "bool", False,
              help="ADD the Switch-Transformer load-balancing loss (coef 0.01) to the "
                   "training loss; the reference computes it and never adds it"),
        Param("task", "Task (single preset)", "choice", "t1", choices=list(TASK_NAMES),
              help="which canonical demo set the single preset trains on; the claims suite "
                   "always runs t1"),
        Param("steps", "Training steps per arm", "int", 20000, min=50, max=60000, step=50),
        Param("lr", "Learning rate", "float", 3e-4, min=1e-5, max=3e-3,
              help="AdamW peak LR (linear warmup, cosine to 0 - nfv.train.fit)"),
        Param("batch", "Batch size", "int", 64, min=8, max=128, step=8),
        Param("demos", "Demos (first n episodes)", "int", 200, min=8, max=400, step=4,
              help="subset of the canonical set by episode order; clamped to what lesson 15 "
                   "recorded"),
        Param("text_dropout", "Instruction dropout p", "float", 0.0, min=0.0, max=0.5,
              help="train-time: per sample, replace the instruction with the blank text "
                   "with this probability - the conditioning dropout classifier-free "
                   "guidance needs (round 2; 0.0 = the round-1 model exactly)"),
        Param("seed", "Seed", "int", 0, min=0, max=9999,
              help="init + batch sampling + flow noise; suite arms use seed + 101*i"),
    ],  # fmt: skip
    eval_params=[
        Param("episodes", "Episodes per arm", "int", 100, min=2, max=200, step=1,
              help="held-out seeds per group, SHARED across groups; the claims want 100"),
        Param("eval_seed", "Eval seed", "int", 0, min=0, max=99,
              help="shifts the window inside the task's eval-seed block"),
        Param("video", "Render a rollout video", "bool", True),
        Param("force_noise", "Wrench at act time", "choice", "off", choices=NOISE_CHOICES,
              help="corrupt the wrench the policy reads (the checkpoint is not retrained); "
                   "while 'off', an fvlmoe arm is additionally evaluated at zero and "
                   "gaussian - the twin / FACT claim rows. shuffle is a batch-of-1 no-op at "
                   "act time (the lesson's deliberately-wrong exercise)"),
        Param("ckpt_b", "Compare checkpoint (optional)", "ckpt", None,
              ckpt_task="nano-force-vla",
              help="a second nano-force-vla checkpoint rolled on the same seeds; its arms "
                   "appear as 'B: ...' rows (rendered, never in claims)"),
        Param("cfg_w", "CFG weight on the instruction", "float", 1.0, min=0.5, max=5.0,
              help="act-time classifier-free guidance: each Euler step combines "
                   "v_blank + w*(v_cond - v_blank); 1.0 = the plain conditioned model (the "
                   "blank branch is skipped). Applies to every evaluated group; no retrain"),
        Param("n_exec_override", "Executed steps override", "int", 0, min=0, max=16,
              help="act-time only: 0 = the checkpoint's trained n_exec; 1..16 = execute this "
                   "many steps of each chunk before re-planning (clamped to the chunk "
                   "length) - how stale the one wrench sample per decision can get"),
    ],  # fmt: skip
    charts=[
        {"id": "loss", "title": "Flow-matching loss",
         "series": [{"key": "loss", "label": "single"}]
         + [{"key": f"{k}_loss", "label": route} for k, route in SUITE],
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
    lesson="nano-force-vla.html",
)  # fmt: skip
registry.register(SPEC)
