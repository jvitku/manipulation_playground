"""The claims framework (plan §6): quantitative claims as data, verdicts as a pure function.

Every quantitative claim a course module makes — the papers' ablation rows scaled to mini and
the course's own design targets — is a :class:`Claim`. An evaluation fills the claims it can
measure and writes one row per claim into ``result.claims``; the ``claims`` viewer renders the
board. A claim's ``expect`` is a comparison string over its arm names:

    "a >= 0.95"           one arm against a margin
    "a - b >= 0.15"       a gap between two arms (whitespace optional)
    "a <= 0.2"            an upper bound
    "0.3 <= a <= 0.7"     a window, both bounds inclusive (the M1 "T1 must sit mid-range" gate)

:func:`verdict` evaluates it to ``"CONFIRMED"`` / ``"NOT CONFIRMED"``, or ``"INCONCLUSIVE"``
when a supplied confidence interval of the measured left-hand side straddles the margin (the
comparison holds at one end of the interval and fails at the other). For the window form the
CI logic is interval-against-interval: a CI entirely inside the window is CONFIRMED, one
disjoint from it is NOT CONFIRMED, any overlap across a bound is INCONCLUSIVE. A scenario that
cannot measure a claim marks its row :data:`NOT_MEASURED` instead of calling :func:`verdict`.

``M0_CLAIMS`` are the testbed's own three claims (module ``force-sim``, lesson 15),
``M1_CLAIMS`` the nanoVLA module's (``nano-vla``, lesson 16) and ``M2_CLAIMS`` the
nanoForceVLA module's (``nano-force-vla``, lesson 17: the ForceVLA Table-3 mini, the twin and
FACT force-to-noise checks and the plan's M2 gate); later modules append theirs in the same
registry style (M3 lesson 18, M4 lesson 19, M5 lesson 20 ``lora-vla``: LoRA on SmolVLA).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

VERDICTS = ("CONFIRMED", "NOT CONFIRMED", "INCONCLUSIVE")
NOT_MEASURED = "NOT MEASURED"  # row verdict for a claim the scenario did not measure

# "a >= 0.95" | "a - b >= 0.15" | "a <= 0.2"  (arm names, one optional difference, a margin)
_EXPECT = re.compile(
    r"^\s*(?P<a>[A-Za-z_]\w*)\s*(?:-\s*(?P<b>[A-Za-z_]\w*)\s*)?(?P<op>>=|<=)\s*"
    r"(?P<margin>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
# "0.3 <= a <= 0.7"  (one arm inside a closed window — e.g. the M1 T1-mid-range gate)
_EXPECT_RANGE = re.compile(
    rf"^\s*(?P<lo>{_NUM})\s*<=\s*(?P<a>[A-Za-z_]\w*)\s*<=\s*(?P<hi>{_NUM})\s*$"
)


@dataclass(frozen=True)
class Claim:
    """One quantitative claim: what is compared, against what margin, and on whose word."""

    id: str
    statement: str
    module: str  # lesson id the claim belongs to (e.g. "force-sim")
    task: str  # "t0".."t3" or "all"
    arms: dict = field(default_factory=dict)  # arm name -> config/scenario description
    metric: str = "success"
    expect: str = "a >= 0.95"  # comparison over the arm names (see module docstring)
    source: str = ""  # where the expectation comes from (paper row, plan section)
    url: str = ""  # link for the claims board ("" = no link)


def _parse(expect: str) -> tuple[str, str | None, str, float]:
    m = _EXPECT.match(expect)
    if not m:
        raise ValueError(f"malformed expect {expect!r} (want 'a >= 0.95' or 'a-b >= 0.15')")
    return m["a"], m["b"], m["op"], float(m["margin"])


def _parse_range(expect: str) -> tuple[str, float, float] | None:
    """(arm, lo, hi) when ``expect`` is the window form "lo <= a <= hi", else None."""
    m = _EXPECT_RANGE.match(expect)
    return (m["a"], float(m["lo"]), float(m["hi"])) if m else None


def measured_value(expect: str, values: dict) -> float:
    """The left-hand side of ``expect`` under ``values`` (arm name -> measured metric)."""
    rng = _parse_range(expect)
    if rng:
        return float(values[rng[0]])
    a, b, _, _ = _parse(expect)
    return float(values[a]) - float(values[b]) if b else float(values[a])


def verdict(expect: str, values: dict, ci: tuple[float, float] | None = None) -> str:
    """CONFIRMED / NOT CONFIRMED from the point value; INCONCLUSIVE when ``ci`` straddles.

    ``values`` maps arm names to measured metrics; ``ci`` is an optional (lo, hi) confidence
    interval of the WHOLE left-hand side (for "a-b" claims that is the interval of the
    difference, the caller's job). The verdict flips to INCONCLUSIVE only when the comparison
    disagrees between the two interval ends — a CI lying entirely on one side stays decisive.
    For the window form "lo <= a <= hi": the CI entirely inside the window is CONFIRMED,
    disjoint from it NOT CONFIRMED, overlapping a bound INCONCLUSIVE.
    """
    rng = _parse_range(expect)
    if rng:
        _, lo, hi = rng
        if ci is not None:
            cl, ch = float(ci[0]), float(ci[1])
            if lo <= cl and ch <= hi:
                return "CONFIRMED"
            if ch < lo or cl > hi:  # the whole interval misses the window
                return "NOT CONFIRMED"
            return "INCONCLUSIVE"
        return "CONFIRMED" if lo <= measured_value(expect, values) <= hi else "NOT CONFIRMED"
    _, _, op, margin = _parse(expect)
    ok = (lambda v: v >= margin) if op == ">=" else (lambda v: v <= margin)
    if ci is not None:
        lo, hi = float(ci[0]), float(ci[1])
        if ok(lo) != ok(hi):
            return "INCONCLUSIVE"
        return "CONFIRMED" if ok(lo) else "NOT CONFIRMED"
    return "CONFIRMED" if ok(measured_value(expect, values)) else "NOT CONFIRMED"


# ---- the testbed's claims (module "force-sim", lesson 15) ----------------------------------

M0_CLAIMS: tuple[Claim, ...] = (
    Claim(
        id="m0.expert_solves_every_task",
        statement="The scripted expert solves every task (T0-T3) under 0.5 mm DART noise",
        module="force-sim",
        task="all",
        arms={"a": {"policy": "scripted expert", "noise_mm": 0.5, "scenario": "expert_rollouts"}},
        metric="per-task success rate (the minimum over the evaluated tasks is tested)",
        expect="a >= 0.95",
        source="course plan §8, M0 gate (docs/plans/force_vla_school_plan.md)",
    ),
    Claim(
        id="m0.hybrid_holds_5n",
        statement="The hybrid controller holds a commanded 5 N on a surface of unknown height",
        module="force-sim",
        task="t2",
        arms={
            "a": {
                "controller": "hybrid",
                "f_target_n": 5.0,
                "hidden_heights_mm": [-8.0, 0.0, 8.0],
                "scenario": "hybrid_hold_5N",
            }
        },
        metric="max over the three hidden heights of |mean F_n - 5| (N)",
        expect="a <= 1.0",
        source="course plan §8, M0 gate (docs/plans/force_vla_school_plan.md)",
    ),
    Claim(
        id="m0.position_cannot_hold_force",
        statement="A position-only controller cannot hold a commanded force on an unknown "
        "surface (it presses K_p times its height error, or misses the surface); the hybrid "
        "controller holds it on every surface",
        module="force-sim",
        task="t2",
        arms={
            "a": {"controller": "hybrid (the wipe expert)", "task": "t2"},
            "b": {
                "controller": "position-only, oracle press depth",
                "task": "t2",
                "scenario": "position_vs_hybrid_t2",
            },
        },
        metric="fraction of T2 episodes that HOLD the force band (inside the 1.5 N band for "
        ">= 80 % of contact time, T2's own tracking criterion; a force abort counts as not "
        "held), hybrid minus position. The position arm can only hold when the hidden height "
        "happens to land in its ~5 mm band window (band width / contact stiffness) of the "
        "16 mm range - measured 15/100 episodes even with the oracle press depth - so the "
        "gap, not the single arm, is the decisive quantity",
        expect="a-b >= 0.5",
        source="course plan §2 + the ForceVLA2 motivation, arXiv 2603.15169",
        url="https://arxiv.org/abs/2603.15169",
    ),
)


# ---- the nanoVLA module's claims (module "nano-vla", lesson 16) -----------------------------

M1_CLAIMS: tuple[Claim, ...] = (
    Claim(
        id="m1.nanovla_solves_visible",
        statement="nanoVLA (vision + words, no force input) solves the fully visible T0 "
        "reach-and-grasp — the architecture works where the camera sees everything",
        module="nano-vla",
        task="t0",
        arms={"a": {"arm": "A t0 default", "force_in": "none", "task": "t0"}},
        metric="closed-loop success on held-out T0 seeds",
        expect="a >= 0.95",
        source="course plan §8, M1 gate (docs/plans/force_vla_school_plan.md)",
    ),
    Claim(
        id="m1.hidden_threshold_blinds_vla",
        statement="T1's hidden click threshold leaves a vision-only policy mid-range: good "
        "enough to find and press the right button, blind to when the click comes. Above the "
        "window the task premise broke (vision solves T1) and T1 must be re-randomised",
        module="nano-vla",
        task="t1",
        arms={"a": {"arm": "B t1 default", "force_in": "none", "task": "t1"}},
        metric="closed-loop success on held-out T1 seeds (a two-sided window: too low = the "
        "model never learned the task, too high = the threshold is not hidden enough)",
        expect="0.30 <= a <= 0.70",
        source="course plan §8, M1 gate (docs/plans/force_vla_school_plan.md)",
    ),
    Claim(
        id="m1.language_is_load_bearing",
        statement="The two-button T1 needs the instruction: the same model trained and "
        "evaluated with the text blanked loses at least 20 points — the words, not the "
        "pixels, carry which button to press",
        module="nano-vla",
        task="t1",
        arms={
            "a": {"arm": "B t1 default", "force_in": "none", "task": "t1"},
            "b": {"arm": "C t1 no-language", "force_in": "none", "task": "t1",
                  "blank_text": True},
        },
        metric="T1 success, default instruction minus blanked instruction (arm C trains and "
        "rolls out on a constant information-free text)",
        expect="a-b >= 0.2",
        source="course plan §7, lesson-16 row ('language needed on two-button T1'); "
        "the v1 plan's module 1",
    ),
)


# ---- the nanoForceVLA module's claims (module "nano-force-vla", lesson 17) ------------------

M2_CLAIMS: tuple[Claim, ...] = (
    Claim(
        id="m2.force_in_beats_none",
        statement="A force input beats no force on T1: the FVLMoE route gains at least "
        "15 points over the force-blind baseline trained on the same demos — the headline "
        "of ForceVLA's Table 3, scaled to mini",
        module="nano-force-vla",
        task="t1",
        arms={
            "a": {"arm": "fvlmoe", "force_in": "fvlmoe", "task": "t1", "force_noise": "off"},
            "b": {"arm": "none", "force_in": "none", "task": "t1", "force_noise": "off"},
        },
        metric="T1 success on shared held-out seeds, fvlmoe minus none",
        expect="a-b >= 0.15",
        source="ForceVLA Table 3 (plug insertion: FVLMoE 80% vs no-force 45%), "
        "arXiv 2505.22159",
        url="https://arxiv.org/abs/2505.22159",
    ),
    Claim(
        id="m2.twin_gap",
        statement="The force input is load-bearing: the SAME fvlmoe checkpoint evaluated "
        "with its wrench zeroed (the force-blind twin) loses at least 20 points — the model "
        "uses force, it does not merely carry the parameters",
        module="nano-force-vla",
        task="t1",
        arms={
            "a": {"arm": "fvlmoe", "force_noise": "off"},
            "b": {"arm": "fvlmoe", "force_noise": "zero",
                  "note": "one checkpoint, no retrain — act-time corruption"},
        },
        metric="T1 success of one fvlmoe checkpoint, raw wrench minus zeroed wrench",
        expect="a-b >= 0.20",
        source="course plan §8, M2 gate (docs/plans/force_vla_school_plan.md); the twin "
        "protocol of the v1 plan",
    ),
    Claim(
        id="m2.noise_drop",
        statement="Replacing the wrench with Gaussian noise at act time drops T1 by at "
        "least 10 points — FACT's force-to-noise test: a model that truly reads force is "
        "hurt by a lying sensor, not just by a silent one",
        module="nano-force-vla",
        task="t1",
        arms={
            "a": {"arm": "fvlmoe", "force_noise": "off"},
            "b": {"arm": "fvlmoe", "force_noise": "gaussian",
                  "note": "N(0, 1 N) added per component at act time"},
        },
        metric="T1 success of one fvlmoe checkpoint, raw wrench minus noise-corrupted wrench",
        expect="a-b >= 0.10",
        source="FACT (force-to-noise dropped ForceVLA's key insertion 42.5% to 15%), "
        "arXiv 2608.01402",
        url="https://arxiv.org/abs/2608.01402",
    ),
    Claim(
        id="m2.moe_before_vlm_collapses",
        statement="The MoE placed BEFORE the VLM does not beat the plain concat-after "
        "baseline: ForceVLA's Table 3 measured that row at 0% — force must not enter the "
        "VLM pathway. Reported either way: a mini-scale surprise here is a finding",
        module="nano-force-vla",
        task="t1",
        arms={
            "a": {"arm": "prefix_moe", "force_in": "prefix_moe", "task": "t1"},
            "b": {"arm": "concat_after", "force_in": "concat_after", "task": "t1"},
        },
        metric="T1 success, prefix_moe minus concat_after (expected non-positive)",
        expect="a-b <= 0",
        source="ForceVLA Table 3 (MoE before the VLM: 0% vs concat-after 60%), "
        "arXiv 2505.22159",
        url="https://arxiv.org/abs/2505.22159",
    ),
    Claim(
        id="m2.fvlmoe_gate",
        statement="The full FVLMoE route reaches 80% on T1 — the plan's M2 gate, scaled "
        "from the paper's plug-insertion 80%. The bar is fixed: a red row here is the "
        "course's honest result, not a reason to move the bar",
        module="nano-force-vla",
        task="t1",
        arms={"a": {"arm": "fvlmoe", "force_in": "fvlmoe", "task": "t1",
                    "force_noise": "off"}},
        metric="T1 success on held-out seeds",
        expect="a >= 0.80",
        source="course plan §8, M2 gate (docs/plans/force_vla_school_plan.md); "
        "ForceVLA Table 3's 80%",
        url="https://arxiv.org/abs/2505.22159",
    ),
)


# ---- the nanoForceVLA2 module's claims (module "nano-force-vla2", lesson 18) -----------------

_FV2_URL = "https://arxiv.org/abs/2603.15169"

M3_CLAIMS: tuple[Claim, ...] = (
    Claim(
        id="m3.hybrid_tracks_force",
        statement="With a force OUTPUT, the policy holds a commanded force: full ForceVLA2 "
        "(forcevla2 + hybrid head + force prompt) solves the T2 wipe in at least 75% of "
        "episodes AND tracks the commanded force (3 N gently / 8 N firmly) with a contact "
        "RMSE under 1 N - the plan's M3 gate. Both halves must hold",
        module="nano-force-vla2",
        task="t2",
        arms={
            "a": {"arm": "B forcevla2+hybrid", "force_in": "forcevla2", "head": "hybrid",
                  "force_prompt": True, "task": "t2"},
            "b": {"arm": "B forcevla2+hybrid", "metric": "contact force RMSE (N)"},
        },
        metric="T2 success (a) on held-out seeds AND contact force RMSE |F_n - F_target| "
        "over contact steps (b <= 1.0 N)",
        expect="a >= 0.75",
        source="course plan §8, M3 gate (docs/plans/force_vla_school_plan.md); ForceVLA2 "
        "arXiv 2603.15169 (clean board 70%, force-controlled wiping)",
        url=_FV2_URL,
    ),
    Claim(
        id="m3.position_head_cannot",
        statement="Force INPUTS without a force OUTPUT cannot hold a commanded force: the "
        "same ForceVLA2 inputs (prompt + encoder + MoE) with a position-only head solve the "
        "T2 wipe in at most 20% of episodes - the row ForceVLA2 never ran",
        module="nano-force-vla2",
        task="t2",
        arms={"a": {"arm": "C forcevla2+position", "force_in": "forcevla2",
                    "head": "position", "force_prompt": True, "task": "t2"}},
        metric="T2 success on held-out seeds",
        expect="a <= 0.20",
        source="course plan §2/§6 (the position-head control row); the M0 finding that a "
        "position controller holds the band in only ~15% of hidden heights",
        url=_FV2_URL,
    ),
    Claim(
        id="m3.force_output_is_the_lever",
        statement="The force output is the lever: on identical inputs, the hybrid head beats "
        "the position head by at least 30 points on T2",
        module="nano-force-vla2",
        task="t2",
        arms={
            "a": {"arm": "B forcevla2+hybrid", "head": "hybrid"},
            "b": {"arm": "C forcevla2+position", "head": "position"},
        },
        metric="T2 success, B minus C, shared held-out seeds",
        expect="a-b >= 0.30",
        source="course plan §6 (isolating the force output); ForceVLA2 arXiv 2603.15169 "
        "does not run this ablation",
        url=_FV2_URL,
    ),
    Claim(
        id="m3.beats_forcevla",
        statement="ForceVLA2 beats ForceVLA on the commanded-force task: B gains at least 30 "
        "points over lesson 17's fvlmoe + position head trained on the same T2 demos",
        module="nano-force-vla2",
        task="t2",
        arms={
            "a": {"arm": "B forcevla2+hybrid", "force_in": "forcevla2"},
            "b": {"arm": "A fvlmoe+position", "force_in": "fvlmoe", "head": "position"},
        },
        metric="T2 success, B minus A, shared held-out seeds",
        expect="a-b >= 0.30",
        source="ForceVLA2 arXiv 2603.15169 (66% vs ForceVLA 35% average over 5 tasks)",
        url=_FV2_URL,
    ),
    Claim(
        id="m3.gate_tilts_to_force",
        statement="The dense gate tilts toward the force expert in contact: B's mean "
        "force-expert gate weight is higher on decisions taken in contact than in free space",
        module="nano-force-vla2",
        task="t2",
        arms={
            "a": {"arm": "B forcevla2+hybrid", "bucket": "contact (F_n > 0.5 N)"},
            "b": {"arm": "B forcevla2+hybrid", "bucket": "free space"},
        },
        metric="force-expert weight averaged over token groups, contact minus free "
        "(both numbers reported)",
        expect="a-b >= 0",
        source="ForceVLA2 arXiv 2603.15169 (cross-scale MoE: modality experts, token-wise "
        "gate); course plan §8 M3 'gate weights visibly tilting'",
        url=_FV2_URL,
    ),
)

# M4 (lesson 19, `force-rl`): runtime residual RL on the frozen nanoVLA. Bars fixed by the
# plan §8 M4 gate; the expected direction follows ForceRFT (a bounded residual on a frozen
# force-conditioned VLA, twin critics). Arms: a = the residual(s), b = the frozen base, all on
# the same shared held-out seeds.
_FRFT_URL = "https://arxiv.org/abs/2609.22840"

M4_CLAIMS: tuple[Claim, ...] = (
    Claim(
        id="m4.residual_improves_t3",
        statement="A bounded residual learned online on top of the frozen base raises T3 "
        "success by at least 10 points (median over the RL seeds)",
        module="force-rl",
        task="t3",
        arms={"a": {"arm": "base + residual (median seed)"}, "b": {"arm": "frozen base"}},
        metric="T3 success on shared held-out seeds, median-over-seeds residual minus base",
        expect="a-b >= 0.10",
        source="course plan §8 M4 gate (+10 pts over 5 seeds); ForceRFT arXiv 2609.22840",
        url=_FRFT_URL,
    ),
    Claim(
        id="m4.peak_force_down",
        statement="The residual makes contact gentler: mean episode peak |F| at least 20 % "
        "below the frozen base's",
        module="force-rl",
        task="t3",
        arms={"a": {"arm": "1 - peak(residual, median over seeds) / peak(base)"}},
        metric="relative reduction of the mean episode peak |F|",
        expect="a >= 0.20",
        source="course plan §8 M4 gate (peak force down >= 20 %); ForceRFT arXiv 2609.22840",
        url=_FRFT_URL,
    ),
    Claim(
        id="m4.residual_quiet_in_free_space",
        statement="The residual stays near zero in free space and acts at contact: its mean "
        "magnitude away from contact is at most a third of its magnitude in contact",
        module="force-rl",
        task="t3",
        arms={"a": {"arm": "residual", "bucket": "free (|F| <= 0.5 N) / contact"}},
        metric="mean |residual dp| free space / in contact (mm/step), mean over seeds",
        expect="a <= 0.3333",
        source="course plan §8 M4 gate ('residual ~ 0 in free space'); v1 Module 4",
        url=_FRFT_URL,
    ),
    Claim(
        id="m4.seed_spread_reported",
        statement="The result is reported per RL seed (at least the plan's 5), not as one "
        "mean curve: RL convergence is a seed lottery (lessons 11-13)",
        module="force-rl",
        task="t3",
        arms={"a": {"arm": "number of RL seeds evaluated"}},
        metric="RL seeds evaluated on the shared seeds (per-seed finals in the measured text)",
        expect="a >= 5",
        source="course plan §8 M4 (5 seeds, per-seed curves); lessons 11-13 seed spread",
        url=_FRFT_URL,
    ),
    # ---- the simulated deployment (interventions="expert"): measured only on such a run ----
    Claim(
        id="m4.corrections_unlock_learning",
        statement="With a simulated operator correcting the robot during training (corrections "
        "supervise the actor, critics on autonomous transitions only), the residual raises T3 "
        "success by at least 10 points over the frozen base, evaluated fully autonomously",
        module="force-rl",
        task="t3",
        arms={
            "a": {"arm": "base + residual trained with interventions=expert (median seed)"},
            "b": {"arm": "frozen base"},
        },
        metric="T3 success on the same shared held-out seeds (no operator at eval), "
        "median-over-seeds residual minus base",
        expect="a-b >= 0.10",
        source="ForceRFT arXiv 2609.22840 (human corrections supervise the residual actor); "
        "HIL-SERL arXiv 2410.21845",
        url=_FRFT_URL,
    ),
    Claim(
        id="m4.autonomy_rises",
        statement="The operator is needed less as training goes on: the intervention rate in "
        "the last 20 % of training is at most half of the rate in the first 20 %",
        module="force-rl",
        task="t3",
        arms={"a": {"arm": "operator steps / env steps in the last / first 20 % of training"}},
        metric="ratio of step-weighted intervention rates (last 20 % / first 20 %)",
        expect="a <= 0.5",
        source="HIL-SERL arXiv 2410.21845 (intervention rate falls as the policy improves); "
        "ForceRFT arXiv 2609.22840",
        url="https://arxiv.org/abs/2410.21845",
    ),
)

# M5 (lesson 20, `lora-vla`): LoRA fine-tuning of the 450 M SmolVLA on the course demos. Bars
# fixed before the runs (outputs/lora_vla/PREREG.txt names the task they are judged on: T3, or
# T0 if every arm is <= 5 % on T3). Arms: A = head only (new-embodiment projections), B = head
# + LoRA r=16 on the VLM and action-expert attention, C = lerobot's expert-only full fine-tune,
# D = every parameter trainable (memory probe only), all at the same batch.
_LORA_URL = "https://arxiv.org/abs/2106.09685"
_SMOLVLA_URL = "https://arxiv.org/abs/2506.01844"

M5_CLAIMS: tuple[Claim, ...] = (
    Claim(
        id="m5.lora_param_fraction",
        statement="LoRA adapts the 450 M-parameter VLA while training at most 2 % of its "
        "parameters (adapters + the new-embodiment projections)",
        module="lora-vla",
        task="all",
        arms={"a": {"arm": "B lora: trainable / total parameters"}},
        metric="trainable parameter fraction of arm B",
        expect="a <= 0.02",
        source="LoRA arXiv 2106.09685 (GPT-3 175B: 0.01 % of the parameters); SmolVLA "
        "arXiv 2506.01844 (450 M, expert-only fine-tuning)",
        url=_LORA_URL,
    ),
    Claim(
        id="m5.lora_matches_expert_ft",
        statement="LoRA on the attention layers reaches the success of lerobot's standard "
        "fine-tune (whole action expert, ~22x the trainable parameters) within 10 points",
        module="lora-vla",
        task="all",
        arms={"a": {"arm": "B lora"}, "b": {"arm": "C expert_full"}},
        metric="closed-loop success on the same 100 held-out seeds, B minus C",
        expect="a-b >= -0.10",
        source="LoRA arXiv 2106.09685 Table 2-4 (LoRA on par with or better than full "
        "fine-tuning); course plan, lesson 20",
        url=_LORA_URL,
    ),
    Claim(
        id="m5.lora_beats_head_only",
        statement="Adapting the network's internals matters: LoRA beats training only the "
        "new-embodiment projections by at least 10 points",
        module="lora-vla",
        task="all",
        arms={"a": {"arm": "B lora"}, "b": {"arm": "A head (linear-probe baseline)"}},
        metric="closed-loop success on the same 100 held-out seeds, B minus A",
        expect="a-b >= 0.10",
        source="LoRA arXiv 2106.09685 (adapter / prefix baselines); intrinsic dimension arXiv "
        "2012.13255 (the update needs a subspace, not just new heads)",
        url=_LORA_URL,
    ),
    Claim(
        id="m5.lora_memory",
        statement="LoRA fine-tuning needs at most 60 % of the GPU memory of the expert-only "
        "full fine-tune at the same batch",
        module="lora-vla",
        task="all",
        arms={"a": {"arm": "peak allocated VRAM, B / C"}},
        metric="torch.cuda.max_memory_allocated during training, B / C, same batch",
        expect="a <= 0.6",
        source="LoRA arXiv 2106.09685 §4.2 (GPT-3: 1.2 TB -> 350 GB training VRAM, ~2/3 saved)",
        url=_LORA_URL,
    ),
    Claim(
        id="m5.full_ft_does_not_fit",
        statement="Training every parameter of the 450 M VLA does not fit a 10 GB consumer GPU "
        "at the batch the other arms use (CUDA OOM, or a peak above 9.5 GB)",
        module="lora-vla",
        task="all",
        arms={"a": {"arm": "D full: peak allocated GB (an OOM counts as 10 GB)"}},
        metric="peak allocated VRAM of 3 full-fine-tune steps (fp32 master weights + AdamW), "
        "same batch",
        expect="a >= 9.5",
        source="SmolVLA arXiv 2506.01844 (designed to train on consumer GPUs only with the VLM "
        "frozen); LoRA arXiv 2106.09685 §4.2",
        url=_SMOLVLA_URL,
    ),
    Claim(
        id="m5.update_is_low_rank",
        statement="What full fine-tuning changes is approximately low-rank: in the expert's "
        "attention layers the top-16 singular values of ΔW = W_after - W_before carry at least "
        "half of its energy",
        module="lora-vla",
        task="all",
        arms={"a": {"arm": "C expert_full: sum of the top-16 s_i^2 / sum of all s_i^2"}},
        metric="share of ||ΔW||_F^2 in the top-16 singular values, mean over the measured layers",
        expect="a >= 0.5",
        source="intrinsic dimension arXiv 2012.13255; LoRA arXiv 2106.09685 §7 (ΔW has a very "
        "low intrinsic rank)",
        url="https://arxiv.org/abs/2012.13255",
    ),
    Claim(
        id="m5.merge_is_exact",
        statement="Merging the adapters into the weights (W + (alpha/r) B A) changes nothing: "
        "the merged network computes the adapter network's output, with zero added latency",
        module="lora-vla",
        task="all",
        arms={"a": {"arm": "B lora: max |v_adapter - v_merged| on a real batch, fp32"}},
        metric="max absolute difference of the predicted flow velocity, fp32, TF32 off",
        expect="a <= 1e-4",
        source="LoRA arXiv 2106.09685 §4.1 (no additional inference latency)",
        url=_LORA_URL,
    ),
)
