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
registry style.
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
