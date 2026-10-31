"""The claims framework (plan §6): quantitative claims as data, verdicts as a pure function.

Every quantitative claim a course module makes — the papers' ablation rows scaled to mini and
the course's own design targets — is a :class:`Claim`. An evaluation fills the claims it can
measure and writes one row per claim into ``result.claims``; the ``claims`` viewer renders the
board. A claim's ``expect`` is a comparison string over its arm names:

    "a >= 0.95"       one arm against a margin
    "a - b >= 0.15"   a gap between two arms (whitespace optional)
    "a <= 0.2"        an upper bound

:func:`verdict` evaluates it to ``"CONFIRMED"`` / ``"NOT CONFIRMED"``, or ``"INCONCLUSIVE"``
when a supplied confidence interval of the measured left-hand side straddles the margin (the
comparison holds at one end of the interval and fails at the other). A scenario that cannot
measure a claim marks its row :data:`NOT_MEASURED` instead of calling :func:`verdict`.

``M0_CLAIMS`` are the testbed's own three claims (module ``force-sim``, lesson 15); later
modules append theirs (fv.table3.*, the twin gap, ...) in the same registry style.
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


def measured_value(expect: str, values: dict) -> float:
    """The left-hand side of ``expect`` under ``values`` (arm name -> measured metric)."""
    a, b, _, _ = _parse(expect)
    return float(values[a]) - float(values[b]) if b else float(values[a])


def verdict(expect: str, values: dict, ci: tuple[float, float] | None = None) -> str:
    """CONFIRMED / NOT CONFIRMED from the point value; INCONCLUSIVE when ``ci`` straddles.

    ``values`` maps arm names to measured metrics; ``ci`` is an optional (lo, hi) confidence
    interval of the WHOLE left-hand side (for "a-b" claims that is the interval of the
    difference, the caller's job). The verdict flips to INCONCLUSIVE only when the comparison
    disagrees between the two interval ends — a CI lying entirely on one side stays decisive.
    """
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
