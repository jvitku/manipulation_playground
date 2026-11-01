"""Task "nano-force-vla2" (ladder 18): the 3-arm T2 suite, the hybrid executor, M3 claims."""

import json

import pytest
import torch

from lab.core.jobs import read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task_nfv3 as tn
from nfv.claims import M3_CLAIMS, NOT_MEASURED, VERDICTS, _parse
from nfv.data.record import record_demos

ALLOWED = {*VERDICTS, NOT_MEASURED}
# tiny CPU budgets: shrunken architecture (ARCH_OVERRIDES hook), 8 t2 demos, 100 steps/arm
TINY_ARCH = {"chunk": 8, "n_exec": 4, "euler_steps": 3, "trunk_layers": 1, "expert_layers": 1}
TINY_COMMON = {"force_in": "forcevla2", "head": "hybrid", "force_prompt": True,
               "expert_inputs": "all", "task": "t2", "lr": 1e-3, "batch": 8, "demos": 8,
               "seed": 0}  # fmt: skip
TINY_SUITE = {"preset": "claims_suite", "steps": 100, **TINY_COMMON}
SUITE_LABELS = [a.label for a in tn.SUITE]


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "nano-force-vla2", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def lab_data(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("nfv3_data")
    mp.setenv("LAB_DATA", str(root))
    mp.setattr(tn, "ARCH_OVERRIDES", dict(TINY_ARCH))
    mp.setattr(tn, "CB_SEEDS", 1)
    record_demos("t2", 8, root / "nfv" / "demos", noise_mm=0.5)
    yield root
    mp.undo()


@pytest.fixture(scope="module")
def suite_trained(lab_data, tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("nfv3_suite") / "run", TINY_SUITE)
    tn.train(ctx)
    return ctx


@pytest.fixture(scope="module")
def suite_result(suite_trained, tmp_path_factory):
    ev = make_ctx(
        tmp_path_factory.mktemp("nfv3_eval") / "eval",
        {"ckpt": str(suite_trained.dir / "ckpt" / "last.pt"), "episodes": 2, "eval_seed": 0,
         "video": False, "prompt": "machine"},
    )  # fmt: skip
    tn.evaluate(ev)
    return json.loads((ev.dir / "result.json").read_text())


def test_frozen_prompt_keeps_the_first_word(suite_trained, tmp_path_factory):
    ev = make_ctx(
        tmp_path_factory.mktemp("nfv3_frozen") / "eval",
        {"ckpt": str(suite_trained.dir / "ckpt" / "last.pt"), "episodes": 1, "eval_seed": 0,
         "video": False, "prompt": "frozen"},
    )  # fmt: skip
    tn.evaluate(ev)
    r = json.loads((ev.dir / "result.json").read_text())
    assert r["prompt"] == "frozen"
    for tr in r["episodes_traces"]:
        if not tr["group"].startswith("A"):
            assert set(tr["subtask"]) == {"approach"}


def test_spec_shape():
    spec = registry.get("nano-force-vla2")
    assert spec.order == 18 and spec.viewer == "fvrollout"
    assert spec.lesson == "nano-force-vla2.html"
    assert spec.title == "ForceVLA2: force as an output"
    assert {"preset", "force_in", "head", "force_prompt", "expert_inputs", "task", "steps",
            "lr", "batch", "demos", "seed"} == {p.name for p in spec.params}  # fmt: skip
    assert {"episodes", "eval_seed", "video", "prompt"} == {p.name for p in spec.eval_params}
    d = registry.validate_params(spec, {}, which="train")
    assert d["force_in"] == "forcevla2" and d["head"] == "hybrid" and d["force_prompt"]
    assert d["task"] == "t2" and d["steps"] == 12000
    assert spec.estimate(d) > 0


def test_suite_arms_and_configs(suite_trained):
    assert read_status(suite_trained.dir)["state"] == "done"
    ck = torch.load(suite_trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert ck["kind"] == "nfv-suite" and ck["arms"] == SUITE_LABELS
    want = {
        "A fvlmoe+position": ("fvlmoe", "position", False, 4),
        "B forcevla2+hybrid": ("forcevla2", "hybrid", True, 11),
        "C forcevla2+position": ("forcevla2", "position", True, 4),
    }
    for label, (fi, head, prompt, act_dim) in want.items():
        a = ck[label]["arch"]
        assert (a["force_in"], a["head"], a["force_prompt"], a["act_dim"]) == (
            fi,
            head,
            prompt,
            act_dim,
        )
        assert ck[label]["task"] == "t2" and a["chunk"] == 8


def test_eval_schema(suite_result):
    r = suite_result
    assert r["scenario"] == "claims_suite" and r["n_episodes"] == 6
    assert set(r["arms"]) == set(SUITE_LABELS)
    for s in r["arms"].values():
        assert s["n"] == 2 and 0 <= s["success"] <= 1 and len(s["wilson95"]) == 2
        assert {"abort_rate", "force_rmse_contact", "track_frac", "coverage",
                "per_instruction", "peak_force"} <= set(s)  # fmt: skip
        assert ("xy_select_free" in s) == (s["head"] == "hybrid")
    assert set(r["gate"]) == {"B forcevla2+hybrid", "C forcevla2+position"}
    for g in r["gate"].values():
        for bucket in ("contact", "free"):
            assert set(g[bucket]) == {"decisions", "vision", "text", "state", "force"}
    tr = r["episodes_traces"]
    assert tr and {"t", "fz", "f_target", "mode", "subtask", "events"} <= set(tr[0])
    hyb = [t for t in tr if t["group"].startswith("B")]
    assert hyb and set(hyb[0]["subtask"]) <= {"approach", "wipe"}  # the prompt's words


def test_claims_rows(suite_result):
    rows = suite_result["claims"]
    assert [c["id"] for c in rows] == [c.id for c in M3_CLAIMS]
    assert len(rows) == 5
    for row in rows:
        assert row["verdict"] in ALLOWED
        assert row["verdict"] != NOT_MEASURED or row["id"] == "m3.gate_tilts_to_force"
    for c in M3_CLAIMS:
        _parse(c.expect)  # every bar is a parseable expression
        assert "2603.15169" in c.source or "2603.15169" in c.url


def test_both_verdicts():
    assert tn._both("CONFIRMED", "CONFIRMED") == "CONFIRMED"
    assert tn._both("CONFIRMED", "NOT CONFIRMED") == "NOT CONFIRMED"
    assert tn._both("INCONCLUSIVE", "CONFIRMED") == "INCONCLUSIVE"
