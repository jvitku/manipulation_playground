"""Task "force-sim" (ladder 15): the contact testbed - demo recording, scenarios, claims."""

import json

import pytest

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task_nfv0 as tn
from nfv.claims import M0_CLAIMS, NOT_MEASURED, VERDICTS, measured_value, verdict

TINY = {"task": "all", "demos_per_task": 3, "noise": 0.5, "seed": 0}
ALLOWED = {*VERDICTS, NOT_MEASURED}


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "force-sim", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def lab_data(tmp_path_factory):
    """Module-scoped LAB_DATA redirect (demos must land in the tmp data root)."""
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("nfv0_data")
    mp.setenv("LAB_DATA", str(root))
    yield root
    mp.undo()


@pytest.fixture(scope="module")
def trained(lab_data, tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("nfv0_run") / "run", TINY)
    tn.train(ctx)
    return ctx


@pytest.fixture(scope="module")
def hold_result(trained, tmp_path_factory):
    ev = make_ctx(
        tmp_path_factory.mktemp("nfv0_eval") / "eval",
        {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "scenario": "hybrid_hold_5N",
         "eval_task": "all", "episodes": 3, "eval_seed": 0, "video": False},
    )  # fmt: skip
    tn.evaluate(ev)
    return json.loads((ev.dir / "result.json").read_text())


def test_spec_shape():
    spec = registry.get("force-sim")
    assert spec.order == 15 and spec.viewer == "fvrollout" and spec.vecgrids is False
    assert spec.lesson == "force-sim.html"
    assert {"task", "demos_per_task", "noise", "seed"} == {p.name for p in spec.params}
    assert {"scenario", "eval_task", "episodes", "eval_seed", "video"} == {
        p.name for p in spec.eval_params
    }
    scen = next(p for p in spec.eval_params if p.name == "scenario")
    assert set(scen.choices) == set(tn.SCENARIOS) and scen.default == "expert_rollouts"
    assert [c["id"] for c in spec.charts] == ["demos", "ok", "speed"]
    assert {s["key"] for s in spec.charts[0]["series"]} == {
        "t0_demos",
        "t1_demos",
        "t2_demos",
        "t3_demos",
    }
    assert spec.estimate(registry.validate_params(spec, {}, which="train")) > 0


def test_train_records_the_canonical_demo_sets(trained, lab_data):
    assert read_status(trained.dir)["state"] == "done"
    for task in ("t0", "t1", "t2", "t3"):
        d = lab_data / "nfv" / "demos" / task
        index = json.loads((d / "index.json").read_text())
        assert index["n"] == 3 and index["task"] == task
        assert len(list(d.glob("ep_*.npz"))) >= 3
        # provenance on top of record.py's fields
        assert index["run_id"] == trained.dir.name
        assert index["params"]["demos_per_task"] == 3
        assert "git" in index and index["tried"] >= index["n"]
    rows = read_metrics(trained.dir)
    steps = [r["step"] for r in rows]
    assert steps == sorted(steps)  # one monotonic stream across the four tasks
    for task in ("t0", "t1", "t2", "t3"):
        own = [r for r in rows if f"{task}_demos" in r]
        assert own and own[-1][f"{task}_demos"] == 3
        assert 0 < own[-1][f"{task}_ok_of_tries"] <= 1.0
        assert own[-1][f"{task}_s_per_demo"] > 0


def test_demo_set_checkpoint(trained):
    import torch

    ck = torch.load(trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert ck["kind"] == "nfv-demo-set"
    assert set(ck["tasks"]) == {"t0", "t1", "t2", "t3"}
    assert all(d["n"] == 3 for d in ck["tasks"].values())
    index = json.loads((trained.dir / "ckpt" / "index.json").read_text())
    assert index[-1]["label"] == "demo set"


def test_hold_result_schema(hold_result):
    r = hold_result
    assert {"scenario", "episodes", "n_episodes", "fps", "video", "outcomes", "peak_force",
            "tracking", "claims", "episodes_traces"} <= set(r)  # fmt: skip
    assert r["scenario"] == "hybrid_hold_5N" and r["n_episodes"] == 3 and r["video"] is None
    o = r["outcomes"]
    assert 0.0 <= o["success"] <= 1.0 and isinstance(o["counts"], dict)
    lo, hi = o["wilson95"]
    assert 0.0 <= lo <= hi <= 1.0
    assert r["peak_force"]["max"] >= r["peak_force"]["mean"] > 0
    heights = r["tracking"]["per_height"]
    assert [h["dz_mm"] for h in heights] == [-8.0, 0.0, 8.0]
    assert sum(h["n"] for h in heights) == 3
    # the demo-set marker checkpoint is read for provenance only
    assert r["demo_set"]["tasks"] == {"t0": 3, "t1": 3, "t2": 3, "t3": 3}


def test_hold_claims_rows(hold_result):
    rows = hold_result["claims"]
    assert [c["id"] for c in rows] == [c.id for c in M0_CLAIMS]
    for c in rows:
        assert {"id", "statement", "metric", "expected", "measured", "verdict", "source",
                "url"} <= set(c)  # fmt: skip
        assert c["verdict"] in ALLOWED
    by_id = {c["id"]: c for c in rows}
    c2 = by_id["m0.hybrid_holds_5n"]
    assert c2["verdict"] in VERDICTS  # measured by this scenario
    assert "N" in c2["measured"]
    assert by_id["m0.expert_solves_every_task"]["verdict"] == NOT_MEASURED
    assert by_id["m0.position_cannot_hold_force"]["verdict"] == NOT_MEASURED
    assert by_id["m0.expert_solves_every_task"]["measured"] == "not measured in this run"


def test_hold_traces_bounded(hold_result):
    traces = hold_result["episodes_traces"]
    assert 0 < len(traces) <= 10
    for tr in traces:
        n = len(tr["fz"])
        assert 0 < n <= 31  # the 3 s hold window (+ the abort edge case)
        assert len(tr["t"]) == len(tr["f_target"]) == len(tr["mode"]) == len(tr["subtask"]) == n
        assert set(tr["mode"]) <= {"position", "hybrid"}
        assert all(v == -5.0 for v in tr["f_target"])  # the commanded 5 N press
        assert isinstance(tr["events"], list)
        assert tr["outcome"] and tr["task"] == "t2"


def test_verdict_cases():
    assert verdict("a >= 0.95", {"a": 1.0}) == "CONFIRMED"
    assert verdict("a >= 0.95", {"a": 0.5}) == "NOT CONFIRMED"
    assert verdict("a <= 0.2", {"a": 0.1}) == "CONFIRMED"
    assert verdict("a <= 0.2", {"a": 0.3}) == "NOT CONFIRMED"
    assert verdict("a-b >= 0.15", {"a": 0.8, "b": 0.4}) == "CONFIRMED"
    assert verdict("a - b >= 0.15", {"a": 0.5, "b": 0.4}) == "NOT CONFIRMED"
    assert measured_value("a-b >= 0.15", {"a": 0.8, "b": 0.4}) == pytest.approx(0.4)


def test_verdict_inconclusive_when_ci_straddles():
    # 20/20 successes: the point value passes 0.95 but the Wilson CI dips below it
    assert verdict("a >= 0.95", {"a": 1.0}, ci=(0.839, 1.0)) == "INCONCLUSIVE"
    assert verdict("a >= 0.95", {"a": 1.0}, ci=(0.963, 1.0)) == "CONFIRMED"
    assert verdict("a >= 0.95", {"a": 0.5}, ci=(0.3, 0.7)) == "NOT CONFIRMED"
    assert verdict("a <= 0.2", {"a": 0.15}, ci=(0.093, 0.233)) == "INCONCLUSIVE"
    assert verdict("a-b >= 0.5", {"a": 1.0, "b": 0.1}, ci=(0.538, 0.972)) == "CONFIRMED"


def test_verdict_rejects_malformed_expect():
    with pytest.raises(ValueError):
        verdict("a > 0.95", {"a": 1.0})  # only >= and <= exist
    with pytest.raises(ValueError):
        verdict("a+b >= 0.1", {"a": 1.0, "b": 1.0})


def test_stop_is_stopped(lab_data, tmp_path):
    ctx = make_ctx(tmp_path / "run", TINY)
    ctx._stop = True
    tn.train(ctx)
    assert read_status(ctx.dir)["state"] == "stopped"


def test_estimate_uses_calib(monkeypatch):
    spec = registry.get("force-sim")
    p = registry.validate_params(spec, {}, which="train")
    monkeypatch.setattr(
        registry,
        "calib",
        lambda tid: {"s_per_demo": 1.0, "task_overhead_s": 0.0, "overhead_s": 0.0},
        raising=True,
    )
    assert spec.estimate({**p, "demos_per_task": 50}) == pytest.approx(200.0)  # four tasks
    assert spec.estimate({**p, "task": "t2", "demos_per_task": 50}) == pytest.approx(50.0)
