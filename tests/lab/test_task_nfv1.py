"""Task "nano-vla" (ladder 16): nanoVLA single + claims-suite training, eval, M1 claims."""

import json

import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task_nfv1 as tn
from nfv.claims import M1_CLAIMS, NOT_MEASURED, VERDICTS, measured_value, verdict
from nfv.data.record import record_demos

ALLOWED = {*VERDICTS, NOT_MEASURED}
# tiny CPU budgets: single = 200 steps on 20 t0 demos, suite = 3 arms x 120 steps on 12 demos
TINY_COMMON = {"lr": 1e-3, "batch": 16, "chunk": 8, "n_exec": 4, "euler_steps": 5, "seed": 0}
TINY_SINGLE = {"preset": "single", "task": "t0", "steps": 200, "demos": 20, **TINY_COMMON}
TINY_SUITE = {"preset": "claims_suite", "task": "t1", "steps": 120, "demos": 12, **TINY_COMMON}
SUITE_LABELS = ["A t0 default", "B t1 default", "C t1 no-language"]


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "nano-vla", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def lab_data(tmp_path_factory):
    """Module-scoped LAB_DATA redirect with tiny canonical demo sets (t0: 20, t1: 12)."""
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("nfv1_data")
    mp.setenv("LAB_DATA", str(root))
    mp.setattr(tn, "CB_SEEDS", 4)  # the periodic closed-loop evaluation stays CPU-cheap
    record_demos("t0", 20, root / "nfv" / "demos", noise_mm=0.5)
    record_demos("t1", 12, root / "nfv" / "demos", noise_mm=0.5)
    yield root
    mp.undo()


@pytest.fixture(scope="module")
def single_trained(lab_data, tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("nfv1_single") / "run", TINY_SINGLE)
    tn.train(ctx)
    return ctx


@pytest.fixture(scope="module")
def suite_trained(lab_data, tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("nfv1_suite") / "run", TINY_SUITE)
    tn.train(ctx)
    return ctx


@pytest.fixture(scope="module")
def suite_result(suite_trained, tmp_path_factory):
    ev = make_ctx(
        tmp_path_factory.mktemp("nfv1_eval") / "eval",
        {"ckpt": str(suite_trained.dir / "ckpt" / "last.pt"), "episodes": 2, "eval_seed": 0,
         "video": False},
    )  # fmt: skip
    tn.evaluate(ev)
    return json.loads((ev.dir / "result.json").read_text())


def test_spec_shape():
    spec = registry.get("nano-vla")
    assert spec.order == 16 and spec.viewer == "fvrollout" and spec.vecgrids is False
    assert spec.lesson == "nano-vla.html"
    assert {"preset", "task", "steps", "lr", "batch", "demos", "chunk", "n_exec",
            "euler_steps", "seed"} == {p.name for p in spec.params}  # fmt: skip
    assert {"episodes", "eval_seed", "video", "cfg_w", "n_exec_override"} == {
        p.name for p in spec.eval_params
    }
    cw = next(p for p in spec.eval_params if p.name == "cfg_w")
    assert cw.kind == "float" and cw.default == 1.0 and (cw.min, cw.max) == (0.5, 5.0)
    ne = next(p for p in spec.eval_params if p.name == "n_exec_override")
    assert ne.kind == "int" and ne.default == 0 and (ne.min, ne.max) == (0, 16)
    preset = next(p for p in spec.params if p.name == "preset")
    assert preset.choices == ["single", "claims_suite"] and preset.default == "single"
    assert [c["id"] for c in spec.charts] == ["loss", "eval", "speed", "lr"]
    assert {s["key"] for s in spec.charts[0]["series"]} == {"loss", "a_loss", "b_loss", "c_loss"}
    assert spec.estimate(registry.validate_params(spec, {}, which="train")) > 0
    with pytest.raises(ValueError):  # the cross-param check: n_exec must be <= chunk
        registry.validate_params(spec, {"chunk": 8, "n_exec": 16}, which="train")


def test_single_train_loss_drops(single_trained):
    assert read_status(single_trained.dir)["state"] == "done"
    rows = read_metrics(single_trained.dir)
    losses = [r["loss"] for r in rows if "loss" in r]
    assert len(losses) >= 5
    first, last = sum(losses[:2]) / 2, sum(losses[-2:]) / 2
    assert last < 0.75 * first, f"loss only moved {first:.4f} -> {last:.4f}"
    evals = [r["eval_success"] for r in rows if "eval_success" in r]
    assert evals and all(0.0 <= v <= 1.0 for v in evals)  # the final-step evaluation point
    assert any("lr" in r and "steps_per_s" in r for r in rows)


def test_single_checkpoint_payload(single_trained):
    ck = torch.load(single_trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert {"model", "ema", "norm", "arch", "params", "gen"} <= set(ck)
    assert ck["gen"] == "flow" and ck["arch"]["chunk"] == 8 and ck["arch"]["force_in"] == "none"
    assert ck["params"]["preset"] == "single" and ck["params"]["task"] == "t0"


def test_suite_one_checkpoint_three_labelled_arms(suite_trained):
    assert read_status(suite_trained.dir)["state"] == "done"
    ck = torch.load(suite_trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert ck["kind"] == "nfv-suite" and ck["arms"] == SUITE_LABELS
    for label, task, blank in zip(
        SUITE_LABELS, ("t0", "t1", "t1"), (False, False, True), strict=True
    ):
        arm = ck[label]
        assert {"model", "ema", "arch", "task", "blank_text"} <= set(arm)
        assert arm["task"] == task and arm["blank_text"] is blank
        assert arm["arch"]["chunk"] == 8  # identical config across arms (C differs by text)
    index = json.loads((suite_trained.dir / "ckpt" / "index.json").read_text())
    assert [row["label"] for row in index] == [
        "A t0 default trained",
        "B t1 default trained",
        "final",
    ]
    assert "c_loss" in index[-1]["metrics"]


def test_suite_metrics_streams(suite_trained):
    rows = read_metrics(suite_trained.dir)
    steps = [r["step"] for r in rows]
    assert steps == sorted(steps)  # one monotonic stream across the three arms
    for key in ("a_loss", "b_loss", "c_loss"):
        own = [r[key] for r in rows if key in r]
        assert own, f"no {key} rows"
        assert all(v > 0 for v in own)


def test_suite_eval_result_schema(suite_result):
    r = suite_result
    assert {"scenario", "episodes", "n_episodes", "fps", "video", "arms", "outcomes",
            "peak_force", "claims", "episodes_traces"} <= set(r)  # fmt: skip
    assert r["scenario"] == "claims_suite" and r["video"] is None
    assert r["episodes"] == 2 and r["n_episodes"] == 6
    assert set(r["arms"]) == set(SUITE_LABELS)
    for label in SUITE_LABELS:
        arm = r["arms"][label]
        assert arm["n"] == 2 and 0.0 <= arm["success"] <= 1.0
        lo, hi = arm["wilson95"]
        assert 0.0 <= lo <= hi <= 1.0
        assert arm["peak_force"]["max"] >= arm["peak_force"]["mean"] >= 0
    assert r["arms"]["C t1 no-language"]["blank_text"] is True
    assert "pressed_counts" in r["arms"]["B t1 default"]  # which buttons actually clicked
    assert set(r["outcomes"]["per_task"]) == set(SUITE_LABELS)


def test_suite_eval_claims_rows(suite_result):
    rows = suite_result["claims"]
    assert [c["id"] for c in rows] == [c.id for c in M1_CLAIMS]
    for c in rows:
        assert {"id", "statement", "metric", "expected", "measured", "verdict", "source",
                "url"} <= set(c)  # fmt: skip
        assert c["verdict"] in ALLOWED
    by_id = {c["id"]: c for c in rows}
    # the suite provides every arm, so all three M1 claims are measured (verdict, not a gap)
    for cid in ("m1.nanovla_solves_visible", "m1.hidden_threshold_blinds_vla",
                "m1.language_is_load_bearing"):  # fmt: skip
        assert by_id[cid]["verdict"] in VERDICTS
        assert by_id[cid]["measured"] != "not measured in this run"


def test_suite_eval_traces(suite_result):
    traces = suite_result["episodes_traces"]
    assert 0 < len(traces) <= 10
    assert {tr["group"] for tr in traces} == set(SUITE_LABELS)  # spread round-robin over arms
    for tr in traces:
        n = len(tr["fz"])
        assert 0 < n <= 101
        assert len(tr["t"]) == len(tr["mode"]) == len(tr["subtask"]) == n
        assert set(tr["mode"]) == {"position"}  # nanoVLA commands no wrench
        assert "f_target" not in tr  # omitted: the viewer draws no command line
        assert tr["outcome"] and tr["task"] in ("t0", "t1")
        assert isinstance(tr["events"], list)


def test_suite_eval_default_recipe_recorded(suite_result):
    # the act-time knobs default off and are echoed (old eval configs lack them: .get path)
    assert suite_result["cfg_w"] == 1.0 and suite_result["n_exec_override"] == 0


def test_eval_knobs_roundtrip(suite_trained, tmp_path):
    """cfg_w / n_exec_override reach the rebuilt policies and are echoed in result.json."""
    ev = make_ctx(
        tmp_path / "eval",
        {"ckpt": str(suite_trained.dir / "ckpt" / "last.pt"), "episodes": 1, "eval_seed": 0,
         "video": False, "cfg_w": 2.0, "n_exec_override": 2},
    )  # fmt: skip
    tn.evaluate(ev)
    r = json.loads((ev.dir / "result.json").read_text())
    assert r["cfg_w"] == 2.0 and r["n_exec_override"] == 2
    assert set(r["arms"]) == set(SUITE_LABELS)
    for arm in r["arms"].values():
        assert arm["n"] == 1 and 0.0 <= arm["success"] <= 1.0


def test_verdict_window_form():
    # the window form added for the M1 T1 gate: lo <= a <= hi
    assert verdict("0.30 <= a <= 0.70", {"a": 0.5}) == "CONFIRMED"
    assert verdict("0.30 <= a <= 0.70", {"a": 0.8}) == "NOT CONFIRMED"
    assert verdict("0.30 <= a <= 0.70", {"a": 0.2}) == "NOT CONFIRMED"
    assert measured_value("0.30 <= a <= 0.70", {"a": 0.44}) == pytest.approx(0.44)
    # CI inside the window / across a bound / disjoint from it
    assert verdict("0.30 <= a <= 0.70", {"a": 0.5}, ci=(0.4, 0.6)) == "CONFIRMED"
    assert verdict("0.30 <= a <= 0.70", {"a": 0.65}, ci=(0.55, 0.75)) == "INCONCLUSIVE"
    assert verdict("0.30 <= a <= 0.70", {"a": 0.9}, ci=(0.83, 0.95)) == "NOT CONFIRMED"
    assert verdict("0.30 <= a <= 0.70", {"a": 0.5}, ci=(0.2, 0.8)) == "INCONCLUSIVE"
    with pytest.raises(ValueError):
        verdict("0.3 < a < 0.7", {"a": 0.5})  # only the <= window form exists


def test_stop_is_stopped(lab_data, tmp_path):
    ctx = make_ctx(tmp_path / "run", {**TINY_SINGLE, "steps": 50})
    ctx._stop = True
    tn.train(ctx)
    assert read_status(ctx.dir)["state"] == "stopped"


def test_estimate_uses_calib(monkeypatch):
    spec = registry.get("nano-vla")
    p = registry.validate_params(spec, {}, which="train")
    monkeypatch.setattr(
        registry,
        "calib",
        lambda tid: {
            "steps_per_s": 100.0,
            "load_s_per_demo": 0.0,
            "ep_s": 0.0,
            "overhead_s": 0.0,
        },  # fmt: skip
        raising=True,
    )
    assert spec.estimate({**p, "steps": 1000, "preset": "single"}) == pytest.approx(10.0)
    assert spec.estimate({**p, "steps": 1000, "preset": "claims_suite"}) == pytest.approx(30.0)
