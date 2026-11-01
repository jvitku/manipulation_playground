"""Task "nano-force-vla" (ladder 17): the Table-3 suite, noise-row eval, routing, M2 claims."""

import json

import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task_nfv2 as tn
from nfv.claims import M2_CLAIMS, NOT_MEASURED, VERDICTS, verdict
from nfv.data.record import record_demos

ALLOWED = {*VERDICTS, NOT_MEASURED}
# tiny CPU budgets: shrunken architecture (ARCH_OVERRIDES hook), 12 t1 demos, 120 steps/arm
TINY_ARCH = {"chunk": 8, "n_exec": 4, "euler_steps": 5, "trunk_layers": 2, "expert_layers": 2}
TINY_COMMON = {"moe": "top1_capacity", "n_experts": 2, "balance_loss": False, "task": "t1",
               "lr": 1e-3, "batch": 16, "demos": 12, "seed": 0}  # fmt: skip
TINY_SUITE = {"preset": "claims_suite", "force_in": "fvlmoe", "steps": 120, **TINY_COMMON}
# the single arm doubles as the text_dropout train check (round 2's one training knob)
TINY_SINGLE = {"preset": "single", "force_in": "none", "steps": 200, "text_dropout": 0.25,
               **TINY_COMMON}  # fmt: skip
SUITE_LABELS = ["none", "prefix_linear", "prefix_moe", "concat_after", "fvlmoe"]
NOISE_LABELS = ["fvlmoe [zero]", "fvlmoe [gaussian]"]


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "nano-force-vla", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def lab_data(tmp_path_factory):
    """Module-scoped LAB_DATA redirect: a tiny canonical t1 demo set + the shrunken arch."""
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("nfv2_data")
    mp.setenv("LAB_DATA", str(root))
    mp.setattr(tn, "ARCH_OVERRIDES", dict(TINY_ARCH))
    mp.setattr(tn, "CB_SEEDS", 2)  # the single preset's closed-loop points stay CPU-cheap
    record_demos("t1", 12, root / "nfv" / "demos", noise_mm=0.5)
    yield root
    mp.undo()


@pytest.fixture(scope="module")
def suite_trained(lab_data, tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("nfv2_suite") / "run", TINY_SUITE)
    tn.train(ctx)
    return ctx


@pytest.fixture(scope="module")
def single_trained(lab_data, tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("nfv2_single") / "run", TINY_SINGLE)
    tn.train(ctx)
    return ctx


@pytest.fixture(scope="module")
def suite_result(suite_trained, single_trained, tmp_path_factory):
    """One eval over the suite ckpt + the single ckpt as ckpt_b: 8 groups x 2 episodes."""
    ev = make_ctx(
        tmp_path_factory.mktemp("nfv2_eval") / "eval",
        {"ckpt": str(suite_trained.dir / "ckpt" / "last.pt"),
         "ckpt_b": str(single_trained.dir / "ckpt" / "last.pt"),
         "episodes": 2, "eval_seed": 0, "video": False, "force_noise": "off"},
    )  # fmt: skip
    tn.evaluate(ev)
    return json.loads((ev.dir / "result.json").read_text())


def test_spec_shape():
    spec = registry.get("nano-force-vla")
    assert spec.order == 17 and spec.viewer == "fvrollout" and spec.vecgrids is False
    assert spec.lesson == "nano-force-vla.html"
    assert {"preset", "force_in", "moe", "n_experts", "balance_loss", "task", "steps", "lr",
            "batch", "demos", "text_dropout", "seed"} == {p.name for p in spec.params}  # fmt: skip
    assert {"episodes", "eval_seed", "video", "force_noise", "ckpt_b", "cfg_w",
            "n_exec_override"} == {p.name for p in spec.eval_params}  # fmt: skip
    td = next(p for p in spec.params if p.name == "text_dropout")
    assert td.kind == "float" and td.default == 0.0 and (td.min, td.max) == (0.0, 0.5)
    cw = next(p for p in spec.eval_params if p.name == "cfg_w")
    assert cw.kind == "float" and cw.default == 1.0 and (cw.min, cw.max) == (0.5, 5.0)
    ne = next(p for p in spec.eval_params if p.name == "n_exec_override")
    assert ne.kind == "int" and ne.default == 0 and (ne.min, ne.max) == (0, 16)
    preset = next(p for p in spec.params if p.name == "preset")
    assert preset.choices == ["single", "claims_suite"] and preset.default == "single"
    fi = next(p for p in spec.params if p.name == "force_in")
    assert fi.choices == ["fvlmoe", "prefix_linear", "prefix_moe", "concat_after",
                          "prompt_bins", "none"] and fi.default == "fvlmoe"  # fmt: skip
    ne = next(p for p in spec.params if p.name == "n_experts")
    assert ne.kind == "choice" and ne.choices == [2, 4, 8] and ne.default == 4
    fn = next(p for p in spec.eval_params if p.name == "force_noise")
    assert fn.choices == ["off", "zero", "shuffle", "gaussian"] and fn.default == "off"
    ckb = next(p for p in spec.eval_params if p.name == "ckpt_b")
    assert ckb.kind == "ckpt" and ckb.default is None  # optional: the dropdown gets "none"
    assert ckb.ckpt_task == "nano-force-vla"
    assert [c["id"] for c in spec.charts] == ["loss", "eval", "speed", "lr"]
    assert {s["key"] for s in spec.charts[0]["series"]} == {
        "loss",
        "a_loss",
        "b_loss",
        "c_loss",
        "d_loss",
        "e_loss",
    }
    assert spec.estimate(registry.validate_params(spec, {}, which="train")) > 0


def test_suite_one_checkpoint_five_labelled_arms(suite_trained):
    assert read_status(suite_trained.dir)["state"] == "done"
    ck = torch.load(suite_trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert ck["kind"] == "nfv-suite" and ck["arms"] == SUITE_LABELS
    for label in SUITE_LABELS:
        arm = ck[label]
        assert {"model", "ema", "arch", "task", "force_in"} <= set(arm)
        assert arm["task"] == "t1" and arm["force_in"] == label
        assert arm["arch"]["force_in"] == label and arm["arch"]["n_experts"] == 2
        assert arm["arch"]["chunk"] == 8  # the test hook reached the built config
        if label in ("prefix_moe", "fvlmoe"):  # the capacity cap at TRAINING batch size
            assert 0.0 <= arm["train_fraction_dropped"] <= 1.0
            assert len(arm["train_expert_counts"]) == 2
        else:
            assert "train_fraction_dropped" not in arm
    index = json.loads((suite_trained.dir / "ckpt" / "index.json").read_text())
    assert [row["label"] for row in index] == [
        "none trained", "prefix_linear trained", "prefix_moe trained",
        "concat_after trained", "final",
    ]  # fmt: skip
    assert "e_loss" in index[-1]["metrics"] and "e_drop" in index[-1]["metrics"]


def test_suite_metrics_streams(suite_trained):
    rows = read_metrics(suite_trained.dir)
    steps = [r["step"] for r in rows]
    assert steps == sorted(steps)  # one monotonic stream across the five arms
    for key in ("a_loss", "b_loss", "c_loss", "d_loss", "e_loss"):
        own = [r[key] for r in rows if key in r]
        assert own, f"no {key} rows"
        assert all(v > 0 for v in own)


def test_suite_eval_result_schema(suite_result):
    r = suite_result
    assert {"scenario", "episodes", "n_episodes", "fps", "video", "arms", "outcomes",
            "peak_force", "routing", "claims", "episodes_traces"} <= set(r)  # fmt: skip
    assert r["scenario"] == "claims_suite" and r["video"] is None
    assert r["episodes"] == 2 and r["n_episodes"] == 16  # 5 arms + 2 noise rows + ckpt_b
    assert set(r["arms"]) == {*SUITE_LABELS, *NOISE_LABELS, "B: single"}
    assert r["cfg_w"] == 1.0 and r["n_exec_override"] == 0  # the default act-time recipe
    for s in r["arms"].values():
        assert s["n"] == 2 and 0.0 <= s["success"] <= 1.0
        lo, hi = s["wilson95"]
        assert 0.0 <= lo <= hi <= 1.0
        assert s["peak_force"]["max"] >= s["peak_force"]["mean"] >= 0
        assert "pressed_counts" in s and "per_instruction" in s  # every group is t1
        bm = s["button_match"]  # instruction following, separated from press control
        assert bm["matched"] <= bm["pressed"] <= s["n"]
        assert bm["rate"] is None if bm["pressed"] == 0 else 0.0 <= bm["rate"] <= 1.0
    assert r["arms"]["fvlmoe"]["force_noise"] == "off"
    assert r["arms"]["fvlmoe [zero]"]["force_noise"] == "zero"
    assert r["arms"]["fvlmoe [gaussian]"]["force_noise"] == "gaussian"
    assert r["arms"]["B: single"]["force_in"] == "none"  # ckpt_b wired through + rendered
    assert set(r["outcomes"]["per_task"]) == set(r["arms"])


def test_suite_eval_claims_rows(suite_result):
    rows = suite_result["claims"]
    assert [c["id"] for c in rows] == [c.id for c in M2_CLAIMS]
    for c in rows:
        assert {"id", "statement", "metric", "expected", "measured", "verdict", "source",
                "url"} <= set(c)  # fmt: skip
        assert c["verdict"] in ALLOWED
    by_id = {c["id"]: c for c in rows}
    # the suite + noise rows measure all five M2 claims (a verdict, never "not measured")
    for cid in ("m2.force_in_beats_none", "m2.twin_gap", "m2.noise_drop",
                "m2.moe_before_vlm_collapses", "m2.fvlmoe_gate"):  # fmt: skip
        assert by_id[cid]["verdict"] in VERDICTS, cid
        assert by_id[cid]["measured"] != "not measured in this run"


def test_suite_eval_routing_stats(suite_result):
    routing = suite_result["routing"]
    # the LIMoE routes report expert usage; the ckpt_b "none" arm and linear routes do not
    assert {"prefix_moe", "fvlmoe", "fvlmoe [zero]", "fvlmoe [gaussian]"} == set(routing)
    for label, st in routing.items():
        assert st["n_experts"] == 2
        total = 0
        for side in ("contact", "free"):
            b = st[side]
            total += b["decisions"]
            assert len(b["expert_frac"]) == 2 and len(b["force_token_expert_frac"]) == 2
            if b["decisions"]:
                assert sum(b["expert_frac"]) == pytest.approx(1.0, abs=2e-3)
                assert sum(b["force_token_expert_frac"]) == pytest.approx(1.0, abs=2e-3)
                # act time = batch 1 = groups of 1: the capacity floor never binds (docstring)
                assert b["fraction_dropped"] == 0.0
                assert b["force_token_kept_frac"] == 1.0
        assert total > 0, f"{label}: no decisions recorded"


def test_suite_eval_traces(suite_result):
    traces = suite_result["episodes_traces"]
    assert 0 < len(traces) <= 10
    assert len({tr["group"] for tr in traces}) >= 8  # spread round-robin over the 8 groups
    for tr in traces:
        n = len(tr["fz"])
        assert 0 < n <= 101
        assert len(tr["t"]) == len(tr["mode"]) == len(tr["subtask"]) == n
        assert set(tr["mode"]) == {"position"}  # a force INPUT commands no wrench
        assert "f_target" not in tr  # omitted: the viewer draws no command line
        assert tr["task"] == "t1" and tr["outcome"]
        assert tr["force_noise"] in ("off", "zero", "gaussian")
        assert isinstance(tr["events"], list)


def test_single_train_loss_drops_and_payload(single_trained):
    assert read_status(single_trained.dir)["state"] == "done"
    rows = read_metrics(single_trained.dir)
    losses = [r["loss"] for r in rows if "loss" in r]
    assert len(losses) >= 5
    first, last = sum(losses[:2]) / 2, sum(losses[-2:]) / 2
    assert last < 0.75 * first, f"loss only moved {first:.4f} -> {last:.4f}"
    evals = [r["eval_success"] for r in rows if "eval_success" in r]
    assert evals and all(0.0 <= v <= 1.0 for v in evals)  # the final-step evaluation point
    ck = torch.load(single_trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert {"model", "ema", "norm", "arch", "params", "gen"} <= set(ck)
    assert ck["arch"]["force_in"] == "none" and ck["params"]["preset"] == "single"
    assert ck["arch"]["text_dropout"] == 0.25  # the round-2 training knob reached the config


def test_eval_knobs_roundtrip(suite_trained, tmp_path):
    """cfg_w / n_exec_override reach every rebuilt policy and are echoed in result.json."""
    ev = make_ctx(
        tmp_path / "eval",
        {"ckpt": str(suite_trained.dir / "ckpt" / "last.pt"), "episodes": 1, "eval_seed": 0,
         "video": False, "force_noise": "off", "cfg_w": 2.0, "n_exec_override": 2},
    )  # fmt: skip
    tn.evaluate(ev)
    r = json.loads((ev.dir / "result.json").read_text())
    assert r["cfg_w"] == 2.0 and r["n_exec_override"] == 2
    assert {*SUITE_LABELS, *NOISE_LABELS} == set(r["arms"])
    for s in r["arms"].values():
        assert s["n"] == 1 and 0.0 <= s["success"] <= 1.0


def test_verdict_nonpositive_gap_form():
    # the m2.moe_before_vlm_collapses grammar: a-b <= 0 (reported either way)
    assert verdict("a-b <= 0", {"a": 0.1, "b": 0.4}) == "CONFIRMED"
    assert verdict("a-b <= 0", {"a": 0.5, "b": 0.4}) == "NOT CONFIRMED"
    assert verdict("a-b <= 0", {"a": 0.3, "b": 0.4}, ci=(-0.2, 0.1)) == "INCONCLUSIVE"


def test_stop_is_stopped(lab_data, tmp_path):
    ctx = make_ctx(tmp_path / "run", {**TINY_SUITE, "steps": 50})
    ctx._stop = True
    tn.train(ctx)
    assert read_status(ctx.dir)["state"] == "stopped"


def test_estimate_uses_calib(monkeypatch):
    spec = registry.get("nano-force-vla")
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
    assert spec.estimate({**p, "steps": 1000, "preset": "claims_suite"}) == pytest.approx(50.0)
