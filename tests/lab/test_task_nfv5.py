"""Task "lora-vla" (ladder 20): SmolVLA + LoRA - spec shape, the M5 claims rows, and the
train/eval orchestration end to end against a FAKE worker (tests/lab/fake_lora_worker.py), so
the test needs neither /opt/vla nor a GPU nor the SmolVLA weights. CPU, a few seconds."""

import json
import sys
from pathlib import Path

import pytest

from lab.core.jobs import read_metrics
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task_nfv5 as tn
from nfv.claims import M5_CLAIMS, NOT_MEASURED, VERDICTS

ALLOWED = {*VERDICTS, NOT_MEASURED}
ROOT = Path(__file__).resolve().parents[2]


def make_ctx(d: Path, task: str, params: dict) -> RunContext:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": task, "params": params}))
    return RunContext(d)


@pytest.fixture
def fake_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("NFV_VLA_PY", sys.executable)
    monkeypatch.setenv("NFV_LORA_WORKER", "tests.lab.fake_lora_worker")
    monkeypatch.setenv("LAB_DATA", str(tmp_path / "labdata"))
    demo = tmp_path / "labdata" / "nfv" / "demos" / "t3"
    demo.mkdir(parents=True)
    (demo / "index.json").write_text(json.dumps({"task": "t3", "n": 0, "episodes": []}))
    monkeypatch.setattr(tn, "POLL_S", 0.05)
    return tmp_path


# ------------------------------------------------------------------------------------- spec
def test_spec_shape():
    s = registry.get("lora-vla")
    assert s is tn.SPEC
    assert (s.order, s.viewer, s.lesson) == (20, "fvrollout", "lora-vla.html")
    assert (ROOT / "lab/web/lessons" / s.lesson).is_file()
    names = {p.name for p in s.params}
    assert {"preset", "mode", "r", "alpha", "task", "steps", "batch", "lr", "seed"} <= names
    assert {p.name for p in s.eval_params} == {"episodes", "eval_seed", "video", "parallel"}
    p = registry.validate_params(s, {}, which="train")
    assert (p["preset"], p["mode"], p["r"], p["task"]) == ("single", "lora", 16, "t3")
    assert next(x for x in s.params if x.name == "r").choices == [4, 8, 16, 32]
    assert registry.validate_params(s, {}, which="eval")["episodes"] == 100
    with pytest.raises(ValueError):
        registry.validate_params(s, {"r": 12}, which="train")
    assert {c["id"] for c in s.charts} >= {"loss", "vram", "speed"}


def test_lesson_placeholder_has_the_question():
    txt = (ROOT / "lab/web/lessons/lora-vla.html").read_text()
    assert "<h3>The question</h3>" in txt and "<p>" in txt


def test_estimate_uses_calib(monkeypatch):
    monkeypatch.setattr(registry, "calib", lambda tid: {
        "samples_per_s": {"head": 10.0, "lora": 10.0, "expert_full": 10.0}, "load_s": 0.0,
        "probe_s": 5.0, "overhead_s": 0.0})  # fmt: skip
    base = registry.validate_params(tn.SPEC, {"steps": 100, "batch": 10}, which="train")
    assert tn.estimate(base) == pytest.approx(100.0)  # 100 steps x 10 / 10 samples/s
    suite = {**base, "preset": "claims_suite"}
    assert tn.estimate(suite) == pytest.approx(3 * 100.0 + 2 * 5.0)


def test_suite_and_vla_constants_agree():
    """task_nfv5 keeps its own copies (the lab python never imports lerobot); they must match."""
    from nfv import vla_smolvla as V

    assert tuple(m for _, _, m in tn.SUITE) == V.TRAIN_MODES == tn.TRAIN_MODES
    assert set(V.MODE_LR) >= set(V.TRAIN_MODES)


# ----------------------------------------------------------------------------------- claims
def _arm(mode, success, ci, trainable, total, vram):
    return {"label": mode, "mode": mode, "trainable": trainable, "total": total,
            "success": success, "wilson95": ci, "peak_vram_gb": vram, "batch": 16, "r": 16,
            "lora_layers": 128}  # fmt: skip


def test_claims_rows_schema_and_verdicts():
    arms = [_arm("head", 0.20, [0.13, 0.29], 1_635_152, 450_046_176, 3.21),
            _arm("lora", 0.62, [0.52, 0.71], 4_563_792, 452_974_816, 2.00),
            _arm("expert_full", 0.60, [0.50, 0.69], 99_880_992, 450_046_176, 4.74)]  # fmt: skip
    result = {
        "probe_full": {"fp32": {"fits": False, "peak_alloc_gb": 8.96, "batch": 16},
                       "bf16": {"fits": True, "peak_alloc_gb": 7.1}},
        "lora_spectrum": {"full_mean_energy_top16": 0.4,
                          "layers": [{"name": "x.lm_expert.layers.0.q", "full":
                                      {"energy_top_16": 0.4}}]},
        "merge_check": {"fp32_max_abs": 2e-6, "fp32_out_abs_mean": 0.7, "bf16_max_abs": 0.04},
    }  # fmt: skip
    rows = tn.claims_rows(arms, {}, result)
    assert [r["id"] for r in rows] == [c.id for c in M5_CLAIMS]
    keys = {"id", "statement", "metric", "expected", "measured", "verdict", "source", "url"}
    for r in rows:
        assert set(r) == keys and r["verdict"] in ALLOWED
    v = {r["id"]: r["verdict"] for r in rows}
    assert v["m5.lora_param_fraction"] == "CONFIRMED"  # 1.0 %
    # +2 pts, but the gap's CI (-17..+21 from two Wilson intervals) straddles the -10 bar
    assert v["m5.lora_matches_expert_ft"] == "INCONCLUSIVE"
    assert v["m5.lora_beats_head_only"] == "CONFIRMED"  # CI 0.23..0.58 above 0.10
    assert v["m5.lora_memory"] == "CONFIRMED"  # 2.00 / 4.74 = 0.42
    assert v["m5.full_ft_does_not_fit"] == "CONFIRMED"  # OOM
    assert v["m5.update_is_low_rank"] == "NOT CONFIRMED"  # 0.4 < 0.5
    assert v["m5.merge_is_exact"] == "CONFIRMED"


def test_claims_unmeasured_without_eval_or_probe():
    arms = [_arm("lora", None, None, 4_563_792, 452_974_816, 3.6)]
    rows = {r["id"]: r for r in tn.claims_rows(arms, {}, {})}
    assert rows["m5.lora_param_fraction"]["verdict"] == "CONFIRMED"
    for cid in (
        "m5.lora_matches_expert_ft",
        "m5.lora_beats_head_only",
        "m5.lora_memory",
        "m5.full_ft_does_not_fit",
        "m5.update_is_low_rank",
        "m5.merge_is_exact",
    ):
        assert rows[cid]["verdict"] == NOT_MEASURED, cid


def test_full_ft_that_fits_is_not_confirmed():
    res = {"probe_full": {"fp32": {"fits": True, "peak_alloc_gb": 8.0, "batch": 4}}}
    row = {r["id"]: r for r in tn.claims_rows([], {}, res)}["m5.full_ft_does_not_fit"]
    assert row["verdict"] == "NOT CONFIRMED" and "fits" in row["measured"]


def test_pick_video_seed_prefers_lora_win_over_head_fail():
    def ev(mode, oks):
        return {"mode": mode, "episodes": [{"seed": 10 + i, "success": o}
                                           for i, o in enumerate(oks)]}  # fmt: skip

    evs = {"A": ev("head", [True, False, False]), "B": ev("lora", [True, False, True]),
           "C": ev("expert_full", [False, True, True])}  # fmt: skip
    assert tn.pick_video_seed(evs, ["A", "B", "C"]) == 12
    evs["B"] = ev("lora", [False, False, False])
    assert tn.pick_video_seed(evs, ["A", "B", "C"]) == 10  # first seed any arm solves
    for k in evs:
        evs[k] = ev(evs[k]["mode"], [False] * 3)
    assert tn.pick_video_seed(evs, ["A", "B", "C"]) == 10


# ------------------------------------------------------------------ orchestration (fake worker)
def test_suite_train_and_eval_end_to_end(fake_worker):
    tmp = fake_worker
    params = registry.validate_params(tn.SPEC, {"preset": "claims_suite", "steps": 40,
                                                "batch": 4}, which="train")  # fmt: skip
    ctx = make_ctx(tmp / "run", "lora-vla", params)
    tn.train(ctx)
    st = json.loads((tmp / "run" / "status.json").read_text())
    assert st["state"] == "done", st
    rows = read_metrics(tmp / "run")
    keys = {k for r in rows for k in r}
    assert {"a_loss", "b_loss", "c_loss", "b_vram_gb", "c_steps_per_s"} <= keys
    index = json.loads((tmp / "run" / "ckpt" / "index.json").read_text())
    assert [e["label"] for e in index][-1] == "final" and len(index) == 3
    import torch

    ck = torch.load(tmp / "run" / "ckpt" / "last.pt", weights_only=False)
    assert ck["kind"] == "nfv-lora-suite" and [a["mode"] for a in ck["arms"]] == list(
        tn.TRAIN_MODES
    )
    assert ck["probe_full"]["fp32"]["fits"] is False
    assert all((tmp / "run" / a["adapter"]).is_file() for a in ck["arms"])

    eparams = registry.validate_params(tn.SPEC, {"episodes": 10}, which="eval")
    eparams["ckpt"] = str(tmp / "run" / "ckpt" / "last.pt")
    ectx = make_ctx(tmp / "eval", "lora-vla", eparams)
    tn.evaluate(ectx)
    est = json.loads((tmp / "eval" / "status.json").read_text())
    assert est["state"] == "done", est
    res = json.loads((tmp / "eval" / "result.json").read_text())
    assert res["video"] == "rollout.mp4" and (tmp / "eval" / "rollout.mp4").stat().st_size > 0
    assert not list((tmp / "eval" / "workers").glob("video_*.mp4"))  # segments cleaned up
    assert set(res["arms"]) == {"A head", "B lora", "C expert_full"}
    assert res["arms"]["B lora"]["success"] == 0.5 and res["arms"]["A head"]["success"] == 0.2
    assert [a["mode"] for a in res["lora_arms"]] == list(tn.TRAIN_MODES)
    assert res["lora_arms"][1]["trainable_pct"] == pytest.approx(1.0075, abs=1e-3)
    sp = res["lora_spectrum"]
    assert sp["r"] == 16 and len(sp["layers"]) == 2
    assert all("lora" in L and "full" in L for L in sp["layers"])
    assert sp["layers"][0]["lora"]["energy_top_r"] == pytest.approx(1.0)
    assert 0 < len(res["episodes_traces"]) <= 10
    tr = res["episodes_traces"][0]
    assert {"task", "seed", "group", "outcome", "t", "fz", "mode", "subtask", "events"} <= set(tr)
    assert res["outcomes"]["per_task"]["B lora"]["n"] == 10
    v = {r["id"]: r["verdict"] for r in res["claims"]}
    assert v["m5.lora_beats_head_only"] in ALLOWED and v["m5.merge_is_exact"] == "CONFIRMED"
    assert v["m5.full_ft_does_not_fit"] == "CONFIRMED"
    # fake outcomes: A solves the first 2 seeds, B the first 5 -> the first B-win / A-fail seed
    assert res["video_seed"] == 80002


def test_single_preset_trains_one_arm(fake_worker):
    tmp = fake_worker
    form = {"preset": "single", "mode": "head", "steps": 20, "batch": 2}
    params = registry.validate_params(tn.SPEC, form, which="train")
    ctx = make_ctx(tmp / "run1", "lora-vla", params)
    tn.train(ctx)
    assert json.loads((tmp / "run1" / "status.json").read_text())["state"] == "done"
    keys = {k for r in read_metrics(tmp / "run1") for k in r}
    assert "loss" in keys and "a_loss" not in keys
    assert not (tmp / "run1" / "probe_full_fp32.json").exists()  # single: no D probe by default
