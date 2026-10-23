"""Task "attention" (id `attention`, ladder 6): key-value lookup with one attention layer."""

import json

import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task_attention as ta

TINY = {"heads": 2, "d_k": 16, "n_pairs": 4, "steps": 1500, "lr": 3e-3, "seed": 0}


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "attention", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("attn") / "run", TINY)
    ta.train(ctx)
    return ctx


def test_spec_shape():
    spec = registry.get("attention")
    assert spec.order == 6 and spec.live == "attnlive" and spec.vecgrids is True
    assert spec.viewer == "attnstep"
    names = [p.name for p in spec.params + spec.eval_params]
    assert len(names) == len(set(names))
    assert {"heads", "d_k", "n_pairs", "steps", "lr", "seed"} == {p.name for p in spec.params}
    assert {"eval_seed", "eval_n_pairs"} == {p.name for p in spec.eval_params}
    assert registry.get("flow").live == ""  # the field defaults off for other tasks


def test_train_reaches_target(trained):
    assert read_status(trained.dir)["state"] == "done"
    rows = [r for r in read_metrics(trained.dir) if "acc" in r]
    assert rows[-1]["acc"] >= 0.99


def test_metrics_rows_small(trained):
    lines = (trained.dir / "metrics.jsonl").read_text().splitlines()
    assert all(len(line) < 8192 for line in lines)
    snaps = [json.loads(line) for line in lines if '"attn"' in line]
    assert 40 <= len(snaps) <= 80
    assert all("qkv" in s and "loss" in s for s in snaps)
    assert len(snaps[0]["attn"]) == TINY["heads"]


def test_row_cap_at_largest_config(tmp_path):
    ctx = make_ctx(tmp_path / "run", {**TINY, "heads": 4, "d_k": 32, "n_pairs": 8, "steps": 20})
    ta.train(ctx)
    lines = (ctx.dir / "metrics.jsonl").read_text().splitlines()
    assert lines and max(len(line) for line in lines) < 8192


def test_eval_roundtrip(tmp_path, trained):
    ck = torch.load(trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert {"heads", "d_k", "n_keys", "n_vals"} <= set(ck["meta"])
    ev = make_ctx(
        tmp_path / "eval",
        {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "eval_seed": 3, "eval_n_pairs": 6},
    )
    ta.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["n_pairs"] == 6 and 0.0 <= res["acc"] <= 1.0
    assert set(res["acc_by_n"]) >= {"2", "4", "8"}
    pr = res["probe"]
    assert len(pr["tokens"]) == 7 and pr["tokens"][-1]["label"].startswith("?")
    assert pr["pred"] in range(ck["meta"]["n_vals"]) and "target" in pr
    names = [s["name"] for s in res["stages"]]
    assert names == ["embed", "qkv", "scores", "softmax", "mixture", "readout"]
    st = {s["name"]: s for s in res["stages"]}
    H, dk, T = 2, 16, 7
    assert len(st["qkv"]["q"]) == H and len(st["qkv"]["k"][0]) == T
    assert len(st["qkv"]["k"][0][0]) == dk
    assert len(st["scores"]["pre"]) == H and len(st["scores"]["pre"][0]) == T
    assert abs(sum(st["softmax"]["attn"][0]) - 1) < 1e-3
    assert len(st["readout"]["probs"]) == ck["meta"]["n_vals"]


def test_stage_attention_matches_mha(trained):
    ck = torch.load(trained.dir / "ckpt" / "last.pt", weights_only=False)
    model = ta.build(ck["meta"])
    model.load_state_dict(ck["state"])
    keys, vals, qpos = ta.make_batch(2, 5, ck["meta"], torch.Generator().manual_seed(1), "cpu")
    stages = ta.forward_stages(model, keys[:1], vals[:1], qpos[:1])
    _, _, attn = model(keys[:1], vals[:1], qpos[:1])
    got = torch.tensor(stages["softmax"]["attn"])
    assert torch.allclose(got, attn[0, :, 0], atol=2e-4)


def test_stop_is_stopped(tmp_path):
    ctx = make_ctx(tmp_path / "run", {**TINY, "steps": 1000})
    ctx._stop = True
    ta.train(ctx)
    assert read_status(ctx.dir)["state"] == "stopped"


def test_estimate_uses_calib(monkeypatch):
    spec = registry.get("attention")
    p = registry.validate_params(spec, {}, which="train")
    base = spec.estimate(p)
    assert base > 0
    monkeypatch.setattr(
        registry, "calib", lambda tid: {"steps_per_s": 1.0, "overhead_s": 0.0}, raising=True
    )
    assert spec.estimate({**p, "steps": 100}) == pytest.approx(100.0)
