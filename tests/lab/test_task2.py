"""Task 2 (flow): tiny train + eval (K-sweep, trace) + DDPM head."""

import json

import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task2_flow as t2

TINY = {
    "head": "flow",
    "sample_steps": 4,
    "demos": 64,
    "steps": 80,
    "lr": 1e-3,
    "batch": 32,
    "horizon": 8,
    "n_exec": 8,
    "expert_noise": 0.1,
    "eval_every": 80,
}


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "flow", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("flow") / "run", TINY)
    t2.train(ctx)
    return ctx


def test_task2_registered_and_estimate():
    spec = registry.get("flow")
    assert spec.viewer == "flow"
    assert spec.estimate(registry.validate_params(spec, {}, which="train")) > 0


def test_task2_train_tiny(trained):
    ctx = trained
    assert read_status(ctx.dir)["state"] == "done"
    keys = {k for r in read_metrics(ctx.dir) for k in r}
    assert {"t1_loss", "t1_val_loss", "t2_loss", "t2_success", "t1_collision"} <= keys
    ck = torch.load(ctx.dir / "ckpt" / "last.pt", weights_only=False)
    assert ck["gen"] == "flow" and ck["t1"]["arch"]["horizon"] == 1
    assert ck["t2"]["arch"]["horizon"] == 8


def test_task2_eval_sweep_keys(tmp_path, trained):
    tctx = trained
    ev = make_ctx(
        tmp_path / "eval",
        {"ckpt": str(tctx.dir / "ckpt" / "last.pt"), "n_episodes": 32, "eval_steps": 4},
    )
    t2.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert list(res["sweep"]["euler"]) == ["1", "2", "4", "8", "16", "32"]
    assert all(0 <= v <= 1 for v in res["sweep"]["euler"].values())
    assert res["head"] == "flow" and res["sample_steps"] == 4
    tr = res["trace"]
    assert len(tr["xs"]) == 4 + 1 and len(tr["xs"][0]) == len(tr["c"]) <= 64
    assert len(tr["xs"][0][0]) == 2
    assert set(res["t1"]) >= {"collision", "points"} and "episodes" in res["t2"]


def test_task2_ddpm_head_runs(tmp_path):
    ctx = make_ctx(tmp_path / "run", {**TINY, "head": "ddpm", "steps": 40, "eval_every": 40})
    t2.train(ctx)
    assert read_status(ctx.dir)["state"] == "done"
    ck = torch.load(ctx.dir / "ckpt" / "last.pt", weights_only=False)
    assert ck["gen"] == "ddpm"
    ev = make_ctx(tmp_path / "eval", {"ckpt": str(ctx.dir / "ckpt" / "last.pt"), "n_episodes": 16})
    t2.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["head"] == "ddpm" and len(res["sweep"]["euler"]) == 6


def test_task2_stop_is_stopped(tmp_path):
    ctx = make_ctx(tmp_path / "run", {**TINY, "steps": 40, "eval_every": 40})
    real = ctx.should_continue
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > 50:
            ctx._stop = True
        return real()

    ctx.should_continue = flaky
    t2.train(ctx)
    assert read_status(ctx.dir)["state"] == "stopped"
