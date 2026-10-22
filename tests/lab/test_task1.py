"""Task 1 (bc-mse): tiny train + eval-from-checkpoint-arch + estimate."""

import json

import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task1_bc_mse as t1

TINY = {
    "demos": 64,
    "steps": 60,
    "lr": 1e-3,
    "batch": 32,
    "horizon": 8,
    "expert_noise": 0.1,
    "eval_every": 30,
}


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "bc-mse", "params": params}))
    return RunContext(d)


def test_task1_train_tiny(tmp_path):
    ctx = make_ctx(tmp_path / "run", TINY)
    t1.train(ctx)
    assert read_status(ctx.dir)["state"] == "done"
    rows = read_metrics(ctx.dir)
    keys = {k for r in rows for k in r}
    assert {"t1_loss", "t1_val_loss", "t2_loss", "t2_success", "t1_collision"} <= keys
    ck = torch.load(ctx.dir / "ckpt" / "last.pt", weights_only=False)
    assert "arch" in ck and "gen" in ck
    assert ck["t2"]["arch"]["horizon"] == 8 and ck["t1"]["arch"]["horizon"] == 1


def test_task1_eval_uses_ckpt_arch(tmp_path):
    tctx = make_ctx(tmp_path / "run", TINY)
    t1.train(tctx)
    ev = make_ctx(
        tmp_path / "eval",
        {"ckpt": str(tctx.dir / "ckpt" / "last.pt"), "n_episodes": 50, "horizon": 1},
    )
    t1.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert set(res) == {"t1", "t2"}
    for k in ("collision", "left_frac"):
        assert 0 <= res["t1"][k] <= 1
    assert len(res["t1"]["points"]) <= 500 and len(res["t1"]["points"][0]) == 3
    for k in ("success", "collision", "timeout"):
        assert 0 <= res["t2"][k] <= 1
    lo, hi = res["t2"]["wilson95"]
    assert 0 <= lo <= hi <= 1
    eps = res["t2"]["episodes"]
    assert 0 < len(eps) <= 24 and eps[0]["outcome"] in ("success", "collision", "timeout")


def test_task1_estimate_positive():
    spec = registry.get("bc-mse")
    assert spec.estimate(registry.validate_params(spec, {}, which="train")) > 0


def test_task1_metrics_steps_monotonic_across_phases(tmp_path):
    ctx = make_ctx(tmp_path / "run", TINY)
    t1.train(ctx)
    steps = [r["step"] for r in read_metrics(ctx.dir)]
    assert steps == sorted(steps) and max(steps) == 2 * TINY["steps"]
    assert any(s > TINY["steps"] for s in steps)


def test_task1_stop_during_t2_is_stopped_not_done(tmp_path):
    ctx = make_ctx(tmp_path / "run", TINY)
    real = ctx.should_continue
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > TINY["steps"] + 10:  # inside the T2 phase
            ctx._stop = True
        return real()

    ctx.should_continue = flaky
    t1.train(ctx)
    assert read_status(ctx.dir)["state"] == "stopped"


def test_task1_eval_fewer_episodes_than_view_cap(tmp_path):
    tctx = make_ctx(tmp_path / "run", TINY)
    t1.train(tctx)
    ev = make_ctx(tmp_path / "eval", {"ckpt": str(tctx.dir / "ckpt" / "last.pt"), "n_episodes": 10})
    t1.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert 0 < len(res["t2"]["episodes"]) <= 10
