"""Task 6 (td3): TD3 from scratch on the arm hidden-hole task - tiny train, stop, eval + video."""

import json

import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task6_td3 as t6

TINY = {"env_steps": 400, "seed": 0, "expl_noise": 0.2, "force_penalty": 0.05, "time_cost": 0.1,
        "shaping_weight": 2.0, "use_force": True, "eval_every": 200,
        "eval_episodes": 2}  # fmt: skip


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "td3", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("t6") / "run", TINY)
    t6.train(ctx)
    return ctx


def test_task6_registered_and_estimate():
    spec = registry.get("td3")
    assert spec.order == 11 and spec.viewer == "video" and spec.lesson == "td3.html"
    assert next(p for p in spec.params if p.name == "env_steps").long is True
    assert next(p for p in spec.params if p.name == "env_steps").default == 100_000
    assert next(p for p in spec.params if p.name == "eval_every").default == 10_000
    full = spec.estimate(registry.validate_params(spec, {}, which="train"))
    small = spec.estimate(registry.validate_params(spec, {"env_steps": 20_000}, which="train"))
    assert full > small > 0


def test_task6_train_tiny(trained):
    assert read_status(trained.dir)["state"] == "done"
    rows = read_metrics(trained.dir)
    steps = [r["step"] for r in rows]
    assert steps == sorted(steps) and steps[-1] <= 400
    keys = {k for r in rows for k in r}
    assert {"ep_return", "ep_return_ma", "eval_success", "eval_return"} <= keys
    evals = [r for r in rows if "eval_success" in r]
    assert [r["step"] for r in evals] == [200, 400]
    # losses are throttled: one flush per LOSS_EVERY updates, plus a final partial flush
    assert {"critic_loss", "q1_mean", "target_mean"} <= keys
    assert len([r for r in rows if "critic_loss" in r]) <= 400 // t6.LOSS_EVERY + 2
    ck = torch.load(trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert {"actor", "critic", "cfg", "obs_dim", "act_dim", "meta"} <= set(ck)
    m = ck["meta"]
    assert m["algo"] == "td3" and m["use_force"] is True and m["history"] == 4
    assert m["t"] == 400 and (t6.NORM_PATH).exists()
    labels = [e["label"] for e in json.loads((trained.dir / "ckpt" / "index.json").read_text())]
    assert "best" in labels or "last" in labels
    assert len(labels) <= 2  # best + last only


def test_task6_stop_saves_usable_ckpt(tmp_path):
    ctx = make_ctx(tmp_path / "run", {**TINY, "env_steps": 100_000, "eval_every": 50_000})
    real = ctx.should_continue
    calls = {"n": 0}

    def stop_after_a_while():
        calls["n"] += 1
        return real() and calls["n"] < 30

    ctx.should_continue = stop_after_a_while
    t6.train(ctx)
    assert read_status(ctx.dir)["state"] == "stopped"
    from fvb.policy.td3 import TD3Policy

    pol = TD3Policy(str(ctx.dir / "ckpt" / "last.pt"), "cpu")
    assert pol.use_force is True


def test_task6_eval_and_video(tmp_path, trained):
    ev = make_ctx(tmp_path / "eval", {"ckpt": str(trained.dir / "ckpt" / "last.pt"),
                                      "n_episodes": 2, "render_n": 1, "eval_seed": 0})  # fmt: skip
    t6.evaluate(ev)
    assert read_status(ev.dir)["state"] == "done"
    assert (ev.dir / "rollout.mp4").stat().st_size > 1000
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["video"] == "rollout.mp4" and res["trained_with_force"] is True
    assert res["force_zeroed"] is False and res["offset_sigma_mm"] == 1.5
    assert sum(res["counts"].values()) == 2 and len(res["wilson95"]) == 2
    assert len(res["episodes"]) == 1
    e = res["episodes"][0]
    assert len(e["force"]) == len(e["act"]) == len(e["depth_mm"]) and len(e["force"][0]) == 6
