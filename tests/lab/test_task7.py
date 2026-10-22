"""Task 7 (td3-bc): TD3 + demos + decaying BC weight on top of task 6's loop."""

import json

import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task7_td3bc as t7

TINY = {"env_steps": 400, "seed": 0, "expl_noise": 0.2, "force_penalty": 0.05, "time_cost": 0.1,
        "shaping_weight": 2.0, "use_force": True, "eval_every": 200, "eval_episodes": 2,
        "n_demos": 2, "bc_weight_start": 2.5, "bc_weight_floor": 1.0,
        "bc_decay_steps": 300}  # fmt: skip


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "td3-bc", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("t7") / "run", TINY)
    t7.train(ctx)
    return ctx


def test_task7_registered_and_estimate():
    spec = registry.get("td3-bc")
    assert spec.order == 8 and spec.viewer == "video" and spec.lesson == "td3-bc.html"
    d = {p.name: p.default for p in spec.params}
    assert d["n_demos"] == 100 and d["bc_weight_start"] == 2.5 and d["bc_weight_floor"] == 1.0
    assert "bc" in [c["id"] for c in spec.charts]
    base = spec.estimate(registry.validate_params(spec, {}, which="train"))
    more = spec.estimate(registry.validate_params(spec, {"n_demos": 300}, which="train"))
    assert more > base > 0


def test_bc_weight_formula():
    assert t7.bc_weight_at(0, 2.5, 1.0, 100) == 2.5
    assert t7.bc_weight_at(50, 2.5, 1.0, 100) == pytest.approx(1.75)
    assert t7.bc_weight_at(100, 2.5, 1.0, 100) == 1.0 == t7.bc_weight_at(10**6, 2.5, 1.0, 100)


def test_task7_train_tiny(trained):
    assert read_status(trained.dir)["state"] == "done"
    rows = read_metrics(trained.dir)
    steps = [r["step"] for r in rows]
    assert steps == sorted(steps) and steps[-1] <= 400
    keys = {k for r in rows for k in r}
    assert {"demos_collected", "ep_return", "eval_success", "critic_loss", "bc_loss"} <= keys
    assert [r["demos_collected"] for r in rows if "demos_collected" in r] == [1, 2]
    assert [r["step"] for r in rows if "eval_success" in r] == [200, 400]
    ck = torch.load(trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert ck["meta"]["bc"]["n_demos"] == 2 and ck["meta"]["t"] == 400


def test_task7_bc_weight_decays(trained):
    ws = [r["bc_weight"] for r in read_metrics(trained.dir) if "bc_weight" in r]
    assert len(ws) >= 2
    assert ws == sorted(ws, reverse=True) and ws[0] < 2.5 and ws[0] > ws[-1]
    assert ws[-1] == pytest.approx(1.0, abs=0.05)  # reached the floor, never below


def test_task7_eval_reuses_task6(tmp_path, trained):
    ev = make_ctx(tmp_path / "eval", {"ckpt": str(trained.dir / "ckpt" / "last.pt"),
                                      "n_episodes": 2, "render_n": 0, "eval_seed": 0})  # fmt: skip
    registry.get("td3-bc").evaluate(ev)
    assert read_status(ev.dir)["state"] == "done"
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["trained_with_force"] is True and sum(res["counts"].values()) == 2
