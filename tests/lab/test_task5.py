"""Task E1 (ta-vla): TAVLAMini with the torque token on the 2-D contact T4 world."""

import json

import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task5_tavla as t5

TINY = {
    "demos": 60, "steps": 80, "d": 64, "batch": 16, "lr": 1e-3, "eval_every": 80,
    "torque_pos": "suffix", "zero_torque": False, "aux_weight": 0.1,
}  # fmt: skip


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "ta-vla", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("t5") / "run", TINY)
    t5.train(ctx)
    return ctx


def test_task5_registered_and_estimate():
    spec = registry.get("ta-vla")
    assert spec.order == 10 and spec.viewer == "tavla" and spec.lesson == "ta-vla.html"
    p = registry.validate_params(spec, {}, which="train")
    assert p["torque_pos"] == "suffix" and p["d"] == 128 and p["demos"] == 800
    assert spec.estimate(p) > 0


def test_task5_train_tiny(trained):
    assert read_status(trained.dir)["state"] == "done"
    keys = {k for r in read_metrics(trained.dir) for k in r}
    assert {"loss", "val_loss", "aux", "lr", "success", "c0", "c1", "c2", "c3"} <= keys
    ck = torch.load(trained.dir / "ckpt" / "last.pt", weights_only=False)
    a = ck["arch"]
    assert a["torque_pos"] == "suffix" and a["zero_torque"] is False and a["aux_dim"] == 16
    assert a["image_hw"] == 32 and a["d"] == 64 and a["torque_dim"] == 20 and a["vocab"] > 0
    assert ck["gen"] == "flow" and {"model", "ema", "norm"} <= set(ck)


@pytest.mark.parametrize("pos", ["none", "prefix", "suffix"])
def test_task5_torque_positions(tmp_path, pos):
    ctx = make_ctx(tmp_path / "run", {**TINY, "steps": 4, "demos": 24, "torque_pos": pos,
                                      "eval_every": 1000, "aux_weight": 0.0})  # fmt: skip
    t5.train(ctx)
    assert read_status(ctx.dir)["state"] == "done"
    ck = torch.load(ctx.dir / "ckpt" / "last.pt", weights_only=False)
    assert ck["arch"]["torque_pos"] == pos and ck["arch"]["aux_dim"] == 0
    m = t5.build(ck["arch"])
    assert (m.torque_pos == pos) and (hasattr(m, "torque_mlp") == (pos != "none"))


def test_task5_eval_per_task_breakdown(tmp_path, trained):
    ev = make_ctx(tmp_path / "eval",
                  {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "n_episodes": 24,
                   "eval_steps": 2})  # fmt: skip
    t5.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert len(res["commands"]) == 4 and sum(c["n"] for c in res["commands"]) == 24
    assert res["trained_with_force"] is True and res["force_zeroed"] is False
    for e in res["episodes"]:
        assert isinstance(e["task"], str) and len(e["holes"]) == 2
        assert len(e["act"]) == len(e["pos"]) - 1


def test_task5_eval_blind_and_filter(tmp_path, trained):
    ev = make_ctx(tmp_path / "eval",
                  {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "n_episodes": 12,
                   "eval_steps": 2, "tasks": "press", "blind": True})  # fmt: skip
    t5.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["force_zeroed"] is True and res["trained_with_force"] is True
    assert all(e["task"].startswith("press") for e in res["episodes"])
    assert res["commands"][0]["n"] == 0
