"""Task 3 (flow-bc): tiny train, episode payload, zeroed-force twin."""

import json

import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task3_flow_bc as t3

TINY = {
    "demos": 40, "steps": 100, "lr": 1e-3, "batch": 32, "horizon": 16, "n_exec": 8,
    "use_force": True, "sample_steps": 2, "eval_every": 50,
}  # fmt: skip
EPISODE_KEYS = {"z0", "holes", "target", "est", "pos", "force", "outcome"}


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "flow-bc", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("t3") / "run", TINY)
    t3.train(ctx)
    return ctx


def test_task3_registered_and_estimate():
    spec = registry.get("flow-bc")
    assert spec.viewer == "contact2d"
    assert spec.estimate(registry.validate_params(spec, {}, which="train")) > 0
    assert "eval_every" in {p.name for p in spec.params}


def test_task3_train_tiny(trained):
    assert read_status(trained.dir)["state"] == "done"
    keys = {k for r in read_metrics(trained.dir) for k in r}
    assert {"loss", "val_loss", "success"} <= keys
    ck = torch.load(trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert ck["arch"]["obs_dim"] == 22 and ck["gen"] == "flow"


def test_task3_eval_episode_payload(tmp_path, trained):
    ev = make_ctx(
        tmp_path / "eval",
        {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "n_episodes": 20, "eval_steps": 2},
    )
    t3.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert 0 < len(res["episodes"]) <= 16
    for e in res["episodes"]:
        assert set(e) == EPISODE_KEYS
        assert len(e["pos"]) == len(e["force"]) >= 1 and len(e["pos"][0]) == 2
    assert res["force_zeroed"] is False and res["trained_with_force"] is True
    assert sum(res["counts"].values()) == 20
    assert len(res["wilson95"]) == 2


def test_task3_force_zeroed_twin(tmp_path, trained):
    ev = make_ctx(
        tmp_path / "eval",
        {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "n_episodes": 10, "force_zeroed": True,
         "eval_steps": 2},
    )  # fmt: skip
    t3.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["force_zeroed"] is True


def test_task3_trained_without_force_flags(tmp_path):
    tr = make_ctx(tmp_path / "run", {**TINY, "use_force": False, "steps": 50, "eval_every": 50})
    t3.train(tr)
    ev = make_ctx(tmp_path / "eval", {"ckpt": str(tr.dir / "ckpt" / "last.pt"), "n_episodes": 8,
                                      "eval_steps": 2})  # fmt: skip
    t3.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["force_zeroed"] is True and res["trained_with_force"] is False
