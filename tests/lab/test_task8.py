"""Task 13 (residual-rl): ckpt param validation, tiny base -> tiny residual run, standalone ckpt,
residual_off evaluation, base/delta in the episode payload."""

import json
import shutil

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task3_flow_bc as t3
from lab.tasks import task8_residual as t8

BASE = {
    "demos": 40, "steps": 100, "lr": 1e-3, "batch": 32, "horizon": 16, "n_exec": 8,
    "use_force": True, "sample_steps": 2, "eval_every": 100,
}  # fmt: skip
TINY = {
    "env_steps": 400, "residual_scale": 0.3, "seed": 0, "expl_noise": 0.1, "force_penalty": 0.05,
    "time_cost": 0.01, "shaping_weight": 1.0, "eval_every": 150, "eval_episodes": 20,
}  # fmt: skip


def make_ctx(d, params, task="residual-rl"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": task, "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("t8")
    base = make_ctx(root / "base", BASE, "flow-bc")
    t3.train(base)
    res = make_ctx(root / "res", {**TINY, "base_ckpt": str(base.dir / "ckpt" / "last.pt")})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(t8, "N_ENVS", 2)  # few scenes, so whole episodes finish inside 400 transitions
        t8.train(res)
    return base, res


def test_task8_registered_and_estimate():
    spec = registry.get("residual-rl")
    assert spec.order == 13 and spec.viewer == "contact2d" and spec.lesson == "residual-rl.html"
    p = next(p for p in spec.params if p.name == "base_ckpt")
    assert p.kind == "ckpt" and p.ckpt_task == "flow-bc"
    steps = next(p for p in spec.params if p.name == "env_steps")
    assert steps.default == 60_000 and steps.long is True
    scale = next(p for p in spec.params if p.name == "residual_scale")
    assert (scale.min, scale.max, scale.default) == (0.05, 1.0, 0.3)
    form = registry.validate_params(spec, {}, which="train")
    assert form["base_ckpt"] == ""  # estimate must work before a base is picked
    assert spec.estimate(form) > spec.estimate({**form, "env_steps": 2000}) > 0


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_DATA", str(tmp_path))
    from lab.server import app as srv

    started = []
    monkeypatch.setattr(srv.jobs, "start_job", lambda kind, d: started.append(d))
    c = TestClient(srv.create_app())
    c.started = started
    return c


def _fake_ckpt(root, task, run="r1", name="last.pt"):
    d = root / "runs" / task / run / "ckpt"
    d.mkdir(parents=True)
    (d / name).write_bytes(b"x")
    return f"{run}/{name}"


BAD = [None, "", "nope/last.pt", "../x/last.pt", "r1/../../x", "td3/r1/last.pt"]


@pytest.mark.parametrize("ckpt", BAD)
def test_task8_requires_base_ckpt(client, ckpt):
    params = {} if ckpt is None else {"base_ckpt": ckpt}
    r = client.post("/api/tasks/residual-rl/runs", json={"params": params})
    assert r.status_code == 422, r.text
    assert not client.started


def test_task8_foreign_ckpt_rejected(client, tmp_path):
    cid = _fake_ckpt(tmp_path, "td3")  # exists, but under another task
    r = client.post("/api/tasks/residual-rl/runs", json={"params": {"base_ckpt": cid}})
    assert r.status_code == 422 and "flow-bc" in r.text


def test_task8_valid_ckpt_resolved_to_path(client, tmp_path):
    cid = _fake_ckpt(tmp_path, "flow-bc")
    r = client.post("/api/tasks/residual-rl/runs", json={"params": {"base_ckpt": cid}})
    assert r.status_code == 200, r.text
    cfg = json.loads((client.started[0] / "config.json").read_text())
    assert cfg["params"]["base_ckpt"] == str((tmp_path / "runs/flow-bc/r1/ckpt/last.pt").resolve())


def test_task8_trainer_fails_on_unusable_base(tmp_path):
    ctx = make_ctx(tmp_path / "r", {**TINY, "base_ckpt": str(tmp_path / "missing.pt")})
    t8.train(ctx)
    assert read_status(ctx.dir)["state"] == "failed"
    torch.save({"model": 1}, tmp_path / "bad.pt")
    ctx = make_ctx(tmp_path / "r2", {**TINY, "base_ckpt": str(tmp_path / "bad.pt")})
    t8.train(ctx)
    assert read_status(ctx.dir)["state"] == "failed"


def test_task8_train_tiny(runs):
    _, res = runs
    assert read_status(res.dir)["state"] == "done"
    rows = read_metrics(res.dir)
    assert rows[0]["step"] == 0 and "base_success" in rows[0]
    evals = [r for r in rows if "eval_success" in r]
    assert len(evals) >= 2 and evals[-1]["step"] >= 400
    assert {r["base_success"] for r in rows if "base_success" in r} == {rows[0]["base_success"]}
    keys = {k for r in rows for k in r}
    assert {"critic_loss", "q1_mean", "eval_success", "ep_return_ma", "train_success_ma"} <= keys
    ck = torch.load(res.dir / "ckpt" / "last.pt", weights_only=False)
    assert {"actor", "critic", "base", "residual_scale"} <= set(ck)
    assert ck["obs_dim"] == 24 and ck["meta"]["base_success"] == rows[0]["base_success"]


def test_task8_ckpt_is_standalone(runs, tmp_path):
    base, res = runs
    shutil.copytree(res.dir / "ckpt", tmp_path / "ckpt")
    shutil.rmtree(base.dir)  # the task-3 run is gone
    ev = make_ctx(tmp_path / "ev", {"ckpt": str(tmp_path / "ckpt" / "last.pt"), "n_episodes": 10,
                                    "eval_steps": 2}, "residual-rl")  # fmt: skip
    t8.evaluate(ev)
    assert read_status(ev.dir)["state"] == "done"


def test_task8_eval_payload_and_residual_off(runs, tmp_path):
    _, res = runs
    out = {}
    for off in (False, True):
        ev = make_ctx(tmp_path / f"ev{off}", {"ckpt": str(res.dir / "ckpt" / "last.pt"),
                                              "n_episodes": 20, "eval_steps": 2,
                                              "residual_off": off}, "residual-rl")  # fmt: skip
        t8.evaluate(ev)
        out[off] = json.loads((ev.dir / "result.json").read_text())
    r = out[False]
    assert sum(r["counts"].values()) == 20 and len(r["wilson95"]) == 2
    assert r["residual_off"] is False and r["base_success"] is not None
    for e in r["episodes"]:
        n = len(e["pos"]) - 1
        assert len(e["act"]) == len(e["base_act"]) == len(e["delta"]) == n
        summed = np.clip(np.array(e["base_act"]) + np.array(e["delta"]), -1, 1)
        assert np.allclose(summed, e["act"], atol=2.5e-3)
    off = out[True]
    assert off["residual_off"] is True
    for e in off["episodes"]:
        assert e["act"] == e["base_act"] and not np.any(np.array(e["delta"]))


def test_task8_lesson_served_with_schematic():
    from lab.server.app import create_app

    r = TestClient(create_app()).get("/lessons/residual-rl.html")
    assert r.status_code == 200 and "<svg" in r.text and "2409.00588" in r.text
    assert "2502.02538" in r.text and "<html" not in r.text.lower()
