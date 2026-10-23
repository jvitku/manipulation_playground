"""Task 4 (force-token): Stage-1 gantry pipeline wrapped - tiny train, video eval, ckpt rebuild."""

import json

import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.core.video import frames_to_mp4
from lab.tasks import registry
from lab.tasks import task4_force_token as t4

TINY = {"demos": 5, "model": "act", "use_force": True, "epochs": 2, "lr": 3e-4, "H": 10, "K": 5,
        "noise_mm": 0.5}  # fmt: skip


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "force-token", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("t4") / "run", TINY)
    t4.train(ctx)
    return ctx


def test_task4_registered_and_estimate():
    spec = registry.get("force-token")
    assert spec.order == 5 and spec.viewer == "video"
    assert spec.estimate(registry.validate_params(spec, {}, which="train")) > 0
    assert spec.lesson == "force-token.html"


def test_frames_to_mp4(tmp_path):
    import numpy as np

    p = tmp_path / "a.mp4"
    frames_to_mp4((np.full((32, 48, 3), i * 20, np.uint8) for i in range(10)), p, fps=10)
    assert p.stat().st_size > 100


def test_task4_train_tiny(trained):
    assert read_status(trained.dir)["state"] == "done"
    rows = read_metrics(trained.dir)
    steps = [r["step"] for r in rows]
    assert steps == sorted(steps)
    assert [r["collected"] for r in rows if "collected" in r] == list(range(1, 6))
    assert {"train_l1", "val_l1", "val_mae_mm"} <= {k for r in rows for k in r}
    assert len([r for r in rows if "val_l1" in r]) == 2
    ck = torch.load(trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert {"state_dict", "norm", "meta"} <= set(ck)
    m = ck["meta"]
    assert m["model"] == "act" and m["H"] == 10 and m["K"] == 5 and m["use_force"] is True
    assert m["obs_dim"] == 18 and m["act_dim"] == 3


def test_task4_stop_is_stopped(tmp_path):
    ctx = make_ctx(tmp_path / "run", TINY)
    ctx._stop = True
    t4.train(ctx)
    assert read_status(ctx.dir)["state"] == "stopped"


def test_task4_eval_video_written(tmp_path, trained):
    ev = make_ctx(tmp_path / "eval", {"ckpt": str(trained.dir / "ckpt" / "last.pt"),
                                      "n_episodes": 2, "render_n": 1, "seed": 0})  # fmt: skip
    t4.evaluate(ev)
    assert read_status(ev.dir)["state"] == "done"
    assert (ev.dir / "rollout.mp4").stat().st_size > 1000
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["video"] == "rollout.mp4" and res["trained_with_force"] is True
    assert res["force_zeroed"] is False
    assert sum(res["counts"].values()) == 2 and len(res["wilson95"]) == 2
    assert len(res["episodes"]) == 1
    e = res["episodes"][0]
    assert len(e["force"]) == len(e["act"]) and len(e["force"][0]) == 6
    assert len(e["obs_force"][0]) == 9 and e["outcome"] in res["counts"]
    assert sum(b["n"] for b in res["buckets"]) == 2


def test_task4_ckpt_rebuild(tmp_path, trained):
    """Eval rebuilds the architecture from ckpt meta only: form-style params cannot change it."""
    ev = make_ctx(tmp_path / "eval", {"ckpt": str(trained.dir / "ckpt" / "last.pt"),
                                      "n_episodes": 1, "render_n": 0, "model": "mlp", "H": 3,
                                      "K": 2, "use_force": False})  # fmt: skip
    t4.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert (res["model"], res["H"], res["K"]) == ("act", 10, 5)
    assert res["trained_with_force"] is True and res["force_zeroed"] is False
    assert res["video"] is None and res["episodes"] == []


def test_task4_force_zeroed_twin(tmp_path, trained):
    ev = make_ctx(tmp_path / "eval", {"ckpt": str(trained.dir / "ckpt" / "last.pt"),
                                      "n_episodes": 1, "render_n": 0,
                                      "force_zeroed": True})  # fmt: skip
    t4.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["force_zeroed"] is True and res["trained_with_force"] is True


def test_task4_default_noise_is_zero():
    spec = registry.get("force-token")
    assert registry.validate_params(spec, {}, which="train")["noise_mm"] == 0.0


def test_task4_buckets_open_ended_at_large_sigma(tmp_path, trained):
    ev = make_ctx(tmp_path / "eval", {"ckpt": str(trained.dir / "ckpt" / "last.pt"),
                                      "n_episodes": 12, "render_n": 0,
                                      "offset_sigma_mm": 3.0})  # fmt: skip
    t4.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert sum(b["n"] for b in res["buckets"]) == 12
    assert res["buckets"][-1]["hi"] is None


def test_task4_render_failure_keeps_stats(tmp_path, trained, monkeypatch):
    def boom(frames, path, fps=20):
        next(iter(frames))  # start rendering, then fail mid-way
        (tmp_path / "x").write_text("")
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(t4, "frames_to_mp4", boom)
    ev = make_ctx(tmp_path / "eval", {"ckpt": str(trained.dir / "ckpt" / "last.pt"),
                                      "n_episodes": 3, "render_n": 2})  # fmt: skip
    t4.evaluate(ev)
    assert read_status(ev.dir)["state"] == "done"
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["video"] is None and sum(res["counts"].values()) == 3
    assert len(res["episodes"]) == 2 and all("force" not in e for e in res["episodes"])
    assert not (ev.dir / "rollout.mp4").exists()


def test_frames_to_mp4_error_removes_partial(tmp_path):
    import numpy as np

    def gen():
        yield np.zeros((32, 48, 3), np.uint8)
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        frames_to_mp4(gen(), tmp_path / "a.mp4")
    assert list(tmp_path.iterdir()) == []


def test_task4_stop_mid_epochs_keeps_usable_ckpt(tmp_path):
    ctx = make_ctx(tmp_path / "run", {**TINY, "epochs": 5})
    real = ctx.should_continue
    from lab.core.runctx import POLL_S  # noqa: F401

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > TINY["demos"] + 2:  # collection ok, epoch 0 and 1 run, then stop
            ctx._stop = True
        return real()

    ctx.should_continue = flaky
    t4.train(ctx)
    assert read_status(ctx.dir)["state"] == "stopped"
    ck = torch.load(ctx.dir / "ckpt" / "last.pt", weights_only=False)
    assert ck["epoch"] >= 1 and ck["meta"]["model"] == "act"
