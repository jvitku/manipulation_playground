"""RunContext: metrics, pause/resume/stop control, checkpoints."""

import json
import threading
import time

import torch

from lab.core.jobs import read_metrics, read_status, send_control
from lab.core.runctx import RunContext


def make_ctx(tmp_path, params=None):
    (tmp_path / "config.json").write_text(json.dumps({"task": "dummy", "params": params or {}}))
    return RunContext(tmp_path)


def wait_for(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_log_and_read_metrics_roundtrip(tmp_path):
    ctx = make_ctx(tmp_path, {"lr": 1e-3})
    assert ctx.params == {"lr": 1e-3} and ctx.task_id == "dummy"
    for i in range(3):
        ctx.log(i, loss=1.0 / (i + 1))
    rows = read_metrics(tmp_path)
    assert [r["step"] for r in rows] == [0, 1, 2]
    assert len(read_metrics(tmp_path, since=2)) == 1


def test_read_metrics_skips_partial_line(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.log(1, loss=1.0)
    ctx.log(2, loss=0.5)
    path = tmp_path / "metrics.jsonl"
    with path.open("a") as f:
        f.write('{"step": 4, "loss"')
    assert len(read_metrics(tmp_path)) == 2
    with path.open("a") as f:
        f.write(": 0.1}\n")
    rows = read_metrics(tmp_path, since=2)
    assert rows == [{"step": 4, "loss": 0.1}]


def test_pause_resume_stop(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.set_status(state="running")
    iters = []
    result = {}

    def loop():
        while ctx.should_continue():
            iters.append(1)
            time.sleep(0.02)
        result["stopped"] = True

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    assert wait_for(lambda: len(iters) > 2)
    send_control(tmp_path, "pause")
    assert wait_for(lambda: read_status(tmp_path)["state"] == "paused")
    n = len(iters)
    time.sleep(0.3)
    assert len(iters) <= n + 1
    send_control(tmp_path, "run")
    assert wait_for(lambda: len(iters) > n + 3)
    assert read_status(tmp_path)["state"] == "running"
    send_control(tmp_path, "stop")
    t.join(5)
    assert result.get("stopped")


def test_save_ckpt_index(tmp_path):
    ctx = make_ctx(tmp_path)
    p1 = ctx.save_ckpt(10, {"w": torch.zeros(2)})
    p2 = ctx.save_ckpt(20, {"w": torch.ones(2)}, label="best", metrics={"loss": 0.1})
    assert p1.name == "step_00000010.pt" and p2.name == "step_00000020.pt"
    index = json.loads((tmp_path / "ckpt" / "index.json").read_text())
    assert [e["step"] for e in index] == [10, 20]
    assert index[1]["label"] == "best" and index[1]["metrics"] == {"loss": 0.1}
    assert index[1]["file"] == "step_00000020.pt" and "created" in index[1]
    last = torch.load(tmp_path / "ckpt" / "last.pt", weights_only=True)
    assert torch.equal(last["w"], torch.ones(2))
