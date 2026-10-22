"""Job runner: liveness, status reading, run_job error handling, spawning."""

import json
import re
import time
from types import SimpleNamespace

from lab import run_job
from lab.core.jobs import new_job_dir, read_status, send_control, start_job
from lab.tasks import registry


def write_config(job_dir, task, params=None):
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "config.json").write_text(json.dumps({"task": task, "params": params or {}}))


def test_stale_pid_is_failed(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "running", "pid": 2**22 + 12345}))
    st = read_status(tmp_path)
    assert st["state"] == "failed" and "died" in st["msg"]


def test_read_status_missing(tmp_path):
    assert read_status(tmp_path)["state"] == "pending"


def test_new_job_dir_and_control(tmp_path):
    d = new_job_dir(tmp_path, "sort")
    assert d.parent == tmp_path / "sort"
    assert re.fullmatch(r"\d{6}-\d{6}-[0-9a-f]{4}", d.name)
    d.mkdir(parents=True)
    send_control(d, "pause")
    assert json.loads((d / "control.json").read_text()) == {"cmd": "pause"}


def test_runner_exception_sets_failed(tmp_path):
    def boom(ctx):
        raise ValueError("kaboom")

    registry.register(SimpleNamespace(id="boom", train=boom, evaluate=boom))
    write_config(tmp_path, "boom")
    run_job.main(["train", str(tmp_path)])
    st = read_status(tmp_path)
    assert st["state"] == "failed" and "kaboom" in st["msg"]


def test_runner_normal_return_is_done(tmp_path):
    def ok(ctx):
        ctx.log(0, loss=1.0)

    registry.register(SimpleNamespace(id="ok", train=ok, evaluate=ok))
    write_config(tmp_path, "ok")
    run_job.main(["eval", str(tmp_path)])
    assert read_status(tmp_path)["state"] == "done"


def test_start_job_spawns_detached(tmp_path):
    # unknown task -> subprocess fails fast; proves spawn, pid, log and failure path
    write_config(tmp_path, "no_such_task")
    pid = start_job("train", tmp_path)
    assert pid > 0 and int((tmp_path / "job.pid").read_text()) == pid
    for _ in range(100):
        if read_status(tmp_path)["state"] == "failed":
            break
        time.sleep(0.1)
    assert read_status(tmp_path)["state"] == "failed"
    assert (tmp_path / "job.log").exists()


def test_start_job_never_clobbers_child_state(tmp_path):
    # child finishes (done) before/while the parent records its pid: terminal state must survive
    def ok(ctx):
        pass

    registry.register(SimpleNamespace(id="ok", train=ok, evaluate=ok))
    write_config(tmp_path, "ok")
    run_job.main(["train", str(tmp_path)])
    (tmp_path / "job.pid").write_text("12345")  # what a late parent write looks like
    assert read_status(tmp_path)["state"] == "done"


def test_pid_file_gives_liveness_before_child_writes_status(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "running"}))
    (tmp_path / "job.pid").write_text(str(2**22 + 12345))
    assert read_status(tmp_path)["state"] == "failed"


def test_atomic_write_concurrent_writers(tmp_path):
    import threading

    from lab.core.runctx import atomic_write_json

    errs = []

    def w(i):
        try:
            for _ in range(50):
                atomic_write_json(tmp_path / "c.json", {"i": i})
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and "i" in json.loads((tmp_path / "c.json").read_text())
    assert [p.name for p in tmp_path.iterdir()] == ["c.json"]
