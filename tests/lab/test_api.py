import time

import pytest
from fastapi.testclient import TestClient

import tests.lab.dummy_task  # noqa: F401  (registers "dummy")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_DATA", str(tmp_path))
    monkeypatch.setenv("LAB_EXTRA_TASKS", "tests.lab.dummy_task")
    from lab.server.app import create_app

    return TestClient(create_app())


def _wait(client, url, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(url).json()
        if r["status"]["state"] in ("done", "failed", "stopped"):
            return r
        time.sleep(0.2)
    raise AssertionError("timeout")


def _start(client, **params):
    r = client.post("/api/tasks/dummy/runs", json={"params": params})
    assert r.status_code == 200, r.text
    return r.json()["run_id"]


def test_list_tasks(client):
    ids = [t["id"] for t in client.get("/api/tasks").json()]
    assert "dummy" in ids and "bc-mse" in ids
    t = next(t for t in client.get("/api/tasks").json() if t["id"] == "dummy")
    assert t["running"] is False and t["title"] == "Dummy"


def test_start_run_and_poll(client):
    rid = _start(client, steps=5, delay=0.05)
    r = _wait(client, f"/api/runs/dummy/{rid}")
    assert r["status"]["state"] == "done"
    assert len(r["metrics"]) == 5 and r["next"] == 5
    r2 = client.get(f"/api/runs/dummy/{rid}?since=3").json()
    assert len(r2["metrics"]) == 2 and r2["next"] == 5
    d = client.get("/api/tasks/dummy").json()
    assert d["runs"][0]["run_id"] == rid and d["runs"][0]["state"] == "done"
    assert d["checkpoints"][0]["id"].startswith(rid + "/")
    assert d["spec"]["lesson_url"] == "/lessons/dummy.html"
    assert d["spec"]["params"][0]["name"] == "steps"


def test_control_pause_resume(client):
    rid = _start(client, steps=400, delay=0.05)
    url = f"/api/runs/dummy/{rid}"
    assert client.post(url + "/control", json={"cmd": "pause"}).status_code == 204
    t0 = time.time()
    while client.get(url).json()["status"]["state"] != "paused":
        assert time.time() - t0 < 15
        time.sleep(0.2)
    assert client.get("/api/tasks").json()  # still serving
    client.post(url + "/control", json={"cmd": "run"})
    t0 = time.time()
    while client.get(url).json()["status"]["state"] != "running":
        assert time.time() - t0 < 15
        time.sleep(0.2)
    client.post(url + "/control", json={"cmd": "stop"})
    assert _wait(client, url)["status"]["state"] == "stopped"


def test_invalid_params_422(client):
    r = client.post("/api/tasks/dummy/runs", json={"params": {"bogus": 1}})
    assert r.status_code == 422 and "bogus" in r.text


def test_two_concurrent_runs_distinct_dirs(client):
    a = _start(client, steps=3)
    b = _start(client, steps=3)
    assert a != b
    for rid in (a, b):
        assert _wait(client, f"/api/runs/dummy/{rid}")["status"]["state"] == "done"


def test_estimate_label(client):
    r = client.post("/api/tasks/dummy/estimate", json={"params": {"steps": 500}}).json()
    assert r["seconds"] == 7000 and "overnight" in r["label"]
    r = client.post("/api/tasks/dummy/estimate", json={"params": {"steps": 10}}).json()
    assert r["label"] == "~2 min"
    r = client.post("/api/tasks/dummy/estimate", json={"params": {"steps": 250}}).json()
    assert "long run" in r["label"]
    r = client.post("/api/tasks/dummy/estimate", json={"params": {"steps": 1}}).json()
    assert r["label"] == "~14 s"  # below a minute: seconds
    r = client.post("/api/tasks/dummy/estimate", json={"params": {"steps": 3}}).json()
    assert r["label"] == "~42 s"


def test_list_has_est_label_and_last_state(client):
    row = next(t for t in client.get("/api/tasks").json() if t["id"] == "dummy")
    assert row["est_label"] == "~1 min"  # 5 steps * 14 s = 70 s
    assert row["last_state"] is None
    rid = _start(client, steps=3)
    _wait(client, f"/api/runs/dummy/{rid}")
    row = next(t for t in client.get("/api/tasks").json() if t["id"] == "dummy")
    assert row["last_state"] == "done" and not row["running"]


def test_favicon(client):
    for path in ("/favicon.ico", "/favicon.svg"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg+xml")
    assert "favicon.svg" in client.get("/").text


def test_reference_ckpts_allowlist(client, tmp_path, monkeypatch):
    from lab.server import app as appmod
    from lab.tasks import registry

    (tmp_path / "ref.pt").write_bytes(b"x")
    spec = registry.get("dummy")
    monkeypatch.setattr(appmod, "REPO", tmp_path)
    monkeypatch.setattr(
        spec, "reference_ckpts", {"a": ("ref.pt", "reference (test)"), "b": ("nope.pt", "x")}
    )
    cks = client.get("/api/tasks/dummy").json()["checkpoints"]
    assert [c["id"] for c in cks] == ["reference/a"] and cks[0][
        "reference"
    ]  # missing file not listed
    assert appmod._resolve_ckpt("dummy", "reference/a") == (tmp_path / "ref.pt").resolve()
    for bad in ("reference/b", "reference/../ref.pt", "reference/..", "reference/zzz"):
        r = client.post("/api/tasks/dummy/evals", json={"ckpt": bad})
        assert r.status_code == 404, bad


def test_eval_roundtrip(client):
    rid = _start(client, steps=3)
    _wait(client, f"/api/runs/dummy/{rid}")
    ck = client.get("/api/tasks/dummy").json()["checkpoints"][0]["id"]
    r = client.post("/api/tasks/dummy/evals", json={"ckpt": ck, "params": {"n": 2}})
    assert r.status_code == 200, r.text
    eid = r.json()["eval_id"]
    t0 = time.time()
    while True:
        e = client.get(f"/api/evals/dummy/{eid}").json()
        if e["result"] is not None:
            break
        assert time.time() - t0 < 30
        time.sleep(0.2)
    assert e["result"]["n"] == 2 and e["result"]["ckpt"].endswith(ck.split("/")[1])
    assert client.get(f"/api/evals/dummy/{eid}/file/result.json").status_code == 200
    assert client.get(f"/api/evals/dummy/{eid}/file/..%2Fx").status_code in (404, 422)
    bad = client.post("/api/tasks/dummy/evals", json={"ckpt": "nope/x.pt", "params": {}})
    assert bad.status_code == 404


def test_gpu_endpoint_shape(client):
    r = client.get("/api/gpu").json()
    assert set(r) == {"used_mb", "total_mb"}
    assert isinstance(r["used_mb"], int) and isinstance(r["total_mb"], int)


def _handler(client, path_suffix, method="GET"):
    """Route endpoint, called directly with a literal '..' (HTTP clients normalise it)."""
    for r in client.app.routes:
        if getattr(r, "path", "").endswith(path_suffix) and method in getattr(r, "methods", ()):
            return r.endpoint
    raise AssertionError(path_suffix)


def test_eval_ckpt_traversal_404(client):
    rid = _start(client, steps=2)
    _wait(client, f"/api/runs/dummy/{rid}")
    for bad in ("../x", "..", "../..", f"{rid}/..", f"{rid}/../index.json", "./x", "/etc/passwd"):
        r = client.post("/api/tasks/dummy/evals", json={"ckpt": bad, "params": {}})
        assert r.status_code == 404, bad


def test_run_id_dotdot_404(client):
    from fastapi import HTTPException

    h = _handler(client, "/api/runs/{task_id}/{run_id}")
    for bad in ("..", "."):
        with pytest.raises(HTTPException) as e:
            h("dummy", bad)
        assert e.value.status_code == 404


def test_eval_file_dotdot_404(client):
    from fastapi import HTTPException

    h = _handler(client, "/file/{name}")
    with pytest.raises(HTTPException) as e:
        h("dummy", "..", "..")
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        h("dummy", "x", "..")
    assert e.value.status_code == 404


def test_startup_sweeps_interrupted_runs(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("LAB_DATA", str(tmp_path))
    monkeypatch.setenv("LAB_EXTRA_TASKS", "tests.lab.dummy_task")
    d = tmp_path / "runs" / "dummy" / "260101-000000-aaaa"
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": "running", "pid": 1}))
    from lab.server.app import create_app

    with TestClient(create_app()) as c:  # entering the context runs the startup hook
        st = c.get("/api/runs/dummy/260101-000000-aaaa").json()["status"]
    assert st["state"] == "failed" and st["msg"] == "server restarted"


def test_list_tasks_survives_broken_estimate(client, monkeypatch):
    from lab.tasks import registry

    spec = registry.get("dummy")
    monkeypatch.setattr(spec, "estimate", lambda p: 1 / 0, raising=False)
    r = client.get("/api/tasks")
    assert r.status_code == 200
    assert next(t for t in r.json() if t["id"] == "dummy")["est_label"] is None
