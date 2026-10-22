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
