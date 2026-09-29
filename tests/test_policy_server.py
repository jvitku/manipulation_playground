"""PLAN §13 V0: sim <-> policy transport (stdlib TCP, version-independent array encoding)."""

import threading
import time

import numpy as np
import pytest

from fvb.vla.transport import PolicyClient, PolicyServer, pack, unpack


def test_pack_round_trip_preserves_arrays_and_structure():
    obs = {
        "images": {"front": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)},
        "state": np.arange(16, dtype=np.float32),
        "torque_hist": np.random.randn(10, 15),
        "task": "sort the parts",
        "k": np.int64(3),
    }
    back = unpack(pack(obs))
    assert back["task"] == "sort the parts" and back["k"] == 3
    assert back["images"]["front"].dtype == np.uint8
    assert np.array_equal(back["images"]["front"], obs["images"]["front"])
    assert np.array_equal(back["torque_hist"], obs["torque_hist"])
    # records carry only builtins: nothing numpy-version-specific gets pickled
    rec = pack(obs)["state"]
    assert set(rec) == {"__ndarray__", "dtype", "shape", "data"} and isinstance(rec["data"], bytes)


def _serve(port, act, reset=None, n=1):
    srv = PolicyServer(act, reset, host="127.0.0.1", port=port)
    t = threading.Thread(target=srv.serve, kwargs={"max_connections": n}, daemon=True)
    t.start()
    return t


def test_round_trip_and_latency():
    calls = {"reset": 0}
    _serve(
        6311,
        act=lambda o: np.tile(o["state"][:7], (50, 1)),
        reset=lambda: calls.__setitem__("reset", calls["reset"] + 1),
    )
    c = PolicyClient(port=6311, timeout=2.0)
    c.reset()
    obs = {
        "images": {
            "front": np.zeros((256, 256, 3), np.uint8),
            "wrist": np.zeros((256, 256, 3), np.uint8),
        },
        "state": np.arange(16, dtype=np.float32),
        "task": "t",
    }
    t0 = time.perf_counter()
    for _ in range(20):
        a = c.act(obs)
    rtt = (time.perf_counter() - t0) / 20
    c.close()
    assert a.shape == (50, 7) and np.allclose(a[0], np.arange(7))
    assert calls["reset"] == 1
    assert rtt < 0.2  # V0 gate: transport well under 200 ms with two 256^2 images


def test_server_errors_are_reported_not_swallowed():
    def bad(obs):
        raise ValueError("boom")

    _serve(6312, act=bad)
    c = PolicyClient(port=6312, timeout=2.0)
    with pytest.raises(RuntimeError, match="boom"):
        c.act({"state": np.zeros(3)})
    c.close()


def test_timeout_is_an_infrastructure_error():
    _serve(6313, act=lambda o: (time.sleep(0.5), np.zeros((50, 7)))[1])
    c = PolicyClient(port=6313, timeout=0.1)
    with pytest.raises(TimeoutError):
        c.act({"state": np.zeros(3)})
