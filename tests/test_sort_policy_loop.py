"""Closed-loop plumbing: SortBoltsNuts -> TCP policy server -> action (PLAN §14, V5/V7 harness).

The served policy here is a stub (it records what it receives and returns a fixed action), so
the test checks the I/O contract and the loop, not a learned policy.
"""

import threading

import numpy as np
import pytest

pytest.importorskip("robosuite")

import robosuite as suite  # noqa: E402

import fvb.envs  # noqa: E402, F401
from fvb.logging.sort_episode import LOWDIM  # noqa: E402
from fvb.vla.sort_io import STATE_KEYS, policy_obs, run_episode  # noqa: E402
from fvb.vla.transport import PolicyClient, PolicyServer  # noqa: E402


def test_state_layout_matches_recorder():
    assert sum(LOWDIM[k][0] for k in STATE_KEYS) == 16


def test_loop_through_server():
    seen = []

    def act(obs):
        seen.append(obs)
        return np.r_[0.0, 0.0, 0.2, 0.0, 0.0, 0.0, -1.0]

    server = PolicyServer(act, port=6021)
    th = threading.Thread(target=server.serve, kwargs={"max_connections": 1}, daemon=True)
    th.start()
    env = suite.make(
        "SortBoltsNuts",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["sortview", "robot0_eye_in_hand"],
        camera_heights=64,
        camera_widths=64,
        seed=0,
    )
    client = PolicyClient(port=6021, timeout=5.0)
    try:
        z0 = None
        res = run_episode(env, client, max_steps=10)
        z0 = seen[0]["observation.state"][7 + 2]
        z1 = env._observables["robot0_eef_pos"].obs[2]
    finally:
        client.close()
        env.close()
    assert res["steps"] == 10 and res["outcome"] == "running"
    o = seen[0]
    assert o["observation.state"].shape == (16,) and o["observation.state"].dtype == np.float32
    assert o["observation.images.front"].shape == (64, 64, 3)
    assert o["observation.images.wrist"].dtype == np.uint8
    assert o["observation.torque_hist"].shape == (150,)
    assert o["task"].startswith("sort the parts")
    assert z1 > z0 + 0.01  # the served +z action moved the hand up
    assert res["latency_p50_s"] < 0.2


def test_policy_obs_without_cameras():
    obs = {
        "robot0_joint_pos": np.zeros(7),
        "robot0_eef_pos": np.ones(3),
        "robot0_eef_quat": np.zeros(4),
        "robot0_gripper_qpos": np.zeros(2),
    }
    o = policy_obs(obs, "t")
    assert o["observation.state"][7:10].tolist() == [1, 1, 1]
    assert "observation.images.front" not in o and o["observation.tactile"].shape == (32,)
