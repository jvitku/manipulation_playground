"""V4: recorded SortBoltsNuts episodes validate (PLAN §14 test_sort_expert, recording half).

A short expert episode with small cameras is recorded and checked: schema, monotonic time,
video frame count and size, segments within the episode.
"""

import json

import numpy as np
import pytest

pytest.importorskip("robosuite")

import robosuite as suite  # noqa: E402

import fvb.envs  # noqa: E402, F401
from fvb.logging.sort_episode import (  # noqa: E402
    STATUS_CODES,
    SortEpisodeWriter,
    part_segments,
    validate_episode,
)
from fvb.policy.sort_expert import NO_NOISE, SortExpert  # noqa: E402


def test_recorded_episode_validates(tmp_path):
    cams = ("sortview", "robot0_eye_in_hand")
    env = suite.make(
        "SortBoltsNuts",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=list(cams),
        camera_heights=64,
        camera_widths=64,
        seed=2,
    )
    try:
        obs = env.reset()
        ex = SortExpert(NO_NOISE, 2)
        ex.reset(env)
        w = SortEpisodeWriter(tmp_path / "ep", 2, cameras=cams, meta={"synthetic": True})
        for _ in range(60):
            a = ex.act()
            w.add(float(env.sim.data._data.time), obs, a, env)
            obs, _, done, info = env.step(a)
        segs = part_segments(ex.log, np.asarray(w.rows["part_status"]), env.parts)
        w.close({"outcome": info["outcome"], "segments": segs})
    finally:
        env.close()
    meta = validate_episode(tmp_path / "ep")
    assert meta["n_steps"] == 60 and meta["synthetic"]
    assert len(meta["segments"]) >= 1 and meta["segments"][0]["start"] >= 0
    data = np.load(tmp_path / "ep" / "data.npz")
    assert data["torque_hist"].shape == (60, 150) and data["tactile"].shape == (60, 32)
    assert json.loads((tmp_path / "ep" / "meta.json").read_text())["git_sha"] is not None


def test_part_segments_from_log():
    class P:
        def __init__(self, body, kind):
            self.body, self.kind = body, kind

    parts = [P("bolt0", "bolt"), P("nut0", "nut")]
    status = np.zeros((100, 2), int)
    status[40:, 1] = STATUS_CODES["in_bucket"]
    log = [(1, "above", 1), (50, "above", 0), (90, "regrasp_up", 0)]
    segs = part_segments(log, status, parts)
    assert segs[0] == {
        "part": "nut0",
        "kind": "nut",
        "start": 1,
        "end": 40,
        "success": True,
        "instruction": "put the nut in bucket A",
    }
    assert segs[1]["part"] == "bolt0" and not segs[1]["success"] and segs[1]["end"] == 100
