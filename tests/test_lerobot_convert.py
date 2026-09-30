"""Recorded sort episodes convert to a LeRobotDataset that loads back (VLA image only)."""

import json

import numpy as np
import pytest

pytest.importorskip("lerobot")
pytest.importorskip("datasets")
av = pytest.importorskip("av")

from fvb.logging.sort_episode import LOWDIM  # noqa: E402
from fvb.vla.lerobot_convert import convert  # noqa: E402


def _fake_episode(d, n=30, hw=32, seed=0):
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    arrays = {k: rng.normal(size=(n, *shape)) for k, shape in LOWDIM.items()}
    arrays["t"] = np.arange(n) * 0.05
    arrays["action"] = np.clip(arrays["action"], -1, 1)
    arrays["part_status"] = np.zeros((n, 2), int)
    np.savez(d / "data.npz", **arrays)
    for cam in ("sortview", "robot0_eye_in_hand"):
        with av.open(str(d / f"{cam}.mp4"), "w") as c:
            s = c.add_stream("libx264", rate=20)
            s.width = s.height = hw
            s.pix_fmt = "yuv420p"
            for i in range(n):
                img = np.full((hw, hw, 3), i * 8 % 255, np.uint8)
                for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                    c.mux(p)
            for p in s.encode():
                c.mux(p)
    meta = {
        "seed": seed,
        "n_steps": n,
        "fps": 20,
        "cameras": ["sortview", "robot0_eye_in_hand"],
        "instruction": "sort the parts: bolts into the hole, nuts into bucket A",
        "segments": [
            {
                "part": "nut0",
                "kind": "nut",
                "start": 2,
                "end": 12,
                "success": True,
                "instruction": "put the nut in bucket A",
            },
            {
                "part": "bolt0",
                "kind": "bolt",
                "start": 12,
                "end": 30,
                "success": False,
                "instruction": "put the bolt in the hole",
            },
        ],
    }
    (d / "meta.json").write_text(json.dumps(meta))


@pytest.mark.parametrize("mode,eps,frames", [("segments", 2, 20), ("episode", 2, 60)])
def test_convert_and_load(tmp_path, mode, eps, frames):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw = [tmp_path / "raw" / "ep_0", tmp_path / "raw" / "ep_1"]
    for i, d in enumerate(raw):
        _fake_episode(d, seed=i)
    out = convert(raw, tmp_path / "ds", "local/test", mode)
    assert out == {"episodes": eps, "frames": frames}
    ds = LeRobotDataset("local/test", root=tmp_path / "ds")
    assert len(ds) == frames and ds.num_episodes == eps
    item = ds[0]
    assert item["observation.state"].shape == (16,)
    assert item["observation.torque_hist"].shape == (150,)
    assert item["action"].shape == (7,)
    assert item["observation.images.front"].shape == (3, 32, 32)
    want = "put the nut in bucket A" if mode == "segments" else "sort the parts"
    assert item["task"].startswith(want)


def test_convert_without_force(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw = [tmp_path / "raw" / "ep_0"]
    _fake_episode(raw[0])
    convert(raw, tmp_path / "ds", "local/test", "segments", force=False)
    ds = LeRobotDataset("local/test", root=tmp_path / "ds")
    assert "observation.torque_hist" not in ds.features and "observation.state" in ds.features
