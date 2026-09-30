"""Recorded SortBoltsNuts episodes -> LeRobotDataset (PLAN §13, V4/V5/V7). Runs in the VLA image.

Two modes:

* ``episode``  - one LeRobot episode per recorded episode, task = the episode instruction;
* ``segments`` - one LeRobot episode per *successful per-part segment* (``meta.json``), task =
                 the part's sub-instruction ("put the bolt in the hole"), so failed episodes
                 still contribute the parts they sorted.

Features (names follow LeRobot / SmolVLA conventions; images stay at the recorded size, the
policy's processor resizes):

* ``observation.images.front`` / ``observation.images.wrist`` - video (H, W, 3);
* ``observation.state``   float32 (16): joint pos (7), eef pos (3), eef quat (4), gripper (2);
* ``observation.tau_ext`` float32 (7); ``observation.torque_hist`` float32 (150);
  ``observation.tactile`` float32 (32) - phase-2 force inputs, unused by phase-1 policies;
* ``action``              float32 (7): OSC delta pose + gripper, as the expert commanded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fvb.vla.sort_io import CAMERAS, STATE_KEYS  # one layout for training and evaluation


def read_video(path: Path) -> np.ndarray:
    import av

    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


FORCE_KEYS = ("observation.tau_ext", "observation.torque_hist", "observation.tactile")


def features(image_hw: tuple[int, int], cameras: list[str], force: bool = True) -> dict:
    h, w = image_hw
    f = {
        CAMERAS[c]: {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
        for c in cameras
    }
    f["observation.state"] = {"dtype": "float32", "shape": (16,), "names": None}
    f["observation.tau_ext"] = {"dtype": "float32", "shape": (7,), "names": None}
    f["observation.torque_hist"] = {"dtype": "float32", "shape": (150,), "names": None}
    f["observation.tactile"] = {"dtype": "float32", "shape": (32,), "names": None}
    f["action"] = {"dtype": "float32", "shape": (7,), "names": None}
    if not force:  # phase 1: LeRobot types every extra observation.* key as robot state
        for k in FORCE_KEYS:
            del f[k]
    return f


def convert(
    episode_dirs: list[Path],
    root: Path,
    repo_id: str = "local/sort_synth",
    mode: str = "segments",
    force: bool = True,
    only_success: bool = False,
) -> dict:
    """Write a LeRobotDataset at ``root``; returns counts."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    assert mode in ("episode", "segments")
    ds = None
    n_eps = n_frames = 0
    for ep in sorted(Path(p) for p in episode_dirs):
        meta = json.loads((ep / "meta.json").read_text())
        if only_success and meta.get("outcome") != "success":
            continue
        data = np.load(ep / "data.npz")
        cams = [c for c in meta["cameras"] if c in CAMERAS]
        videos = {c: read_video(ep / f"{c}.mp4") for c in cams}
        if ds is None:
            hw = next(iter(videos.values())).shape[1:3] if videos else (0, 0)
            ds = LeRobotDataset.create(
                repo_id=repo_id,
                fps=meta.get("fps", 20),
                features=features(hw, cams, force),
                root=root,
                robot_type="panda",
                use_videos=True,
            )
        if mode == "episode":
            spans = [(0, meta["n_steps"], meta["instruction"])]
        else:
            spans = [
                (s["start"], s["end"], s["instruction"])
                for s in meta.get("segments", [])
                if s["success"] and s["end"] > s["start"]
            ]
        state = np.concatenate([data[k] for k in STATE_KEYS], axis=1).astype(np.float32)
        for a, b, task in spans:
            for i in range(a, b):
                frame = {
                    "observation.state": state[i],
                    "observation.tau_ext": data["tau_ext"][i].astype(np.float32),
                    "observation.torque_hist": data["torque_hist"][i].astype(np.float32),
                    "observation.tactile": data["tactile"][i].astype(np.float32),
                    "action": data["action"][i].astype(np.float32),
                    "task": task,
                }
                if not force:
                    for k in FORCE_KEYS:
                        del frame[k]
                for c in cams:
                    frame[CAMERAS[c]] = videos[c][i]
                ds.add_frame(frame)
            ds.save_episode()
            n_eps += 1
            n_frames += b - a
    if ds is not None:
        ds.finalize()
    return {"episodes": n_eps, "frames": n_frames}
