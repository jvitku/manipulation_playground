"""Recorded SortBoltsNuts episodes (PLAN §13, V4): the demo format for BC, TD3+BC and the VLA.

One episode = a directory with

* ``data.npz``   low-dimensional streams at control rate (20 Hz), time-major:
                 ``t``, ``action`` (7), ``eef_pos`` (3), ``eef_quat`` (4), ``gripper_qpos`` (2),
                 ``joint_pos`` (7), ``joint_vel`` (7), ``tau_ext`` (7), ``torque_hist`` (150 =
                 10 frames x 15), ``tactile`` (32 = 2 pads x 4 x 4, max over the step),
                 ``n_sorted`` and ``part_status`` (n_parts, int codes of ``STATUS_CODES``);
* ``<camera>.mp4`` one video per camera (H.264, frame i = observation at step i);
* ``meta.json``  seed, instruction, outcome, per-part *segments* (start/end step, success,
                 sub-instruction), expert noise, ``synthetic``, git SHA, package versions.

Segments make per-part demos possible: a failed episode still contributes the parts it sorted.
``validate_episode`` is the schema check used by the tests and the LeRobot converter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fvb.logging.episode import git_sha, package_versions

LOWDIM = {
    "t": (),
    "action": (7,),
    "eef_pos": (3,),
    "eef_quat": (4,),
    "gripper_qpos": (2,),
    "joint_pos": (7,),
    "joint_vel": (7,),
    "tau_ext": (7,),
    "torque_hist": (150,),
    "tactile": (32,),
    "n_sorted": (),
}
STATUS_CODES = {
    s: i
    for i, s in enumerate(
        ["in_bin", "moving", "grasped", "seated", "vanished", "in_bucket", "dropped", "jammed"]
    )
}


@dataclass
class SortEpisodeWriter:
    out_dir: Path
    seed: int
    cameras: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    fps: int = 20

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows: dict[str, list] = {k: [] for k in LOWDIM}
        self.rows["part_status"] = []
        self._writers = {}
        if self.cameras:
            import imageio.v2 as imageio

            for cam in self.cameras:
                self._writers[cam] = imageio.get_writer(
                    self.out_dir / f"{cam}.mp4",
                    fps=self.fps,
                    codec="libx264",
                    quality=8,
                    macro_block_size=1,
                )

    def add(self, t: float, obs: dict, action: np.ndarray, env) -> None:
        """Record the observation the action was chosen from, and the action."""
        r = self.rows
        r["t"].append(t)
        r["action"].append(np.asarray(action, float))
        r["eef_pos"].append(obs["robot0_eef_pos"])
        r["eef_quat"].append(obs["robot0_eef_quat"])
        r["gripper_qpos"].append(obs["robot0_gripper_qpos"])
        r["joint_pos"].append(obs["robot0_joint_pos"])
        r["joint_vel"].append(obs["robot0_joint_vel"])
        r["tau_ext"].append(obs.get("tau_ext", np.zeros(7)))
        r["torque_hist"].append(obs.get("torque_hist", np.zeros(150)))
        r["tactile"].append(obs.get("tactile", np.zeros(32)))
        r["n_sorted"].append(env.n_sorted())
        r["part_status"].append([STATUS_CODES[p.status] for p in env.parts])
        for cam, w in self._writers.items():
            w.append_data(np.ascontiguousarray(obs[f"{cam}_image"][::-1]))  # robosuite: upside down

    def close(self, extra_meta: dict[str, Any] | None = None) -> Path:
        for w in self._writers.values():
            w.close()
        arrays = {k: np.asarray(v) for k, v in self.rows.items()}
        np.savez_compressed(self.out_dir / "data.npz", **arrays)
        meta = {
            "seed": self.seed,
            "n_steps": len(self.rows["t"]),
            "cameras": list(self.cameras),
            "fps": self.fps,
            "git_sha": git_sha(),
            "versions": package_versions(),
            **self.meta,
            **(extra_meta or {}),
        }
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        return self.out_dir


def part_segments(expert_log, part_status: np.ndarray, parts) -> list[dict]:
    """Per-part segments from the expert's phase log (step, phase, part) and the status stream.

    A segment runs from the step the expert selected a part to the step it was sorted
    (success) or the step the expert moved on (failure).
    """
    from fvb.envs.sort_bolts_nuts import SUB_INSTRUCTIONS

    sorted_codes = {STATUS_CODES["vanished"], STATUS_CODES["in_bucket"]}
    starts = [(s, p) for s, ph, p in expert_log if ph == "above" and p is not None]
    segs = []
    for k, (s0, p) in enumerate(starts):
        s_next = starts[k + 1][0] if k + 1 < len(starts) else len(part_status)
        done = np.flatnonzero(np.isin(part_status[s0:, p], list(sorted_codes)))
        success = len(done) > 0 and s0 + done[0] <= s_next + 60
        end = int(s0 + done[0]) if success else int(s_next)
        segs.append(
            {
                "part": parts[p].body,
                "kind": parts[p].kind,
                "start": int(s0),
                "end": end,
                "success": bool(success),
                "instruction": SUB_INSTRUCTIONS[parts[p].kind],
            }
        )
    return segs


def validate_episode(ep_dir: Path) -> dict:
    """Schema check: shapes, monotonic time, finite values, video frame count and size."""
    ep_dir = Path(ep_dir)
    meta = json.loads((ep_dir / "meta.json").read_text())
    data = np.load(ep_dir / "data.npz")
    n = meta["n_steps"]
    assert n > 0
    for k, shape in LOWDIM.items():
        a = data[k]
        assert a.shape == (n, *shape), (k, a.shape)
        assert np.all(np.isfinite(a)), k
    assert data["part_status"].shape[0] == n
    assert np.all(np.diff(data["t"]) > 0), "time must increase"
    assert np.all(np.abs(data["action"]) <= 1.0 + 1e-9)
    for cam in meta["cameras"]:
        import imageio.v2 as imageio

        r = imageio.get_reader(ep_dir / f"{cam}.mp4")
        frames = sum(1 for _ in r)
        first = r.get_data(0)
        r.close()
        assert frames == n, (cam, frames, n)
        assert first.ndim == 3 and first.shape[2] == 3
    for seg in meta.get("segments", []):
        assert 0 <= seg["start"] <= seg["end"] <= n
    return meta
