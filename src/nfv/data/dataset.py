"""Demo dataset: npz shards -> RAM (images stay uint8) -> torch batches with action chunks.

Loading keeps every array exactly as stored — images as uint8 [T, 96, 96, 3] (~8 MB per
episode, so 200 demos of one task are ~1.6 GB of RAM) — and normalises ONLY at batch time:
images become float32 in [0, 1] and move to channel-first [B, 3, 96, 96] for torch convs.
Wrenches/state are served raw (z-scoring is a model concern and lives with the model).

Batch (``sample_batch(batch_size, rng, chunk=16)``), B = batch, H = chunk length:
    rgb_static     f32 [B, 3, 96, 96]   /255, channel-first
    rgb_wrist      f32 [B, 3, 96, 96]
    state          f32 [B, 8]
    wrench         f32 [B, 6]
    wrench_window  f32 [B, 20, 6]
    dp             f32 [B, H, 3]        CLEAN expert actions t .. t+H-1 (the DART label)
    gripper        f32 [B, H]
    f_target       f32 [B, H, 6]
    s_fv2          f32 [B, H]
    s_lin          f32 [B, H]
    pad_mask       bool [B, H]          True = real step, False = past the episode end
                                        (padded rows are zero; mask the loss with this)
    subtask        i64 [B]              the step's subtask label

Chunk starts are sampled uniformly over ALL steps of all episodes (a start near the episode
end simply gets a short mask) with the caller's ``numpy`` Generator, so a training run is
reproducible from one seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

_OBS_KEYS = ("rgb_static", "rgb_wrist", "state", "wrench", "wrench_window")
_CHUNK_KEYS = ("dp", "gripper", "f_target", "s_fv2", "s_lin")


class DemoDataset:
    """All shards of one task directory, held in RAM."""

    def __init__(self, root: str | Path, task: str):
        self.dir = Path(root) / task
        index_path = self.dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"no index.json under {self.dir} — record demos first")
        self.index = json.loads(index_path.read_text())
        self.episodes: list[dict[str, np.ndarray]] = []
        for ep in self.index["episodes"]:
            with np.load(self.dir / ep["file"]) as z:
                self.episodes.append(
                    {k: z[k] for k in (*_OBS_KEYS, *_CHUNK_KEYS, "dp_exec", "subtask")}
                )
        self.lengths = np.array([e["dp"].shape[0] for e in self.episodes])
        # flat (episode, t) index of every step = every possible chunk start
        self.starts = np.array(
            [(i, t) for i, n in enumerate(self.lengths) for t in range(n)], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def n_steps(self) -> int:
        return int(self.lengths.sum())

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator, chunk: int = 16
    ) -> dict[str, torch.Tensor]:
        """Uniform chunk starts -> torch batch; see the module docstring for the legend."""
        picks = self.starts[rng.integers(0, len(self.starts), size=batch_size)]
        out: dict[str, list] = {k: [] for k in (*_OBS_KEYS, *_CHUNK_KEYS, "pad_mask", "subtask")}
        for i, t in picks:
            ep, n = self.episodes[i], int(self.lengths[i])
            for k in _OBS_KEYS:
                out[k].append(ep[k][t])
            valid = min(chunk, n - t)
            for k in _CHUNK_KEYS:
                a = ep[k][t : t + valid].astype(np.float32)
                if valid < chunk:  # zero-pad past the episode end; pad_mask marks it
                    pad = np.zeros((chunk - valid, *a.shape[1:]), dtype=np.float32)
                    a = np.concatenate([a, pad], axis=0)
                out[k].append(a)
            mask = np.zeros(chunk, dtype=bool)
            mask[:valid] = True
            out["pad_mask"].append(mask)
            out["subtask"].append(np.int64(ep["subtask"][t]))
        batch: dict[str, torch.Tensor] = {}
        for k in ("rgb_static", "rgb_wrist"):  # uint8 HWC -> float CHW in [0, 1], batch time
            u8 = torch.from_numpy(np.stack(out[k]))  # [B, 96, 96, 3] uint8
            batch[k] = u8.permute(0, 3, 1, 2).float() / 255.0  # [B, 3, 96, 96]
        for k in ("state", "wrench", "wrench_window", *_CHUNK_KEYS):
            batch[k] = torch.from_numpy(np.stack(out[k]).astype(np.float32))
        batch["pad_mask"] = torch.from_numpy(np.stack(out["pad_mask"]))
        batch["subtask"] = torch.from_numpy(np.stack(out["subtask"]))
        return batch
