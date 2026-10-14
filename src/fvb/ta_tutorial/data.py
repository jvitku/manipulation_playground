"""Episodes -> training samples, the way LeRobot builds them for pi0 / SmolVLA / ACT.

Sample i of an episode holds the observation at step i and the action CHUNK a[i : i + H]. Near
the episode end the chunk runs past the last step; those positions are padded (repeat the last
action) and masked out of the loss (``mask`` = 0), as LeRobot's ``action_is_pad`` does.

Normalisation: every observation vector (state, torque history, the future-force target) is
z-scored with the training set's per-dimension mean / std. Actions are already in [-1, 1]. The
torque history gets the same treatment as everything else - in our SmolVLA runs the raw history
mixed a 47 N grip force with 1 N joint torques, a scale problem the network has to undo first.

``aux`` (TA-VLA's auxiliary target): the sensed force over the next ``F`` steps, flattened.
"""

from __future__ import annotations

import numpy as np
import torch

OBS_KEYS = ("obs", "state", "torque", "scene", "lang")


class ChunkDataset:
    def __init__(
        self, episodes: list[dict], horizon: int, future: int = 8, device="cpu", norm=None
    ):
        self.H, self.F = horizon, future
        cols: dict[str, list] = {k: [] for k in (*OBS_KEYS, "x0", "mask", "aux", "ep")}
        for j, ep in enumerate(episodes):
            T = len(ep["action"])
            idx = np.arange(T)[:, None] + np.arange(horizon)[None]
            cols["mask"].append((idx < T).astype(np.float32))
            cols["x0"].append(ep["action"][np.minimum(idx, T - 1)])
            if "force" in ep:
                fi = np.minimum(np.arange(T)[:, None] + np.arange(future)[None], T - 1)
                cols["aux"].append(ep["force"][fi].reshape(T, -1))
            for k in OBS_KEYS:
                if k in ep:
                    cols[k].append(ep[k])
            cols["ep"].append(np.full(T, j))
        self.t = {}
        for k, v in cols.items():
            if v:
                a = np.concatenate(v)
                self.t[k] = torch.as_tensor(a, device=device)
        self.norm = norm or {
            k: (self.t[k].float().mean(0), self.t[k].float().std(0).clamp_min(1e-3))
            for k in ("obs", "state", "torque", "aux")
            if k in self.t
        }
        self.n = len(self.t["x0"])
        self.device = device

    def cond(self, raw: dict) -> dict:
        """Normalise a dict of raw observation tensors (training batch or live policy input)."""
        out = {}
        for k, v in raw.items():
            if k in self.norm:
                m, s = self.norm[k]
                out[k] = (v.float() - m.to(v.device)) / s.to(v.device)
            else:
                out[k] = v
        return out

    def batch(self, idx: torch.Tensor):
        raw = {k: self.t[k][idx] for k in OBS_KEYS if k in self.t}
        aux = None
        if "aux" in self.t:
            m, s = self.norm["aux"]
            aux = (self.t["aux"][idx] - m) / s
        return self.cond(raw), self.t["x0"][idx].float(), self.t["mask"][idx], aux


def split_episodes(eps: list[dict], val_frac: float, seed: int):
    r = np.random.default_rng(seed)
    perm = r.permutation(len(eps))
    nv = max(1, int(round(len(eps) * val_frac)))
    return [eps[i] for i in perm[nv:]], [eps[i] for i in perm[:nv]]
