"""Windows of (H past observations, K future actions) from expert episodes, with normalisation."""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fvb.policy.task import ACT_DIM, FORCE_IDX, OBS_DIM


@dataclass
class Norm:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray

    def to_json(self) -> dict:
        return {
            k: getattr(self, k).tolist() for k in ("obs_mean", "obs_std", "act_mean", "act_std")
        }

    @classmethod
    def from_json(cls, d: dict) -> Norm:
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in d.items()})


def load_episodes(data_dir: str | Path) -> list[dict]:
    eps = []
    for f in sorted(glob.glob(str(Path(data_dir) / "ep*_policy.npz"))):
        z = np.load(f)
        obs, act = z["obs"].astype(np.float32), z["action"].astype(np.float32)
        assert obs.shape[1] == OBS_DIM and act.shape[1] == ACT_DIM, (obs.shape, act.shape)
        eps.append({"obs": obs, "action": act, "success": bool(z["success"]), "path": f})
    if not eps:
        raise FileNotFoundError(f"no ep*_policy.npz in {data_dir}")
    return eps


def split_episodes(eps: list[dict], val_frac: float, seed: int) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eps))
    n_val = max(1, int(round(val_frac * len(eps))))
    val = [eps[i] for i in idx[:n_val]]
    train = [eps[i] for i in idx[n_val:]]
    return train, val


def apply_ablation(obs: np.ndarray, use_force: bool) -> np.ndarray:
    if use_force:
        return obs
    o = obs.copy()
    o[..., FORCE_IDX] = 0.0
    return o


def compute_norm(train: list[dict], use_force: bool) -> Norm:
    obs = np.concatenate([apply_ablation(e["obs"], use_force) for e in train])
    act = np.concatenate([e["action"] for e in train])
    return Norm(
        obs.mean(0), np.maximum(obs.std(0), 1e-6), act.mean(0), np.maximum(act.std(0), 1e-6)
    )


def make_windows(eps: list[dict], H: int, K: int, use_force: bool, norm: Norm):
    """Returns X (N,H,OBS), Y (N,K,ACT), M (N,K) validity mask. Start is padded by repeating
    the first observation; the action target past the episode end is masked out."""
    X, Y, M = [], [], []
    for e in eps:
        obs = (apply_ablation(e["obs"], use_force) - norm.obs_mean) / norm.obs_std
        act = (e["action"] - norm.act_mean) / norm.act_std
        T = len(obs)
        pad = np.repeat(obs[:1], H - 1, axis=0)
        obs_p = np.concatenate([pad, obs])
        for t in range(T):
            X.append(obs_p[t : t + H])
            y = np.zeros((K, act.shape[1]), np.float32)
            m = np.zeros(K, np.float32)
            n = min(K, T - t)
            y[:n] = act[t : t + n]
            m[:n] = 1.0
            Y.append(y)
            M.append(m)
    return np.stack(X), np.stack(Y), np.stack(M)


def save_meta(path: Path, **kw) -> None:
    path.write_text(
        json.dumps(kw, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    )
