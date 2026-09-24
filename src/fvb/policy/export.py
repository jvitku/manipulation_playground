"""Stage 1 G3: export expert datasets (PLAN §5 episodes + policy obs/actions) to robomimic HDF5.

Layout (robomimic conventions, written with h5py only)::

    data/                      attrs: total, env_args (JSON), task_spec
      demo_<i>/                attrs: num_samples, seed, success, hole_xy_m
        actions   (T, A)       policy-facing actions (target deltas, m)
        rewards   (T,)         1.0 on the last step of a successful episode, else 0
        dones     (T,)         1 on the last step
        obs/<group>  (T, d)    named slices of the policy observation (fvb.policy.spec)
        obs/s5_<key> (T, ...)  PLAN §5 control-rate arrays (ee_pos, ft_comp, ...)
        obs/ft_raw_hf (T, n_sub, 6)  physics-rate raw wrench, grouped per control step
    mask/train, mask/valid     demo names (bytes), split by episode
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from fvb.policy.data import split_episodes
from fvb.policy.spec import spec_of_dataset

S5_KEYS = (
    "t",
    "ee_pos",
    "ee_quat",
    "ee_vel",
    "ft_raw",
    "ft_world",
    "ft_comp",
    "contact_wrench",
    "n_contacts",
    "joint_torque",
)


def export_hdf5(
    data_dir: str | Path, out_path: str | Path, val_frac: float = 0.1, seed: int = 0
) -> dict:
    import h5py

    data_dir, out_path = Path(data_dir), Path(out_path)
    spec = spec_of_dataset(data_dir)
    cfg_path = data_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    pol_files = sorted(glob.glob(str(data_dir / "ep*_policy.npz")))
    if not pol_files:
        raise FileNotFoundError(f"no ep*_policy.npz in {data_dir}")
    names = [f"demo_{i}" for i in range(len(pol_files))]
    train, val = split_episodes(names, val_frac, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = 0
    with h5py.File(out_path, "w") as f:
        data = f.create_group("data")
        for name, pf in zip(names, pol_files, strict=True):
            pol = np.load(pf)
            s5_path = Path(pf.replace("_policy.npz", ".npz"))
            s5 = np.load(s5_path)
            meta = json.loads(s5_path.with_suffix(".json").read_text())
            obs, act = pol["obs"], pol["action"]
            T = len(obs)
            if len(s5["t"]) != T:
                raise ValueError(f"{s5_path}: §5 has {len(s5['t'])} rows, policy obs has {T}")
            g = data.create_group(name)
            g.create_dataset("actions", data=act)
            success = bool(pol["success"])
            rewards = np.zeros(T, np.float32)
            rewards[-1] = float(success)
            dones = np.zeros(T, np.int64)
            dones[-1] = 1
            g.create_dataset("rewards", data=rewards)
            g.create_dataset("dones", data=dones)
            og = g.create_group("obs")
            for grp, sl in spec.obs_groups.items():
                og.create_dataset(grp, data=obs[:, sl])
            for k in S5_KEYS:
                if k in s5.files:
                    og.create_dataset(f"s5_{k}", data=s5[k])
            hf = s5["ft_raw_hf"]
            n_sub = len(hf) // T
            og.create_dataset(
                "ft_raw_hf", data=hf[: n_sub * T].reshape(T, n_sub, 6), compression="gzip"
            )
            g.attrs["num_samples"] = T
            g.attrs["seed"] = int(meta.get("seed", -1))
            g.attrs["success"] = success
            g.attrs["hole_xy_m"] = np.asarray(meta["config"].get("hole_xy", [np.nan, np.nan]))
            n_samples += T
        data.attrs["total"] = n_samples
        data.attrs["task_spec"] = spec.name
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": f"fvb_{spec.name}_hidden_hole",
                "type": "fvb",
                "env_kwargs": cfg.get("task", {}),
                "obs_groups": {k: [v.start, v.stop] for k, v in spec.obs_groups.items()},
            }
        )
        mask = f.create_group("mask")
        mask.create_dataset("train", data=np.array(train, dtype="S"))
        mask.create_dataset("valid", data=np.array(val, dtype="S"))
    return {
        "path": str(out_path),
        "demos": len(names),
        "samples": n_samples,
        "train": len(train),
        "valid": len(val),
        "task_spec": spec.name,
    }
