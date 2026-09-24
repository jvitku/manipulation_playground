"""Stage 1 G3: robomimic-style HDF5 export of expert datasets."""

import json

import h5py
import numpy as np

from fvb.logging.episode import EpisodeLogger
from fvb.policy.export import export_hdf5
from fvb.policy.spec import GANTRY
from fvb.policy.task import GantryTask, TaskParams


def _write_dataset(root, seeds):
    p = TaskParams()
    (root / "config.json").write_text(json.dumps({"task_spec": "gantry", "task": {"kp": p.kp}}))
    for i, seed in enumerate(seeds):
        task = GantryTask(p, seed)
        log = EpisodeLogger(config={"hole_xy": task.hole_xy.tolist()}, seed=seed)
        obs, act, done, reason = [], [], False, None
        while not done:
            obs.append(task.observe())
            a = task.expert_action()
            act.append(a.astype(np.float32))
            done, reason = task.step(a, log)
        log.success = reason == "success"
        log.save(root / f"ep{i:04d}.npz")
        np.savez_compressed(
            root / f"ep{i:04d}_policy.npz",
            obs=np.stack(obs),
            action=np.stack(act),
            success=np.bool_(log.success),
        )


def test_export_layout_and_round_trip(tmp_path):
    _write_dataset(tmp_path, seeds=(11, 12, 13))
    info = export_hdf5(tmp_path, tmp_path / "out.hdf5", val_frac=0.34, seed=0)
    assert info["demos"] == 3 and info["train"] + info["valid"] == 3 and info["valid"] == 1
    with h5py.File(tmp_path / "out.hdf5", "r") as f:
        data = f["data"]
        env_args = json.loads(data.attrs["env_args"])
        assert env_args["env_name"] == "fvb_gantry_hidden_hole"
        assert data.attrs["task_spec"] == "gantry"
        total = 0
        for i in range(3):
            g = data[f"demo_{i}"]
            pol = np.load(tmp_path / f"ep{i:04d}_policy.npz")
            T = g.attrs["num_samples"]
            total += T
            assert np.array_equal(g["actions"][:], pol["action"])
            # the named observation groups reassemble the policy observation exactly
            parts = [g["obs"][k][:] for k in GANTRY.obs_groups]
            assert np.array_equal(np.concatenate(parts, axis=1), pol["obs"])
            assert g["dones"][-1] == 1 and g["dones"][:-1].sum() == 0
            assert g["rewards"][-1] == float(g.attrs["success"])
            hf = g["obs"]["ft_raw_hf"]
            assert hf.shape[0] == T and hf.shape[2] == 6
            assert g["obs"]["s5_ft_comp"].shape == (T, 6)
            assert len(g.attrs["hole_xy_m"]) == 2
        assert data.attrs["total"] == total
        names = sorted([*f["mask/train"][:], *f["mask/valid"][:]])
        assert names == [b"demo_0", b"demo_1", b"demo_2"]
