"""Demo recording round-trip: expert -> npz shards -> DemoDataset -> torch batch."""

import json

import numpy as np
import torch

from nfv.data.dataset import DemoDataset
from nfv.data.record import record_demos
from nfv.seeds import DEMO_SEED0, EVAL_SEED0, TASK_NAMES, demo_seeds, eval_seeds


def test_seed_ranges_disjoint():
    for task in TASK_NAMES:
        demo = set(demo_seeds(task, 500))
        for other in TASK_NAMES:
            assert not demo & set(eval_seeds(other, 500))  # demo never meets eval
            if other != task:
                assert not demo & set(demo_seeds(other, 500))  # task blocks don't collide
    assert DEMO_SEED0 > 999 and EVAL_SEED0 >= DEMO_SEED0 + 4 * 10_000


def test_record_and_dataset_roundtrip(tmp_path):
    index = record_demos("t0", n=2, out_dir=tmp_path, noise_mm=0.5)
    assert index["n"] == 2 and all(ep["success"] for ep in index["episodes"])
    assert json.loads((tmp_path / "t0" / "index.json").read_text())["n"] == 2

    ds = DemoDataset(tmp_path, "t0")
    assert len(ds) == 2 and ds.n_steps == sum(ep["steps"] for ep in index["episodes"])

    # round-trip: what the dataset holds is byte-for-byte what the shard stored
    with np.load(tmp_path / "t0" / index["episodes"][0]["file"]) as z:
        t = int(z["dp"].shape[0])
        assert z["rgb_static"].dtype == np.uint8 and z["rgb_static"].shape == (t, 96, 96, 3)
        assert z["wrench_window"].shape == (t, 20, 6)
        assert np.array_equal(ds.episodes[0]["dp"], z["dp"])
        assert np.array_equal(ds.episodes[0]["rgb_wrist"], z["rgb_wrist"])
        assert not np.array_equal(z["dp"], z["dp_exec"])  # DART: label is clean, exec is noisy
        assert np.allclose(z["dp"], z["dp_exec"], atol=5e-3)  # ... by ~0.5 mm of noise

    batch = ds.sample_batch(4, np.random.default_rng(0), chunk=8)
    assert batch["rgb_static"].shape == (4, 3, 96, 96)
    assert batch["rgb_static"].dtype == torch.float32
    assert 0.0 <= batch["rgb_static"].min() and batch["rgb_static"].max() <= 1.0
    assert batch["rgb_static"].max() > 0.1  # images are not empty
    assert batch["state"].shape == (4, 8)
    assert batch["wrench_window"].shape == (4, 20, 6)
    assert batch["dp"].shape == (4, 8, 3) and batch["f_target"].shape == (4, 8, 6)
    assert batch["gripper"].shape == (4, 8) and batch["s_fv2"].shape == (4, 8)
    assert batch["pad_mask"].dtype == torch.bool and batch["pad_mask"].shape == (4, 8)
    assert batch["subtask"].dtype == torch.int64

    # a chunk starting at the last step is padded and masked, never truncated
    last_ep, last_t = len(ds.episodes) - 1, int(ds.lengths[-1]) - 1
    ds.starts = np.array([(last_ep, last_t)])
    tail = ds.sample_batch(1, np.random.default_rng(1), chunk=8)
    assert bool(tail["pad_mask"][0, 0]) and not bool(tail["pad_mask"][0, 1])
    assert float(tail["dp"][0, 1:].abs().sum()) == 0.0
