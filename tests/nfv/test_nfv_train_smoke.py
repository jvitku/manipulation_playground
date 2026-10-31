"""Smoke train (CPU): 8 real T0 demos -> 150 steps at batch 16 -> the loss drops > 30 %.

This is the end-to-end plumbing check of milestone M1 — recorder -> dataset -> batcher ->
Policy.loss -> fit — not a success-rate evaluation (that is the lesson-16 gate on the GPU).
It also prints the measured steps/s so the 20 k-step wall-clock gate can be projected.
"""

import numpy as np

from nfv.data.dataset import DemoDataset
from nfv.data.record import record_demos
from nfv.model import ModelConfig
from nfv.train import fit

N_DEMOS = 8
STEPS = 150
BATCH = 16


def test_smoke_train_loss_drops(tmp_path):
    index = record_demos("t0", N_DEMOS, tmp_path, noise_mm=0.5)
    assert index["n"] == N_DEMOS
    ds = DemoDataset(tmp_path, "t0")
    policy, history = fit(
        ModelConfig(),
        ds,
        None,
        steps=STEPS,
        lr=1e-3,
        batch=BATCH,
        log_every=10,
        ckpt_every=0,
        seed=0,
        device="cpu",
    )
    losses = [row["loss"] for row in history if "loss" in row]
    rates = [row["steps_per_s"] for row in history if "steps_per_s" in row]
    first = float(np.mean(losses[:2]))
    last = float(np.mean(losses[-2:]))
    print(f"smoke train: demos={index['n']} (tried {index['tried']}), steps={STEPS}, batch={BATCH}")
    print(f"loss curve (every 10 steps): {[round(x, 4) for x in losses]}")
    print(
        f"first~{first:.4f} -> last~{last:.4f} ({(1 - last / first) * 100:.1f} % drop), "
        f"median {np.median(rates):.2f} steps/s CPU"
    )
    assert last < 0.7 * first, f"loss only dropped {first:.4f} -> {last:.4f}"
