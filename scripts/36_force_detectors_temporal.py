#!/usr/bin/env python
"""V8, last attempt: temporal and ensemble detectors on the same events and episode split as
``scripts/28_force_detectors.py`` (VLA or train image: needs torch).

Features per event (``27_force_labels.py``): torque_hist 10 x 15 (tau_ext 7 + tactile summary 8,
2 s at 5 Hz), tactile 32, tau_ext 7, wrench history 10 x 6. Models:

* ``cnn``: 1-D convolution over the 10 history frames (channels: 15 torque/tactile + 6 wrench),
  global pooling, concatenated with the 39 snapshot features, MLP head;
* ``mlp_ens`` / ``cnn_ens``: 5 members with different seeds, probabilities averaged.

Training: AdamW, class-balanced BCE, early stopping on 20 % of the training seeds (never the
held-out seeds). Held-out accuracy with a 95 % seed-bootstrap interval. Gate: > 95 %.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def split_inputs(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist = X[:, :150].reshape(-1, 10, 15)
    wrench = X[:, 189:249].reshape(-1, 10, 6)
    seq = np.concatenate([hist, wrench], axis=2)  # (n, 10 frames, 21 channels)
    static = X[:, 150:189]  # tactile 32 + tau_ext 7
    return seq, static


class CNN(nn.Module):
    def __init__(self, ch: int = 21, n_static: int = 39, w: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch, w, 3, padding=1), nn.GELU(), nn.Conv1d(w, w, 3, padding=1), nn.GELU()
        )
        self.head = nn.Sequential(nn.Linear(2 * w + n_static, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, seq, static):
        h = self.conv(seq.transpose(1, 2))  # (n, w, 10)
        z = torch.cat([h.mean(2), h[:, :, -1], static], dim=1)  # mean + latest frame
        return self.head(z).squeeze(1)


class MLP(nn.Module):
    def __init__(self, n: int = 249, w: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, w), nn.GELU(), nn.Linear(w, w), nn.GELU(), nn.Linear(w, 1)
        )

    def forward(self, seq, static, flat):
        return self.net(flat).squeeze(1)


def train(kind, data, y, tr, va, seed, epochs=400):
    torch.manual_seed(seed)
    net = CNN() if kind == "cnn" else MLP()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    pos = y[tr].mean()
    wts = torch.where(torch.as_tensor(y[tr]) == 1, 0.5 / pos, 0.5 / (1 - pos)).float()
    yt = torch.as_tensor(y, dtype=torch.float32)
    best, best_state, bad = -1.0, None, 0
    g = torch.Generator().manual_seed(seed)
    idx_tr = torch.as_tensor(np.flatnonzero(tr))
    for _ in range(epochs):
        net.train()
        perm = idx_tr[torch.randperm(len(idx_tr), generator=g)]
        for b in perm.split(128):
            logit = net(data[0][b], data[1][b]) if kind == "cnn" else net(None, None, data[2][b])
            w = wts[torch.searchsorted(idx_tr, b)]
            loss = (
                w * nn.functional.binary_cross_entropy_with_logits(logit, yt[b], reduction="none")
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = ((predict(kind, net, data, va) > 0.5) == (y[va] == 1)).mean()
        if acc > best + 1e-4:
            best, best_state, bad = acc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 40:
                break
    net.load_state_dict(best_state)
    return net


@torch.no_grad()
def predict(kind, net, data, sel):
    net.eval()
    b = torch.as_tensor(np.flatnonzero(sel))
    out = net(data[0][b], data[1][b]) if kind == "cnn" else net(None, None, data[2][b])
    return torch.sigmoid(out).numpy()


def seed_boot(correct, seeds, n=5000):
    rng = np.random.default_rng(0)
    u = np.unique(seeds)
    per = {s: correct[seeds == s] for s in u}
    accs = []
    for _ in range(n):
        pick = rng.choice(u, len(u))
        c = np.concatenate([per[s] for s in pick])
        accs.append(c.mean())
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="outputs/v8_labels")
    ap.add_argument("--glob", default="part*/events.npz")
    ap.add_argument("--test-frac", type=float, default=0.3)
    args = ap.parse_args()
    torch.set_num_threads(4)

    d = [np.load(p) for p in sorted(Path(args.dir).glob(args.glob))]
    X = np.concatenate([e["X"] for e in d]).astype(np.float32)
    y = np.concatenate([e["y"] for e in d]).astype(np.float32)
    kind = np.concatenate([e["kind"] for e in d])
    seed = np.concatenate([e["seed"] for e in d])
    seeds = np.unique(seed)
    rng = np.random.default_rng(0)  # same held-out seeds as 28_force_detectors.py
    test_seeds = set(rng.choice(seeds, int(round(len(seeds) * args.test_frac)), replace=False))
    is_test = np.array([s in test_seeds for s in seed])
    train_seeds = np.array(sorted(set(seeds) - test_seeds))
    val_seeds = set(
        np.random.default_rng(1).choice(train_seeds, len(train_seeds) // 5, replace=False)
    )
    is_val = np.array([s in val_seeds for s in seed])

    out = {}
    for k, name in ((0, "grasp_stable"), (1, "seated")):
        sel = kind == k
        tr, va, te = sel & ~is_test & ~is_val, sel & is_val, sel & is_test
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = (X - mu) / sd
        seq, static = split_inputs(Z)
        data = (torch.as_tensor(seq), torch.as_tensor(static), torch.as_tensor(Z))
        res = {"n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())}
        probs = {}
        for model in ("mlp", "cnn"):
            members = [predict(model, train(model, data, y, tr, va, s), data, te) for s in range(5)]
            probs[model] = members[0]
            probs[f"{model}_ens"] = np.mean(members, 0)
        for m, p in probs.items():
            correct = (p > 0.5) == (y[te] == 1)
            tpr = correct[y[te] == 1].mean()
            tnr = correct[y[te] == 0].mean()
            res[m] = {
                "acc": float(correct.mean()),
                "balanced": float((tpr + tnr) / 2),
                "seed_boot95": seed_boot(correct, seed[te]),
            }
        out[name] = res
        print(
            name, json.dumps({m: round(res[m]["acc"], 3) for m in probs}), f"n_test={res['n_test']}"
        )
    (Path(args.dir) / "detectors_temporal.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
