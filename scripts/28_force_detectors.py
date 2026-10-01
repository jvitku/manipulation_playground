#!/usr/bin/env python
"""V8 gate: simple detectors for "grasp stable" and "seated" from torque / tactile features.

Reads the events from scripts/27_force_labels.py (``<dir>/part*/events.npz``), splits by episode
seed (70 % train, 30 % held out), fits an L2 logistic regression per event kind (numpy, features
z-scored on the training split) and reports held-out accuracy and balanced accuracy, with
ablations: torque only (tau_ext + its history), tactile only, and the majority-class baseline.
Gate (PLAN §13 V8): > 95 % on held-out episodes. Writes <dir>/detectors.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# feature layout from 27_force_labels.py: torque_hist (10 frames x [tau_ext 7, tactile summary 8]),
# tactile (32), tau_ext (7)
HIST = np.arange(150).reshape(10, 15)
GROUPS = {
    "all": np.arange(189),
    "torque": np.r_[HIST[:, :7].ravel(), np.arange(182, 189)],
    "tactile": np.r_[HIST[:, 7:].ravel(), np.arange(150, 182)],
}


def fit_logreg(X, y, l2=1e-2, iters=3000, lr=0.5):
    w = np.zeros(X.shape[1])
    b = 0.0
    pos = max(y.mean(), 1e-3)
    sw = np.where(y == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-3))  # class-balanced
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = sw * (p - y)
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b


def scores(y, pred):
    acc = float(np.mean(pred == y))
    tpr = float(np.mean(pred[y == 1] == 1)) if np.any(y == 1) else float("nan")
    tnr = float(np.mean(pred[y == 0] == 0)) if np.any(y == 0) else float("nan")
    return {"acc": acc, "balanced_acc": (tpr + tnr) / 2, "tpr": tpr, "tnr": tnr}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="outputs/v8_labels")
    ap.add_argument("--test-frac", type=float, default=0.3)
    args = ap.parse_args()

    parts = sorted(Path(args.dir).glob("part*/events.npz"))
    d = [np.load(p) for p in parts]
    X = np.concatenate([e["X"] for e in d])
    y = np.concatenate([e["y"] for e in d]).astype(float)
    kind = np.concatenate([e["kind"] for e in d])
    seed = np.concatenate([e["seed"] for e in d])
    seeds = np.unique(seed)
    rng = np.random.default_rng(0)
    test_seeds = set(rng.choice(seeds, int(round(len(seeds) * args.test_frac)), replace=False))
    is_test = np.array([s in test_seeds for s in seed])
    out = {
        "events": int(len(y)),
        "train_seeds": int(len(seeds) - len(test_seeds)),
        "test_seeds": int(len(test_seeds)),
    }
    for k, name in ((0, "grasp_stable"), (1, "seated")):
        sel = kind == k
        tr, te = sel & ~is_test, sel & is_test
        res = {
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "pos_rate_train": float(y[tr].mean()),
            "pos_rate_test": float(y[te].mean()),
        }
        maj = float(y[tr].mean() >= 0.5)
        res["majority"] = scores(y[te], np.full(te.sum(), maj))
        for g, cols in GROUPS.items():
            mu, sd = X[tr][:, cols].mean(0), X[tr][:, cols].std(0) + 1e-6
            w, b = fit_logreg((X[tr][:, cols] - mu) / sd, y[tr])
            p = 1 / (1 + np.exp(-(((X[te][:, cols] - mu) / sd) @ w + b)))
            res[g] = scores(y[te], (p > 0.5).astype(float))
        out[name] = res
        print(
            name,
            json.dumps({g: round(res[g]["acc"], 3) for g in ("majority", *GROUPS)}),
            f"(test n={res['n_test']}, positives {res['pos_rate_test']:.2f})",
        )
    (Path(args.dir) / "detectors.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
