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


def fit_mlp(X, y, hidden=128, iters=6000, lr=3e-3, l2=1e-3, seed=0):
    """One hidden layer (tanh), class-balanced logistic loss, full-batch Adam (numpy)."""
    rng = np.random.default_rng(seed)
    p = {
        "W1": rng.normal(0, 1 / np.sqrt(X.shape[1]), (X.shape[1], hidden)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(0, 1 / np.sqrt(hidden), hidden),
        "b2": np.zeros(1),
    }
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    pos = max(y.mean(), 1e-3)
    sw = np.where(y == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-3))
    for t in range(1, iters + 1):
        h = np.tanh(X @ p["W1"] + p["b1"])
        q = 1 / (1 + np.exp(-(h @ p["w2"] + p["b2"])))
        g = sw * (q - y) / len(y)
        dh = np.outer(g, p["w2"]) * (1 - h**2)
        grads = {
            "w2": h.T @ g + l2 * p["w2"],
            "b2": np.array([g.sum()]),
            "W1": X.T @ dh + l2 * p["W1"],
            "b1": dh.sum(0),
        }
        for k in p:
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            p[k] -= lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v[k] / (1 - 0.999**t)) + 1e-8)
    return lambda Z: 1 / (1 + np.exp(-(np.tanh(Z @ p["W1"] + p["b1"]) @ p["w2"] + p["b2"])))


def scores(y, pred):
    acc = float(np.mean(pred == y))
    tpr = float(np.mean(pred[y == 1] == 1)) if np.any(y == 1) else float("nan")
    tnr = float(np.mean(pred[y == 0] == 0)) if np.any(y == 0) else float("nan")
    return {"acc": acc, "balanced_acc": (tpr + tnr) / 2, "tpr": tpr, "tnr": tnr}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="outputs/v8_labels")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--glob", default="part*/events.npz")
    args = ap.parse_args()

    parts = sorted(Path(args.dir).glob(args.glob))
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
            if g == "all":
                net = fit_mlp((X[tr][:, cols] - mu) / sd, y[tr])
                res["mlp"] = scores(y[te], (net((X[te][:, cols] - mu) / sd) > 0.5).astype(float))
        out[name] = res
        print(
            name,
            json.dumps({g: round(res[g]["acc"], 3) for g in ("majority", *GROUPS, "mlp")}),
            f"(test n={res['n_test']}, positives {res['pos_rate_test']:.2f})",
        )
    (Path(args.dir) / "detectors.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
