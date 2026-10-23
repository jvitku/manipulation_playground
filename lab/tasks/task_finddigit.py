"""Task "Find the digit" (id `find-digit`, ladder 9): a query token reads an image by attention.

A VLA in miniature. The input is a 28x56 canvas with two MNIST digits of DISTINCT classes side by
side, plus a command: a digit class 0-9 ("where is the 4?"). The answer is 0 = left, 1 = right,
2 = absent. A third of the samples are absent (the query is drawn from the 8 classes not shown).

Model: the canvas is cut into 7x7 patches (4x8 = 32 tokens), linearly embedded, a learned position
vector is added and `depth` `minitf.Block`s mix the patches (self-attention, as in task 7). The
command is a learned embedding of the query class (10 rows): ONE `minitf.MHA` lets that single
query token cross-attend over the 32 patch tokens (no CLS slot; the query is not part of the keys,
so its attention row holds exactly 32 weights summing to 1), and a LayerNorm + linear head turns
`query + readout` into the 3-way answer. Loss = cross-entropy over the three answers. Everything
trains jointly from scratch. d_model = heads * d_k.

Checkpoint (`last.pt`): {"meta": {"patch", "depth", "heads", "d_model", "steps"}, "state": ...}.
`evaluate` rebuilds the model from `meta` alone.

Metrics rows: `loss` (train batch) every ~steps/300 steps; `val_acc` and `absent_acc` (accuracy on
the absent cases only; same fixed 2000-canvas held-out set built from the test digits) every
~steps/15 steps. About SNAPSHOTS (60) rows also carry
    "attn":  [heads][1][32] the query's cross-attention row of every head over the patches (row by
             row of the 4x8 grid) for the probe canvas, "names": ["head 0", ...],
    "grid": [4, 8], "img": [28][56] uint8 probe pixels, "query": the probe's query class.
The probe is fixed: the first test digits of two different classes, the query names the right one.

result.json written by `evaluate` (viewer `finddigit`):
    {"heads", "depth", "grid": [4, 8], "noise", "query": "random" | "0".."9",
     "acc", "absent_acc" (held-out canvases, `n_eval` of them, with the same noise),
     "examples": [{"pixels": [28][56] uint8 (noise included), "query", "truth", "pred",
                   "conf" (softmax prob. of pred), "probs": [3], "classes": [left, right],
                   "attn": [heads][4][8] the query's weight on every patch,
                   "vec": {"query": [[d_model]] the query token, "patches": [idx...] the
                           (<= 8) most attended patches, "keys": [[d_model]...] their K vectors}}]}
Answers: 0 left, 1 right, 2 absent. Query "random" draws the digit like training does; with a fixed
digit the examples cycle truth left, right, absent so every case shows up.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lab.core.runctx import atomic_write_json
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.minitf import MHA, Block
from lab.tasks.mnist_data import load_mnist
from lab.tasks.task_vit import norm, patchify

PATCH, GRID = 7, (4, 8)
H, W = 28, 56
BATCH = 128
SNAPSHOTS, LOG_ROWS, VAL_ROWS = 60, 300, 15
VAL_N, VAL_SEED = 2000, 12345
FALLBACK_STEPS_PER_S, OVERHEAD_S = 60.0, 4.0
ANSWERS = ["left", "right", "absent"]


# ---- data ------------------------------------------------------------------------------------
def _by_class(labels) -> list[np.ndarray]:
    labels = np.asarray(labels)
    return [np.flatnonzero(labels == c) for c in range(10)]


def make_canvases(imgs, labels, n, rng, query=None, truth=None, pools=None):
    """n canvases [n, 28, 56] uint8 of two MNIST digits of distinct classes side by side.

    Returns (canvas, query [n], truth [n], classes [n, 2] = (left class, right class)).
    truth 0 = query is the left digit, 1 = the right one, 2 = absent (query is neither). `truth`
    defaults to uniform over the three answers (P(absent) = 1/3); `query` (int 0-9) pins the
    command, otherwise it is read off the construction (the left / right class, or a class
    drawn from the 8 not shown). `imgs` / `labels` are the pool the digits come from
    (`pools` = precomputed per-class index lists of `labels`).
    """
    pools = pools if pools is not None else _by_class(labels)
    if truth is None:
        truth = rng.integers(0, 3, n)
    truth = np.asarray(truth)
    step = rng.integers(1, 10, n)  # nonzero offset: a second class different from the first
    if query is None:
        left = rng.integers(0, 10, n)
        right = (left + step) % 10
        q = np.where(truth == 0, left, right)
        absent = np.flatnonzero(truth == 2)
        for _ in range(100):  # the query of an absent case: a class that is neither shown
            cand = rng.integers(0, 10, len(absent))
            q[absent] = cand
            bad = (cand == left[absent]) | (cand == right[absent])
            if not bad.any():
                break
            absent = absent[bad]
    else:
        q = np.full(n, int(query))
        left = np.where(truth == 0, q, (q + step) % 10)  # truth 1 / 2: any class but q
        right = np.where(truth == 1, q, (q + step) % 10)
        both = np.flatnonzero(truth == 2)  # absent: two distinct classes, neither is q
        for _ in range(100):
            right[both] = (q[both] + rng.integers(1, 10, len(both))) % 10
            bad = right[both] == left[both]
            if not bad.any():
                break
            both = both[bad]
        one = np.flatnonzero(truth == 0)  # left = q, right = another class
        right[one] = (q[one] + step[one]) % 10
        two = np.flatnonzero(truth == 1)
        left[two] = (q[two] + step[two]) % 10
    classes = np.stack([left, right], 1)
    canvas = np.zeros((n, H, W), np.uint8)
    for side in range(2):
        for c in range(10):
            m = np.flatnonzero(classes[:, side] == c)
            if len(m):
                pick = pools[c][rng.integers(0, len(pools[c]), len(m))]
                canvas[m, :, side * 28 : (side + 1) * 28] = imgs[pick]
    return canvas, q.astype(np.int64), truth.astype(np.int64), classes.astype(np.int64)


def add_noise(canvas, noise: float, rng):
    """Gaussian pixel noise, std = noise * 255, clipped to the valid pixel range (uint8)."""
    if noise <= 0:
        return np.array(canvas, dtype=np.uint8)
    x = np.asarray(canvas, np.float32) + rng.normal(0.0, noise * 255.0, np.shape(canvas))
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ---- model -----------------------------------------------------------------------------------
class FindDigit(nn.Module):
    def __init__(self, depth: int, heads: int, d_model: int):
        super().__init__()
        self.embed = nn.Linear(PATCH * PATCH, d_model)
        self.pos = nn.Parameter(torch.randn(1, GRID[0] * GRID[1], d_model) * 0.02)
        self.blocks = nn.ModuleList(Block(d_model, heads) for _ in range(depth))
        self.ln_kv = nn.LayerNorm(d_model)
        self.query = nn.Embedding(10, d_model)  # the command: one learned vector per class
        self.readout = MHA(d_model, heads)  # the query token cross-attends over the patches
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 3)

    def encode(self, imgs):
        """normalised float [B, 28, 56] -> patch tokens [B, 32, D] (what the query reads)."""
        x = self.embed(patchify(imgs, PATCH)) + self.pos
        for blk in self.blocks:
            x, _ = blk(x)
        return self.ln_kv(x)

    def read(self, q, tokens):
        """q: [B] query classes -> (logits [B, 3], attn [B, H, 1, 32])."""
        qt = self.query(q)[:, None]  # [B, 1, D]
        out, attn = self.readout(qt, tokens)
        return self.head(self.norm(qt + out)[:, 0]), attn

    def forward(self, imgs, q):
        return self.read(q, self.encode(imgs))


def build(meta: dict) -> FindDigit:
    return FindDigit(int(meta["depth"]), int(meta["heads"]), int(meta["d_model"]))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict(model, canvas, query, dev):
    """uint8 canvases + query classes -> (probs [n, 3], attn [n, heads, 32]) on the cpu."""
    model.eval()
    probs, attns = [], []
    for i in range(0, len(canvas), 500):
        x = norm(canvas[i : i + 500]).to(dev)
        logits, a = model(x, torch.as_tensor(query[i : i + 500]).to(dev))
        probs.append(logits.softmax(-1).cpu())
        attns.append(a[:, :, 0].cpu())
    return torch.cat(probs), torch.cat(attns)


def score(model, canvas, query, truth, dev) -> tuple[float, float]:
    """(accuracy, accuracy on the absent cases only)."""
    pred = predict(model, canvas, query, dev)[0].argmax(-1).numpy()
    hit = pred == truth
    absent = truth == 2
    return float(hit.mean()), float(hit[absent].mean()) if absent.any() else 0.0


def train(ctx) -> None:
    P = ctx.params
    steps = int(P["steps"])
    meta = {"patch": PATCH, "depth": int(P["depth"]), "heads": int(P["heads"]),
            "d_model": int(P["d_model"]), "steps": steps}  # fmt: skip
    dev, seed = _device(), int(P.get("seed", 0))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xtr, ytr = load_mnist("train")
    n = min(int(P["train_n"]), len(xtr))
    sel = np.random.default_rng(seed).permutation(len(xtr))[:n]
    pool_x, pool_pools = np.array(xtr[sel]), _by_class(ytr[sel])
    xte, yte = load_mnist("test")
    te_pools = _by_class(yte)
    vc, vq, vt, _ = make_canvases(xte, yte, VAL_N, np.random.default_rng(VAL_SEED), pools=te_pools)
    # the probe: a fixed canvas of the first two test digits of different classes; the query is
    # the right one, so a trained model should light the right half of the grid
    i0 = 0
    i1 = int(np.flatnonzero(yte != yte[0])[0])
    probe_u8 = np.concatenate([xte[i0], xte[i1]], 1).astype(np.uint8)
    probe_q = int(yte[i1])
    probe = norm(probe_u8[None]).to(dev)
    probe_qt = torch.tensor([probe_q], device=dev)
    names = [f"head {h}" for h in range(meta["heads"])]
    model = build(meta).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(P["lr"]), weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * max(0.02, 1 - s / steps)
    )
    log_every = max(1, steps // LOG_ROWS)
    val_every = log_every * max(1, round(steps / VAL_ROWS / log_every))
    snap_every = log_every * max(1, LOG_ROWS // SNAPSHOTS)
    ctx.set_status(state="running", message="training")
    val = 0.0
    for step in range(1, steps + 1):
        if not ctx.should_continue():
            ctx.set_status(state="stopped")
            return
        model.train()
        cv, q, t, _ = make_canvases(pool_x, None, BATCH, rng, pools=pool_pools)
        logits, _ = model(norm(cv).to(dev), torch.as_tensor(q).to(dev))
        loss = F.cross_entropy(logits, torch.as_tensor(t).to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            row = {"loss": float(loss)}
            if step % val_every == 0 or step == steps:
                val, absent = score(model, vc, vq, vt, dev)
                row["val_acc"], row["absent_acc"] = val, absent
            if step % snap_every == 0 or step == steps:
                model.eval()
                with torch.no_grad():
                    a = model(probe, probe_qt)[1][0, :, 0]  # [heads, 32]
                row["attn"] = a[:, None, :].double().round(decimals=3).tolist()
                row.update(names=names, grid=list(GRID), img=probe_u8.tolist(), query=probe_q)
            ctx.log(step, **row)
            ctx.set_status(progress=step / steps, message=f"step {step}/{steps} val {val:.3f}")
    ctx.save_ckpt(steps, {"meta": meta, "state": model.state_dict()}, label="final",
                  metrics={"val_acc": val})  # fmt: skip
    ctx.done(f"trained: val acc {val:.3f}")


def evaluate(ctx) -> None:
    P = ctx.params
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    meta = ck["meta"]  # the model is rebuilt from the checkpoint alone
    dev = _device()
    model = build(meta)
    model.load_state_dict(ck["state"])
    model.to(dev).eval()
    noise = float(P.get("noise", 0))
    qsel = str(P.get("query", "random"))
    fixed = None if qsel == "random" else int(qsel)
    rng = np.random.default_rng(int(P.get("eval_seed", 0)))
    xte, yte = load_mnist("test")
    pools = _by_class(yte)
    n_ex = int(P["n_examples"])
    # accuracy over a natural held-out set (same query rule, same noise)
    tr = np.random.default_rng(VAL_SEED + 1 + int(P.get("eval_seed", 0)))
    cv, q, t, _ = make_canvases(xte, yte, VAL_N, tr, query=fixed, pools=pools)
    acc, absent_acc = score(model, add_noise(cv, noise, tr), q, t, dev)
    # the examples cycle left / right / absent so an absent case is always among them
    cycle = np.arange(n_ex) % 3
    cv, q, t, cls = make_canvases(xte, yte, n_ex, rng, query=fixed, truth=cycle, pools=pools)
    cv = add_noise(cv, noise, rng)
    probs, attn = predict(model, cv, q, dev)
    with torch.no_grad():
        tokens = model.encode(norm(cv).to(dev))
        qv = model.query(torch.as_tensor(q).to(dev))
        keys = model.readout.k(tokens)
    gh, gw = GRID
    examples = []
    for j in range(n_ex):
        pred = int(probs[j].argmax())
        top = np.sort(np.argsort(-attn[j].mean(0).numpy())[:8])
        examples.append({
            "pixels": cv[j].tolist(), "query": int(q[j]), "truth": int(t[j]), "pred": pred,
            "conf": round(float(probs[j, pred]), 4),
            "probs": probs[j].double().round(decimals=4).tolist(),
            "classes": cls[j].tolist(),
            "attn": attn[j].reshape(-1, gh, gw).double().round(decimals=4).tolist(),
            "vec": {"query": [qv[j].cpu().double().round(decimals=3).tolist()],
                    "patches": top.tolist(),
                    "keys": keys[j, top].cpu().double().round(decimals=3).tolist()},
        })  # fmt: skip
    atomic_write_json(ctx.dir / "result.json", {
        "heads": int(meta["heads"]), "depth": int(meta["depth"]), "grid": list(GRID),
        "noise": noise, "query": qsel, "acc": acc, "absent_acc": absent_acc, "n_eval": VAL_N,
        "examples": examples,
    })  # fmt: skip
    ctx.done(f"evaluated: acc {acc:.3f}, absent acc {absent_acc:.3f}")


def estimate(params: dict) -> float:
    cal = registry.calib("find-digit")
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    return params["steps"] / sps + cal.get("overhead_s", OVERHEAD_S)


SPEC = TaskSpec(
    id="find-digit",
    order=9,
    title="Find the digit",
    subtitle="A command token reads an image: is the queried digit left, right or absent?",
    params=[
        Param("depth", "Transformer blocks", "choice", 2, choices=[1, 2]),
        Param("heads", "Heads", "choice", 2, choices=[2, 4]),
        Param("d_model", "Model width d_model", "int", 64, min=32, max=128, step=32),
        Param("lr", "Learning rate", "float", 1e-3, min=3e-4, max=3e-3),
        Param("steps", "Training steps", "int", 3000, min=500, max=10000, step=100),
        Param("train_n", "Training digits", "int", 10000, min=1000, max=60000, step=1000),
        Param("seed", "Seed", "int", 0, min=0, max=9999),
    ],
    eval_params=[
        Param("eval_seed", "Eval seed", "int", 0, min=0, max=9999),
        Param("n_examples", "Canvases to show", "int", 8, min=1, max=24, step=1),
        Param("noise", "Pixel noise (std, fraction of full scale)", "float", 0.0,
              min=0.0, max=0.5, step=0.05),
        Param("query", "Query digit", "choice", "random",
              choices=["random"] + [str(i) for i in range(10)]),
    ],
    charts=[
        {"id": "loss", "title": "Cross-entropy loss",
         "series": [{"key": "loss", "label": "train batch"}], "log_y": True},
        {"id": "val_acc", "title": "Validation accuracy (left / right / absent)",
         "series": [{"key": "val_acc", "label": "2000 held-out canvases"}], "log_y": False},
        {"id": "absent_acc", "title": "Accuracy on the absent cases only",
         "series": [{"key": "absent_acc", "label": "query not on the canvas"}], "log_y": False},
    ],
    viewer="finddigit",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="find-digit.html",
    vecgrids=True,
    live="fdlive",
)  # fmt: skip
registry.register(SPEC)
