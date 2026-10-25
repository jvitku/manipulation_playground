"""Task "A tiny Vision Transformer" (id `vit-mnist`, ladder 7): MNIST digits via patch attention.

The image is cut into square patches (`patch` 7 -> 4x4 = 16 patches, 14 -> 2x2 = 4), every patch is
flattened and linearly embedded, a learned position vector is added to each (attention itself is
order-blind), and a learned CLS token is put in front. `depth` pre-norm `minitf.Block`s mix the
tokens; the final CLS vector goes through a LayerNorm and a linear head to 10 classes. Loss =
cross-entropy. The CLS token is the only one that is read out, so the CLS row of each attention
map ("where does the summary token look?") shows what the model uses. d_model = heads * d_k.

Checkpoint (`last.pt`): {"meta": {"patch", "depth", "heads", "d_model", "steps"}, "state": ...}.
`evaluate` rebuilds the model from `meta` alone.

Metrics rows: `loss` (train batch) every ~steps/300 steps; `val_acc` (accuracy on a fixed 1000-image
held-out set, the first 1000 test digits) every ~steps/15 steps. About SNAPSHOTS (60) rows also
carry
    "attn":  [depth*heads][1][1 + gh*gw] the CLS row of every layer's every head (layer-major;
             index 0 = CLS looking at itself, then the patches row by row) for the probe digit,
    "names": ["layer 0 head 0", ...], "grid": [gh, gw], "img": [28][28] uint8 probe pixels.
That is about 3.5 KB at the largest config, so a metrics line stays far below 8 KB.

result.json written by `evaluate` (viewer `vitattn`), the digits are random test images:
    {"patch", "depth", "heads", "grid": [gh, gw], "acc" (whole 10000-image test set),
     "examples": [{"pixels": [28][28] uint8, "label", "pred", "conf" (softmax prob. of pred),
                   "cls_self": [depth][heads] (CLS weight on itself),
                   "attn": [depth][heads][gh][gw]  CLS weight on every patch (not renormalised:
                   sum + cls_self = 1)}]}
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lab.core.runctx import atomic_write_json
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.minitf import Block
from lab.tasks.mnist_data import load_mnist

IMG, BATCH = 28, 128
SNAPSHOTS, LOG_ROWS, VAL_ROWS = 60, 300, 15
VAL_N = 1000
MEAN, STD = 0.1307, 0.3081
FALLBACK_STEPS_PER_S, OVERHEAD_S = 60.0, 4.0


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def patchify(imgs: torch.Tensor, patch: int) -> torch.Tensor:
    """[B, H, W] -> [B, N, patch*patch] flattened patches, row-major over the patch grid.

    Square 28x28 digits give 28/patch squared patches; find-digit's 28x56 canvas gives 4x8.
    """
    b, hh, ww = imgs.shape
    gh, gw = hh // patch, ww // patch
    x = imgs.reshape(b, gh, patch, gw, patch).permute(0, 1, 3, 2, 4)  # [B, gy, gx, py, px]
    return x.reshape(b, gh * gw, patch * patch)


class ViT(nn.Module):
    def __init__(self, patch: int, depth: int, heads: int, d_model: int):
        super().__init__()
        self.patch, n = patch, (IMG // patch) ** 2
        self.embed = nn.Linear(patch * patch, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n + 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(Block(d_model, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 10)

    def input_tokens(self, imgs):
        """imgs -> [B, 1 + n, d_model]: CLS + patch embeddings + position (the blocks' input)."""
        x = self.embed(patchify(imgs, self.patch))
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        return x

    def forward(self, imgs):
        """imgs: normalised float [B, 28, 28] -> (logits [B, 10], attn list of [B, H, T, T])."""
        x = self.input_tokens(imgs)
        attns = []
        for blk in self.blocks:
            x, a = blk(x)
            attns.append(a)
        return self.head(self.norm(x)[:, 0]), attns


def build(meta: dict) -> ViT:
    return ViT(int(meta["patch"]), int(meta["depth"]), int(meta["heads"]), int(meta["d_model"]))


def norm(u8) -> torch.Tensor:
    """uint8 images -> normalised float tensor (a copy: the MNIST arrays are read-only)."""
    return (torch.as_tensor(np.array(u8, dtype=np.float32)) / 255.0 - MEAN) / STD


def _cls_rows(attns) -> torch.Tensor:
    """list over layers of [B, H, T, T] -> [B, depth, H, T]: the CLS query row of every head."""
    return torch.stack([a[:, :, 0] for a in attns], 1)


@torch.no_grad()
def accuracy(model, x, y, dev) -> float:
    model.eval()
    hit = 0
    for i in range(0, len(x), 1000):
        hit += int((model(x[i : i + 1000].to(dev))[0].argmax(-1).cpu() == y[i : i + 1000]).sum())
    return hit / len(x)


def train(ctx) -> None:
    P = ctx.params
    steps = int(P["steps"])
    meta = {"patch": int(P["patch"]), "depth": int(P["depth"]), "heads": int(P["heads"]),
            "d_model": int(P["d_model"]), "steps": steps}  # fmt: skip
    dev, seed = _device(), int(P.get("seed", 0))
    torch.manual_seed(seed)
    xtr, ytr = load_mnist("train")
    n = min(int(P["train_n"]), len(xtr))
    sel = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(seed))[:n].numpy()
    x, y = norm(xtr[sel]).to(dev), torch.as_tensor(ytr[sel].astype(np.int64)).to(dev)
    xte, yte = load_mnist("test")
    xv, yv = norm(xte[:VAL_N]), torch.as_tensor(yte[:VAL_N].astype(np.int64))
    probe_u8 = np.array(xte[0])  # the fixed probe digit (a 7)
    probe = norm(probe_u8)[None].to(dev)
    g = IMG // meta["patch"]
    names = [
        f"layer {layer} head {h}" for layer in range(meta["depth"]) for h in range(meta["heads"])
    ]
    model = build(meta).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(P["lr"]), weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * max(0.02, 1 - s / steps)
    )
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
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
        idx = torch.randint(n, (BATCH,), generator=gen, device=dev)
        loss = F.cross_entropy(model(x[idx])[0], y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            row = {"loss": float(loss)}
            if step % val_every == 0 or step == steps:
                val = row["val_acc"] = accuracy(model, xv, yv, dev)
            if step % snap_every == 0 or step == steps:
                model.eval()
                with torch.no_grad():
                    rows = _cls_rows(model(probe)[1])[0]  # [depth, H, T]
                row["attn"] = (
                    rows.reshape(-1, 1, rows.shape[-1]).double().round(decimals=3).tolist()
                )
                row.update(names=names, grid=[g, g], img=probe_u8.tolist())
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
    ctx.set_status(message="evaluating: scoring the MNIST test set")
    xte, yte = load_mnist("test")
    acc = accuracy(model, norm(xte), torch.as_tensor(yte.astype(np.int64)), dev)
    gen = torch.Generator().manual_seed(int(P.get("eval_seed", 0)))
    idx = torch.randperm(len(xte), generator=gen)[: int(P["n_examples"])]
    g = IMG // int(meta["patch"])
    D, H = int(meta["depth"]), int(meta["heads"])
    with torch.no_grad():
        logits, attns = model(norm(xte[idx.numpy()]).to(dev))
        probs, rows = logits.softmax(-1).cpu(), _cls_rows(attns).cpu()  # rows [B, D, H, T]
        toks = model.input_tokens(norm(xte[idx.numpy()]).to(dev)).cpu()  # [B, 1 + n, d_model]
    examples = []
    for j, i in enumerate(idx.tolist()):
        pred = int(probs[j].argmax())
        examples.append({
            "pixels": np.array(xte[i]).tolist(), "label": int(yte[i]), "pred": pred,
            "conf": round(float(probs[j, pred]), 4),
            "cls_self": rows[j, :, :, 0].double().round(decimals=4).tolist(),
            "vec": {"cls": toks[j, :1].double().round(decimals=3).tolist(),
                    "patches": toks[j, 1:].double().round(decimals=3).tolist()},
            "attn": rows[j, :, :, 1:].reshape(D, H, g, g).double().round(decimals=4).tolist(),
        })  # fmt: skip
    atomic_write_json(ctx.dir / "result.json", {
        "patch": int(meta["patch"]), "depth": D, "heads": H, "grid": [g, g], "acc": acc,
        "examples": examples,
    })  # fmt: skip
    ctx.done(f"evaluated: test acc {acc:.3f}")


def estimate(params: dict) -> float:
    cal = registry.calib("vit-mnist")
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    return params["steps"] / sps + cal.get("overhead_s", OVERHEAD_S)


SPEC = TaskSpec(
    id="vit-mnist",
    order=7,
    title="A tiny Vision Transformer",
    subtitle="Read digits with patches, a CLS token and attention - see where CLS looks.",
    params=[
        Param("patch", "Patch size (px)", "choice", 7, choices=[7, 14]),
        Param("depth", "Transformer blocks", "choice", 2, choices=[1, 2]),
        Param("heads", "Heads", "choice", 2, choices=[2, 4]),
        Param("d_model", "Model width d_model", "int", 64, min=32, max=128, step=32),
        Param("lr", "Learning rate", "float", 1e-3, min=3e-4, max=3e-3),
        Param("steps", "Training steps", "int", 2000, min=200, max=8000, step=100),
        Param("train_n", "Training images", "int", 10000, min=1000, max=60000, step=1000),
        Param("seed", "Seed", "int", 0, min=0, max=9999),
    ],
    eval_params=[
        Param("eval_seed", "Eval seed", "int", 0, min=0, max=9999),
        Param("n_examples", "Digits to show", "int", 8, min=1, max=24, step=1),
    ],
    charts=[
        {"id": "loss", "title": "Cross-entropy loss",
         "series": [{"key": "loss", "label": "train batch"}], "log_y": True},
        {"id": "val_acc", "title": "Validation accuracy",
         "series": [{"key": "val_acc", "label": "1000 held-out digits"}], "log_y": False},
    ],
    viewer="vitattn",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="vit-mnist.html",
    vecgrids=True,
    live="attnlive",
)  # fmt: skip
registry.register(SPEC)
