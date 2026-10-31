"""Task "Attention is a lookup" (id `attention`, ladder 6): learn to fetch a value by its key.

Each example is `n_pairs` tokens plus one query token. A pair token is the SUM of a key embedding
and a value embedding (the key and its value fused into one vector); the query token is the key
embedding of one of the keys plus a learned "?" vector standing in for the missing value. Keys in
one example are distinct (drawn from N_KEYS), values are random (N_VALS). There are no position
embeddings, so the model can only find the answer by matching the query's key against the pair
tokens' keys. The model: embeddings -> ONE `minitf.MHA` (the query token is the only query; keys
and values come from all n_pairs + 1 tokens) -> a linear readout of the mixed vector -> a
classifier over the values. Loss = cross-entropy on the value. d_model = heads * d_k.

Checkpoint (`last.pt`): {"meta": {"heads", "d_k", "n_keys", "n_vals", "n_pairs", "steps"},
"state": state_dict}. `evaluate` rebuilds the model from `meta` alone.

Metrics rows: `loss`, `acc` (accuracy on the fresh training batch; data is generated on the fly,
so it is also held-out accuracy). About SNAPSHOTS (60) rows per run also carry
    "attn": [H][1][T] attention of the query token over the T = n_pairs + 1 tokens (the fixed
            probe example; `minitf.attn_snapshot`), and
    "qkv":  {"labels": [T token labels], "heads": [{"q": [[dk]], "k": [T][dk], "v": [T][dk]}]}
            for the same probe (heads dropped from the end if the row would exceed QKV_BUDGET
            numbers, so a metrics line stays far below 8 KB).

result.json written by `evaluate` (viewer `attnstep`):
    {"n_pairs", "heads", "d_k", "acc", "acc_by_n": {"2".."8" and n_pairs: accuracy},
     "probe": {"tokens": [{"label", "key", "value"}...] (last = the query: "?k<key>"),
               "query_key", "target", "pred"},
     "stages": [                                    # the inference stepper, in order
       {"name": "embed",   "tokens": [T][d_model]},
       {"name": "qkv",     "q": [H][1][dk], "k": [H][T][dk], "v": [H][T][dk]},
       {"name": "scores",  "pre": [H][T]},         # q.k / sqrt(d_k), before softmax
       {"name": "softmax", "attn": [H][T]},
       {"name": "mixture", "per_head": [H][dk], "concat": [d_model], "projected": [d_model]},
       {"name": "readout", "logits": [V], "probs": [V], "pred", "target"}]}
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from lab.core.runctx import atomic_write_json
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.minitf import MHA, attn_snapshot

N_KEYS, N_VALS = 16, 16
BATCH = 256
SNAPSHOTS = 60  # attention snapshots per run
LOG_ROWS = 300  # loss/acc rows per run
QKV_BUDGET = 700  # max numbers in a row's qkv payload
EVAL_N, EVAL_BATCH = 2000, 1000
FALLBACK_STEPS_PER_S, OVERHEAD_S = 100.0, 3.0


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class AttnLookup(nn.Module):
    def __init__(self, heads: int, d_k: int, n_keys: int, n_vals: int):
        super().__init__()
        d = heads * d_k
        self.ek, self.ev = nn.Embedding(n_keys, d), nn.Embedding(n_vals, d)
        self.qtok = nn.Parameter(torch.randn(d) * 0.1)  # the "?" that replaces the value
        self.mha = MHA(d, heads)
        self.readout = nn.Linear(d, n_vals)

    def tokens(self, keys, vals, qpos):
        """[B, n+1, D]: fused (key + value) pair tokens, then the query token last."""
        qk = keys.gather(1, qpos[:, None])[:, 0]
        query = (self.ek(qk) + self.qtok)[:, None]
        return torch.cat([self.ek(keys) + self.ev(vals), query], 1)

    def forward(self, keys, vals, qpos):
        # legend: B = batch, n = pairs, T = n + 1 tokens, D = d_model, H = heads, V = n_vals
        x = self.tokens(keys, vals, qpos)  # keys [B, n], vals [B, n], qpos [B] -> x [B, T, D]
        out, attn = self.mha(x[:, -1:], x)  # q from the query token [B, 1, D]; attn [B, H, 1, T]
        return self.readout(out[:, 0]), x, attn  # out [B, 1, D] -> logits [B, V]


def build(meta: dict) -> AttnLookup:
    return AttnLookup(
        int(meta["heads"]), int(meta["d_k"]), int(meta["n_keys"]), int(meta["n_vals"])
    )


def make_batch(b: int, n: int, meta: dict, gen: torch.Generator, dev):
    """keys [b, n] (distinct per row), vals [b, n], qpos [b] (which pair is asked for)."""
    nk, nv = int(meta["n_keys"]), int(meta["n_vals"])
    keys = torch.rand(b, nk, generator=gen, device=dev).argsort(1)[:, :n]
    vals = torch.randint(nv, (b, n), generator=gen, device=dev)
    qpos = torch.randint(n, (b,), generator=gen, device=dev)
    return keys, vals, qpos


def _r(t, nd=4):
    return t.detach().double().round(decimals=nd).tolist()


def _qkv(model, x):
    """Per-head Q (query token only), K, V of tokens x [1, T, D] -> three [H][.][dk] tensors."""
    m = model.mha
    q, k, v = m.split(m.q(x[:, -1:])), m.split(m.k(x)), m.split(m.v(x))
    return q[0], k[0], v[0]  # [H, 1|T, dk]


def forward_stages(model, keys, vals, qpos) -> dict:
    """Every intermediate of one example (batch of 1), by name; see the module docstring."""
    m = model.mha
    with torch.no_grad():
        x = model.tokens(keys, vals, qpos)
        q, k, v = _qkv(model, x)
        scores = (q @ k.transpose(-2, -1))[:, 0] / m.dh**0.5  # [H, T]
        attn = scores.softmax(-1)
        per_head = (attn[:, None] @ v)[:, 0]  # [H, dk]
        concat = per_head.reshape(-1)
        projected = m.out(concat)
        logits = model.readout(projected)
    return {
        "embed": {"tokens": _r(x[0])},
        "qkv": {"q": _r(q), "k": _r(k), "v": _r(v)},
        "scores": {"pre": _r(scores)},
        "softmax": {"attn": _r(attn)},
        "mixture": {"per_head": _r(per_head), "concat": _r(concat), "projected": _r(projected)},
        "readout": {
            "logits": _r(logits),
            "probs": _r(logits.softmax(-1)),
            "pred": int(logits.argmax()),
            "target": int(vals[0, qpos[0]]),
        },
    }


def _labels(keys, vals, qpos) -> list[str]:
    q = int(keys[0, qpos[0]])
    return [f"k{int(a)}:v{int(b)}" for a, b in zip(keys[0], vals[0], strict=True)] + [f"?k{q}"]


def _probe(meta: dict, n: int, seed: int, dev):
    return make_batch(1, n, meta, torch.Generator(device=dev).manual_seed(seed), dev)


def _qkv_payload(model, keys, vals, qpos) -> dict:
    with torch.no_grad():
        x = model.tokens(keys, vals, qpos)
        q, k, v = _qkv(model, x)
    H, T, dk = q.shape[0], k.shape[1], q.shape[2]
    keep = max(1, min(H, QKV_BUDGET // (dk * (1 + 2 * T))))
    heads = [{"q": _r(q[h], 2), "k": _r(k[h], 2), "v": _r(v[h], 2)} for h in range(keep)]
    return {"labels": _labels(keys, vals, qpos), "heads": heads}


@torch.no_grad()
def accuracy(model, meta: dict, n: int, count: int, seed: int, dev) -> float:
    # legend: b = chunk of examples, n = pairs, V = n_vals
    model.eval()
    gen = torch.Generator(device=dev).manual_seed(seed)
    hit = tot = 0
    while tot < count:
        b = min(EVAL_BATCH, count - tot)
        keys, vals, qpos = make_batch(b, n, meta, gen, dev)  # keys/vals [b, n], qpos [b]
        pred = model(keys, vals, qpos)[0].argmax(-1)  # logits [b, V] -> pred [b]
        hit += int((pred == vals.gather(1, qpos[:, None])[:, 0]).sum())
        tot += b
    return hit / tot


def train(ctx) -> None:
    # legend: B = BATCH examples per step, n = pairs per example, V = n_vals value classes
    P = ctx.params
    steps, n = int(P["steps"]), int(P["n_pairs"])
    meta = {"heads": int(P["heads"]), "d_k": int(P["d_k"]), "n_keys": N_KEYS, "n_vals": N_VALS,
            "n_pairs": n, "steps": steps}  # fmt: skip
    dev = _device()
    torch.manual_seed(int(P.get("seed", 0)))
    model = build(meta).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(P["lr"]), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * max(0.05, 1 - s / steps)
    )
    gen = torch.Generator(device=dev).manual_seed(int(P.get("seed", 0)) + 1)
    probe = _probe(meta, n, 12345, dev)
    log_every = max(1, steps // LOG_ROWS)
    snap_every = log_every * max(1, LOG_ROWS // SNAPSHOTS)
    ctx.set_status(state="running", message="training")
    acc = 0.0
    for step in range(1, steps + 1):
        if not ctx.should_continue():
            ctx.set_status(state="stopped")
            return
        model.train()
        keys, vals, qpos = make_batch(BATCH, n, meta, gen, dev)  # keys/vals [B, n], qpos [B]
        target = vals.gather(1, qpos[:, None])[:, 0]  # the value under the queried key: [B]
        logits, _, _ = model(keys, vals, qpos)  # [B, V]
        loss = F.cross_entropy(logits, target)  # [B, V] vs [B] -> scalar
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            acc = float((logits.argmax(-1) == target).float().mean())
            row = {"loss": float(loss), "acc": acc}
            if step % snap_every == 0 or step == steps:
                model.eval()
                with torch.no_grad():
                    attn = model(*probe)[2]
                row["attn"] = attn_snapshot(attn)
                row["qkv"] = _qkv_payload(model, *probe)
            ctx.log(step, **row)
            ctx.set_status(progress=step / steps, message=f"step {step}/{steps} acc {acc:.3f}")
    ctx.save_ckpt(
        steps, {"meta": meta, "state": model.state_dict()}, label="final", metrics={"acc": acc}
    )
    ctx.done(f"trained: acc {acc:.3f}")


def evaluate(ctx) -> None:
    P = ctx.params
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    meta = ck["meta"]  # the model is rebuilt from the checkpoint alone
    dev = _device()
    model = build(meta)
    model.load_state_dict(ck["state"])
    model.to(dev).eval()
    ctx.set_status(message=f"evaluating: accuracy sweep over 2-8 pairs, {EVAL_N} examples each")
    n, seed = int(P.get("eval_n_pairs", meta["n_pairs"])), int(P.get("eval_seed", 0))
    acc_by_n = {str(k): accuracy(model, meta, k, EVAL_N, seed + k, dev) for k in range(2, 9)}
    acc = acc_by_n.get(str(n)) or accuracy(model, meta, n, EVAL_N, seed + n, dev)
    acc_by_n[str(n)] = acc
    keys, vals, qpos = _probe(meta, n, seed, dev)
    st = forward_stages(model, keys, vals, qpos)
    labels = _labels(keys, vals, qpos)
    toks = [{"label": lab, "key": int(k), "value": int(v)}
            for lab, k, v in zip(labels[:-1], keys[0], vals[0], strict=True)]  # fmt: skip
    q = int(keys[0, qpos[0]])
    toks.append({"label": labels[-1], "key": q, "value": None})
    names = ["embed", "qkv", "scores", "softmax", "mixture", "readout"]
    result = {
        "n_pairs": n, "heads": meta["heads"], "d_k": meta["d_k"], "acc": acc, "acc_by_n": acc_by_n,
        "probe": {"tokens": toks, "query_key": q, "target": st["readout"]["target"],
                  "pred": st["readout"]["pred"]},
        "stages": [{"name": s, **st[s]} for s in names],
    }  # fmt: skip
    atomic_write_json(ctx.dir / "result.json", result)
    ctx.done(f"evaluated: acc {acc:.3f}")


def estimate(params: dict) -> float:
    cal = registry.calib("attention")
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    return params["steps"] / sps + cal.get("overhead_s", OVERHEAD_S)


SPEC = TaskSpec(
    id="attention",
    order=6,
    title="Attention is a lookup",
    subtitle="Fetch the value stored under a key with one attention layer - watch Q, K and V form.",
    params=[
        Param("heads", "Heads", "choice", 2, choices=[1, 2, 4]),
        Param("d_k", "Key size d_k (per head)", "int", 16, min=8, max=32, step=8),
        Param("n_pairs", "Key-value pairs per example", "int", 4, min=2, max=8, step=1),
        Param("steps", "Training steps", "int", 1500, min=200, max=10000, step=100),
        Param("lr", "Learning rate", "float", 3e-3, min=3e-4, max=1e-2),
        Param("seed", "Seed", "int", 0, min=0, max=9999),
    ],
    eval_params=[
        Param("eval_seed", "Eval seed", "int", 0, min=0, max=9999),
        Param("eval_n_pairs", "Pairs in the probe", "int", 4, min=2, max=12, step=1,
              help="the model has no position embeddings, so it can be tested on more pairs"),
    ],  # fmt: skip
    charts=[
        {"id": "loss", "title": "Cross-entropy loss",
         "series": [{"key": "loss", "label": "train"}], "log_y": True},
        {"id": "acc", "title": "Lookup accuracy",
         "series": [{"key": "acc", "label": "train batch"}], "log_y": False},
    ],
    viewer="attnstep",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="attention.html",
    vecgrids=True,
    live="attnlive",
)  # fmt: skip
registry.register(SPEC)
