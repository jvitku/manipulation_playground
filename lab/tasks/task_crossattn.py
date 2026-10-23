"""Task "Cross-attention: translate a date" (id `cross-attn`, ladder 8): char-level seq2seq.

The source is a date written one of three ways ("23 Oct 2026", "Oct 23, 2026", "23/10/2026",
uniform choice), the target is always ISO ("2026-10-23"). Dates span 1970-01-01 .. 2070-12-31
(36 890 days). Vocabulary: one id per character plus PAD / BOS / EOS. The split is BY DATE: a hash
of the day number puts ~10% of the days in the held-out set, so `exact` (held-out exact-match) is
measured on days the model has never seen, in any of the three formats.

Model: char embedding + learned positions -> ONE encoder `minitf.Block` (self-attention + MLP)
= the memory; decoder: char embedding + learned positions -> ONE `minitf.CrossBlock` (causal
self-attention, cross-attention into the memory, MLP) -> final LayerNorm -> linear head over the
vocabulary. Q of the cross-attention comes from the decoder, K and V from the encoder. Training is
teacher forcing: the decoder reads BOS + the target and predicts the target + EOS, cross-entropy
ignoring PAD. Inference (`greedy_decode`) starts from BOS alone and feeds back its own argmax until
EOS. d_model = any multiple of heads.

Checkpoint (`last.pt`): {"meta": {"d_model", "heads", "steps"}, "state": state_dict}.

Metrics rows: `loss` every step block; about 20 rows carry `exact` (held-out exact-match, greedy
decode of EXACT_N held-out examples); about SNAPSHOTS (60) rows carry
    "attn":  [H][Tq][Tk] cross-attention of the fixed probe pair (teacher-forced), rows = decoder
             positions (BOS, then the target chars), columns = source characters, and
    "qkv":   {"labels": [Tk source char labels], "heads": []} (labels for the heatmap hover;
             empty heads = labels only; live Q/K/V is deferred, see the eval-side `vec`).

result.json written by `evaluate` (viewer `xattn`):
    {"d_model", "heads", "exact", "n_eval", "seed",
     "examples": [{"source", "target", "decoded", "correct",
                   "src_labels": [S chars of the padded source],
                   "in_labels":  [L decoder inputs: "<bos>" then the decoded chars],
                   "out_labels": [L predicted chars: the decoded chars then "<eos>"],
                   "enc_attn":   [H][S][S],     # encoder self-attention
                   "dec_attn":   [H][L][L],     # decoder causal self-attention (row = position)
                   "cross_attn": [H][L][S],     # row t = what position t read to emit out_labels[t]
                   "vec": {"q": [H][L][d_k],    # decoder-side Q per head and output position
                           "k": [H][S][d_k],    # encoder-side K per head and source position
                           "v": [H][S][d_k]}}]} # encoder-side V (3 decimals)
The matrices come from re-running the model on BOS + the greedy output, which under the causal mask
is the same computation as the incremental decode.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from lab.core.runctx import atomic_write_json
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.minitf import Block, CrossBlock, attn_snapshot, causal_mask

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FORMATS = ("{d:02d} {m} {y}", "{m} {d:02d}, {y}", "{d:02d}/{n:02d}/{y}")
FIRST, LAST = dt.date(1970, 1, 1).toordinal(), dt.date(2070, 12, 31).toordinal()
HELD_OUT_FRAC = 10  # one day in HELD_OUT_FRAC is held out
CHARS = sorted(set("0123456789-/, " + "".join(MONTHS)))
PAD, BOS, EOS = 0, 1, 2
ITOS = ["<pad>", "<bos>", "<eos>"] + CHARS
STOI = {c: i for i, c in enumerate(ITOS)}
SRC_LEN, TGT_LEN = 12, 10  # padded source length; ISO target length
MAX_OUT = TGT_LEN + 1  # target chars + EOS

BATCH = 128
SNAPSHOTS, LOG_ROWS, EXACT_ROWS = 60, 300, 20
EXACT_N, EVAL_N = 500, 2000
PROBE_DATE, PROBE_FMT = dt.date(2026, 10, 23).toordinal(), 1
FALLBACK_STEPS_PER_S, OVERHEAD_S = 100.0, 3.0


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---- data ---------------------------------------------------------------------------------------


def is_held_out(ordinal: int) -> bool:
    """Deterministic split by day (Knuth multiplicative hash): a day is in exactly one split."""
    return (ordinal * 2654435761 % 2**32) * HELD_OUT_FRAC // 2**32 == 0


@lru_cache(maxsize=1)
def split_days() -> tuple[tuple[int, ...], tuple[int, ...]]:
    days = range(FIRST, LAST + 1)
    return (tuple(d for d in days if not is_held_out(d)), tuple(d for d in days if is_held_out(d)))


def format_pair(ordinal: int, fmt: int) -> tuple[str, str]:
    """(source string in format `fmt`, ISO target) for a day number."""
    d = dt.date.fromordinal(ordinal)
    src = FORMATS[fmt].format(d=d.day, m=MONTHS[d.month - 1], n=d.month, y=d.year)
    return src, d.isoformat()


def date_pairs(split: str, n: int, seed: int) -> list[tuple[str, str, int]]:
    """The generator: n random (source, target, day number) triples from `split` ('train' or
    'heldout'); the source format is drawn uniformly per sample."""
    days = split_days()[0 if split == "train" else 1]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(days), (n,), generator=g).tolist()
    fm = torch.randint(len(FORMATS), (n,), generator=g).tolist()
    return [(*format_pair(days[i], f), days[i]) for i, f in zip(idx, fm, strict=True)]


def encode_src(s: str) -> list[int]:
    return [STOI[c] for c in s] + [PAD] * (SRC_LEN - len(s))


def encode_tgt(s: str) -> list[int]:
    return [STOI[c] for c in s] + [EOS]


def decode_ids(ids) -> str:
    out = []
    for i in ids:
        if i == EOS:
            break
        if i > EOS:
            out.append(ITOS[i])
    return "".join(out)


@lru_cache(maxsize=2)
def _table(split: str):
    """All (day, format) pairs of a split, CPU: src [N, 3, SRC_LEN], tgt [N, MAX_OUT]."""
    days = split_days()[0 if split == "train" else 1]
    src = torch.tensor(
        [[encode_src(format_pair(d, f)[0]) for f in range(len(FORMATS))] for d in days]
    )
    tgt = torch.tensor([encode_tgt(format_pair(d, 0)[1]) for d in days])
    return src, tgt


@lru_cache(maxsize=4)
def _table_on(split: str, dev: str):
    return tuple(t.to(dev) for t in _table(split))


def make_batch(split: str, b: int, gen: torch.Generator, dev):
    src, tgt = _table_on(split, str(dev))
    i = torch.randint(len(tgt), (b,), generator=gen, device=dev)
    f = torch.randint(len(FORMATS), (b,), generator=gen, device=dev)
    return src[i, f], tgt[i]


@lru_cache(maxsize=32)
def _causal(length: int, dev: str):
    return causal_mask(length).to(dev)  # built once per (length, device), not per step


# ---- model --------------------------------------------------------------------------------------


class DateTranslator(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        V = len(ITOS)
        self.emb_s, self.pos_s = nn.Embedding(V, d_model), nn.Embedding(SRC_LEN, d_model)
        self.emb_t, self.pos_t = nn.Embedding(V, d_model), nn.Embedding(MAX_OUT, d_model)
        self.enc, self.dec = Block(d_model, heads), CrossBlock(d_model, heads)
        self.ln_f, self.head = nn.LayerNorm(d_model), nn.Linear(d_model, V)

    def encode(self, src):
        """src [B, S] -> (memory [B, S, D], encoder self-attention [B, H, S, S])."""
        x = self.emb_s(src) + self.pos_s(torch.arange(src.shape[1], device=src.device))
        return self.enc(x)

    def decode(self, dec_in, memory):
        """dec_in [B, L] (BOS + chars so far) -> (logits [B, L, V], self_attn, cross_attn)."""
        L = dec_in.shape[1]
        x = self.emb_t(dec_in) + self.pos_t(torch.arange(L, device=dec_in.device))
        mask = _causal(L, str(dec_in.device))
        x, self_attn, cross_attn = self.dec(x, memory, mask)
        return self.head(self.ln_f(x)), self_attn, cross_attn

    def forward(self, src, dec_in):
        memory, enc_attn = self.encode(src)
        logits, self_attn, cross_attn = self.decode(dec_in, memory)
        return logits, enc_attn, self_attn, cross_attn


def build(meta: dict) -> DateTranslator:
    return DateTranslator(int(meta["d_model"]), int(meta["heads"]))


def teacher_inputs(tgt):
    """Decoder input for teacher forcing: BOS then the target shifted right (the last EOS drops)."""
    return torch.cat([torch.full_like(tgt[:, :1], BOS), tgt[:, :-1]], 1)


def loss_fn(logits, tgt):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=PAD)


@torch.no_grad()
def greedy_batch(model, src):
    """src [B, S] ids -> generated ids [B, <= MAX_OUT] (EOS included, PAD after it). Starts from BOS
    alone; the decoder only ever sees BOS and its own earlier outputs."""
    model.eval()
    memory, _ = model.encode(src)
    ids = torch.full((src.shape[0], 1), BOS, dtype=torch.long, device=src.device)
    done = torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)
    for _ in range(MAX_OUT):
        nxt = model.decode(ids, memory)[0][:, -1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        ids = torch.cat([ids, nxt[:, None]], 1)
        done = done | (nxt == EOS)
        if bool(done.all()):
            break
    return ids[:, 1:]


def greedy_decode(model, src: str) -> str:
    """Translate one source string; builds the output from BOS only and stops at EOS."""
    dev = next(model.parameters()).device
    ids = torch.tensor([encode_src(src)], device=dev)
    return decode_ids(greedy_batch(model, ids)[0].tolist())


@torch.no_grad()
def exact_match(model, pairs) -> float:
    dev = next(model.parameters()).device
    src = torch.tensor([encode_src(s) for s, _, _ in pairs], device=dev)
    out = greedy_batch(model, src).tolist()
    return sum(decode_ids(o) == t for o, (_, t, _) in zip(out, pairs, strict=True)) / len(pairs)


# ---- training / evaluation ----------------------------------------------------------------------


def _label(i: int) -> str:
    return {PAD: "·", BOS: "<bos>", EOS: "<eos>"}.get(i) or ("␣" if ITOS[i] == " " else ITOS[i])


def _probe(dev):
    s, t = format_pair(PROBE_DATE, PROBE_FMT)
    return torch.tensor([encode_src(s)], device=dev), torch.tensor([encode_tgt(t)], device=dev)


def train(ctx) -> None:
    P = ctx.params
    steps = int(P["steps"])
    meta = {"d_model": int(P["d_model"]), "heads": int(P["heads"]), "steps": steps}
    dev = _device()
    torch.manual_seed(int(P.get("seed", 0)))
    model = build(meta).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(P["lr"]), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * max(0.02, 1 - s / steps)
    )
    gen = torch.Generator(device=dev).manual_seed(int(P.get("seed", 0)) + 1)
    held = date_pairs("heldout", EXACT_N, 777)
    p_src, p_tgt = _probe(dev)
    p_labels = [_label(i) for i in p_src[0].tolist()]
    log_every = max(1, steps // LOG_ROWS)
    snap_every = log_every * max(1, round(steps / SNAPSHOTS / log_every))
    exact_every = log_every * max(1, round(steps / EXACT_ROWS / log_every))
    ctx.set_status(state="running", message="training")
    exact = 0.0
    for step in range(1, steps + 1):
        if not ctx.should_continue():
            ctx.set_status(state="stopped")
            return
        model.train()
        src, tgt = make_batch("train", BATCH, gen, dev)
        loss = loss_fn(model(src, teacher_inputs(tgt))[0], tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            row = {"loss": float(loss)}
            if step % exact_every == 0 or step == steps:
                exact = exact_match(model, held)
                row["exact"] = exact
            if step % snap_every == 0 or step == steps:
                model.eval()
                with torch.no_grad():
                    cross = model(p_src, teacher_inputs(p_tgt))[3]
                row["attn"] = attn_snapshot(cross)
                row["qkv"] = {"labels": p_labels, "heads": []}
            ctx.log(step, **row)
            ctx.set_status(progress=step / steps, message=f"step {step}/{steps} exact {exact:.3f}")
    ctx.save_ckpt(
        steps, {"meta": meta, "state": model.state_dict()}, label="final", metrics={"exact": exact}
    )
    ctx.done(f"trained: held-out exact-match {exact:.3f}")


def _r(t, nd=4):
    return t.detach().double().round(decimals=nd).tolist()


@torch.no_grad()
def _example(model, src_str: str, tgt_str: str, dev) -> dict:
    src = torch.tensor([encode_src(src_str)], device=dev)
    gen = greedy_batch(model, src)[0].tolist()
    if EOS in gen:
        gen = gen[: gen.index(EOS) + 1]
    inp = torch.tensor([[BOS] + gen[:-1]], device=dev)
    seen = {}
    mha = model.dec.cross_attn
    hook = mha.register_forward_pre_hook(lambda m, a: seen.update(xq=a[0], xkv=a[1]))
    try:
        _, enc_attn, self_attn, cross_attn = model(src, inp)
    finally:
        hook.remove()
    xq, xkv = seen["xq"], seen["xkv"]
    q, k, v = (mha.split(f(x))[0] for f, x in ((mha.q, xq), (mha.k, xkv), (mha.v, xkv)))
    decoded = decode_ids(gen)
    return {
        "source": src_str, "target": tgt_str, "decoded": decoded, "correct": decoded == tgt_str,
        "src_labels": [_label(i) for i in src[0].tolist()],
        "in_labels": [_label(i) for i in inp[0].tolist()],
        "out_labels": [_label(i) for i in gen],
        "enc_attn": _r(enc_attn[0], 3), "dec_attn": _r(self_attn[0], 3),
        "cross_attn": _r(cross_attn[0], 3),
        "vec": {"q": _r(q, 3), "k": _r(k, 3), "v": _r(v, 3)},
    }  # fmt: skip


def evaluate(ctx) -> None:
    P = ctx.params
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    meta = ck["meta"]  # the model is rebuilt from the checkpoint alone
    dev = _device()
    model = build(meta)
    model.load_state_dict(ck["state"])
    model.to(dev).eval()
    seed, k = int(P.get("eval_seed", 0)), int(P.get("n_examples", 6))
    exact = exact_match(model, date_pairs("heldout", EVAL_N, 1000 + seed))
    shown = date_pairs("heldout", k, 5000 + seed)
    result = {
        "d_model": meta["d_model"], "heads": meta["heads"], "exact": exact, "n_eval": EVAL_N,
        "seed": seed, "examples": [_example(model, s, t, dev) for s, t, _ in shown],
    }  # fmt: skip
    atomic_write_json(ctx.dir / "result.json", result)
    ctx.done(f"evaluated: held-out exact-match {exact:.3f}")


def estimate(params: dict) -> float:
    cal = registry.calib("cross-attn")
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    return params["steps"] / sps + cal.get("overhead_s", OVERHEAD_S)


SPEC = TaskSpec(
    id="cross-attn",
    order=8,
    title="Cross-attention: translate a date",
    subtitle="Turn '23 Oct 2026' into '2026-10-23' - see which source character each output reads.",
    params=[
        Param("d_model", "Model width d_model", "int", 64, min=32, max=128, step=32),
        Param("heads", "Heads", "choice", 4, choices=[1, 2, 4]),
        Param("steps", "Training steps", "int", 1500, min=200, max=10000, step=100),
        Param("lr", "Learning rate", "float", 2e-3, min=3e-4, max=1e-2),
        Param("seed", "Seed", "int", 0, min=0, max=9999),
    ],
    eval_params=[
        Param("eval_seed", "Eval seed", "int", 0, min=0, max=9999),
        Param("n_examples", "Examples to show", "int", 6, min=1, max=12, step=1),
    ],
    charts=[
        {"id": "loss", "title": "Cross-entropy loss (teacher forcing)",
         "series": [{"key": "loss", "label": "train"}], "log_y": True},
        {"id": "exact", "title": "Held-out exact-match (greedy decode, unseen dates)",
         "series": [{"key": "exact", "label": "held-out dates"}], "log_y": False},
    ],
    viewer="xattn",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="cross-attn.html",
    vecgrids=True,
    live="attnlive",
)  # fmt: skip
registry.register(SPEC)
