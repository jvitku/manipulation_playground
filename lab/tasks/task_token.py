"""Task "Tokenization: actions as tokens" (id `tokenize`, ladder 14): one problem, three
action interfaces.

The "diagonal two-mode" problem: obs c ~ U(-0.4, 0.4) (shape (B, 1)); the expert picks a side
s = +-1 and acts a = (c + s*T1_HALF_GAP, 0.5 + s*Z_GAP) + N(0, 0.05^2), Z_GAP = 0.25. BOTH action
dims correlate through s - that is the point of the task:
  cont  a continuous regression head (MSE) averages the two modes - straight into the obstacle;
  bins  per-dim independent bins capture each marginal, but sample x and z independently, so
        about half the sampled (x, z) pairs mix the sides (invalid pairing);
  ar    autoregressive tokens (sample tok_x, then tok_z GIVEN tok_x) capture the joint - the
        same mechanism that lets a language model keep a sentence coherent token by token.
Actions already live in [-1, 1] (x in [-0.9, 0.9], z in {0.25, 0.75}), so nothing is z-scored:
the bins/token quantisation grid over [-1, 1] IS the normalisation.

Three heads, trained sequentially with fresh seeds, `steps` Adam steps each:
  cont  MLP 1 -> 256 -> 256 -> 2 (GELU), MSE on the raw actions;
  bins  the same trunk -> 2*n_bins logits; each expert dim clipped to [-1, 1] and quantised into
        n_bins uniform bins; CE per dim. Decode: per-dim softmax sample at `temperature`
        (temperature <= 0.01 -> argmax); action = bin center;
  ar    the action as a 2-token sequence over vocab n_bins: tiny causal transformer
        (`minitf.Block` x 2, 2 heads, d_model param); sequence [obs-token, tok_x] -> predict
        [tok_x, tok_z]; obs token = Linear(1 -> d_model) of c, token embedding
        nn.Embedding(n_bins, d_model), head Linear(d_model -> n_bins). Decode: sample tok_x from
        P(.|c), then tok_z from P(.|c, tok_x), same temperature rule; action = bin centers.

Checkpoint payload (`last.pt`), task 1's multi-part layout:
    {"cont": {"state", "loss"}, "bins": {...}, "ar": {...},
     "arch": {"n_bins", "d_model", "hidden", "n_layers", "n_heads"}, "params": <train params>}
Mid-run checkpoints hold only the heads trained so far; `evaluate` rebuilds every model from
`arch`, never from the current form params.

result.json written by `evaluate` (viewer `tokens`; probe c = 0.0):
    {"n_bins", "d_model", "temperature",
     "bin_edges": [n_bins+1], "bin_centers": [n_bins],
     "heads": {
       "cont": {"collision", "pair_valid", "mode_err", "left_frac", "points": [[c, ax, az]] <=500},
       "bins": {...same..., "probe": {"c", "px": [n_bins], "pz": [n_bins]}},
       "ar":   {...same..., "probe": {"c", "px": [n_bins],
                "pz_given": {"left": [n_bins], "right": [n_bins]}, "tok_left", "tok_right"}}},
     "probe_samples": {"cont": [[ax, az]] <=200, "bins": [...], "ar": [...]},  # at the probe c
     "embedding": {"table": [n_bins][d_model], "d": d_model},  # the AR token-embedding weight
     "expert_example": {"c", "a": [ax, az], "tok": [ix, iz]},
     "text_demo": {"vocab": <cross-attn date task's character vocab, one string in id order>,
                   "sample": "23 Oct 2026"}}
px/pz are the actual softmax probabilities at the probe (no temperature); pz_given.left/right are
the AR step-2 distributions after feeding tok_x = the argmax token of the left / right lobe of px
(shipped as tok_left/tok_right). Metrics: collision = mean(|ax - c| < T1_OBSTACLE), pair_valid =
mean(sign(ax - c) == sign(az - 0.5)), mode_err = mean distance to the nearest expert mode,
left_frac = mean(ax < c).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fvb.ta_tutorial.toy import T1_HALF_GAP, T1_OBSTACLE
from lab.core.runctx import atomic_write_json
from lab.tasks import registry, task_crossattn
from lab.tasks.base import Param, TaskSpec
from lab.tasks.minitf import Block, causal_mask

Z_GAP = 0.25
EXPERT_NOISE = 0.05
HIDDEN = 256
N_LAYERS, N_HEADS = 2, 2
HEADS = ("cont", "bins", "ar")
HEAD_LABEL = {
    "cont": "continuous regression (MSE)",
    "bins": "per-dim bins (CE per dimension)",
    "ar": "autoregressive tokens (tiny transformer)",
}
ARGMAX_TEMP = 0.01  # temperature at or below this decodes with argmax
PROBE_C = 0.0
MAX_POINTS, N_PROBE = 500, 200
LOG_ROWS = 100  # loss rows per head
TEXT_SAMPLE = "23 Oct 2026"
# Fallbacks used when lab/tasks/calib.json has no "tokenize" entry yet (conservative CPU guesses).
FALLBACK_STEPS_PER_S, OVERHEAD_S = 100.0, 5.0


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---- problem ------------------------------------------------------------------------------------


def diag_data(n: int, gen: torch.Generator, dev) -> tuple[torch.Tensor, torch.Tensor]:
    """c [n, 1] and expert actions a [n, 2] of the diagonal two-mode problem."""
    c = (torch.rand(n, 1, generator=gen, device=dev) - 0.5) * 0.8
    s = torch.randint(0, 2, (n, 1), generator=gen, device=dev).float() * 2 - 1
    a = torch.cat([c + s * T1_HALF_GAP, 0.5 + s * Z_GAP], 1)
    return c, a + torch.randn(n, 2, generator=gen, device=dev) * EXPERT_NOISE


def diag_metrics(c: np.ndarray, a: np.ndarray) -> dict:
    """c (n,), a (n, 2): collision, x/z pairing validity, nearest-mode distance, mode balance."""
    dx, dz = a[:, 0] - c, a[:, 1] - 0.5
    d = [np.hypot(a[:, 0] - (c + s * T1_HALF_GAP), a[:, 1] - (0.5 + s * Z_GAP)) for s in (-1, 1)]
    return {
        "collision": float(np.mean(np.abs(dx) < T1_OBSTACLE)),
        "pair_valid": float(np.mean(np.sign(dx) == np.sign(dz))),
        "mode_err": float(np.minimum(d[0], d[1]).mean()),
        "left_frac": float((dx < 0).mean()),
    }


# ---- tokenisation -------------------------------------------------------------------------------


def bin_edges(n_bins: int) -> np.ndarray:
    # legend: K = n_bins
    return np.linspace(-1.0, 1.0, n_bins + 1)  # [K + 1] uniform edges over [-1, 1]


def bin_centers(n_bins: int) -> np.ndarray:
    # legend: K = n_bins
    e = bin_edges(n_bins)  # [K + 1]
    return (e[:-1] + e[1:]) / 2  # midpoints: [K + 1] -> [K]


def quantize(a: torch.Tensor, n_bins: int) -> torch.Tensor:
    """[-1, 1] values (clipped first) -> bin indices, same shape, int64."""
    # legend: B = batch, K = n_bins; a is the action batch [B, 2] here
    return (((a.clamp(-1, 1) + 1) / 2) * n_bins).long().clamp(0, n_bins - 1)  # [B, 2] -> [B, 2] int


def _pick(logits: torch.Tensor, temperature: float, gen: torch.Generator) -> torch.Tensor:
    """Sample token ids [B] from logits [B, V] at `temperature` (<= ARGMAX_TEMP -> argmax)."""
    if temperature <= ARGMAX_TEMP:
        return logits.argmax(-1)
    return torch.multinomial((logits / temperature).softmax(-1), 1, generator=gen)[:, 0]


# ---- models -------------------------------------------------------------------------------------


def _mlp(out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(1, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
        nn.Linear(HIDDEN, out_dim),
    )  # fmt: skip


class ARTokenHead(nn.Module):
    """Tiny causal transformer over [obs-token, tok_x]; logits [B, T, n_bins] per position."""

    # legend: B = batch, T = seq len (1 or 2), d = d_model, K = n_bins (the vocab)

    def __init__(self, arch: dict):
        super().__init__()
        n_bins, d = int(arch["n_bins"]), int(arch["d_model"])
        self.obs = nn.Linear(1, d)
        self.emb = nn.Embedding(n_bins, d)
        self.pos = nn.Embedding(2, d)
        self.blocks = nn.ModuleList(
            [Block(d, int(arch["n_heads"])) for _ in range(int(arch["n_layers"]))]
        )
        self.ln_f, self.head = nn.LayerNorm(d), nn.Linear(d, n_bins)

    def forward(self, c: torch.Tensor, tok_x: torch.Tensor | None = None) -> torch.Tensor:
        x = self.obs(c)[:, None]  # obs token: c [B, 1] -> [B, d] -> [B, 1, d]
        if tok_x is not None:
            x = torch.cat([x, self.emb(tok_x)[:, None]], 1)  # + tok_x emb [B, 1, d] -> [B, 2, d]
        x = x + self.pos(torch.arange(x.shape[1], device=x.device))  # + pos [T, d] -> [B, T, d]
        mask = causal_mask(x.shape[1]).to(x.device)  # causal [T, T]
        for b in self.blocks:
            x, _ = b(x, mask)  # [B, T, d] -> [B, T, d]
        return self.head(self.ln_f(x))  # [B, T, d] -> logits [B, T, K]


def _build(name: str, arch: dict) -> nn.Module:
    if name == "cont":
        return _mlp(2)
    if name == "bins":
        return _mlp(2 * int(arch["n_bins"]))
    return ARTokenHead(arch)


def _load_head(ck: dict, name: str, dev: str) -> nn.Module:
    m = _build(name, ck["arch"])
    m.load_state_dict(ck[name]["state"])
    return m.to(dev).eval()


# ---- per-head losses and decoding ---------------------------------------------------------------


def _loss_cont(model, c, a, n_bins):
    return F.mse_loss(model(c), a)


def _loss_bins(model, c, a, n_bins):
    # legend: B = batch, K = n_bins; obs c [B, 1], expert actions a [B, 2]
    logits = model(c).view(-1, 2, n_bins)  # [B, 2*K] -> [B, 2, K]
    tok = quantize(a, n_bins)  # [B, 2] -> [B, 2] int targets
    # independent CE per dim: logits[:, i] [B, K] vs tok[:, i] [B]; mean of the two -> scalar
    return (F.cross_entropy(logits[:, 0], tok[:, 0]) + F.cross_entropy(logits[:, 1], tok[:, 1])) / 2


def _loss_ar(model, c, a, n_bins):
    # legend: B = batch, K = n_bins; obs c [B, 1], expert actions a [B, 2]
    tok = quantize(a, n_bins)  # [B, 2] -> [B, 2] int targets
    logits = model(c, tok[:, 0])  # teacher forcing: c [B, 1] + expert tok_x [B] -> [B, 2, K]
    # CE per position: pos 0 predicts tok_x, pos 1 predicts tok_z; each [B, K] vs [B] -> scalar
    return (F.cross_entropy(logits[:, 0], tok[:, 0]) + F.cross_entropy(logits[:, 1], tok[:, 1])) / 2


LOSS = {"cont": _loss_cont, "bins": _loss_bins, "ar": _loss_ar}


@torch.no_grad()
def bins_sample(model, c, n_bins, temperature, gen):
    """Per-dim independent softmax sample -> token ids tx, tz [B]."""
    # legend: B = batch, K = n_bins; obs c [B, 1]
    logits = model(c).view(-1, 2, n_bins)  # [B, 2*K] -> [B, 2, K]
    # x and z drawn apart: each dim's logits [B, K] -> token ids [B]
    return _pick(logits[:, 0], temperature, gen), _pick(logits[:, 1], temperature, gen)


@torch.no_grad()
def ar_sample(model, c, n_bins, temperature, gen):
    """tok_x from P(.|c), then tok_z from P(.|c, tok_x) -> token ids tx, tz [B]."""
    # legend: B = batch, K = n_bins; obs c [B, 1]
    tx = _pick(model(c)[:, 0], temperature, gen)  # step 1: [B, 1, K] -> [B, K] -> ids [B]
    tz = _pick(model(c, tx)[:, 1], temperature, gen)  # step 2: [B, 2, K] -> [B, K] -> ids [B]
    return tx, tz


@torch.no_grad()
def sample_actions(name: str, model, c, n_bins, centers, temperature, gen) -> torch.Tensor:
    """Decoded actions [B, 2] of one head for obs c [B, 1] (cont is deterministic)."""
    if name == "cont":
        return model(c).clamp(-1, 1)
    sampler = bins_sample if name == "bins" else ar_sample
    tx, tz = sampler(model, c, n_bins, temperature, gen)
    return torch.stack([centers[tx], centers[tz]], 1)


# ---- training / evaluation ----------------------------------------------------------------------


def train(ctx) -> None:
    P = ctx.params
    steps, batch = int(P["steps"]), int(P["batch"])
    lr, seed = float(P["lr"]), int(P.get("seed", 0))
    n_bins, d_model = int(P["n_bins"]), int(P["d_model"])
    arch = {"n_bins": n_bins, "d_model": d_model, "hidden": HIDDEN,
            "n_layers": N_LAYERS, "n_heads": N_HEADS}  # fmt: skip
    dev = _device()
    total = len(HEADS) * steps
    log_every = max(1, steps // LOG_ROWS)
    holder: dict = {}
    for i, name in enumerate(HEADS):
        ctx.set_status(state="running", message=f"training head {i + 1}/3: {HEAD_LABEL[name]}")
        torch.manual_seed(seed + 101 * (i + 1))  # fresh init per head
        model = _build(name, arch).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        gen = torch.Generator(device=dev).manual_seed(seed + 7919 * (i + 1))
        loss_fn = LOSS[name]
        last = 0.0
        for step in range(1, steps + 1):
            if not ctx.should_continue():
                ctx.set_status(state="stopped")
                return
            model.train()
            c, a = diag_data(batch, gen, dev)
            loss = loss_fn(model, c, a, n_bins)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = float(loss)
            if step % log_every == 0 or step == steps:
                g = i * steps + step
                ctx.log(g, **{f"{name}_loss": last})
                ctx.set_status(
                    progress=g / total,
                    message=f"head {i + 1}/3 ({name}) step {step}/{steps} loss {last:.4f}",
                )
        holder[name] = {"state": model.state_dict(), "loss": last}
        payload = {**holder, "arch": arch, "params": ctx.params}
        label = "final" if name == HEADS[-1] else f"{name} trained"
        metrics = {f"{k}_loss": round(v["loss"], 4) for k, v in holder.items()}
        ctx.save_ckpt((i + 1) * steps, payload, label=label, metrics=metrics)
    ctx.done("trained 3 heads: " + ", ".join(f"{k} loss {holder[k]['loss']:.4f}" for k in HEADS))


def _r4(x) -> list:
    return np.round(np.asarray(x, float), 4).tolist()


def evaluate(ctx) -> None:
    P = ctx.params
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    arch = ck["arch"]  # every model is rebuilt from the checkpoint alone
    n_bins, d_model = int(arch["n_bins"]), int(arch["d_model"])
    eval_n, seed = int(P.get("eval_n", 4000)), int(P.get("eval_seed", 0))
    temperature = float(P.get("temperature", 1.0))
    dev = _device()
    models = {name: _load_head(ck, name, dev) for name in HEADS}
    centers_np = bin_centers(n_bins)
    centers = torch.as_tensor(centers_np, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)

    ctx.set_status(message="evaluating: sampling 3 heads over the c grid")
    c = np.linspace(-0.4, 0.4, eval_n).astype(np.float32)
    ct = torch.as_tensor(c[:, None], device=dev)
    sel = np.linspace(0, eval_n - 1, min(eval_n, MAX_POINTS)).astype(int)
    heads = {}
    for name in HEADS:
        a = sample_actions(name, models[name], ct, n_bins, centers, temperature, gen).cpu().numpy()
        m = diag_metrics(c, a)
        heads[name] = {**{k: round(v, 4) for k, v in m.items()},
                       "points": [_r4([c[i], a[i, 0], a[i, 1]]) for i in sel]}  # fmt: skip

    # probes at c = PROBE_C: the distributions the two token heads actually hold there
    ctx.set_status(message="evaluating: probe distributions and samples at c = 0")
    c0 = torch.full((1, 1), PROBE_C, device=dev)
    with torch.no_grad():
        bl = models["bins"](c0).view(2, n_bins).softmax(-1).cpu().numpy()
        px_ar = models["ar"](c0)[:, 0].softmax(-1)[0].cpu().numpy()
    left, right = centers_np < PROBE_C, centers_np > PROBE_C
    tok_left = int(np.where(left, px_ar, -np.inf).argmax())
    tok_right = int(np.where(right, px_ar, -np.inf).argmax())
    pz_given = {}
    for side, tok in (("left", tok_left), ("right", tok_right)):
        with torch.no_grad():
            t = torch.tensor([tok], device=dev)
            pz_given[side] = _r4(models["ar"](c0, t)[:, 1].softmax(-1)[0].cpu().numpy())
    heads["bins"]["probe"] = {"c": PROBE_C, "px": _r4(bl[0]), "pz": _r4(bl[1])}
    heads["ar"]["probe"] = {"c": PROBE_C, "px": _r4(px_ar), "pz_given": pz_given,
                            "tok_left": tok_left, "tok_right": tok_right}  # fmt: skip

    c0b = torch.full((N_PROBE, 1), PROBE_C, device=dev)
    probe_samples = {
        name: _r4(
            sample_actions(name, models[name], c0b, n_bins, centers, temperature, gen)
            .cpu().numpy()
        )
        for name in HEADS
    }  # fmt: skip

    rng = np.random.default_rng(seed)
    ce = float(rng.uniform(-0.4, 0.4))
    s = float(rng.choice([-1.0, 1.0]))
    ae = np.array([ce + s * T1_HALF_GAP, 0.5 + s * Z_GAP]) + rng.normal(0, EXPERT_NOISE, 2)
    tok_e = quantize(torch.as_tensor(ae, dtype=torch.float32), n_bins).tolist()

    result = {
        "n_bins": n_bins, "d_model": d_model, "temperature": temperature,
        "bin_edges": _r4(bin_edges(n_bins)), "bin_centers": _r4(centers_np),
        "heads": heads,
        "probe_samples": probe_samples,
        "embedding": {"table": _r4(models["ar"].emb.weight.detach().cpu().numpy()), "d": d_model},
        "expert_example": {"c": round(ce, 4), "a": _r4(ae), "tok": [int(t) for t in tok_e]},
        "text_demo": {"vocab": "".join(task_crossattn.CHARS), "sample": TEXT_SAMPLE},
    }  # fmt: skip
    atomic_write_json(ctx.dir / "result.json", result)
    ctx.done(
        f"evaluated: pair_valid cont {heads['cont']['pair_valid']:.2f} / "
        f"bins {heads['bins']['pair_valid']:.2f} / ar {heads['ar']['pair_valid']:.2f}"
    )


def estimate(params: dict) -> float:
    cal = registry.calib("tokenize")
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    # steps_per_s is "per-head steps": the run trains three heads of `steps` each.
    return len(HEADS) * params["steps"] / sps + cal.get("overhead_s", OVERHEAD_S)


SPEC = TaskSpec(
    id="tokenize",
    order=14,
    title="Tokenization: actions as tokens",
    subtitle="One problem, three action interfaces: regress it, bin each dim, or spell it out "
             "token by token - only the tokens keep x and z on the same side.",
    params=[
        Param("steps", "Training steps per head", "int", 3000, min=500, max=20000, step=500),
        Param("lr", "Learning rate", "float", 1e-3, min=1e-4, max=1e-2),
        Param("batch", "Batch size", "int", 256, min=64, max=512, step=64),
        Param("n_bins", "Bins per dimension (vocab)", "choice", 17, choices=[9, 17, 33]),
        Param("d_model", "Transformer width d_model", "choice", 32, choices=[16, 32, 64]),
        Param("seed", "Seed", "int", 0, min=0, max=9999),
    ],
    eval_params=[
        Param("eval_seed", "Eval seed", "int", 0, min=0, max=9999),
        Param("eval_n", "Samples across the c grid", "int", 4000, min=500, max=20000, step=500),
        Param("temperature", "Sampling temperature", "float", 1.0, min=0.0, max=2.0, step=0.1),
    ],
    charts=[
        {"id": "loss", "title": "Training loss (MSE / CE per head)",
         "series": [{"key": f"{h}_loss", "label": h} for h in HEADS], "log_y": True},
    ],
    viewer="tokens",
    vecgrids=True,
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="tokenization.html",
)  # fmt: skip
registry.register(SPEC)
