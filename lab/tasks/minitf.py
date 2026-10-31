"""A mini-transformer in plain torch, small enough to read in one sitting.

Every attention layer also returns its attention weights, so a lesson page can show what the
model looked at. Shapes: B = batch, T = tokens, D = d_model, H = heads,
Tq / Tk = query / key length.

Position information is the caller's job (learned tables added to the embeddings), except the
opt-in `rope=True` on `MHA` / `Block` / `CrossBlock` (default off, self-attention only), which
rotates Q and K by position instead - see `rope_rotate`.
"""

import torch
import torch.nn.functional as F
from torch import nn

ROPE_BASE = 10000.0


def rope_rotate(x: torch.Tensor, base: float = ROPE_BASE) -> torch.Tensor:
    """Rotary position embedding (RoPE): rotate each head's dims, taken as consecutive pairs.

    x: [B, H, T, dh]. Pair i of the vector at position m is turned by the angle m * theta_i with
    theta_i = base^(-2i/dh) - low pairs spin fast (local order), high pairs slowly (long range).
    Applied to Q and K (never V) before the scores, so dot(q_m, k_n) depends on the positions
    only through the offset m - n: the model reads *relative* position without any position table.
    """
    # legend: B = batch, H = heads, T = tokens, dh = per-head dim
    dh = x.shape[-1]  # x: [B, H, T, dh]
    theta = base ** (-torch.arange(0, dh, 2, device=x.device) / dh)  # [dh/2]
    ang = torch.arange(x.shape[-2], device=x.device)[:, None] * theta  # [T, dh/2]
    cos, sin = ang.cos(), ang.sin()  # [T, dh/2] each
    x1, x2 = x[..., 0::2], x[..., 1::2]  # [B, H, T, dh/2] each: one 2D point per pair
    out = torch.empty_like(x)  # [B, H, T, dh]
    out[..., 0::2] = x1 * cos - x2 * sin  # rotated x-coords [B, H, T, dh/2]
    out[..., 1::2] = x1 * sin + x2 * cos  # rotated y-coords [B, H, T, dh/2]
    return out  # [B, H, T, dh], same shape as x


class MHA(nn.Module):
    """Multi-head attention. Queries come from `x_q`, keys and values from `x_kv`."""

    def __init__(self, d_model: int, n_heads: int, rope: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        assert not rope or self.dh % 2 == 0  # RoPE rotates dims in pairs
        self.rope = rope
        self.q = nn.Linear(d_model, d_model)  # "what am I looking for?"
        self.k = nn.Linear(d_model, d_model)  # "what do I contain?"
        self.v = nn.Linear(d_model, d_model)  # "what do I hand over if picked?"
        self.out = nn.Linear(d_model, d_model)  # mixes the heads back together

    def split(self, x):  # [B, T, D] -> [B, H, T, dh]
        return x.view(x.shape[0], x.shape[1], self.h, self.dh).transpose(1, 2)

    def forward(self, x_q, x_kv=None, mask=None):
        """mask: bool, broadcastable to [Tq, Tk], True = may attend. Returns (out, attn)."""
        # legend: B = batch, H = heads, Tq / Tk = query / key tokens, D = d_model, dh = D / H
        # RoPE only in self-attention: a separate x_kv stream (cross-attention) has its own
        # position axis, so "query position minus key position" is meaningless across the two.
        use_rope = self.rope and x_kv is None
        x_kv = x_q if x_kv is None else x_kv  # self-attention unless told otherwise
        # project then split heads: [B, Tq|Tk, D] -> q [B, H, Tq, dh], k / v [B, H, Tk, dh]
        q, k, v = self.split(self.q(x_q)), self.split(self.k(x_kv)), self.split(self.v(x_kv))
        if use_rope:
            q, k = rope_rotate(q), rope_rotate(k)
        scores = q @ k.transpose(-2, -1) / self.dh**0.5  # [B, H, Tq, Tk]: how well q matches k
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = scores.softmax(-1)  # [B, H, Tq, Tk]: each query's weights over the keys sum to 1
        mixed = (attn @ v).transpose(1, 2).reshape(x_q.shape)  # weighted sum of values: [B, Tq, D]
        return self.out(mixed), attn  # out [B, Tq, D], attn [B, H, Tq, Tk]


def mlp(d_model: int) -> nn.Sequential:
    hidden = 4 * d_model
    return nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))


class Block(nn.Module):
    """Self-attention, then an MLP, each with a residual connection and a pre-LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, rope: bool = False):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn, self.mlp = MHA(d_model, n_heads, rope=rope), mlp(d_model)

    def forward(self, x, mask=None):
        # legend: B = batch, T = tokens, D = d_model, H = heads; x: [B, T, D]
        h = self.ln1(x)  # [B, T, D], normalised
        a, attn = self.attn(h, mask=mask)  # a [B, T, D], attn [B, H, T, T]
        x = x + a  # residual: [B, T, D]
        return x + self.mlp(self.ln2(x)), attn  # MLP keeps [B, T, D]


class CrossBlock(nn.Module):
    """Decoder block: masked self-attention, cross-attention into `memory`, then an MLP."""

    def __init__(self, d_model: int, n_heads: int, rope: bool = False):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(d_model) for _ in range(3))
        # rope reaches the causal self-attention only; the cross read is position-free either way
        self.self_attn, self.cross_attn = MHA(d_model, n_heads, rope=rope), MHA(d_model, n_heads)
        self.mlp = mlp(d_model)

    def forward(self, x, memory, self_mask=None):
        # legend: B = batch, L = target tokens, S = memory tokens, D = d_model, H = heads
        a, self_attn = self.self_attn(self.ln1(x), mask=self_mask)  # attn [B, H, L, L]
        x = x + a  # [B, L, D]
        c, cross_attn = self.cross_attn(self.ln2(x), memory)  # q from x, k/v from memory [B, S, D]
        x = x + c  # [B, L, D]; cross_attn: [B, H, L, S]
        return x + self.mlp(self.ln3(x)), self_attn, cross_attn  # [B, L, D]


def causal_mask(t: int) -> torch.Tensor:
    """[t, t] bool, True where a position may look: itself and everything before it."""
    return torch.tril(torch.ones(t, t, dtype=torch.bool))


def attn_snapshot(attn: torch.Tensor, max_dim: int = 16) -> list:
    """[B, H, Tq, Tk] -> nested lists [H][Tq'][Tk'] for the UI: batch-mean, pooled to <= max_dim."""
    a = attn.detach().double().mean(0)  # [H, Tq, Tk]
    size = (min(a.shape[1], max_dim), min(a.shape[2], max_dim))
    a = F.adaptive_avg_pool2d(a, size)
    return a.round(decimals=3).tolist()
