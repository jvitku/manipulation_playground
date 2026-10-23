"""A mini-transformer in plain torch, small enough to read in one sitting.

Every attention layer also returns its attention weights, so a lesson page can show what the
model looked at. Shapes: B = batch, T = tokens, D = d_model, H = heads,
Tq / Tk = query / key length.
"""

import torch
import torch.nn.functional as F
from torch import nn


class MHA(nn.Module):
    """Multi-head attention. Queries come from `x_q`, keys and values from `x_kv`."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)  # "what am I looking for?"
        self.k = nn.Linear(d_model, d_model)  # "what do I contain?"
        self.v = nn.Linear(d_model, d_model)  # "what do I hand over if picked?"
        self.out = nn.Linear(d_model, d_model)  # mixes the heads back together

    def split(self, x):  # [B, T, D] -> [B, H, T, dh]
        return x.view(x.shape[0], x.shape[1], self.h, self.dh).transpose(1, 2)

    def forward(self, x_q, x_kv=None, mask=None):
        """mask: bool, broadcastable to [Tq, Tk], True = may attend. Returns (out, attn)."""
        x_kv = x_q if x_kv is None else x_kv  # self-attention unless told otherwise
        q, k, v = self.split(self.q(x_q)), self.split(self.k(x_kv)), self.split(self.v(x_kv))
        scores = q @ k.transpose(-2, -1) / self.dh**0.5  # [B, H, Tq, Tk]: how well q matches k
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = scores.softmax(-1)  # each query's weights over the keys sum to 1
        mixed = (attn @ v).transpose(1, 2).reshape(x_q.shape)  # weighted sum of values
        return self.out(mixed), attn


def mlp(d_model: int) -> nn.Sequential:
    hidden = 4 * d_model
    return nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))


class Block(nn.Module):
    """Self-attention, then an MLP, each with a residual connection and a pre-LayerNorm."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn, self.mlp = MHA(d_model, n_heads), mlp(d_model)

    def forward(self, x, mask=None):
        h = self.ln1(x)
        a, attn = self.attn(h, mask=mask)
        x = x + a
        return x + self.mlp(self.ln2(x)), attn


class CrossBlock(nn.Module):
    """Decoder block: masked self-attention, cross-attention into `memory`, then an MLP."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.self_attn, self.cross_attn = MHA(d_model, n_heads), MHA(d_model, n_heads)
        self.mlp = mlp(d_model)

    def forward(self, x, memory, self_mask=None):
        a, self_attn = self.self_attn(self.ln1(x), mask=self_mask)
        x = x + a
        c, cross_attn = self.cross_attn(self.ln2(x), memory)  # q from x, k/v from memory
        x = x + c
        return x + self.mlp(self.ln3(x)), self_attn, cross_attn


def causal_mask(t: int) -> torch.Tensor:
    """[t, t] bool, True where a position may look: itself and everything before it."""
    return torch.tril(torch.ones(t, t, dtype=torch.bool))


def attn_snapshot(attn: torch.Tensor, max_dim: int = 16) -> list:
    """[B, H, Tq, Tk] -> nested lists [H][Tq'][Tk'] for the UI: batch-mean, pooled to <= max_dim."""
    a = attn.detach().double().mean(0)  # [H, Tq, Tk]
    size = (min(a.shape[1], max_dim), min(a.shape[2], max_dim))
    a = F.adaptive_avg_pool2d(a, size)
    return a.round(decimals=3).tolist()
