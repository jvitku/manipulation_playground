"""The nanoVLA "VLM": 4 pre-norm transformer layers over the [image ; text] prefix.

This file also holds the attention building blocks the whole model is made of (the text
encoder borrows ``EncoderBlock``, the action expert builds its decoder from ``MHA``) and the
``attn_snapshot`` helper the lessons use to show what the model looked at — the same idiom as
``lab/tasks/minitf.py``, written fresh for the school (plan §4).

Shape legend used below: B = batch, T = tokens, D = d_model, NH = heads,
Tq / Tk = query / key tokens, dh = D / NH, F = FFN hidden width.

Masks are boolean, True = "may attend", broadcastable to [B, NH, Tq, Tk]. The two shapes the
model actually uses: a key-padding mask ``key_mask[:, None, None, :]`` ([B, 1, 1, Tk], hides
the text <pad> tokens from every query) and the expert's causal mask ([Tq, Tk] lower
triangle). The trunk itself is never force-touched: both papers found force must stay out of
the VLM pathway (plan §4), so force enters only in the later lessons' fusion modules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MHA(nn.Module):
    """Multi-head attention. Queries come from ``x_q``, keys and values from ``x_kv``."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)  # "what am I looking for?"
        self.k = nn.Linear(d_model, d_model)  # "what do I contain?"
        self.v = nn.Linear(d_model, d_model)  # "what do I hand over if picked?"
        self.out = nn.Linear(d_model, d_model)  # mixes the heads back together

    def split(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D] -> [B, NH, T, dh]
        return x.view(x.shape[0], x.shape[1], self.h, self.dh).transpose(1, 2)

    def forward(self, x_q, x_kv=None, mask=None):
        """mask: bool, broadcastable to [B, NH, Tq, Tk], True = may attend. -> (out, attn)."""
        x_kv = x_q if x_kv is None else x_kv  # self-attention unless told otherwise
        # project then split heads: [B, Tq|Tk, D] -> q [B, NH, Tq, dh], k / v [B, NH, Tk, dh]
        q, k, v = self.split(self.q(x_q)), self.split(self.k(x_kv)), self.split(self.v(x_kv))
        scores = q @ k.transpose(-2, -1) / self.dh**0.5  # [B, NH, Tq, Tk]: match of q against k
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = scores.softmax(-1)  # [B, NH, Tq, Tk]: each query's weights over the keys sum to 1
        mixed = (attn @ v).transpose(1, 2).reshape(x_q.shape)  # weighted sum of values [B, Tq, D]
        return self.out(mixed), attn  # out [B, Tq, D], attn [B, NH, Tq, Tk]


def ffn(d_model: int, hidden: int) -> nn.Sequential:
    """The position-wise MLP of every block: D -> F -> D with a GELU (F = cfg.ffn = 768)."""
    return nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))


class EncoderBlock(nn.Module):
    """Pre-norm encoder layer: self-attention, then the FFN, each behind a residual."""

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn, self.mlp = MHA(d_model, n_heads), ffn(d_model, ffn_hidden)

    def forward(self, x, mask=None):
        a, attn = self.attn(self.ln1(x), mask=mask)  # a [B, T, D], attn [B, NH, T, T]
        x = x + a  # residual: [B, T, D]
        return x + self.mlp(self.ln2(x)), attn  # FFN keeps [B, T, D]


class Trunk(nn.Module):
    """The prefix stack: ``layers`` encoder blocks over [image ; text] -> prefix E.

    Input is the concatenation of the 72 vision tokens and the ≤ 12 text tokens (≤ 84 total);
    ``key_mask`` [B, T] is True on real tokens (all vision tokens, non-<pad> text tokens).
    Returns the final-LayerNormed prefix and every layer's attention map (lesson material).
    """

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, layers: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, ffn_hidden) for _ in range(layers)
        )
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, key_mask: torch.Tensor):
        # tokens [B, T, D]; key_mask [B, T] -> broadcast mask [B, 1, 1, T] (pads hidden as keys)
        mask = key_mask[:, None, None, :]
        attns = []
        x = tokens
        for block in self.blocks:
            x, attn = block(x, mask=mask)  # x [B, T, D], attn [B, NH, T, T]
            attns.append(attn)
        return self.ln_out(x), attns  # E [B, T, D], attns: layers x [B, NH, T, T]


def causal_mask(t: int, device=None) -> torch.Tensor:
    """[t, t] bool, True where a position may look: itself and everything before it."""
    return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))


def attn_snapshot(attn: torch.Tensor, max_dim: int = 16) -> list:
    """[B, NH, Tq, Tk] -> nested lists [NH][Tq'][Tk'] for the UI: batch-mean, pooled <= max_dim."""
    a = attn.detach().double().mean(0)  # [NH, Tq, Tk]
    size = (min(a.shape[1], max_dim), min(a.shape[2], max_dim))
    a = F.adaptive_avg_pool2d(a, size)
    return a.round(decimals=3).tolist()
