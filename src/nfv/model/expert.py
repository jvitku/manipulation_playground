"""The action expert: a small decoder that turns noisy action chunks into velocities.

The suffix is [state token ; H noisy-action tokens] (H = chunk = 16). Each action token is a
linear lift of one noisy action x_tau[t] plus the flow-time embedding tau (the same tau for
the whole chunk) plus a learned suffix position. Four decoder layers run causal self-attention
over the suffix (an action token may read the state and earlier steps, never later ones) and
cross-attention into the conditioning memory — for nanoVLA that memory is the trunk's prefix E;
lesson 18's ForceVLA2 swaps in E_MoE without touching this file (plan §4: the conditioning is
an input, not a property of the expert).

Shape legend: B = batch, H = chunk length, A = act_dim, D = d_model, S = memory tokens (≤ 84),
T = 1 + H suffix tokens, NH = heads.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from nfv.model.trunk import MHA, causal_mask, ffn

TAU_FREQ_MAX = 1000.0  # fastest Fourier feature: ~160 cycles across tau in [0, 1]


class TauEmbed(nn.Module):
    """Flow time tau in [0, 1] -> [B, D]: log-spaced sin/cos features through a 2-layer MLP."""

    def __init__(self, d_model: int):
        super().__init__()
        half = d_model // 2
        # frequencies log-spaced between 1 and TAU_FREQ_MAX rad per unit of tau
        freqs = torch.exp(torch.linspace(0.0, math.log(TAU_FREQ_MAX), half))  # [D/2]
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        ang = tau[:, None] * self.freqs  # [B] -> [B, D/2]
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1)  # [B, D]
        return self.mlp(feats)  # [B, D]


class DecoderBlock(nn.Module):
    """Pre-norm decoder layer: causal self-attention, cross-attention into memory, FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.self_attn = MHA(d_model, n_heads)
        self.cross_attn = MHA(d_model, n_heads)
        self.mlp = ffn(d_model, ffn_hidden)

    def forward(self, x, memory, self_mask, mem_mask):
        a, self_attn = self.self_attn(self.ln1(x), mask=self_mask)  # attn [B, NH, T, T]
        x = x + a  # [B, T, D]
        c, cross_attn = self.cross_attn(self.ln2(x), memory, mask=mem_mask)  # [B, NH, T, S]
        x = x + c  # [B, T, D]
        return x + self.mlp(self.ln3(x)), self_attn, cross_attn  # [B, T, D]


class ActionExpert(nn.Module):
    """Suffix = [state ; H noisy actions] through ``layers`` decoder blocks -> chunk features.

    Returns the features of the H chunk tokens (the state token is dropped — it exists to be
    read, not to predict) plus the last layer's attention maps for the lessons.
    """

    def __init__(
        self, d_model: int, n_heads: int, ffn_hidden: int, layers: int, act_dim: int, chunk: int
    ):
        super().__init__()
        self.chunk = chunk
        self.act_in = nn.Linear(act_dim, d_model)  # noisy action -> token
        self.tau = TauEmbed(d_model)
        self.pos = nn.Parameter(torch.randn(1 + chunk, d_model) * 0.02)  # [T, D] suffix slots
        self.blocks = nn.ModuleList(
            DecoderBlock(d_model, n_heads, ffn_hidden) for _ in range(layers)
        )
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x_tau, tau, state_token, memory, mem_key_mask):
        """x_tau [B, H, A], tau [B], state_token [B, D], memory [B, S, D], mem_key_mask [B, S].

        Returns (feats [B, H, D], self_attn [B, NH, T, T], cross_attn [B, NH, T, S]) with the
        attention maps taken from the last decoder layer.
        """
        b, h, _ = x_tau.shape
        assert h == self.chunk, f"chunk mismatch: got {h}, built for {self.chunk}"
        acts = self.act_in(x_tau) + self.tau(tau)[:, None, :]  # [B, H, D], tau on every token
        x = torch.cat([state_token[:, None, :], acts], dim=1) + self.pos  # suffix [B, T, D]
        self_mask = causal_mask(1 + h, device=x.device)  # [T, T]: state first, then the chunk
        mem_mask = mem_key_mask[:, None, None, :]  # [B, 1, 1, S]: hide the prefix <pad> keys
        for block in self.blocks:
            x, self_attn, cross_attn = block(x, memory, self_mask, mem_mask)
        feats = self.ln_out(x[:, 1:, :])  # [B, H, D]: chunk tokens only
        return feats, self_attn, cross_attn
