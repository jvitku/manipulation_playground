"""PyTorch port of ForceVLA's LIMoEBlock — the FVLMoE fusion block (plan §4, lesson 17).

Provenance: ported from the pinned reference clone at ``third_party/forcevla``
(https://github.com/ft-robotic/ForceVLA, commit 79de88f, Apache License 2.0):

    src/openpi/models/limoe_simple.py                      LIMoEBlock, Encoder1DBlock,
                                                           MlpBlock (the encoder's GELU MLP),
                                                           MoeLayer, _num_groups
    flaxformer/flaxformer/architectures/moe/routing.py     RouterWeights ("w"),
                                                           TokensChooseScatterRouter (top-1,
                                                           jitter 0.01, batch-prioritized),
                                                           _load_balancing_loss, _router_z_loss
    flaxformer/flaxformer/components/dense.py              MlpBlock (the expert: wi -> relu -> wo)

The original files carry the Apache-2.0 notice ("Copyright 2024 Google LLC. Licensed under
the Apache License, Version 2.0 — http://www.apache.org/licenses/LICENSE-2.0; distributed on
an AS-IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND"); this port keeps that license
for the ported logic. Reference module/parameter names are kept so the parity fixture maps
one-to-one (see tests/nfv/test_nfv_limoe_parity.py for the name mapping).

Faithful-port notes (what is kept exactly):
  * pre-norm ViT encoder block: LayerNorm(eps 1e-6) -> multi-head attention (reference: 8
    heads; configurable here) -> dropout 0.1 -> residual -> LayerNorm -> MLP (GELU, tanh
    approximation — flax ``nn.gelu`` default) -> residual;
  * tokens-choose TOP-1 scatter router: multiplicative jitter noise 0.01 on the router INPUT
    (train only), softmax router probabilities, padding tokens (all-zero vectors) down-
    weighted to probability 0 (they still occupy capacity slots — the reference scatter
    router only down-weights, routing.py docstring), batch-prioritized routing (tokens sorted
    by top-1 probability, stable order on ties);
  * grouping: tokens are flattened over (batch, sequence) and split into groups by the
    reference ``_num_groups`` (groups of <= 16; when no viable split exists the reference
    degenerates to groups of 1, where the capacity floor makes the cap inert — kept as-is);
  * capacity: ``max(round(capacity_factor * group_size / n_experts), min_expert_capacity)``
    with capacity_factor 1.0 and min_expert_capacity 4 (reference defaults). A token that
    overflows its expert's buffer gets ZERO expert output (its combine weight is zeroed);
  * experts: flaxformer MlpBlock defaults — wi -> ReLU -> dropout(0.1, mask broadcast over
    tokens) -> wo -> dropout; intermediate width default 2048 (the flaxformer default the
    reference inherits);
  * residual ``x + moe_out`` and the final Linear to ``out_dim``;
  * the auxiliary load-balancing loss is COMPUTED but never added to any training loss —
    exactly like the reference; it is returned in ``aux`` so the course can add it
    (``ModelConfig.balance_loss``).

Deliberate deviations (documented, not silent):
  * fp32 end-to-end. The reference rounds router logits (RouterWeights dtype=bfloat16) and
    expert inputs (MoeLayer dtype=bfloat16) to bfloat16 — a TPU-efficiency choice. The course
    is fp32-only (plan §1), and the parity fixture pins those dtypes to float32 accordingly.
  * ``forward`` returns ``(y, aux)`` instead of the reference's ``(y, None)``; ``aux`` carries
    per-expert token counts and the fraction of tokens dropped by the capacity cap.
  * train/eval comes from ``nn.Module.training`` instead of a ``deterministic`` argument.
  * ``moe`` variants beyond the reference ("top1" without the capacity cap, "soft" dense
    mixing) are course ablations, flagged per call site via the constructor.

Shape legend: B = batch, S = tokens, D = width, E = experts, G = groups, g = group size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MOE_VARIANTS = ("top1_capacity", "top1", "soft")
LN_EPS = 1e-6  # flax LayerNorm default (torch default is 1e-5)


def _num_groups(num_tokens: int, max_group_size: int, num_experts: int) -> int:
    """Verbatim port of limoe_simple._num_groups: the smallest num_groups such that
    num_groups >= num_tokens / max_group_size, num_tokens % num_groups == 0 and
    num_groups % num_experts == 0 — falling back to num_tokens (groups of 1) when no
    such split exists, exactly like the reference loop."""
    min_num_groups = max(num_tokens // max_group_size, num_experts)

    def viable(n: int) -> bool:
        return num_tokens % n == 0 and n % num_experts == 0

    num_groups = min_num_groups
    while num_groups < num_tokens and not viable(num_groups):
        num_groups += 1
    if num_tokens % num_groups > 0:
        raise ValueError(
            f"no viable grouping: num_tokens={num_tokens}, num_experts={num_experts}, "
            f"max_group_size={max_group_size}"
        )
    return num_groups


class EncoderAttention(nn.Module):
    """flax ``nn.MultiHeadDotProductAttention`` equivalent (self-attention, no dropout).

    q/k/v project D -> heads*head_dim, scores are scaled by 1/sqrt(head_dim), softmax over
    keys, then the out projection mixes the heads back to D — the exact flax computation.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide into heads"
        self.h, self.dh = num_heads, d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, S, D] -> [B, S, D]
        b, s, _ = x.shape

        def split(t: torch.Tensor) -> torch.Tensor:  # [B, S, D] -> [B, H, S, dh]
            return t.view(b, s, self.h, self.dh).transpose(1, 2)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        attn = (q @ k.transpose(-2, -1) / self.dh**0.5).softmax(-1)  # [B, H, S, S]
        return self.out((attn @ v).transpose(1, 2).reshape(b, s, -1))


class EncoderMlp(nn.Module):
    """limoe_simple.MlpBlock: Dense -> GELU(tanh, the flax default) -> drop -> Dense -> drop."""

    def __init__(self, d_model: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dense_0 = nn.Linear(d_model, mlp_dim)
        self.dense_1 = nn.Linear(mlp_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(F.gelu(self.dense_0(x), approximate="tanh"))
        return self.drop(self.dense_1(x))


class Encoder1DBlock(nn.Module):
    """limoe_simple.Encoder1DBlock: pre-norm ViT layer (attention + MLP behind residuals)."""

    def __init__(self, d_model: int, mlp_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.encoder_in_norm = nn.LayerNorm(d_model, eps=LN_EPS)
        self.attention = EncoderAttention(d_model, num_heads)
        self.encoder_out_norm = nn.LayerNorm(d_model, eps=LN_EPS)
        self.mlp = EncoderMlp(d_model, mlp_dim, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attention(self.encoder_in_norm(x)))
        return x + self.mlp(self.encoder_out_norm(x))


class ExpertMlp(nn.Module):
    """One expert: flaxformer dense.MlpBlock defaults — wi -> ReLU -> drop -> wo -> drop.

    The reference dropout masks broadcast over the token axis (``broadcast_dims=(-2,)``):
    one mask per expert call, shared by every token the expert processes — kept here.
    """

    def __init__(self, d_model: int, expert_dim: int, dropout: float = 0.1):
        super().__init__()
        self.wi = nn.Linear(d_model, expert_dim)
        self.wo = nn.Linear(expert_dim, d_model)
        self.p = dropout

    def _broadcast_drop(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = F.dropout(torch.ones(1, x.shape[-1], device=x.device, dtype=x.dtype), self.p)
        return x * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [M, D] -> [M, D]
        x = self._broadcast_drop(F.relu(self.wi(x)))
        return self._broadcast_drop(self.wo(x))


class MoeLayer(nn.Module):
    """limoe_simple.MoeLayer + the TokensChooseScatterRouter, over flattened (B*S) tokens.

    ``moe`` picks the routing variant:
      top1_capacity  the reference: groups of <= ``max_group_size``, per-group expert
                     capacity, overflow tokens get zero expert output;
      top1           course ablation: the same top-1 routing with no capacity cap;
      soft           course ablation: dense softmax mix over all experts.

    Returns ``(moe_out [B, S, D], aux)`` where aux holds:
      expert_counts    int64 [E]  tokens each expert actually processed (within capacity)
      fraction_dropped tensor []  reference "fraction_tokens_left_behind"
      balance_loss     tensor []  Switch-Transformer load-balancing loss (computed, never
                                  added here — the reference computes it and never applies it)
      router_z_loss    tensor []  z-loss over the (padding-masked) router logits
      kept             bool [B, S]   token reached an expert (False = dropped or padding-zeroed)
      expert_index     int64 [B, S]  each token's top-1 expert (soft: argmax, for logging)
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        moe: str = "top1_capacity",
        expert_dim: int = 2048,  # flaxformer MlpBlock.intermediate_dim default (the reference's)
        max_group_size: int = 16,  # reference LIMoEBlock hardcodes 16
        capacity_factor: float = 1.0,  # reference train and eval capacity factor
        min_expert_capacity: int = 4,  # flaxformer MoeLayer default
        jitter_noise: float = 0.01,  # reference TokensChooseScatterRouter
        dropout: float = 0.1,
    ):
        super().__init__()
        assert moe in MOE_VARIANTS, f"moe={moe!r} not in {MOE_VARIANTS}"
        self.num_experts = num_experts
        self.moe = moe
        self.max_group_size = max_group_size
        self.capacity_factor = capacity_factor
        self.min_expert_capacity = min_expert_capacity
        self.jitter_noise = jitter_noise
        self.router = nn.Linear(d_model, num_experts)  # RouterWeights "w" (f32, like the port)
        self.experts = nn.ModuleList(
            ExpertMlp(d_model, expert_dim, dropout) for _ in range(num_experts)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        b, s, d = x.shape
        n = b * s
        if self.moe == "top1_capacity":
            groups = _num_groups(n, self.max_group_size, self.num_experts)
            g = n // groups
            capacity = max(
                int(round(self.capacity_factor * g / self.num_experts)),
                self.min_expert_capacity,
            )
        else:
            groups, g, capacity = 1, n, n  # one group, the cap can never bind
        xg = x.reshape(groups, g, d)

        # Router probabilities (reference _compute_router_probabilities): jitter noise is
        # multiplicative on the router INPUT, train only; the experts see the raw tokens.
        router_in = xg
        if self.training and self.jitter_noise > 0:
            lo, hi = 1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            router_in = xg * torch.empty_like(xg).uniform_(lo, hi)
        logits = self.router(router_in)  # [G, g, E]
        # ignore_padding_tokens=True: all-zero tokens are down-weighted (never fully excluded)
        padding = (xg.abs().sum(-1) > 0).to(x.dtype)  # [G, g], 1 = real token
        probs = logits.softmax(-1) * padding.unsqueeze(-1)  # [G, g, E]
        gate, index = probs.max(-1)  # top-1 (ties -> first expert, like jax.lax.top_k)

        # Switch-Transformer load-balancing loss, from the pre-capacity choices (reference
        # computes it right after top-k, before batch-prioritized reordering).
        expert_mask = F.one_hot(index, self.num_experts).to(probs.dtype)  # [G, g, E]
        balance = (expert_mask.mean(1) * probs.mean(1)).mean() * self.num_experts**2  # [] (eq. 4-6)
        masked_logits = logits * padding.unsqueeze(-1)  # reference masks logits for the z-loss
        z_loss = (torch.logsumexp(masked_logits, dim=-1) ** 2).sum() / n

        if self.moe == "soft":  # dense mix: every expert sees every token
            moe_flat = torch.zeros_like(xg)
            for e, expert in enumerate(self.experts):
                moe_flat = moe_flat + probs[..., e : e + 1] * expert(xg)
            kept = padding > 0
            combine = gate * kept
        else:
            if self.moe == "top1_capacity":
                # Batch-prioritized routing: fill expert buffers in order of router confidence.
                order = torch.argsort(-gate, dim=-1, stable=True)  # [G, g]
                idx_sorted = index.gather(-1, order)
                one_hot = F.one_hot(idx_sorted, self.num_experts)  # [G, g, E]
                priority_sorted = (one_hot.cumsum(1) * one_hot).sum(-1) - 1  # buffer slot
                priority = priority_sorted.gather(-1, torch.argsort(order, dim=-1))
                kept = priority < capacity  # overflow tokens get ZERO expert output
            else:
                kept = torch.ones_like(gate, dtype=torch.bool)
            combine = gate * kept
            moe_flat = torch.zeros_like(xg)
            for e, expert in enumerate(self.experts):
                m = (index == e) & kept
                if bool(m.any()):
                    moe_flat = moe_flat.index_put((m,), expert(xg[m]))
            moe_flat = moe_flat * combine.unsqueeze(-1)

        counts = torch.stack([((index == e) & kept).sum() for e in range(self.num_experts)])  # [E]
        aux = {
            "expert_counts": counts,
            "fraction_dropped": 1.0 - kept.float().sum() / n,
            "balance_loss": balance,
            "router_z_loss": z_loss,
            "kept": kept.reshape(b, s),
            "expert_index": index.reshape(b, s),
        }
        return moe_flat.reshape(b, s, d), aux


class LIMoEBlock(nn.Module):
    """limoe_simple.LIMoEBlock: encoder block -> MoE -> residual -> out projection.

    forward: x [B, S, D] -> (y [B, S, out_dim], aux) with aux from the MoeLayer (see above).
    """

    def __init__(
        self,
        d_model: int,
        mlp_dim: int,
        num_heads: int,
        out_dim: int,
        num_experts: int = 4,
        moe: str = "top1_capacity",
        expert_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoderblock = Encoder1DBlock(d_model, mlp_dim, num_heads, dropout)
        self.moe_layer = MoeLayer(
            d_model, num_experts=num_experts, moe=moe, expert_dim=expert_dim, dropout=dropout
        )
        self.out_proj = nn.Linear(d_model, out_dim)  # the reference's final Dense_0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x = self.encoderblock(x)
        moe_out, aux = self.moe_layer(x)
        return self.out_proj(x + moe_out), aux
