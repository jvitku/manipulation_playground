"""Denoisers: a plain MLP, and TAVLAMini - pi0 / SmolVLA's layout with TA-VLA's torque token.

TAVLAMini, as in pi0 (Black et al. 2024) and SmolVLA (Shukor et al. 2025), is ONE attention
sequence processed by TWO sets of weights:

    prefix  = [image patch tokens | language tokens | state token]    -> "VLM" weights
    suffix  = [torque token | H noisy-action tokens (+ time)]         -> "action expert" weights

Every layer computes queries / keys / values per stream with that stream's weights, then attends
jointly under a block mask: prefix tokens see only the prefix; suffix tokens see the prefix and
the whole suffix. So the observation is encoded without ever looking at the noisy actions, which
gives the main inference trick: the prefix keys/values are computed ONCE per observation and
reused for all K denoising steps (``with model.cache(cond): ...``).

The torque token (TA-VLA, Zhang et al. 2025, arXiv 2509.07962): the last few frames of
force / torque readings, flattened, through a 2-layer MLP into one d-dim token. TA-VLA compares
putting it in the encoder (``torque_pos="prefix"``) against the decoder (``"suffix"``) and finds
the decoder better: there it sits next to the actions being denoised at every step.
``zero_torque=True`` keeps the parameters and feeds zeros - the no-force twin.

The auxiliary head (TA-VLA's L = L_action + w * L_torque): a linear map from the mean of the
action tokens' outputs to the FUTURE torque history; it makes the expert's representation
model the contact forces the chunk will cause.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F
from torch import nn


def time_embedding(t: torch.Tensor, dim: int, max_period: float = 100.0) -> torch.Tensor:
    """Sinusoidal embedding of a scalar t in [0, 1] (the noise level), shape (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    ang = 2 * math.pi * t[:, None] * freqs[None] * 10
    return torch.cat([ang.sin(), ang.cos()], dim=1)


class MLPDenoiser(nn.Module):
    """[flattened noisy chunk, observation, time embedding] -> flattened chunk (velocity / noise /
    the action itself for the regression baseline)."""

    def __init__(self, act_dim: int, horizon: int, obs_dim: int, hidden: int = 512, depth: int = 4):
        super().__init__()
        self.act_dim, self.horizon = act_dim, horizon
        layers, n = [], act_dim * horizon + obs_dim + 64
        for _ in range(depth):
            layers += [nn.Linear(n, hidden), nn.GELU()]
            n = hidden
        layers.append(nn.Linear(n, act_dim * horizon))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, cond):
        # B = batch, H = chunk horizon, A = action dims, O = obs dims
        # flatten the chunk [B, H, A] -> [B, H*A]; cat with obs [B, O] and time [B, 64]
        h = torch.cat([x_t.flatten(1), cond["obs"], time_embedding(t, 64)], dim=1)
        return self.net(h).view(-1, self.horizon, self.act_dim), {}  # [B, H*A] -> [B, H, A]


class _Stream(nn.Module):
    def __init__(self, d: int, heads: int, mlp: int = 4):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def qkv_heads(self, x):
        # B = batch, T = tokens, D = width, h = heads, dh = D/h
        b, n, d = x.shape  # x [B, T, D]
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)  # [B, T, 3*D] -> 3 x [B, T, D]
        # split heads: each [B, T, D] -> [B, h, T, dh]
        return [z.view(b, n, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v)]

    def finish(self, x, attn):
        # B = batch, T = tokens, D = width, h = heads, dh = D/h; x [B, T, D], attn [B, h, T, dh]
        b, h, n, dh = attn.shape
        x = x + self.proj(attn.transpose(1, 2).reshape(b, n, h * dh))  # merge heads -> [B, T, D]
        return x + self.mlp(self.ln2(x))  # residual MLP: [B, T, D]


class TwoStreamBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.vlm, self.expert = _Stream(d, heads), _Stream(d, heads)

    def prefix(self, xp):
        # B = batch, P = prefix tokens, D = width, h = heads, dh = D/h; xp [B, P, D]
        q, k, v = self.vlm.qkv_heads(xp)  # each [B, h, P, dh]
        # out [B, P, D]; (k, v) is this layer's KV-cache entry, each [B, h, P, dh]
        return self.vlm.finish(xp, F.scaled_dot_product_attention(q, k, v)), (k, v)

    def suffix(self, xs, kv_prefix):
        # B = batch, S = suffix tokens, P = prefix tokens, D = width, h = heads, dh = D/h
        q, k, v = self.expert.qkv_heads(xs)  # xs [B, S, D] -> each [B, h, S, dh]
        kp, vp = kv_prefix  # cached prefix keys/values, each [B, h, P, dh]
        # the suffix attends to the prefix AND to itself (bidirectionally within the chunk)
        # keys/values cat along tokens: [B, h, P + S, dh]; attn out [B, h, S, dh]
        attn = F.scaled_dot_product_attention(q, torch.cat([kp, k], 2), torch.cat([vp, v], 2))
        return self.expert.finish(xs, attn)  # [B, h, S, dh] -> [B, S, D]


class TAVLAMini(nn.Module):
    def __init__(
        self,
        act_dim: int,
        horizon: int,
        state_dim: int,
        torque_dim: int = 0,
        torque_pos: str = "suffix",  # "none" | "prefix" | "suffix"
        zero_torque: bool = False,
        aux_dim: int = 0,
        image_hw: int = 0,  # 0: no camera
        vocab: int = 0,  # 0: no language
        lang_len: int = 6,
        d: int = 128,
        layers: int = 4,
        heads: int = 4,
    ):
        super().__init__()
        assert torque_pos in ("none", "prefix", "suffix")
        self.act_dim, self.horizon, self.d = act_dim, horizon, d
        self.torque_pos = torque_pos if torque_dim else "none"
        self.zero_torque = zero_torque
        # --- prefix encoders ("VLM" side) ---
        self.state_in = nn.Linear(state_dim, d)
        self.image_hw = image_hw
        if image_hw:
            self.cnn = nn.Sequential(  # 32x32 -> 4x4 = 16 patch tokens
                nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                nn.Conv2d(64, d, 3, 2, 1),
            )  # fmt: skip
            self.img_pos = nn.Parameter(torch.zeros(1, (image_hw // 8) ** 2, d))
        self.vocab = vocab
        if vocab:
            self.lang = nn.Embedding(vocab, d)
            self.lang_pos = nn.Parameter(torch.zeros(1, lang_len, d))
        # --- the torque token ---
        if self.torque_pos != "none":
            self.torque_mlp = nn.Sequential(nn.Linear(torque_dim, d), nn.SiLU(), nn.Linear(d, d))
        # --- suffix ("action expert" side): noisy action + time -> token, as in pi0 ---
        self.action_in = nn.Linear(act_dim, d)
        self.action_time = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.act_pos = nn.Parameter(torch.zeros(1, horizon, d))
        self.blocks = nn.ModuleList(TwoStreamBlock(d, heads) for _ in range(layers))
        self.ln_out = nn.LayerNorm(d)
        self.action_out = nn.Linear(d, act_dim)
        self.aux_head = nn.Linear(d, aux_dim) if aux_dim else None
        for p in (self.act_pos, *([self.img_pos] if image_hw else []),
                  *([self.lang_pos] if vocab else [])):  # fmt: skip
            nn.init.normal_(p, std=0.02)
        self._kv = None

    # ---- the pieces --------------------------------------------------------------------------
    def torque_token(self, cond):
        # B = batch, D = width, Tq = flattened torque-history dims; cond["torque"] [B, Tq]
        tq = cond["torque"]  # [B, Tq]
        if self.zero_torque:
            tq = torch.zeros_like(tq)  # the zeroed twin: same parameters, no signal
        return self.torque_mlp(tq)[:, None, :]  # [B, Tq] -> [B, D] -> one token [B, 1, D]

    def encode_prefix(self, cond):
        """Run the VLM stream over the prefix; returns each layer's (keys, values)."""
        # B = batch, D = width, P = prefix tokens, L = lang tokens, h = heads, dh = D/h
        # P = image patches + L + 1 state token (+ 1 torque token when torque_pos == "prefix")
        toks = []
        if self.image_hw:
            f = self.cnn(cond["image"])  # (B, d, 4, 4)
            toks.append(f.flatten(2).transpose(1, 2) + self.img_pos)  # [B, D, 4, 4] -> [B, 16, D]
        if self.vocab:
            toks.append(self.lang(cond["lang"]) + self.lang_pos)  # ids [B, L] -> [B, L, D]
        toks.append(self.state_in(cond["state"])[:, None, :])  # [B, state_dim] -> [B, 1, D]
        if self.torque_pos == "prefix":
            toks.append(self.torque_token(cond))  # [B, 1, D]
        x = torch.cat(toks, dim=1)  # [B, P, D]
        kvs = []
        for blk in self.blocks:
            x, kv = blk.prefix(x)  # kv = (k, v), each [B, h, P, dh]
            kvs.append(kv)  # the KV cache: one (k, v) per layer
        return kvs

    def forward(self, x_t, t, cond):
        # B = batch, H = chunk horizon, A = action dims, D = width, S = suffix tokens, F = aux dims
        kvs = self._kv if self._kv is not None else self.encode_prefix(cond)  # prefix kv per layer
        a = self.action_in(x_t)  # [B, H, A] -> [B, H, D]
        te = time_embedding(t, self.d)[:, None, :].expand(-1, self.horizon, -1)  # [B] -> [B, H, D]
        s = self.action_time(torch.cat([a, te], dim=-1)) + self.act_pos  # [B, H, 2*D] -> [B, H, D]
        if self.torque_pos == "suffix":
            # prepend the torque token: [B, H, D] -> [B, S, D] with S = 1 + H (else S = H)
            s = torch.cat([self.torque_token(cond).expand(s.shape[0], -1, -1), s], dim=1)
        for blk, kv in zip(self.blocks, kvs, strict=True):
            s = blk.suffix(s, kv)  # [B, S, D] -> [B, S, D]
        h = self.ln_out(s[:, -self.horizon :])  # the action tokens: [B, S, D] -> [B, H, D]
        # aux head: h.mean(1) [B, H, D] -> [B, D] -> future-torque prediction [B, F]
        extra = {"aux": self.aux_head(h.mean(1))} if self.aux_head is not None else {}
        return self.action_out(h), extra  # velocity [B, H, D] -> [B, H, A]

    @contextlib.contextmanager
    def cache(self, cond):
        """Encode the prefix once and reuse it for every denoising step of one sample() call."""
        # B = batch, h = heads, P = prefix tokens, dh = width / heads
        self._kv = self.encode_prefix(cond)  # each layer's (k, v), both [B, h, P, dh]
        try:
            yield
        finally:
            self._kv = None
