"""ACT-lite (history transformer with action-chunk queries) and an MLP baseline. PyTorch."""

from __future__ import annotations

import torch
from torch import nn


class ACTLite(nn.Module):
    """H observation tokens -> transformer encoder -> K learned queries cross-attend -> K actions.

    Shapes (gantry defaults O = 18, A = 3, D = 128): window [B, H, O] -> memory [B, H, D];
    K queries [B, K, D] cross-attend the memory -> [B, K, D] -> action chunk [B, K, A].
    No images, no CVAE (the expert is deterministic). ~0.6 M parameters at the defaults.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        H: int,
        K: int,
        d: int = 128,
        heads: int = 4,
        enc_layers: int = 3,
        dec_layers: int = 2,
        ff: int = 256,
        dropout: float = 0.1,
        input_norm: bool = True,
    ):
        super().__init__()
        self.H, self.K = H, K
        self.embed = nn.Linear(obs_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, H, d))
        self.norm_in = nn.LayerNorm(d) if input_norm else nn.Identity()
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)
        dec = nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.queries = nn.Parameter(torch.zeros(1, K, d))
        self.head = nn.Linear(d, act_dim)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, H, O] -> [B, K, A]
        # legend: B = batch, H = history, K = chunk, O = obs dims, A = action dims, D = width
        h = self.norm_in(self.embed(x) + self.pos)  # embed [B, H, O] -> [B, H, D]; pos [1, H, D]
        mem = self.encoder(h)  # encoder: [B, H, D] -> [B, H, D]
        q = self.queries.expand(x.shape[0], -1, -1)  # [1, K, D] -> [B, K, D]
        out = self.decoder(q, mem)  # cross-attend: [B, K, D] x [B, H, D] -> [B, K, D]
        return self.head(out)  # [B, K, D] -> [B, K, A]


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, H: int, K: int, hidden: int = 256):
        super().__init__()
        self.K, self.act_dim = K, act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim * H, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, K * act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # legend: B = batch, H = history, K = chunk, O = obs dims, A = action dims
        # [B, H, O] -> flatten [B, H*O] -> net [B, K*A] -> view [B, K, A]
        return self.net(x.flatten(1)).view(x.shape[0], self.K, self.act_dim)


def build_model(kind: str, obs_dim: int, act_dim: int, H: int, K: int, **kw) -> nn.Module:
    """``kw``: ACT-lite options (dropout, input_norm); recorded in checkpoints as model_kw."""
    if kind == "act":
        return ACTLite(obs_dim, act_dim, H, K, **kw)
    if kind == "mlp":
        return MLPPolicy(obs_dim, act_dim, H, K)
    raise ValueError(kind)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
