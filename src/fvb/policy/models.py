"""ACT-lite (history transformer with action-chunk queries) and an MLP baseline. PyTorch."""

from __future__ import annotations

import torch
from torch import nn


class ACTLite(nn.Module):
    """H observation tokens -> transformer encoder -> K learned queries cross-attend -> K actions.

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
    ):
        super().__init__()
        self.H, self.K = H, K
        self.embed = nn.Linear(obs_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, H, d))
        self.norm_in = nn.LayerNorm(d)
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)
        dec = nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.queries = nn.Parameter(torch.zeros(1, K, d))
        self.head = nn.Linear(d, act_dim)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B,H,obs) -> (B,K,act)
        h = self.norm_in(self.embed(x) + self.pos)
        mem = self.encoder(h)
        q = self.queries.expand(x.shape[0], -1, -1)
        out = self.decoder(q, mem)
        return self.head(out)


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
        return self.net(x.flatten(1)).view(x.shape[0], self.K, self.act_dim)


def build_model(kind: str, obs_dim: int, act_dim: int, H: int, K: int) -> nn.Module:
    if kind == "act":
        return ACTLite(obs_dim, act_dim, H, K)
    if kind == "mlp":
        return MLPPolicy(obs_dim, act_dim, H, K)
    raise ValueError(kind)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
