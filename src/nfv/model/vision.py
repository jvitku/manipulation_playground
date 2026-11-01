"""Vision tokens: one conv stem shared by both cameras -> 36 tokens each at d = 256.

The stem is 4 stride-2 conv layers (32, 64, 128, 256 channels) with GELUs, so a 96x96 image
shrinks 96 -> 48 -> 24 -> 12 -> 6 and comes out as a 6x6 grid of d-dim feature vectors = 36
tokens per camera. The two cameras share the stem weights (the plan's 0.4 M vision budget is
for the block, not per camera) and are told apart by a learned per-camera embedding; where a
token sits in its image comes from a learned 2-D position embedding (a row table plus a
column table, summed — "2-D" because the index is (row, col), not a flat token number).

Shape legend: B = batch, C = cameras (2), D = d_model, G = grid side (6), N = C * G^2 = 72.

Input convention: the dataset serves float images in [0, 1] (its batch contract), the live
env serves raw uint8 — the stem accepts both and lands them on the same [-0.5, 0.5] range
(uint8 x -> x / 255 - 0.5, float x -> x - 0.5), so act() needs no preprocessing.
"""

from __future__ import annotations

import torch
from torch import nn

STEM_CHANNELS = (32, 64, 128, 256)  # one entry per stride-2 layer; the last must equal D


class ConvStem(nn.Module):
    """4x (stride-2 conv 3x3 + GELU): [B, 3, img, img] -> [B, G^2, D] tokens."""

    def __init__(self, d_model: int, img: int):
        super().__init__()
        assert STEM_CHANNELS[-1] == d_model, "the last stem stage must emit d_model channels"
        assert img % 2 ** len(STEM_CHANNELS) == 0, "img must survive 4 halvings"
        self.grid = img // 2 ** len(STEM_CHANNELS)  # G: 96 -> 6
        layers: list[nn.Module] = []
        c_in = 3
        for c_out in STEM_CHANNELS:
            layers += [nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1), nn.GELU()]
            c_in = c_out
        self.net = nn.Sequential(*layers)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img [B, 3, 96, 96] uint8 (env) or float in [0, 1] (dataset) -> same [-0.5, 0.5] range
        x = img.float() / 255.0 if img.dtype == torch.uint8 else img
        x = x - 0.5  # [B, 3, 96, 96], zero-centred
        x = self.net(x)  # [B, D, G, G]
        return x.flatten(2).transpose(1, 2)  # [B, D, G*G] -> [B, G*G, D]: one token per cell


class VisionEncoder(nn.Module):
    """Both cameras through the shared stem + 2-D position + camera embeddings -> [B, N, D]."""

    def __init__(self, d_model: int, img: int, cameras: int):
        super().__init__()
        self.cameras = cameras
        self.stem = ConvStem(d_model, img)
        g = self.stem.grid
        self.pos_row = nn.Parameter(torch.randn(g, d_model) * 0.02)  # [G, D]
        self.pos_col = nn.Parameter(torch.randn(g, d_model) * 0.02)  # [G, D]
        self.cam_embed = nn.Parameter(torch.randn(cameras, d_model) * 0.02)  # [C, D]

    def forward(self, *imgs: torch.Tensor) -> torch.Tensor:
        """imgs: one [B, 3, img, img] tensor per camera (static, wrist) -> [B, C*G^2, D]."""
        assert len(imgs) == self.cameras, f"expected {self.cameras} camera images"
        pos = self.pos_row[:, None, :] + self.pos_col[None, :, :]  # [G, G, D], 2-D position
        pos = pos.flatten(0, 1)  # [G*G, D], same order as the stem's flatten
        toks = [
            self.stem(img) + pos + self.cam_embed[c]  # [B, G*G, D] per camera
            for c, img in enumerate(imgs)
        ]
        return torch.cat(toks, dim=1)  # [B, C*G^2, D] = [B, 72, 256]
