"""Where the wrench enters the nano trunk: the five force-input routes (lesson 17).

Each route is one row of the papers' "where does force enter" ablations; the config flag
``ModelConfig.force_in`` picks it and ``build`` wires it (every variant is a config, never a
subclass — plan §4). The wrench is the env's compensated+filtered 6-D reading (nfv/sim/env.py
Obs legend), served raw in Newtons / Newton-metres like the state vector.

Routes (paper rows):
    prefix_linear  φ_F = Linear(6 -> d); ONE force token appended to the trunk INPUT prefix
                   [image ; text ; force]. ForceVLA Table 3 "force as an input token".
    prefix_moe     the LIMoE block applied BEFORE the trunk, over [image ; text ; φ_F] —
                   ForceVLA Table 3's "MoE before the VLM" row (0 % success in the paper;
                   kept as the expected-collapse teaching ablation).
    concat_after   E_F = φ_F(wrench) concatenated FEATURE-WISE to every token of the trunk
                   OUTPUT E, fused by one Linear(2d -> d), no MoE — ForceVLA Table 3's
                   "w/o MoE (concat + linear)" row.
    fvlmoe         the paper's FVLMoE and the reference injection (pi0_force.py): the ported
                   LIMoEBlock runs over [E ; E_F] (force token LAST) and the LAST H tokens of
                   its output are ADDED to the action expert's chunk features before the
                   action head — v = head(feats + limoe_out[:, -H:]), the reference's
                   ``action_out_proj(limoe_out[:, -H:] + suffix_out[:, -H:])``. Note the
                   quirk is kept: the last H positions of [E ; E_F] are text/pad tokens plus
                   the force token, whatever H is. (The reference's sample_actions never
                   recomputes a live prefix for the limoe input — its compute_loss and
                   sample_actions disagree; here act() and loss() share one definition.)
    prompt_bins    ForceBench: each wrench axis quantised to 32 bins over a fixed range
                   (forces ±25 N — the controller's abort bound — torques ±2.5 Nm) and
                   appended as 6 extra text-like tokens from a learned embedding table.

``force_noise`` (eval-time twin / FACT checks, applied to the wrench wherever act()/loss()
reads it): off = raw; zero = zeros (== the force-blind twin); shuffle = permute the batch
dimension (a batch of 1 is unchanged — document in the lesson); gaussian = add
N(0, FORCE_NOISE_SIGMA²) per component.

The LIMoE dims at d = 256 (plan §4 "+ForceVLA ≈ +1.9 M"): encoder MLP = cfg.ffn (768, the
width every other nano block uses), experts 4 x (d -> 2d -> d), 8 heads when d divides by 8
(the reference uses 8), out_dim = d. The balance loss is returned in the block's aux and only
added to the training loss when ``cfg.balance_loss`` is set (the reference computes it and
never adds it) with coefficient BALANCE_LOSS_COEF (Switch Transformer's 0.01).
"""

from __future__ import annotations

import torch
from torch import nn

from nfv.model.config import ModelConfig
from nfv.model.limoe_port import LIMoEBlock

WRENCH_DIM = 6
FORCE_BINS = 32  # bins per wrench axis (prompt_bins)
FORCE_RANGE_N = 25.0  # forces quantised over ±25 N (= controllers.F_ABORT)
TORQUE_RANGE_NM = 2.5  # torques quantised over ±2.5 Nm
FORCE_NOISE_SIGMA = 1.0  # gaussian force_noise std, N / Nm
BALANCE_LOSS_COEF = 0.01  # Switch Transformer's auxiliary-loss coefficient

_LIMOE_ROUTES = ("prefix_moe", "fvlmoe")
_PHI_ROUTES = ("prefix_linear", "prefix_moe", "concat_after", "fvlmoe")


def apply_force_noise(
    wrench: torch.Tensor, mode: str, generator: torch.Generator | None = None
) -> torch.Tensor:
    """The twin / FACT wrench corruptions on a [B, 6] wrench (see the module docstring)."""
    if mode == "off":
        return wrench
    if mode == "zero":
        return torch.zeros_like(wrench)
    if mode == "shuffle":
        perm = torch.randperm(wrench.shape[0], generator=generator, device=wrench.device)
        return wrench[perm]
    if mode == "gaussian":
        noise = torch.empty_like(wrench).normal_(0.0, FORCE_NOISE_SIGMA, generator=generator)
        return wrench + noise
    raise ValueError(f"force_noise={mode!r}")


def quantize_wrench(wrench: torch.Tensor) -> torch.Tensor:
    """[B, 6] wrench -> int64 [B, 6] embedding ids: axis * FORCE_BINS + bin (prompt_bins).

    Each axis is clamped to its fixed range (±FORCE_RANGE_N for the 3 forces, ±TORQUE_RANGE_NM
    for the 3 torques) and quantised to FORCE_BINS uniform bins.
    """
    lim = wrench.new_tensor([FORCE_RANGE_N] * 3 + [TORQUE_RANGE_NM] * 3)  # [6]
    unit = (wrench.clamp(-lim, lim) + lim) / (2 * lim)  # [B, 6] in [0, 1]
    bins = (unit * FORCE_BINS).long().clamp(0, FORCE_BINS - 1)  # [B, 6]
    axis = torch.arange(WRENCH_DIM, device=wrench.device)  # [6]
    return axis * FORCE_BINS + bins  # [B, 6]


class ForceInput(nn.Module):
    """The force_in route of one config: parameters + the four hooks NanoVLA calls.

    For ``force_in="none"`` the module has no parameters and every hook is the identity /
    None, so the nanoVLA baseline is bit-identical to the pre-lesson-17 model. The aux dict
    of the most recent LIMoE call (expert counts, fraction dropped, balance loss) is kept on
    ``self.last_aux`` for the loss/logging step that follows the forward.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.route = cfg.force_in
        self.last_aux: dict | None = None
        if self.route in _PHI_ROUTES:
            # the reference's name for φ_F (pi0_force.py: force_in_proj on state[7:13])
            self.force_in_proj = nn.Linear(WRENCH_DIM, cfg.d_model)
        if self.route in _LIMOE_ROUTES:
            self.limoe = LIMoEBlock(
                d_model=cfg.d_model,
                mlp_dim=cfg.ffn,
                num_heads=8 if cfg.d_model % 8 == 0 else cfg.heads,  # reference: 8 heads
                out_dim=cfg.d_model,
                num_experts=cfg.n_experts,
                moe=cfg.moe,
                expert_dim=2 * cfg.d_model,
            )
        if self.route == "concat_after":
            self.fuse = nn.Linear(2 * cfg.d_model, cfg.d_model)
        if self.route == "prompt_bins":
            self.bins = nn.Embedding(WRENCH_DIM * FORCE_BINS, cfg.d_model)

    @property
    def needs_wrench(self) -> bool:
        return self.route != "none"

    def _force_token(self, wrench: torch.Tensor) -> torch.Tensor:  # [B, 6] -> [B, 1, D]
        return self.force_in_proj(wrench)[:, None, :]

    def prefix_tokens(self, wrench: torch.Tensor | None) -> torch.Tensor | None:
        """Extra tokens appended to the trunk INPUT after [image ; text], or None."""
        if self.route in ("prefix_linear", "prefix_moe"):
            return self._force_token(wrench)  # [B, 1, D]
        if self.route == "prompt_bins":
            return self.bins(quantize_wrench(wrench))  # [B, 6, D]
        return None

    def pre_trunk(self, tokens: torch.Tensor) -> torch.Tensor:
        """prefix_moe: the LIMoE pass over the whole prefix before the trunk; else identity."""
        if self.route != "prefix_moe":
            return tokens
        out, self.last_aux = self.limoe(tokens)
        return out

    def post_trunk(self, e: torch.Tensor, wrench: torch.Tensor | None) -> torch.Tensor:
        """concat_after: [E ; E_F broadcast] -> Linear(2d -> d); else identity."""
        if self.route != "concat_after":
            return e
        e_f = self._force_token(wrench).expand(-1, e.shape[1], -1)  # [B, S, D]
        return self.fuse(torch.cat([e, e_f], dim=-1))  # [B, S, D]

    def expert_add(
        self, e: torch.Tensor, wrench: torch.Tensor | None, chunk: int
    ) -> torch.Tensor | None:
        """fvlmoe: LIMoE over [E ; E_F], last ``chunk`` tokens -> add to the chunk features."""
        if self.route != "fvlmoe":
            return None
        limoe_in = torch.cat([e, self._force_token(wrench)], dim=1)  # [B, S+1, D], force LAST
        out, self.last_aux = self.limoe(limoe_in)
        return out[:, -chunk:, :]  # [B, H, D] — the reference's limoe_out[:, -H:]
