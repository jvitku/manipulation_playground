"""ForceVLA2's modules on the nano trunk (lesson 18): fusion encoder + 3-expert MoE, hybrid head.

ForceVLA2 (arXiv 2603.15169, CVPR 2026) has no released code, so everything here is built
from the paper's equations; every choice the paper leaves open is marked **[choice]** and is
repeated in lesson 18. The force prompt (the third ForceVLA2 module) is a sim-side state
machine and lives in ``nfv/sim/subtasks.py``.

Data flow (paper Fig. 2, our names):

    E          = Trunk([Z_v ; Z_l])                    prefix, untouched by force   [B, S, D]
    E_P, E_F   = φ_P(state), φ_F(wrench / scale)       two query tokens             [B, 1, D] each
    E'_state   = CrossAttn(Q = [E_P ; E_F], K = V = E)  state + force READ the scene [B, 2, D]
    E_cond     = [E ; E'_state ; E_F]                  E_F bypasses the attention   [B, S+3, D]
    w          = softmax(gate(E_cond))                 token-wise, 3 weights        [B, S+3, 3]
    E_MoE      = Σ_m w_m · Expert_m(X)                 dense mix, X = E_cond (see
                                                       expert_inputs)               [B, S+3, D]
    cond       = out_proj(E_MoE)                       the action expert's memory   [B, S+3, D]

``cond`` REPLACES E as the action expert's cross-attention memory (ForceVLA instead ADDS a
fused branch to the expert's output — the contrast lesson 18 draws).

[choice] CrossAttn is one multi-head attention (the trunk's ``MHA``) with LayerNorm on the
queries and NO residual: the paper writes E'_state = CrossAttn(·), and the bypass exists
precisely because the attended output carries the raw force only through the attention
weights. So with the cross-attention zeroed, state has no path out of this module and force
still reaches the gate and the experts through E_F (tests/nfv/test_nfv_forcevla2.py).

[choice] Wrench scaling: forces / F_SCALE (25 N, the env's abort bound), torques /
TORQUE_SCALE (2.5 Nm, force_in.py's prompt_bins range) — the paper normalises to ±100 N /
±15 Nm on its arm; the gantry never exceeds the abort bound.

``expert_inputs`` (the paper's expert ablations; the GATE always sees the full E_cond):
    all          X = E_cond                                   (full ForceVLA2)
    vision_only  X = [E ; 0 ; 0]   state + force tokens zeroed  ("experts see vision only", 50 %)
    none         X = [0 ; E'_state ; 0]  only the state tokens  ("neither vision nor force", 36 %)

The experts are named visual / state / force by the paper but are identical MLPs that all see
the same X; only training makes them specialise (the gate tilt lesson 18 plots).

Hybrid head (plan §4, the gantry): one Linear(D -> 11) on the expert's chunk features, the
flow velocity over a = [Δp 3 ; gripper 1 ; f_target 6 ; s 1]. Flow matching runs jointly on
all 11 dims in MODEL units:
    Δp / DP_MAX (0.05 m)                    in [-1, 1]
    gripper * 2 - 1                         in [-1, 1]
    f_target[:3] / F_SCALE (25 N), f_target[3:] / TORQUE_SCALE (2.5 Nm)   in [-1, 1]
    s * 2 - 1                               in [-1, 1]  (env units: s in [0, 1])
``hybrid_from_model`` clamps back to env units. The controller selects a force axis at
|f_target,i| > 0.5 N = 0.02 in model units, so residual sampling noise on the f_target dims
can select axes in free space — watch it in lesson 18 (f_target RMSE in free space).

Shape legend: B = batch, S = prefix tokens (72 vision + T text), D = d_model, H = chunk,
NH = heads, M = 3 experts, A = 11 hybrid action dims.
"""

from __future__ import annotations

import torch
from torch import nn

from nfv.model.trunk import MHA
from nfv.sim.controllers import DP_MAX, F_ABORT

F_SCALE = F_ABORT  # 25 N: the env's abort bound normalises forces (in and out)
TORQUE_SCALE = 2.5  # Nm: force_in.TORQUE_RANGE_NM, the gantry's torque range
WRENCH_DIM = 6
EXPERT_NAMES = ("visual", "state", "force")  # the paper's names; identical MLPs
EXPERT_INPUTS = ("all", "vision_only", "none")
HYBRID_DIM = 3 + 1 + 6 + 1  # [Δp ; gripper ; f_target ; s] = 11


def _wrench_scale(ref: torch.Tensor) -> torch.Tensor:
    """[6]: (F_SCALE x 3, TORQUE_SCALE x 3) on ref's device/dtype."""
    return ref.new_tensor([F_SCALE] * 3 + [TORQUE_SCALE] * 3)


class ForceVLA2Fusion(nn.Module):
    """[E ; state ; wrench] -> (cond [B, S+3, D], cond_mask [B, S+3], aux); module docstring."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        n_heads: int,
        expert_inputs: str = "all",
        n_vision: int = 72,
    ):
        super().__init__()
        if expert_inputs not in EXPERT_INPUTS:
            raise ValueError(f"expert_inputs={expert_inputs!r} not in {EXPERT_INPUTS}")
        self.expert_inputs = expert_inputs
        self.n_vision = n_vision  # first n_vision prefix tokens are vision (gate aux split)
        self.phi_p = nn.Linear(d_state, d_model)  # φ_P: state -> query token E_P
        self.phi_f = nn.Linear(WRENCH_DIM, d_model)  # φ_F: scaled wrench -> query token E_F
        self.ln_q = nn.LayerNorm(d_model)  # pre-norm on the two queries
        self.cross_attn = MHA(d_model, n_heads)  # Q = [E_P ; E_F], K = V = E
        self.gate = nn.Linear(d_model, len(EXPERT_NAMES))  # token-wise logits over 3 experts
        self.experts = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
                )
                for name in EXPERT_NAMES
            }
        )
        self.out_proj = nn.Linear(d_model, d_model)  # E_MoE -> the action expert's memory

    def expert_input(self, e_cond: torch.Tensor, s: int) -> torch.Tensor:
        """The experts' view X of E_cond [B, S+3, D] per ``expert_inputs`` (S = prefix)."""
        if self.expert_inputs == "all":
            return e_cond
        keep = torch.zeros(e_cond.shape[1], 1, dtype=e_cond.dtype, device=e_cond.device)
        if self.expert_inputs == "vision_only":
            keep[:s] = 1.0  # the prefix E only
        else:  # "none": only E'_state (positions S, S+1)
            keep[s : s + 2] = 1.0
        return e_cond * keep  # [B, S+3, D], other tokens zeroed

    def forward(self, e, e_mask, state, wrench):
        """e [B, S, D], e_mask [B, S] bool, state [B, d_state], wrench [B, 6] (N, Nm, raw).

        Returns (cond [B, S+3, D], cond_mask [B, S+3], aux) with aux =
            gate        [B, S+3, 3]  token-wise expert weights (rows sum to 1)
            gate_groups {vision|text|state|force: [B, 3]}  gate averaged per token group
                        (real tokens only; state = the two E'_state tokens, force = bypass)
            cross_attn  [B, NH, 2, S]  what E_P / E_F looked at in the prefix
        """
        b, s, _ = e.shape
        e_p = self.phi_p(state)[:, None, :]  # [B, 1, D]
        e_f = self.phi_f(wrench / _wrench_scale(wrench))[:, None, :]  # [B, 1, D]
        q = torch.cat([e_p, e_f], dim=1)  # [B, 2, D]: the two queries
        e_state, attn = self.cross_attn(self.ln_q(q), e, mask=e_mask[:, None, None, :])
        # e_state [B, 2, D] = E'_state (no residual: see the module docstring)
        e_cond = torch.cat([e, e_state, e_f], dim=1)  # [B, S+3, D]: E_F is the bypass
        real = torch.ones(b, 3, dtype=torch.bool, device=e.device)
        cond_mask = torch.cat([e_mask, real], dim=1)  # [B, S+3]

        w = self.gate(e_cond).softmax(-1)  # [B, S+3, 3]: token-wise weights over experts
        x = self.expert_input(e_cond, s)  # [B, S+3, D]: what the experts see
        outs = torch.stack([ex(x) for ex in self.experts.values()], dim=-1)  # [B, S+3, D, 3]
        e_moe = (outs * w[:, :, None, :]).sum(-1)  # dense mix Σ_m w_m · Expert_m(X) [B, S+3, D]
        cond = self.out_proj(e_moe)  # [B, S+3, D]

        aux = {"gate": w.detach(), "gate_groups": self._gate_groups(w.detach(), e_mask, s)}
        aux["cross_attn"] = attn.detach()
        return cond, cond_mask, aux

    def _gate_groups(self, w, e_mask, s) -> dict[str, torch.Tensor]:
        """Gate weights [B, S+3, 3] averaged per token group -> {group: [B, 3]}."""
        nv = min(self.n_vision, s)

        def mean(lo, hi, mask=None):
            ww = w[:, lo:hi]  # [B, n, 3]
            if mask is None:
                return ww.mean(1)
            m = mask[:, :, None].to(w.dtype)  # [B, n, 1]
            return (ww * m).sum(1) / m.sum(1).clamp_min(1.0)

        return {
            "vision": mean(0, nv),
            "text": mean(nv, s, e_mask[:, nv:s]),
            "state": mean(s, s + 2),
            "force": mean(s + 2, s + 3),
        }


class HybridHead(nn.Module):
    """Chunk features [B, H, D] -> flow velocity over [Δp 3 ; gripper 1 ; f_target 6 ; s 1]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, HYBRID_DIM)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.proj(feats)  # [B, H, D] -> [B, H, 11]


def hybrid_to_model(dp, gripper, f_target, s) -> torch.Tensor:
    """Env units -> model units [B, H, 11]; dp [B,H,3], gripper [B,H], f_target [B,H,6], s [B,H]."""
    return torch.cat(
        [
            dp / DP_MAX,  # [B, H, 3]
            (gripper * 2 - 1)[..., None],  # [B, H, 1]
            f_target / _wrench_scale(f_target),  # [B, H, 6]
            (s * 2 - 1)[..., None],  # [B, H, 1]
        ],
        dim=-1,
    )


def hybrid_from_model(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Model units [.., 11] -> env units, clamped to what the env/controller accept."""
    return {
        "dp": x[..., 0:3].clamp(-1, 1) * DP_MAX,  # [.., 3] m
        "gripper": ((x[..., 3] + 1) / 2).clamp(0, 1),  # [..]
        "f_target": x[..., 4:10].clamp(-1, 1) * _wrench_scale(x),  # [.., 6] N, Nm
        "s": ((x[..., 10] + 1) / 2).clamp(0, 1),  # [..]
    }
