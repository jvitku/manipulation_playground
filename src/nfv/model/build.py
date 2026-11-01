"""Assemble the nanoVLA Policy: prefix (vision + text -> trunk), suffix (state + chunk ->
expert), flow-matching loss, Euler-sampling act(), and the EMA copy that is evaluated.

Two modules live here:

* ``NanoVLA`` — the trainable network. ``prefix()`` encodes [image ; text] once into E
  (≤ 84 tokens); ``velocity()`` runs the action expert over the suffix against that E and
  returns the flow velocity. Splitting the two is what makes Euler sampling cheap: one
  prefix pass, K small suffix passes.
* ``Policy`` — ``NanoVLA`` plus the house conventions: ``loss(batch)`` (flow matching with
  the dataset's pad mask), ``act(obs, instruction)`` (the ActionChunk contract), and an EMA
  copy of the parameters (decay 0.999) that ``act`` evaluates — the lab's convention that
  the evaluated/checkpointed model is the EMA (lab/core/loop.py). The optimizer and the
  parameter budget both refer to ``policy.net`` only; the EMA is a frozen shadow.

Action scaling (fixed, not fitted — a checkpoint needs no data statistics): the env clamps
dp to ±DP_MAX (0.05 m) per axis and the gripper lives in [0, 1], so model units are
dp / DP_MAX in [-1, 1] and gripper * 2 - 1 in [-1, 1]; flow matching runs in model units.

Shape legend: B = batch, H = chunk (16), A = act_dim (4), D = d_model, S = prefix tokens,
T = max_text. Batch contract = nfv.data.dataset plus "text" int64 [B, T] (nfv.train adds it).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from nfv.model.config import ModelConfig
from nfv.model.expert import ActionExpert
from nfv.model.flow import FlowMatching
from nfv.model.force_in import BALANCE_LOSS_COEF, ForceInput, apply_force_noise
from nfv.model.text import BLANK_INSTRUCTION, TextEncoder, tokenize
from nfv.model.trunk import Trunk, attn_snapshot
from nfv.model.vision import VisionEncoder
from nfv.sim.controllers import DP_MAX  # ±0.05 m: the env's per-axis dp clamp

EMA_DECAY = 0.999  # house convention (lab/core/loop.py)


def actions_to_model(dp: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
    """Env units -> model units in [-1, 1]: x0 = [dp / DP_MAX ; gripper * 2 - 1] [B, H, 4]."""
    return torch.cat([dp / DP_MAX, (gripper * 2 - 1)[..., None]], dim=-1)


def actions_from_model(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Model units -> env units, clamped to what the env accepts: (dp [.., 3], gripper [..])."""
    dp = x[..., :3].clamp(-1, 1) * DP_MAX
    gripper = ((x[..., 3] + 1) / 2).clamp(0, 1)
    return dp, gripper


def cfg_velocity(v_cond: torch.Tensor, v_blank: torch.Tensor, w: float) -> torch.Tensor:
    """Classifier-free guidance combination: v = v_blank + w * (v_cond - v_blank).

    w = 1 returns v_cond exactly; w = 0 the unconditioned velocity; w > 1 extrapolates past
    the conditioned prediction along the direction the instruction adds (Ho & Salimans 2022).
    """
    return v_blank + w * (v_cond - v_blank)


class NanoVLA(nn.Module):
    """The ≈ 7 M-parameter network; blocks named after the plan's table (param_table keys)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.vision = VisionEncoder(cfg.d_model, cfg.img, cfg.cameras)
        self.text = TextEncoder(cfg.d_model, cfg.heads, cfg.ffn, cfg.vocab, cfg.max_text)
        self.trunk = Trunk(cfg.d_model, cfg.heads, cfg.ffn, cfg.trunk_layers)
        self.state_in = nn.Linear(cfg.d_state, cfg.d_model)  # state -> one suffix token
        self.expert = ActionExpert(
            cfg.d_model, cfg.heads, cfg.ffn, cfg.expert_layers, cfg.act_dim, cfg.chunk
        )
        self.head = nn.Linear(cfg.d_model, cfg.act_dim)  # chunk features -> velocity
        self.force = ForceInput(cfg)  # the force_in route (no parameters for "none")
        # The blank text's token ids [1, T] (text_dropout / cfg_w read them); non-persistent:
        # a constant of the vocabulary, deliberately absent from every checkpoint state_dict.
        self.register_buffer(
            "blank_text",
            torch.from_numpy(tokenize(BLANK_INSTRUCTION, cfg.max_text))[None],
            persistent=False,
        )

    def prefix(self, rgb_static, rgb_wrist, text_ids, wrench=None):
        """[image ; text (; force tokens)] -> (E [B, S, D], key_mask [B, S], attns list).

        Vision tokens are always real; the text key_mask hides <pad> both inside the trunk
        and later as cross-attention keys. ``wrench`` [B, 6] is required by the force_in
        routes that touch the prefix (prefix_linear / prefix_moe / concat_after /
        prompt_bins — nfv.model.force_in) and ignored otherwise; appended force tokens are
        real (mask True).
        """
        v = self.vision(rgb_static, rgb_wrist)  # [B, 72, D]
        t, t_mask, _ = self.text(text_ids)  # [B, T, D], [B, T]
        tokens = torch.cat([v, t], dim=1)  # [B, S, D], S = 72 + T
        v_mask = torch.ones(v.shape[:2], dtype=torch.bool, device=v.device)  # [B, 72]
        key_mask = torch.cat([v_mask, t_mask], dim=1)  # [B, S]
        extra = self.force.prefix_tokens(wrench)  # [B, k, D] or None (route-dependent)
        if extra is not None:
            tokens = torch.cat([tokens, extra], dim=1)  # [B, S + k, D]
            pad = torch.ones(extra.shape[:2], dtype=torch.bool, device=extra.device)
            key_mask = torch.cat([key_mask, pad], dim=1)  # force tokens are real
        tokens = self.force.pre_trunk(tokens)  # prefix_moe's LIMoE pass; else identity
        e, attns = self.trunk(tokens, key_mask)  # E [B, S, D]
        e = self.force.post_trunk(e, wrench)  # concat_after's fuse; else identity
        return e, key_mask, attns

    def velocity(self, x_tau, tau, state, e, e_mask, return_attn: bool = False, feats_add=None):
        """One flow-velocity evaluation: v [B, H, A] (+ the expert's attention maps).

        ``feats_add`` [B, H, D] is the fvlmoe injection (force_in.ForceInput.expert_add):
        added to the chunk features BEFORE the action head — the reference's
        ``action_out_proj(limoe_out[:, -H:] + suffix_out[:, -H:])``.
        """
        state_token = self.state_in(state)  # [B, d_state] -> [B, D]
        feats, self_attn, cross_attn = self.expert(x_tau, tau, state_token, e, e_mask)
        if feats_add is not None:
            feats = feats + feats_add  # [B, H, D]
        v = self.head(feats)  # [B, H, D] -> [B, H, A]
        if return_attn:
            return v, {"expert_self_attn": self_attn, "expert_cross_attn": cross_attn}
        return v


class Policy(nn.Module):
    """NanoVLA + flow loss + EMA + act(); constructed through nfv.model.config.build(cfg)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.flow = FlowMatching()
        self.net = NanoVLA(cfg)
        self.ema = copy.deepcopy(self.net).eval()  # the evaluated copy (house convention)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def ema_update(self, decay: float = EMA_DECAY) -> None:
        """Shadow <- lerp(shadow, net, 1 - decay) after every optimizer step."""
        for pe, pm in zip(self.ema.parameters(), self.net.parameters(), strict=True):
            pe.lerp_(pm, 1 - decay)

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Flow-matching loss on one dataset batch (+ "text"); aux splits it per action part.

        Uses ``batch``: rgb_static/rgb_wrist [B, 3, 96, 96], state [B, d_state], text [B, T],
        dp [B, H, 3], gripper [B, H], pad_mask [B, H] (True = real step) — plus wrench [B, 6]
        when force_in is on (the dataset serves it; force_noise corrupts it here, the same
        place act() does). With cfg.balance_loss the LIMoE's load-balancing aux is ADDED
        (coef BALANCE_LOSS_COEF) — the reference only computes it (force_in.py docstring).
        With cfg.text_dropout > 0, each sample's instruction is replaced by the blank text
        with that probability (the CFG conditioning dropout); at 0 the branch is never
        entered, so the default training consumes exactly the pre-round-2 RNG stream.
        """
        x0 = actions_to_model(batch["dp"], batch["gripper"])  # [B, H, A] in model units
        text = batch["text"]
        if self.cfg.text_dropout > 0:
            drop = torch.rand(text.shape[0], device=text.device) < self.cfg.text_dropout
            text = torch.where(drop[:, None], self.net.blank_text, text)  # [B, T]
        wrench = None
        if self.net.force.needs_wrench:
            wrench = apply_force_noise(batch["wrench"], self.cfg.force_noise)
        e, e_mask, _ = self.net.prefix(batch["rgb_static"], batch["rgb_wrist"], text, wrench)
        feats_add = self.net.force.expert_add(e, wrench, self.cfg.chunk)  # fvlmoe or None

        def velocity(x_tau, tau):
            return self.net.velocity(x_tau, tau, batch["state"], e, e_mask, feats_add=feats_add)

        per, aux = self.flow.loss(velocity, x0, batch["pad_mask"])  # per-sample [B]
        per_dim = aux["per_dim"]  # [A] masked MSE per action dimension
        loss = per.mean()
        out = {"loss_dp": float(per_dim[:3].mean()), "loss_gripper": float(per_dim[3])}
        moe_aux = self.net.force.last_aux  # set by the LIMoE routes during the forward
        if moe_aux is not None:
            out["moe_fraction_dropped"] = float(moe_aux["fraction_dropped"])
            out["loss_balance"] = float(moe_aux["balance_loss"])
            if self.cfg.balance_loss:
                loss = loss + BALANCE_LOSS_COEF * moe_aux["balance_loss"]
        return loss, out

    @torch.no_grad()
    def act(self, obs, instruction: str, generator: torch.Generator | None = None) -> dict:
        """One Obs + instruction -> the ActionChunk contract (plan §interfaces), via the EMA.

        Returns {"dp": f32 [H, 3] (env units, clamped), "gripper": f32 [H] in [0, 1],
        "f_target": None, "s": None (both land with the lesson-18 hybrid head),
        "aux": attention snapshots (trunk last layer + the final Euler step's expert maps)}.
        The executor is expected to run the first cfg.n_exec of the H steps.

        With cfg.cfg_w != 1 a second, blank-instruction prefix is encoded and every Euler
        step combines the two velocities: v = v_blank + w * (v_cond - v_blank)
        (:func:`cfg_velocity`). The blank branch runs FIRST so force.last_aux and the aux
        attention maps stay the conditioned pass's; at w = 1.0 it is skipped entirely, so the
        default act() is bit-identical to the pre-round-2 model (same RNG stream: the eval-mode
        forwards draw nothing).
        """
        cfg = self.cfg
        dev = next(self.ema.parameters()).device
        assert obs.rgb_static is not None, "act() needs a rendering env (render=True)"
        # uint8 HWC -> uint8 CHW [1, 3, 96, 96]; the vision stem normalises uint8 itself
        rgb_s = torch.from_numpy(np.ascontiguousarray(obs.rgb_static)).permute(2, 0, 1)[None]
        rgb_w = torch.from_numpy(np.ascontiguousarray(obs.rgb_wrist)).permute(2, 0, 1)[None]
        state = torch.from_numpy(np.asarray(obs.state, dtype=np.float32))[None]  # [1, d_state]
        text = torch.from_numpy(tokenize(instruction, cfg.max_text))[None]  # [1, T]
        rgb_s, rgb_w, state, text = (t.to(dev) for t in (rgb_s, rgb_w, state, text))
        wrench = None
        if self.ema.force.needs_wrench:
            wrench = torch.from_numpy(np.asarray(obs.wrench, dtype=np.float32))[None].to(dev)
            wrench = apply_force_noise(wrench, cfg.force_noise, generator=generator)

        w = float(cfg.cfg_w)
        e_b = e_mask_b = feats_add_b = None
        if w != 1.0:  # the CFG blank branch, encoded once per decision (before the cond pass)
            e_b, e_mask_b, _ = self.ema.prefix(rgb_s, rgb_w, self.ema.blank_text, wrench)
            feats_add_b = self.ema.force.expert_add(e_b, wrench, cfg.chunk)
        e, e_mask, trunk_attns = self.ema.prefix(rgb_s, rgb_w, text, wrench)
        feats_add = self.ema.force.expert_add(e, wrench, cfg.chunk)  # fvlmoe or None
        last_attn: dict[str, torch.Tensor] = {}

        def velocity(x_tau, tau):
            v, attn = self.ema.velocity(
                x_tau, tau, state, e, e_mask, return_attn=True, feats_add=feats_add
            )
            last_attn.update(attn)  # keep the final Euler step's maps for aux
            if e_b is None:
                return v
            v_b = self.ema.velocity(x_tau, tau, state, e_b, e_mask_b, feats_add=feats_add_b)
            return cfg_velocity(v, v_b, w)

        x = self.flow.sample(
            velocity, (1, cfg.chunk, cfg.act_dim), cfg.euler_steps, dev, generator=generator
        )  # [1, H, A] in model units
        dp, gripper = actions_from_model(x)  # env units
        aux = {"trunk_attn": attn_snapshot(trunk_attns[-1])}
        aux.update({k: attn_snapshot(v) for k, v in last_attn.items()})
        return {
            "dp": dp[0].cpu().numpy().astype(np.float32),  # [H, 3]
            "gripper": gripper[0].cpu().numpy().astype(np.float32),  # [H]
            "f_target": None,
            "s": None,
            "aux": aux,
        }


def param_table(policy: Policy) -> dict[str, int]:
    """Trainable-parameter count per plan-table block (EMA excluded), plus the total."""

    def count(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    net = policy.net
    table = {
        "vision": count(net.vision),
        "text": count(net.text),
        "trunk": count(net.trunk),
        "state": count(net.state_in),
        "expert": count(net.expert),
        "head": count(net.head),
        "force": count(net.force),  # 0 for force_in="none"
    }
    table["total"] = sum(table.values())
    return table
