"""TA-SmolVLA (PLAN §13 phase 2, after TA-VLA arXiv 2509.07962): SmolVLA with a torque token.

SmolVLA keeps the robot state in the VLM *prefix*; its action expert's *suffix* is the noisy
action chunk + time. TA-VLA's finding is that the torque history works best on the decoder side,
as a single token. Here the 10 x 15 torque history (tau_ext + tactile summary over 2 s,
``observation.torque_hist``) goes through an MLP to one token of the expert's width, prepended to
the suffix; every action token attends to it. The loss and the denoiser read only the last
``chunk_size`` suffix positions, so the extra token changes nothing else.

``zero_token=True`` keeps the architecture and zeroes the token's input - the "no force" twin
with identical parameters (the plan's ablation).

``normalize=True`` z-scores the history first (per-dim ``mean`` / ``std`` buffers, saved with the
checkpoint): unnormalised, the tactile grip force (~47 N) swamps the joint torques (~1 N).

``aux_weight > 0`` adds TA-VLA's auxiliary torque prediction, L = L_action + w * L_torque: a
linear head on the mean of the action expert's chunk outputs predicts the (z-scored) torque
history ``future`` steps ahead (``observation.torque_future``, from the converter's
``future_torque``), so the expert's representation has to model the coming contact forces.
"""

from __future__ import annotations

import torch
from torch import nn

TORQUE_KEY = "observation.torque_hist"
FUTURE_KEY = "observation.torque_future"


class TorqueNorm(nn.Module):
    def __init__(self, dim: int, mean=None, std=None):
        super().__init__()
        m = torch.zeros(dim) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        s = torch.ones(dim) if std is None else torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("mean", m.clone())
        self.register_buffer("std", torch.where(s < 1e-3, torch.ones_like(s), s))

    def forward(self, t):
        return (t - self.mean) / self.std


def make_ta_smolvla(
    policy,
    torque_dim: int = 150,
    zero_token: bool = False,
    normalize: bool = False,
    stats=None,
    aux_weight: float = 0.0,
):
    """Turn a loaded ``SmolVLAPolicy`` into TA-SmolVLA in place and return it.
    ``stats``: ``(mean, std)`` for ``normalize`` (training); at load time the checkpoint's buffers
    overwrite the defaults."""
    model = policy.model
    width = model.vlm_with_expert.expert_hidden_size
    dev = next(model.parameters()).device
    model.torque_mlp = nn.Sequential(
        nn.Linear(torque_dim, width), nn.SiLU(), nn.Linear(width, width)
    ).to(dev)
    if normalize:
        mean, std = stats if stats is not None else (None, None)
        model.torque_norm = TorqueNorm(torque_dim, mean, std).to(dev)
    if aux_weight > 0:
        mean, std = stats if stats is not None else (None, None)
        model.torque_target_norm = TorqueNorm(torque_dim, mean, std).to(dev)
        model.torque_head = nn.Linear(width, torque_dim).to(dev)
        model._aux = {}

        def keep_chunk_out(_mod, args):  # input of action_out_proj = the chunk's expert outputs
            model._aux["h"] = args[0]

        model.action_out_proj.register_forward_pre_hook(keep_chunk_out)
    model._torque = None
    model._zero_token = zero_token
    base_embed_suffix = model.embed_suffix

    def embed_suffix(noisy_actions, timestep):
        embs, pad, att = base_embed_suffix(noisy_actions, timestep)
        t = model._torque
        if t is None:
            return embs, pad, att
        t = t.to(device=embs.device, dtype=torch.float32)
        if model._zero_token:
            t = torch.zeros_like(t)
        t = t.flatten(1)
        if hasattr(model, "torque_norm"):
            t = model.torque_norm(t)
        tok = model.torque_mlp(t)[:, None, :].to(embs.dtype)
        b = embs.shape[0]
        if tok.shape[0] != b:  # sampling may repeat the batch
            tok = tok.expand(b, -1, -1)
        embs = torch.cat([tok, embs], dim=1)
        pad = torch.cat([torch.ones(b, 1, dtype=pad.dtype, device=pad.device), pad], dim=1)
        att = torch.cat([torch.ones(b, 1, dtype=att.dtype, device=att.device), att], dim=1)
        return embs, pad, att

    model.embed_suffix = embed_suffix

    def with_torque(fn):
        def wrapped(batch, *args, **kwargs):
            model._torque = batch.get(TORQUE_KEY)
            try:
                return fn(batch, *args, **kwargs)
            finally:
                model._torque = None

        return wrapped

    fwd = with_torque(policy.forward)
    if aux_weight > 0:

        def fwd_aux(batch, *args, **kwargs):
            model._aux.clear()
            loss, out = fwd(batch, *args, **kwargs)
            target = batch.get(FUTURE_KEY)
            if target is None or "h" not in model._aux:
                return loss, out
            h = model._aux["h"].float().mean(dim=1)
            pred = model.torque_head(h)
            tgt = model.torque_target_norm(target.to(pred.device, torch.float32).flatten(1))
            aux = torch.nn.functional.mse_loss(pred, tgt)
            out = dict(out or {})
            out["torque_aux_loss"] = float(aux.detach())
            return loss + aux_weight * aux, out

        policy.forward = fwd_aux
    else:
        policy.forward = fwd
    policy.predict_action_chunk = with_torque(policy.predict_action_chunk)
    policy.select_action = with_torque(policy.select_action)
    return policy
