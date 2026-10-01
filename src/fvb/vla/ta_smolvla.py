"""TA-SmolVLA (PLAN §13 phase 2, after TA-VLA arXiv 2509.07962): SmolVLA with a torque token.

SmolVLA keeps the robot state in the VLM *prefix*; its action expert's *suffix* is the noisy
action chunk + time. TA-VLA's finding is that the torque history works best on the decoder side,
as a single token. Here the 10 x 15 torque history (tau_ext + tactile summary over 2 s,
``observation.torque_hist``) goes through an MLP to one token of the expert's width, prepended to
the suffix; every action token attends to it. The loss and the denoiser read only the last
``chunk_size`` suffix positions, so the extra token changes nothing else.

``zero_token=True`` keeps the architecture and zeroes the token's input - the "no force" twin
with identical parameters (the plan's ablation).

Not yet included: TA-VLA's auxiliary torque-prediction loss (L = L_action + 0.1 L_torque).
"""

from __future__ import annotations

import torch
from torch import nn

TORQUE_KEY = "observation.torque_hist"


def make_ta_smolvla(policy, torque_dim: int = 150, zero_token: bool = False):
    """Turn a loaded ``SmolVLAPolicy`` into TA-SmolVLA in place and return it."""
    model = policy.model
    width = model.vlm_with_expert.expert_hidden_size
    dev = next(model.parameters()).device
    model.torque_mlp = nn.Sequential(
        nn.Linear(torque_dim, width), nn.SiLU(), nn.Linear(width, width)
    ).to(dev)
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
        tok = model.torque_mlp(t.flatten(1))[:, None, :].to(embs.dtype)
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

    policy.forward = with_torque(policy.forward)
    policy.predict_action_chunk = with_torque(policy.predict_action_chunk)
    policy.select_action = with_torque(policy.select_action)
    return policy
