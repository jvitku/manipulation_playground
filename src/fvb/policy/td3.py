"""TD3 (Fujimoto et al. 2018): twin critics, delayed actor updates, target-policy smoothing.

Small, dependency-free (PyTorch only) implementation for the Stage 1 RL comparison. Actor and
critics are 2-layer 256-unit ReLU MLPs; actions live in [-1, 1].
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class TD3Config:
    hidden: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    batch: int = 256


def mlp(i: int, o: int, h: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, h: int):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, h)

    def forward(self, o):
        return torch.tanh(self.net(o))


class Critic(nn.Module):
    """Two independent Q heads (clipped double-Q)."""

    def __init__(self, obs_dim: int, act_dim: int, h: int):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, h)
        self.q2 = mlp(obs_dim + act_dim, 1, h)

    def forward(self, o, a):
        x = torch.cat([o, a], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros((size, 1), np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros((size, 1), np.float32)  # 1 = terminal (no bootstrap)
        self.size, self.n, self.i = size, 0, 0

    def add(self, o, a, r, o2, terminal: bool) -> None:
        self.o[self.i], self.a[self.i], self.r[self.i] = o, a, r
        self.o2[self.i], self.d[self.i] = o2, float(terminal)
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch: int, rng: np.random.Generator, device):
        idx = rng.integers(0, self.n, batch)
        t = lambda x: torch.as_tensor(x[idx], device=device)  # noqa: E731
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d)


class TD3:
    def __init__(self, obs_dim: int, act_dim: int, cfg: TD3Config, device: str = "cpu"):
        self.cfg, self.device = cfg, device
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.actor = Actor(obs_dim, act_dim, cfg.hidden).to(device)
        self.critic = Critic(obs_dim, act_dim, cfg.hidden).to(device)
        self.actor_t = copy.deepcopy(self.actor)
        self.critic_t = copy.deepcopy(self.critic)
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.updates = 0

    @torch.no_grad()
    def act(self, o: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(o, dtype=torch.float32, device=self.device)[None]
        return self.actor(x)[0].cpu().numpy()

    def update(self, buf: ReplayBuffer, rng: np.random.Generator) -> dict:
        c = self.cfg
        o, a, r, o2, d = buf.sample(c.batch, rng, self.device)
        with torch.no_grad():
            noise = (torch.randn_like(a) * c.target_noise).clamp(-c.noise_clip, c.noise_clip)
            a2 = (self.actor_t(o2) + noise).clamp(-1.0, 1.0)
            q1t, q2t = self.critic_t(o2, a2)
            y = r + c.gamma * (1.0 - d) * torch.min(q1t, q2t)
        q1, q2 = self.critic(o, a)
        c_loss = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
        self.c_opt.zero_grad(set_to_none=True)
        c_loss.backward()
        self.c_opt.step()
        out = {
            "critic_loss": c_loss.item(),
            "q1_mean": q1.mean().item(),
            "target_mean": y.mean().item(),
        }
        self.updates += 1
        if self.updates % c.policy_delay == 0:
            a_loss = -self.critic(o, self.actor(o))[0].mean()
            self.a_opt.zero_grad(set_to_none=True)
            a_loss.backward()
            self.a_opt.step()
            with torch.no_grad():
                for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                    for p, pt in zip(net.parameters(), tgt.parameters(), strict=True):
                        pt.mul_(1 - c.tau).add_(c.tau * p)
            out["actor_loss"] = a_loss.item()
        return out

    def state(self, meta: dict) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "cfg": asdict(self.cfg),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "meta": meta,
        }


class TD3Policy:
    """Deterministic TD3 actor with the BC policies' interface: ``act(raw ARM observation)``
    returns a tip-target delta in metres. Reproduces ArmRLEnv's pipeline (fixed normalisation,
    force zeroing, frame stacking, action scaling), so it plugs into ``ArmTask`` directly."""

    def __init__(self, path: str, device: str = "cpu", untrained_seed: int | None = None):
        import json
        from collections import deque
        from pathlib import Path

        from fvb.policy.data import Norm
        from fvb.policy.spec import ARM

        ck = torch.load(path, map_location=device, weights_only=False)
        self.meta = ck["meta"]
        self.meta.setdefault("model", "td3")
        self.actor = Actor(ck["obs_dim"], ck["act_dim"], ck["cfg"]["hidden"]).to(device)
        if untrained_seed is None:
            self.actor.load_state_dict(ck["actor"])
        else:  # same architecture, fresh random weights
            torch.manual_seed(untrained_seed)
            self.actor = Actor(ck["obs_dim"], ck["act_dim"], ck["cfg"]["hidden"]).to(device)
        self.actor.eval()
        self.device = device
        self.use_force = bool(self.meta["use_force"])
        self.norm = Norm.from_json(json.loads(Path(self.meta["norm"]).read_text()))
        self.force_idx = ARM.force_idx
        self.scale = np.array([self.meta["xy_scale"], self.meta["xy_scale"], self.meta["z_scale"]])
        self.frames: deque = deque(maxlen=int(self.meta["history"]))
        self.task_overrides: dict = {}

    def reset(self) -> None:
        self.frames.clear()

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        o = ((obs - self.norm.obs_mean) / self.norm.obs_std).astype(np.float32)
        if not self.use_force:
            o[self.force_idx] = 0.0
        if not self.frames:
            for _ in range(self.frames.maxlen - 1):
                self.frames.append(o)
        self.frames.append(o)
        x = torch.as_tensor(np.concatenate(list(self.frames)), device=self.device)[None]
        return np.clip(self.actor(x)[0].cpu().numpy(), -1, 1) * self.scale
