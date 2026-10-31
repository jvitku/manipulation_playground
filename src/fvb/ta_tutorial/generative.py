"""Generative action heads: flow matching (pi0 / SmolVLA / TA-VLA) and DDPM (Diffusion Policy).

Both learn a distribution over action chunks ``x0`` (shape ``(B, H, A)``, actions scaled to
about [-1, 1]) conditioned on an observation. Neither predicts the action directly; both train a
network on *noised* versions of the expert chunk and generate by starting from pure noise and
repeatedly applying the network. That is what lets them represent several distinct valid actions
for one observation (go left OR right around an obstacle) where a regression head returns the
average (straight into the obstacle).

The network is called as ``model(x_t, t, cond) -> output`` with ``x_t`` the noisy chunk and ``t``
a float in [0, 1] per sample; ``cond`` is whatever the network needs (observation tensors) and is
passed through untouched.

Flow matching (Lipman et al. 2023, "rectified flow"; the convention below is SmolVLA's / pi0's)
-----------------------------------------------------------------------------------------------
* pick ``t ~ Beta(1.5, 1)`` (more weight on noisy t near 1), noise ``e ~ N(0, I)``;
* interpolate on a straight line:  ``x_t = t * e + (1 - t) * x0``   (t = 1 pure noise, t = 0 data);
* the velocity of that line is ``u = d x_t / d t = e - x0``; train ``v = model(x_t, t)`` with
  ``L = |v - u|^2``;
* sample: start at ``x_1 = e``, integrate ``dx/dt = v`` from t = 1 to 0 with K Euler steps
  ``x <- x - (1/K) * v(x, t)``.

Averaged over all (x0, e) pairs that pass through a point, the regressed velocity is the
*marginal* velocity field, and integrating it transports the noise distribution onto the data
distribution. With one mode the field is nearly straight (one step already works); with several
modes it bends, and you need more steps.

DDPM / DDIM (Ho et al. 2020; Song et al. 2021; Chi et al. 2023 "Diffusion Policy")
---------------------------------------------------------------------------------
* a discrete schedule ``abar_k`` (cosine) for k = 0..T-1 from almost clean to almost pure noise;
* ``x_k = sqrt(abar_k) x0 + sqrt(1 - abar_k) e``; train ``eps = model(x_k, k/T)`` with
  ``L = |eps - e|^2`` (predict the noise);
* sample with DDIM (deterministic, eta = 0) on a subsequence of S timesteps: estimate
  ``x0_hat = (x_k - sqrt(1 - abar_k) eps) / sqrt(abar_k)``, clip it to [-1, 1], move to the next
  (less noisy) timestep. S can be much smaller than T at test time.

The two are the same idea with a different path and parameterisation; flow matching's straight
path is why pi0-style VLAs get away with ~10 integration steps.
"""

from __future__ import annotations

import math

import torch


class FlowMatching:
    name = "flow"

    def __init__(self, beta_a: float = 1.5, beta_b: float = 1.0, t_min: float = 0.001):
        self.beta = torch.distributions.Beta(beta_a, beta_b)
        self.t_min = t_min

    def sample_t(self, n: int, device) -> torch.Tensor:
        return (self.beta.sample((n,)) * (1 - self.t_min) + self.t_min).to(device)

    def loss(self, model, x0: torch.Tensor, cond, mask: torch.Tensor | None = None):
        """Per-sample loss and the model's extra outputs. ``mask`` (B, H): 1 where the chunk
        holds a real action (0 for padding past the end of the episode, as in LeRobot)."""
        # B = batch, H = chunk horizon, A = action dims; x0 [B, H, A] is the expert chunk
        e = torch.randn_like(x0)  # noise [B, H, A]
        t = self.sample_t(x0.shape[0], x0.device)  # noise level per sample [B]
        tt = t[:, None, None]  # [B] -> [B, 1, 1] for broadcasting
        x_t = tt * e + (1 - tt) * x0  # interpolate on the line: [B, H, A]
        v, extra = model(x_t, t, cond)  # predicted velocity [B, H, A]
        err = (v - (e - x0)).pow(2).mean(-1)  # target e - x0: [B, H, A] -> [B, H]
        if mask is not None:
            err = err * mask  # [B, H], padding zeroed
            return err.sum(1) / mask.sum(1).clamp_min(1), extra  # [B, H] -> [B]
        return err.mean(1), extra  # [B, H] -> [B]

    @torch.no_grad()
    def sample(self, model, shape, cond, steps: int = 10, generator=None, trace: bool = False):
        # B = batch, H = chunk horizon, A = action dims; shape = (B, H, A), steps = K
        x = torch.randn(shape, generator=generator, device=_dev(cond, model))  # x at t=1 [B, H, A]
        xs = [x]
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((shape[0],), 1.0 - i * dt, device=x.device)  # [B], t: 1 -> 1/K
            v, _ = model(x, t, cond)  # velocity [B, H, A]
            x = x - dt * v  # Euler step toward the data: [B, H, A]
            if trace:
                xs.append(x)
        return (x, torch.stack(xs)) if trace else x  # [B, H, A]; trace adds [K+1, B, H, A]


class DDPM:
    name = "ddpm"

    def __init__(self, T: int = 100, s: float = 0.008):
        # cosine schedule (Nichol & Dhariwal 2021): abar(k) = cos^2(((k/T + s) / (1 + s)) * pi/2)
        k = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((k / T + s) / (1 + s)) * math.pi / 2) ** 2
        abar = (f / f[0]).clamp(1e-5, 1.0)
        self.T = T
        self.abar = abar[1:].float()  # abar_k for k = 0..T-1 (k = 0 is the least noisy)

    def loss(self, model, x0, cond, mask=None):
        # B = batch, H = chunk horizon, A = action dims; x0 [B, H, A] is the expert chunk
        k = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)  # timestep per sample [B]
        ab = self.abar.to(x0.device)[k][:, None, None]  # [B] -> [B, 1, 1]
        e = torch.randn_like(x0)  # noise [B, H, A]
        x_k = ab.sqrt() * x0 + (1 - ab).sqrt() * e  # noised chunk [B, H, A]
        eps, extra = model(x_k, (k.float() + 1) / self.T, cond)  # predicted noise [B, H, A]
        err = (eps - e).pow(2).mean(-1)  # [B, H, A] -> [B, H]
        if mask is not None:
            err = err * mask
            return err.sum(1) / mask.sum(1).clamp_min(1), extra
        return err.mean(1), extra

    @torch.no_grad()
    def sample(self, model, shape, cond, steps: int = 10, generator=None, trace: bool = False):
        x = torch.randn(shape, generator=generator, device=_dev(cond, model))
        abar = self.abar.to(x.device)
        ks = torch.linspace(self.T - 1, 0, steps).round().long().tolist()
        xs = [x]
        for j, k in enumerate(ks):
            ab = abar[k]
            eps, _ = model(x, torch.full((shape[0],), (k + 1) / self.T, device=x.device), cond)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            if j + 1 < len(ks):
                ab_next = abar[ks[j + 1]]
                eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()  # consistent with the clipped x0
                x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
            else:
                x = x0
            if trace:
                xs.append(x)
        return (x, torch.stack(xs)) if trace else x


def _dev(cond, model):
    for v in cond.values() if isinstance(cond, dict) else [cond]:
        if torch.is_tensor(v):
            return v.device
    return next(model.parameters()).device


class Regression:
    """Baseline with the same interface: the network sees no noise (x_t = 0, t = 0) and is
    trained with MSE to output the chunk itself. It can only represent the conditional MEAN."""

    name = "mse"

    def loss(self, model, x0, cond, mask=None):
        # B = batch, H = chunk horizon, A = action dims; x0 [B, H, A] is the expert chunk
        # the net sees x_t = zeros [B, H, A] and t = zeros [B]; out [B, H, A] is the chunk itself
        out, extra = model(torch.zeros_like(x0), torch.zeros(x0.shape[0], device=x0.device), cond)
        err = (out - x0).pow(2).mean(-1)  # [B, H, A] -> [B, H]
        if mask is not None:
            err = err * mask  # [B, H], padding zeroed
            return err.sum(1) / mask.sum(1).clamp_min(1), extra  # [B, H] -> [B]
        return err.mean(1), extra  # [B, H] -> [B]

    @torch.no_grad()
    def sample(self, model, shape, cond, steps: int = 1, generator=None, trace: bool = False):
        dev = _dev(cond, model)
        out, _ = model(torch.zeros(shape, device=dev), torch.zeros(shape[0], device=dev), cond)
        return (out, torch.stack([out])) if trace else out


def make(name: str):
    return {"flow": FlowMatching, "ddpm": DDPM, "mse": Regression}[name]()
