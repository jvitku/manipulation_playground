"""Flow matching over action chunks — the conventions of ``fvb/ta_tutorial/generative.py``.

Self-contained here so the school's model has no tutorial dependency, but sign- and
shape-compatible with that file (lessons 2 and 4 taught it; this is the same object):

* pick ``tau ~ Beta(1.5, 1)`` (more weight near the noisy end), noise ``e ~ N(0, I)``;
* interpolate on the straight line ``x_tau = tau * e + (1 - tau) * x0`` (tau = 1 pure noise,
  tau = 0 data); its velocity is ``u = d x_tau / d tau = e - x0``;
* train ``v = velocity(x_tau, tau)`` with the MSE ``|v - u|^2``, masked where the chunk is
  padding past the episode end;
* sample: start at ``x = e``, integrate from tau = 1 to 0 with K Euler steps
  ``x <- x - (1/K) * v(x, tau)``.

Averaged over all (x0, e) pairs through a point, the regressed velocity is the marginal
velocity field; with ONE data mode the field is nearly straight, so a single Euler step from
anywhere at tau = 1 already lands on the mode's mean — the unit-test gate of milestone M1.

``velocity`` is a plain callable ``(x_tau [B, H, A], tau [B]) -> v [B, H, A]``: the policy
passes a closure that carries the conditioning (prefix E, state), so this file never sees
images or text. Shape legend: B = batch, H = chunk length, A = act_dim, K = Euler steps.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Velocity = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def interpolate(x0: torch.Tensor, e: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """The line between data and noise: x_tau = tau * e + (1 - tau) * x0, tau broadcast [B]."""
    tt = tau[:, None, None]  # [B] -> [B, 1, 1]
    return tt * e + (1 - tt) * x0  # [B, H, A]


class FlowMatching:
    """Loss and Euler sampler; stateless apart from the tau distribution."""

    def __init__(self, beta_a: float = 1.5, beta_b: float = 1.0, tau_min: float = 0.001):
        self.beta = torch.distributions.Beta(beta_a, beta_b)
        self.tau_min = tau_min  # keep tau off exact 0, where the target is the clean data

    def sample_tau(self, n: int, device) -> torch.Tensor:
        """[n] flow times in (tau_min, 1], Beta(1.5, 1)-distributed (drawn on CPU, moved)."""
        return (self.beta.sample((n,)) * (1 - self.tau_min) + self.tau_min).to(device)

    def loss(self, velocity: Velocity, x0: torch.Tensor, mask: torch.Tensor | None = None):
        """Velocity-matching MSE. ``mask`` [B, H]: True = real step, False = padding.

        Returns (per_sample [B], aux) with aux["per_dim"] the masked MSE split by action
        dimension [A] (detached — lesson material: which action dim is hardest).
        """
        e = torch.randn_like(x0)  # noise [B, H, A]
        tau = self.sample_tau(x0.shape[0], x0.device)  # [B]
        x_tau = interpolate(x0, e, tau)  # [B, H, A]
        v = velocity(x_tau, tau)  # predicted velocity [B, H, A]
        err = (v - (e - x0)).pow(2)  # squared error against the line's velocity [B, H, A]
        if mask is None:
            mask = torch.ones(x0.shape[:2], dtype=torch.bool, device=x0.device)
        m = mask.to(err.dtype)[:, :, None]  # [B, H, 1], padding zeroed
        n_real = m.sum().clamp_min(1)
        aux = {"per_dim": ((err * m).sum(dim=(0, 1)) / n_real).detach()}  # [A]
        per = (err * m).mean(-1).sum(1) / mask.sum(1).clamp_min(1)  # [B, H, A] -> [B]
        return per, aux

    @torch.no_grad()
    def sample(
        self,
        velocity: Velocity,
        shape: tuple[int, int, int],
        steps: int,
        device,
        generator: torch.Generator | None = None,
        trace: bool = False,
    ):
        """K Euler steps from pure noise to an action chunk; shape = (B, H, A), steps = K."""
        x = torch.randn(shape, generator=generator, device=device)  # x at tau = 1 [B, H, A]
        xs = [x]
        dt = 1.0 / steps
        for i in range(steps):
            tau = torch.full((shape[0],), 1.0 - i * dt, device=device)  # [B], tau: 1 -> 1/K
            v = velocity(x, tau)  # [B, H, A]
            x = x - dt * v  # Euler step toward the data: [B, H, A]
            if trace:
                xs.append(x)
        return (x, torch.stack(xs)) if trace else x  # [B, H, A]; trace adds [K+1, B, H, A]
