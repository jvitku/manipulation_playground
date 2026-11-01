"""Flow-matching unit tests, including the M1 gate: one Euler step recovers a unimodal mean.

The gate (plan §8 M1): on a 1-D unimodal toy the trained velocity field is nearly straight,
so a SINGLE Euler step from tau = 1 lands on the data mean — from the mean of the noise
(x = 0) and on average from random noise. That is the measurable reason pi0-style VLAs get
away with ~10 steps.
"""

import torch
from torch import nn

from nfv.model.flow import FlowMatching, interpolate

TOY_MEAN = 0.7  # the unimodal toy data: x0 ~ N(TOY_MEAN, TOY_STD^2)
TOY_STD = 0.05


def test_interpolation_convention():
    """x_tau = tau * e + (1 - tau) * x0: tau = 1 is pure noise, tau = 0 is the data."""
    x0 = torch.randn(4, 3, 2)
    e = torch.randn(4, 3, 2)
    assert torch.equal(interpolate(x0, e, torch.ones(4)), e)
    assert torch.equal(interpolate(x0, e, torch.zeros(4)), x0)
    mid = interpolate(x0, e, torch.full((4,), 0.5))
    assert torch.allclose(mid, 0.5 * e + 0.5 * x0)


def test_loss_mask_ignores_padding():
    """Garbage in the padded chunk tail must not change the loss (same RNG draws both times)."""
    flow = FlowMatching()
    mask = torch.ones(8, 6, dtype=torch.bool)
    mask[:, 4:] = False  # steps 4-5 are padding
    base = torch.randn(8, 6, 3)
    garbage = base.clone()
    garbage[:, 4:] = 123.0  # only where the mask is False

    def velocity(x_tau, tau):  # deterministic in its inputs, so only the pad rows differ
        return 0.1 * x_tau

    torch.manual_seed(0)
    per_a, aux_a = flow.loss(velocity, base, mask)
    torch.manual_seed(0)  # same e and tau draws for the garbage variant
    per_b, aux_b = flow.loss(velocity, garbage, mask)
    assert torch.allclose(per_a, per_b)
    assert torch.allclose(aux_a["per_dim"], aux_b["per_dim"])
    assert per_a.shape == (8,)
    assert aux_a["per_dim"].shape == (3,)


def test_sampler_shapes_and_determinism():
    """Zero velocity returns the start noise; a seeded generator makes sampling repeatable."""
    flow = FlowMatching()

    def zero_velocity(x_tau, tau):
        return torch.zeros_like(x_tau)

    g = torch.Generator().manual_seed(3)
    x, trace = flow.sample(
        zero_velocity, (2, 5, 4), steps=10, device="cpu", generator=g, trace=True
    )
    assert x.shape == (2, 5, 4)
    assert trace.shape == (11, 2, 5, 4)
    assert torch.equal(x, trace[0])  # zero velocity never moves the sample
    g2 = torch.Generator().manual_seed(3)
    x2 = flow.sample(zero_velocity, (2, 5, 4), steps=10, device="cpu", generator=g2)
    assert torch.equal(x, x2)


class ToyVelocity(nn.Module):
    """A tiny MLP velocity field over (x, tau) for the 1-D toy; chunk H = 1, act A = 1."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, x_tau: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_tau[:, 0, :], tau[:, None]], dim=-1)  # [B, 2]
        return self.net(inp)[:, :, None]  # [B, 1] -> [B, 1, 1]


def test_toy_unimodal_one_euler_step_recovers_mean():
    """THE M1 flow gate: train on N(0.7, 0.05); one Euler step recovers the mean.

    At tau = 1 the optimal velocity is v(x, 1) = x - E[x0] (x_1 is pure noise, independent of
    x0), so one Euler step x <- x - 1 * v lands every start on the data mean.
    """
    torch.manual_seed(0)
    flow = FlowMatching()
    model = ToyVelocity()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(600):
        x0 = TOY_MEAN + TOY_STD * torch.randn(256, 1, 1)
        per, _ = flow.loss(model, x0)
        opt.zero_grad()
        per.mean().backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        # the literal gate: one Euler step from the MEAN of the noise (x = 0) at tau = 1
        v = model(torch.zeros(1, 1, 1), torch.ones(1))
        x_one = 0.0 - 1.0 * v.item()
        # and the distributional version: one step from random noise, batch mean
        g = torch.Generator().manual_seed(1)
        x = flow.sample(model, (2048, 1, 1), steps=1, device="cpu", generator=g)
    print(
        f"toy flow: one-step from 0 -> {x_one:.4f}, mean over noise -> {x.mean():.4f} "
        f"(target {TOY_MEAN})"
    )
    assert abs(x_one - TOY_MEAN) < 0.1
    assert abs(float(x.mean()) - TOY_MEAN) < 0.1
