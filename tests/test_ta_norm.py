"""TA-SmolVLA torque normalisation (fvb.vla.ta_smolvla.TorqueNorm; CPU, no weights needed)."""

import pytest

torch = pytest.importorskip("torch")

from fvb.vla.ta_smolvla import TorqueNorm  # noqa: E402


def test_zscores_and_guards_constant_dims():
    n = TorqueNorm(3, mean=[1.0, 47.0, 0.0], std=[2.0, 19.0, 0.0])
    out = n(torch.tensor([[3.0, 47.0, 5.0]]))
    assert torch.allclose(out, torch.tensor([[1.0, 0.0, 5.0]]))


def test_stats_travel_with_the_state_dict():
    a = TorqueNorm(4, mean=[1, 2, 3, 4], std=[1, 1, 2, 2])
    b = TorqueNorm(4)  # what serving builds before loading the checkpoint
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 4)
    assert torch.allclose(a(x), b(x))
    assert set(a.state_dict()) == {"mean", "std"}
