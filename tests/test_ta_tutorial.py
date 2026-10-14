"""TA-VLA tutorial library (src/fvb/ta_tutorial): physics, experts, data, networks, samplers."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(4)

from fvb.ta_tutorial import contact2d as c2d  # noqa: E402
from fvb.ta_tutorial import generative, toy  # noqa: E402
from fvb.ta_tutorial.data import ChunkDataset  # noqa: E402
from fvb.ta_tutorial.nets import MLPDenoiser, TAVLAMini  # noqa: E402
from fvb.ta_tutorial.train import Cfg, train  # noqa: E402


def test_surface_blocks_and_force_is_spring_into_surface():
    env = c2d.Contact2D([0.0], [[0.5]], [400.0], [[0.0, 0.05]], [0], sensor_noise=0.0)
    for _ in range(8):  # reference goes 3 cm below the flat surface
        env.step(np.array([[0.0, -1.0]]))
    assert env.pos[0, 1] == pytest.approx(0.0, abs=1e-9)  # the peg rests on the surface
    assert env.force[0, 1] == pytest.approx(400 * 0.03, rel=1e-6)  # k * penetration, upward
    assert env.force[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_chamfer_pushes_toward_the_hole_and_the_peg_drops_in():
    h = 0.0
    x0 = h - c2d.WH - c2d.WC / 2  # on the left chamfer
    z_chamfer = -(c2d.WC / 2)  # surface height half-way down the chamfer
    env = c2d.Contact2D([0.0], [[h]], [300.0], [[x0, z_chamfer]], [0], sensor_noise=0.0)
    env.step(np.array([[0.0, -0.3]]))  # press 3 mm into the slope
    assert env.force[0, 0] > 0  # reaction points right, toward the hole centre
    for _ in range(40):
        env.step(np.array([[0.3, -0.6]]))
        if env.done[0]:
            break
    assert env.outcome[0] == "success"


def test_pushing_too_hard_aborts():
    env = c2d.Contact2D([0.0], [[0.5]], [1000.0], [[0.0, 0.0]], [0], sensor_noise=0.0)
    for _ in range(5):
        env.step(np.array([[0.0, -1.0]]))
    assert env.outcome[0] == "force_abort"


@pytest.mark.parametrize("kind", ["t3", "t4"])
def test_experts_solve_their_tasks(kind):
    eps = c2d.collect(kind, 100, seed=5)
    assert np.mean([e["success"] for e in eps]) > 0.95


def test_t2_expert_and_both_sides():
    eps = toy.t2_collect(100, 3)
    assert all(e["success"] for e in eps)
    first_dx = np.array([e["action"][:10, 0].mean() for e in eps])
    assert (first_dx < 0).mean() > 0.3 and (first_dx > 0).mean() > 0.3  # both modes present


def test_dataset_pads_and_masks_chunks():
    ep = {
        "obs": np.zeros((5, 3), np.float32),
        "action": np.arange(10, dtype=np.float32).reshape(5, 2),
    }
    ds = ChunkDataset([ep], horizon=4)
    _, x0, mask, _ = ds.batch(torch.tensor([3]))
    assert mask[0].tolist() == [1, 1, 0, 0]
    assert x0[0, 2].tolist() == [8, 9]  # padded with the last action


def _ta(**kw):
    return TAVLAMini(
        act_dim=2, horizon=4, state_dim=2, torque_dim=20, d=32, layers=2, heads=2, **kw
    )


def _cond(b=3):
    return {"state": torch.randn(b, 2), "torque": torch.randn(b, 20)}


def test_torque_token_reaches_the_actions_only_when_not_zeroed():
    torch.manual_seed(0)
    x, t = torch.randn(3, 4, 2), torch.rand(3)
    for pos in ("suffix", "prefix"):
        m = _ta(torque_pos=pos)
        c1, c2 = _cond(), _cond()
        c2["state"] = c1["state"]
        assert (m(x, t, c1)[0] - m(x, t, c2)[0]).abs().max() > 1e-4
    mz = _ta(zero_torque=True)
    c1, c2 = _cond(), _cond()
    c2["state"] = c1["state"]
    assert torch.allclose(mz(x, t, c1)[0], mz(x, t, c2)[0])


def test_prefix_never_sees_the_noisy_actions_and_the_cache_is_exact():
    torch.manual_seed(0)
    m = _ta(aux_dim=16).eval()
    c = _cond()
    kv1 = m.encode_prefix(c)
    m(torch.randn(3, 4, 2), torch.rand(3), c)
    kv2 = m.encode_prefix(c)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(kv1, kv2, strict=True))
    x, t = torch.randn(3, 4, 2), torch.rand(3)
    out, extra = m(x, t, c)
    with m.cache(c):
        out_c, _ = m(x, t, c)
    assert torch.allclose(out, out_c, atol=1e-6) and extra["aux"].shape == (3, 16)


@pytest.mark.parametrize("name", ["flow", "ddpm"])
def test_generative_head_learns_two_modes(name):
    """A short training on T1 must put samples on both sides and few in the gap."""
    c, a = toy.t1_data(4000, np.random.default_rng(0))
    ds = ChunkDataset([{"obs": c[i : i + 1], "action": a[i : i + 1]} for i in range(len(c))], 1)
    gen = generative.make(name)
    torch.manual_seed(0)
    _, ema = train(MLPDenoiser(2, 1, 1, 256, 3), gen, ds, Cfg(steps=3000, lr=1e-3, ema=0.99,
                   log_every=1000, eval_every=10**9), log=lambda *x: None)  # fmt: skip
    ce = torch.zeros(500, 1)
    x = gen.sample(ema, (500, 1, 2), ds.cond({"obs": ce}), 20, torch.Generator().manual_seed(0))
    m = toy.t1_metrics(ce[:, 0].numpy(), x[:, 0].numpy())
    assert 0.25 < m["left_frac"] < 0.75 and m["collision"] < 0.3, m


def test_regression_averages_the_modes():
    c, a = toy.t1_data(4000, np.random.default_rng(0))
    ds = ChunkDataset([{"obs": c[i : i + 1], "action": a[i : i + 1]} for i in range(len(c))], 1)
    gen = generative.make("mse")
    _, ema = train(MLPDenoiser(2, 1, 1, 128, 2), gen, ds, Cfg(steps=1500, lr=1e-3, ema=0.99,
                   log_every=1000, eval_every=10**9), log=lambda *x: None)  # fmt: skip
    x = gen.sample(ema, (200, 1, 2), ds.cond({"obs": torch.zeros(200, 1)}))
    assert toy.t1_metrics(np.zeros(200), x[:, 0].numpy())["collision"] > 0.9


def test_render_shapes_and_marker_colours():
    scene = torch.tensor([[0.0, -0.08, 0.08, 0.0, 0.12]])
    img = c2d.render(scene, noise=0.0)
    assert img.shape == (1, 3, 32, 32)
    assert (img[0, 0] > 0.9).logical_and(img[0, 2] < 0.3).any()  # a red marker
    assert (img[0, 2] > 0.9).logical_and(img[0, 0] < 0.3).any()  # a blue marker
