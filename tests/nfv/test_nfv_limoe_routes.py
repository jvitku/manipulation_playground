"""The five force_in routes (lesson 17): build + forward + grad flow, the fvlmoe injection
point, the capacity cap of the LIMoE port, the moe variants, force_noise, and the +1.9 M
parameter budget of fvlmoe at d = 256 (plan §4).

d_model stays 256 in the policy tests (the conv stem's last stage is fixed at 256); the
models are shrunk through trunk_layers / expert_layers instead. The capacity and variant
tests drive the ported MoeLayer directly at tiny dims with a rigged router.
"""

import numpy as np
import pytest
import torch

from nfv.model import ModelConfig, build
from nfv.model.build import param_table
from nfv.model.force_in import (
    FORCE_BINS,
    apply_force_noise,
    quantize_wrench,
)
from nfv.model.limoe_port import MoeLayer, _num_groups
from nfv.model.text import PAD

ROUTES = ("prefix_linear", "prefix_moe", "concat_after", "fvlmoe", "prompt_bins")
EXTRA_PREFIX_TOKENS = {
    "prefix_linear": 1,
    "prefix_moe": 1,
    "concat_after": 0,
    "fvlmoe": 0,
    "prompt_bins": 6,
}


def small_cfg(route: str = "none", **kw) -> ModelConfig:
    return ModelConfig(trunk_layers=2, expert_layers=2, force_in=route, **kw)


def batch_with_wrench(cfg: ModelConfig, b: int = 2) -> dict[str, torch.Tensor]:
    """The dataset batch contract (+ text, + wrench) on random data."""
    g = torch.Generator().manual_seed(1)
    text = torch.full((b, cfg.max_text), PAD, dtype=torch.int64)
    text[:, :4] = torch.randint(2, cfg.vocab, (b, 4), generator=g)
    batch = {
        "rgb_static": torch.rand(b, 3, cfg.img, cfg.img, generator=g),
        "rgb_wrist": torch.rand(b, 3, cfg.img, cfg.img, generator=g),
        "state": torch.randn(b, cfg.d_state, generator=g),
        "wrench": torch.randn(b, 6, generator=g) * 3.0,
        "text": text,
        "dp": 0.01 * torch.randn(b, cfg.chunk, 3, generator=g),
        "gripper": torch.rand(b, cfg.chunk, generator=g),
        "pad_mask": torch.ones(b, cfg.chunk, dtype=torch.bool),
    }
    return batch


def fake_obs(cfg: ModelConfig):
    """A hand-built Obs (no sim needed): random cameras, zero state, a distinct wrench."""
    from nfv.sim.env import Obs

    rng = np.random.default_rng(0)
    return Obs(
        rgb_static=rng.integers(0, 255, (cfg.img, cfg.img, 3), dtype=np.uint8),
        rgb_wrist=rng.integers(0, 255, (cfg.img, cfg.img, 3), dtype=np.uint8),
        state=np.zeros(cfg.d_state, dtype=np.float32),
        wrench=np.array([1.0, -2.0, -6.0, 0.1, 0.0, -0.1], dtype=np.float32),
        wrench_window=np.zeros((20, 6), dtype=np.float32),
        subtask_id=0,
        t=0.0,
    )


# ---------------------------------------------------------------- the five routes


@pytest.mark.parametrize("route", ROUTES)
def test_route_builds_forward_prefix_shape_and_act(route):
    """Each route builds, extends the prefix as documented, losses finitely and acts."""
    torch.manual_seed(0)
    policy = build(small_cfg(route))
    cfg = policy.cfg
    batch = batch_with_wrench(cfg)
    s_base = cfg.cameras * (cfg.img // 16) ** 2 + cfg.max_text  # 72 + 12
    e, key_mask, _ = policy.net.prefix(
        batch["rgb_static"], batch["rgb_wrist"], batch["text"], batch["wrench"]
    )
    assert e.shape == (2, s_base + EXTRA_PREFIX_TOKENS[route], cfg.d_model)
    assert key_mask.shape == e.shape[:2]
    loss, aux = policy.loss(batch)
    assert loss.shape == () and torch.isfinite(loss)
    chunk = policy.act(fake_obs(cfg), "press the button", torch.Generator().manual_seed(0))
    assert chunk["dp"].shape == (cfg.chunk, 3) and chunk["gripper"].shape == (cfg.chunk,)


@pytest.mark.parametrize("route", ROUTES)
def test_route_reads_the_wrench(route):
    """Changing only the wrench changes the loss (the route is live), unlike force_in=none."""
    torch.manual_seed(0)
    policy = build(small_cfg(route))
    batch = batch_with_wrench(policy.cfg)
    torch.manual_seed(7)  # the flow loss draws noise; same draw for both wrenches
    l1, _ = policy.loss(batch)
    batch2 = dict(batch)
    batch2["wrench"] = batch["wrench"] + 10.0  # big enough to move every quantised bin too
    torch.manual_seed(7)
    l2, _ = policy.loss(batch2)
    assert float(l1) != float(l2)

    torch.manual_seed(0)
    blind = build(small_cfg("none"))
    torch.manual_seed(7)
    b1, _ = blind.loss(batch)
    torch.manual_seed(7)
    b2, _ = blind.loss(batch2)
    assert float(b1) == float(b2)  # the baseline never reads the wrench


@pytest.mark.parametrize("route", ROUTES)
def test_route_grad_flow(route):
    """Every trainable parameter receives a gradient (moe='soft' for the LIMoE routes so all
    experts participate; the top-1 variants are exercised by the other tests)."""
    kw = {"moe": "soft"} if route in ("prefix_moe", "fvlmoe") else {}
    torch.manual_seed(0)
    policy = build(small_cfg(route, **kw))
    policy.net.train()
    loss, _ = policy.loss(batch_with_wrench(policy.cfg))
    policy.net.zero_grad(set_to_none=True)
    loss.backward()
    missing = [n for n, p in policy.net.named_parameters() if p.grad is None]
    assert missing == [], f"{route}: parameters without gradient: {missing}"


def test_balance_loss_flag_adds_the_aux():
    """balance_loss=False reports the aux only (like the reference); True adds it."""
    batch = None
    losses = {}
    for flag in (False, True):
        torch.manual_seed(0)
        policy = build(small_cfg("fvlmoe", balance_loss=flag))
        batch = batch_with_wrench(policy.cfg)
        torch.manual_seed(7)
        loss, aux = policy.loss(batch)
        assert "loss_balance" in aux and "moe_fraction_dropped" in aux
        losses[flag] = (float(loss), aux["loss_balance"])
    l_off, bal = losses[False]
    l_on, _ = losses[True]
    assert l_on == pytest.approx(l_off + 0.01 * bal, rel=1e-5)


# ---------------------------------------------------------------- fvlmoe injection point


def test_fvlmoe_zeroed_limoe_reproduces_the_none_model_exactly():
    """Zeroing the LIMoE output projection makes fvlmoe bit-identical to force_in='none':
    the injection is a pure ADD into the chunk features before the head (the reference's
    v_t = action_out_proj(limoe_out[:, -H:] + suffix_out[:, -H:])) and the route touches
    nothing else — not the prefix, not the expert, not the head."""
    torch.manual_seed(0)
    p_fvl = build(small_cfg("fvlmoe"))
    torch.manual_seed(0)
    p_none = build(small_cfg("none"))
    shared = {k: v for k, v in p_fvl.net.state_dict().items() if not k.startswith("force.")}
    p_none.net.load_state_dict(shared, strict=True)  # same trunk/expert/head weights
    with torch.no_grad():
        p_fvl.net.force.limoe.out_proj.weight.zero_()
        p_fvl.net.force.limoe.out_proj.bias.zero_()

    cfg = p_fvl.cfg
    batch = batch_with_wrench(cfg)
    g = torch.Generator().manual_seed(3)
    x_tau = torch.randn(2, cfg.chunk, cfg.act_dim, generator=g)
    tau = torch.rand(2, generator=g)
    with torch.no_grad():
        e, mask, _ = p_fvl.net.prefix(
            batch["rgb_static"], batch["rgb_wrist"], batch["text"], batch["wrench"]
        )
        e0, mask0, _ = p_none.net.prefix(batch["rgb_static"], batch["rgb_wrist"], batch["text"])
        assert torch.equal(e, e0)  # fvlmoe leaves the prefix untouched
        add = p_fvl.net.force.expert_add(e, batch["wrench"], cfg.chunk)
        assert add is not None and float(add.abs().max()) == 0.0
        v_fvl = p_fvl.net.velocity(x_tau, tau, batch["state"], e, mask, feats_add=add)
        v_none = p_none.net.velocity(x_tau, tau, batch["state"], e0, mask0)
    assert torch.equal(v_fvl, v_none)


# ---------------------------------------------------------------- the capacity cap


def _rigged_moe(moe: str, seed: int = 0) -> MoeLayer:
    """d=8, 2 experts; the router reads only feature 0: x0 > 0 -> expert 0, x0 < 0 -> expert 1,
    with gate confidence growing with |x0| (so batch-prioritized ordering is controlled)."""
    torch.manual_seed(seed)
    layer = MoeLayer(d_model=8, num_experts=2, moe=moe, expert_dim=16).eval()
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.bias.zero_()
        layer.router.weight[0, 0] = 4.0  # logit_e0 = 4 * x0; logit_e1 = 0
    return layer


def _crafted_tokens() -> tuple[torch.Tensor, torch.Tensor]:
    """32 tokens -> two groups of 16 (the reference grouping), expert capacity 8.

    Group 0 (tokens 0..15): 9 tokens prefer expert 0 (x0 = 0.9 .. 0.1 descending), 7 prefer
    expert 1 -> expert 0 overflows by exactly ONE token, the least-confident (x0 = 0.1).
    Group 1 (tokens 16..31): all 16 prefer expert 0 -> exactly 8 dropped (the 8 smallest x0).
    Returns (x [1, 32, 8], expected_kept [1, 32] bool).
    """
    assert _num_groups(32, 16, 2) == 2  # groups of 16, capacity max(round(16/2), 4) = 8
    x = torch.full((1, 32, 8), 0.25)  # nonzero everywhere: no token is "padding"
    x0 = torch.empty(32)
    x0[0:9] = torch.linspace(0.9, 0.1, 9)  # group 0: 9 -> expert 0
    x0[9:16] = -1.0  # group 0: 7 -> expert 1
    x0[16:32] = torch.linspace(1.6, 0.1, 16)  # group 1: 16 -> expert 0
    x[0, :, 0] = x0
    kept = torch.ones(1, 32, dtype=torch.bool)
    kept[0, 8] = False  # group 0's 9th-most-confident expert-0 token
    kept[0, 24:32] = False  # group 1's 8 least-confident tokens
    return x, kept


def test_capacity_drops_exactly_the_overflow():
    """A 9th token routed to a full expert is dropped — the least confident one — and its
    MoE output is exactly zero; the per-expert counts and drop fraction report it."""
    layer = _rigged_moe("top1_capacity")
    x, expected_kept = _crafted_tokens()
    with torch.no_grad():
        out, aux = layer(x)
    assert torch.equal(aux["kept"], expected_kept)
    assert aux["expert_counts"].tolist() == [16, 7]  # e0: 8 + 8 kept; e1: 7
    assert float(aux["fraction_dropped"]) == pytest.approx(9 / 32)
    dropped_rows = out[~expected_kept[:, :, None].expand_as(out)]
    assert float(dropped_rows.abs().max()) == 0.0  # overflow tokens get ZERO expert output
    kept_norms = out[expected_kept].abs().sum(-1)
    assert float(kept_norms.min()) > 0.0  # routed tokens actually got an expert


def test_moe_variants_differ_and_agree_where_they_should():
    """top1 == top1_capacity on kept tokens, differs exactly on the dropped ones; soft
    differs from both (it mixes every expert)."""
    x, kept = _crafted_tokens()
    outs = {}
    for variant in ("top1_capacity", "top1", "soft"):
        layer = _rigged_moe(variant, seed=0)  # same seed -> identical expert weights
        with torch.no_grad():
            outs[variant], _ = layer(x)
    cap, top1, soft = outs["top1_capacity"], outs["top1"], outs["soft"]
    assert torch.allclose(cap[kept], top1[kept])  # the cap only acts on overflow
    assert float(cap[~kept].abs().max()) == 0.0  # capacity: dropped tokens zeroed
    assert float(top1[~kept].abs().max()) > 0.0  # no cap: the same tokens pass through
    assert not torch.allclose(top1, soft)
    assert not torch.allclose(cap, soft)


# ---------------------------------------------------------------- force_noise + prompt bins


def test_force_noise_modes():
    wrench = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3], [-1.0, -2.0, -3.0, 0.0, 0.0, 0.0]])
    assert torch.equal(apply_force_noise(wrench, "off"), wrench)
    zero = apply_force_noise(wrench, "zero")
    assert torch.equal(zero, torch.zeros_like(wrench))  # == the force-blind twin's input
    g = torch.Generator().manual_seed(1)  # a seed whose 2-permutation is the swap
    shuffled = apply_force_noise(wrench, "shuffle", generator=g)
    assert torch.equal(shuffled.sum(0), wrench.sum(0))  # same rows, reordered
    assert torch.equal(shuffled[0], wrench[1]) and torch.equal(shuffled[1], wrench[0])
    g = torch.Generator().manual_seed(0)
    noisy = apply_force_noise(wrench, "gaussian", generator=g)
    assert noisy.shape == wrench.shape and torch.isfinite(noisy).all()
    assert not torch.equal(noisy, wrench)
    with pytest.raises(ValueError, match="force_noise"):
        apply_force_noise(wrench, "bogus")


def test_quantize_wrench_bins():
    """Fixed ranges: ±25 N forces / ±2.5 Nm torques, 32 bins, clamped, axis-offset ids."""
    w = torch.tensor([[-25.0, 0.0, 25.0, -2.5, 0.0, 2.5], [-100.0, 100.0, 0.0, -10.0, 10.0, 0.0]])
    ids = quantize_wrench(w)
    assert ids.shape == (2, 6) and ids.dtype == torch.int64
    # per axis: id = axis * 32 + bin; -range -> bin 0, 0 -> bin 16, +range -> bin 31 (clamped)
    assert ids[0].tolist() == [0, 32 + 16, 64 + 31, 96 + 0, 128 + 16, 160 + 31]
    assert ids[1].tolist() == [0, 32 + 31, 64 + 16, 96 + 0, 128 + 31, 160 + 16]  # clamped
    assert int(ids.max()) < 6 * FORCE_BINS


# ---------------------------------------------------------------- parameter budget


def test_fvlmoe_param_delta_near_plan():
    """fvlmoe adds ≈ +1.9 M at d = 256 (plan §4: nanoVLA 7.0 M -> +ForceVLA 8.9 M); printed
    per block, asserted with a loose bound."""
    torch.manual_seed(0)
    base = build(ModelConfig())
    torch.manual_seed(0)
    fvl = build(ModelConfig(force_in="fvlmoe"))
    t_base, t_fvl = param_table(base), param_table(fvl)
    delta = t_fvl["total"] - t_base["total"]
    for block in t_fvl:
        print(f"{block:>8}: {t_fvl[block] / 1e6:7.3f} M (delta {t_fvl[block] - t_base[block]:+d})")
    print(f"fvlmoe param delta: {delta / 1e6:.3f} M (plan: +1.9 M)")
    assert delta == t_fvl["force"]  # the route adds parameters nowhere else
    assert 1.4e6 <= delta <= 2.4e6
