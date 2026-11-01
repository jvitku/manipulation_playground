"""nanoVLA model tests: shapes, parameter budget, vocabulary coverage, gradients,
the act() contract on a real Env observation, and training determinism.

The random batch below follows the dataset contract (nfv/data/dataset.py legend) plus the
"text" ids nfv.train adds; B = batch, H = chunk (16), T = max_text (12).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nfv.model import ModelConfig, build
from nfv.model.build import actions_from_model, actions_to_model, cfg_velocity, param_table
from nfv.model.text import BLANK_INSTRUCTION, PAD, UNK, VOCAB, tokenize
from nfv.seeds import eval_seeds
from nfv.sim import tasks
from nfv.train import fit, sample_batch_with_text


@pytest.fixture(scope="module")
def policy():
    torch.manual_seed(0)
    return build(ModelConfig())


def random_batch(cfg: ModelConfig, b: int = 2) -> dict[str, torch.Tensor]:
    """A dataset-contract batch of random data (images already float [0, 1], CHW)."""
    g = torch.Generator().manual_seed(1)
    text = torch.full((b, cfg.max_text), PAD, dtype=torch.int64)
    text[:, :4] = torch.randint(2, cfg.vocab, (b, 4), generator=g)  # 4 real words, rest <pad>
    batch = {
        "rgb_static": torch.rand(b, 3, cfg.img, cfg.img, generator=g),
        "rgb_wrist": torch.rand(b, 3, cfg.img, cfg.img, generator=g),
        "state": torch.randn(b, cfg.d_state, generator=g),
        "text": text,
        "dp": 0.01 * torch.randn(b, cfg.chunk, 3, generator=g),
        "gripper": torch.rand(b, cfg.chunk, generator=g),
        "pad_mask": torch.ones(b, cfg.chunk, dtype=torch.bool),
    }
    batch["pad_mask"][-1, cfg.chunk // 2 :] = False  # one row ends mid-chunk
    return batch


def test_param_budget(policy):
    """Total ≈ 7.0 M ± 20 % (plan §4 table); the per-block split is printed for the report."""
    table = param_table(policy)
    for block, n in table.items():
        print(f"{block:>8}: {n / 1e6:7.3f} M")
    assert 0.8 * 7.0e6 <= table["total"] <= 1.2 * 7.0e6
    assert table["total"] == sum(p.numel() for p in policy.net.parameters())


def test_loss_shapes_and_gradients(policy):
    """loss() is a finite scalar and every trainable parameter receives a gradient."""
    batch = random_batch(policy.cfg)
    loss, aux = policy.loss(batch)
    assert loss.shape == () and torch.isfinite(loss)
    assert set(aux) == {"loss_dp", "loss_gripper"}
    policy.net.zero_grad(set_to_none=True)
    loss.backward()
    missing = [n for n, p in policy.net.named_parameters() if p.grad is None]
    assert missing == [], f"parameters without gradient: {missing}"
    for p in policy.ema.parameters():  # the EMA shadow is frozen
        assert not p.requires_grad


def test_prefix_attention_maps(policy):
    """The trunk exposes one attention map per layer; <pad> text keys get zero weight."""
    cfg = policy.cfg
    batch = random_batch(cfg)
    e, key_mask, attns = policy.net.prefix(batch["rgb_static"], batch["rgb_wrist"], batch["text"])
    n_tokens = cfg.cameras * (cfg.img // 16) ** 2 + cfg.max_text  # 72 + 12 = 84
    assert e.shape == (2, n_tokens, cfg.d_model)
    assert len(attns) == cfg.trunk_layers
    for a in attns:
        assert a.shape == (2, cfg.heads, n_tokens, n_tokens)
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)))  # rows are distributions
        assert float(a[:, :, :, ~key_mask[0]].abs().max()) == 0.0  # pad keys never attended
    assert bool(key_mask[:, : n_tokens - cfg.max_text].all())  # vision tokens are all real


def test_action_scaling_roundtrip():
    """Env units -> model units -> env units is the identity inside the env's clamps."""
    dp = torch.tensor([[[0.05, -0.05, 0.012]]])
    gripper = torch.tensor([[0.25]])
    x = actions_to_model(dp, gripper)
    assert x.shape == (1, 1, 4)
    assert float(x.abs().max()) <= 1.0 + 1e-6  # f32 representation of 0.05 / 0.05
    dp2, grip2 = actions_from_model(x)
    assert torch.allclose(dp2, dp) and torch.allclose(grip2, gripper)
    dp3, grip3 = actions_from_model(torch.tensor([[[5.0, -5.0, 0.0, 5.0]]]))  # out of range
    assert float(dp3.abs().max()) <= np.float32(0.05) and 0.0 <= float(grip3) <= 1.0


def test_vocab_covers_tasks_zero_unks():
    """Every instruction tokenizes with zero <unk>; every subtask word is reserved in VOCAB."""
    for task_cls in tasks.TASKS.values():
        for instruction in task_cls.INSTRUCTIONS:
            ids = tokenize(instruction)
            assert UNK not in ids, f"<unk> in {instruction!r}"
            assert (ids != PAD).sum() == len(instruction.split())
        for subtask in task_cls.SUBTASKS:  # lesson-18 force-prompt words
            assert subtask in VOCAB, f"subtask word {subtask!r} missing from VOCAB"


def test_build_refuses_inconsistent_flags():
    """Every lesson-17/18 switch is implemented; combinations that would ignore one refuse.

    (The five lesson-17 force_in routes: tests/nfv/test_nfv_limoe_routes.py; ForceVLA2's
    modules and wiring: tests/nfv/test_nfv_forcevla2.py.)
    """
    with pytest.raises(ValueError, match="11-D"):
        build(ModelConfig(head="hybrid"))  # act_dim left at the position head's 4
    with pytest.raises(ValueError, match="expert_inputs"):
        build(ModelConfig(expert_inputs="none"))  # no ForceVLA2 fusion to ablate
    with pytest.raises(ValueError, match="force_in"):
        build(ModelConfig(force_in="bogus"))


def test_act_contract_on_real_env(policy, render_env):
    """act() on a live T0 observation returns the ActionChunk contract with env-legal values."""
    env = render_env
    obs = env.reset(eval_seeds("t0", 1)[0], task="t0")
    g = torch.Generator().manual_seed(0)
    chunk = policy.act(obs, env.instruction, generator=g)
    assert set(chunk) == {"dp", "gripper", "f_target", "s", "aux"}
    h = policy.cfg.chunk
    assert chunk["dp"].shape == (h, 3) and chunk["dp"].dtype == np.float32
    # the env's per-axis clamp, compared at float32 precision (f32(0.05) > the decimal 0.05)
    assert float(np.abs(chunk["dp"]).max()) <= np.float32(0.05)
    assert chunk["gripper"].shape == (h,) and chunk["gripper"].dtype == np.float32
    assert 0.0 <= float(chunk["gripper"].min()) and float(chunk["gripper"].max()) <= 1.0
    assert chunk["f_target"] is None and chunk["s"] is None  # lesson-18 outputs
    assert {"trunk_attn", "expert_self_attn", "expert_cross_attn"} <= set(chunk["aux"])
    g2 = torch.Generator().manual_seed(0)
    chunk2 = policy.act(obs, env.instruction, generator=g2)
    assert np.array_equal(chunk["dp"], chunk2["dp"])  # seeded sampling is repeatable


class FakeDataset:
    """The DemoDataset duck-type nfv.train reads: episodes/lengths/starts/index."""

    def __init__(self, n_episodes: int = 2, steps: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.episodes = []
        for _ in range(n_episodes):
            self.episodes.append(
                {
                    "rgb_static": rng.integers(0, 255, (steps, 96, 96, 3), dtype=np.uint8),
                    "rgb_wrist": rng.integers(0, 255, (steps, 96, 96, 3), dtype=np.uint8),
                    "state": rng.normal(size=(steps, 8)).astype(np.float32),
                    "wrench": rng.normal(size=(steps, 6)).astype(np.float32),
                    "wrench_window": rng.normal(size=(steps, 20, 6)).astype(np.float32),
                    "dp": (0.01 * rng.normal(size=(steps, 3))).astype(np.float32),
                    "gripper": rng.uniform(size=steps).astype(np.float32),
                    "f_target": np.zeros((steps, 6), dtype=np.float32),
                    "s_fv2": rng.uniform(size=steps).astype(np.float32),
                    "s_lin": rng.uniform(size=steps).astype(np.float32),
                    "subtask": np.zeros(steps, dtype=np.int16),
                }
            )
        self.lengths = np.array([steps] * n_episodes)
        self.starts = np.array(
            [(i, t) for i in range(n_episodes) for t in range(steps)], dtype=np.int64
        )
        self.index = {"episodes": [{"instruction": "pick up the cube"}] * n_episodes}


def test_batcher_adds_text_and_keeps_contract():
    """sample_batch_with_text serves the dataset batch contract plus per-episode text ids."""
    ds = FakeDataset()
    from nfv.train import episode_text_ids

    cfg = ModelConfig()
    text_ids = episode_text_ids(ds, cfg.max_text)
    batch = sample_batch_with_text(ds, text_ids, 5, np.random.default_rng(0), cfg.chunk)
    assert batch["rgb_static"].shape == (5, 3, 96, 96)
    assert float(batch["rgb_static"].max()) <= 1.0  # uint8 -> [0, 1] at batch time
    assert batch["text"].shape == (5, cfg.max_text) and batch["text"].dtype == torch.int64
    assert torch.equal(batch["text"][0], torch.from_numpy(tokenize("pick up the cube")))
    assert batch["dp"].shape == (5, cfg.chunk, 3)
    # the fake episodes are 8 steps, the chunk is 16: every row has a zero-padded tail
    assert batch["pad_mask"][:, 0].all() and not batch["pad_mask"][:, -1].any()
    assert float(batch["dp"][~batch["pad_mask"]].abs().max()) == 0.0


def test_fit_determinism_same_seed_same_losses():
    """The M1 reproducibility contract: same seed -> the same loss at step 3 (and 1, 2)."""
    ds = FakeDataset()
    cfg = ModelConfig()
    _, h1 = fit(cfg, ds, None, steps=3, lr=1e-4, batch=2, log_every=1, ckpt_every=0, seed=7)
    _, h2 = fit(cfg, ds, None, steps=3, lr=1e-4, batch=2, log_every=1, ckpt_every=0, seed=7)
    l1 = [row["loss"] for row in h1 if "loss" in row]
    l2 = [row["loss"] for row in h2 if "loss" in row]
    assert len(l1) == 3
    assert l1 == pytest.approx(l2, abs=1e-9)


# ---- lesson 17 round 2: instruction dropout + classifier-free guidance ----------------------


def _small_cfg(**kw) -> ModelConfig:
    """A CPU-cheap config for the CFG act() tests (1+1 layers, chunk 4, 2 Euler steps)."""
    return ModelConfig(trunk_layers=1, expert_layers=1, chunk=4, n_exec=2, euler_steps=2, **kw)


def _fake_obs(seed: int = 7) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        rgb_static=rng.integers(0, 256, (96, 96, 3), dtype=np.uint8),
        rgb_wrist=rng.integers(0, 256, (96, 96, 3), dtype=np.uint8),
        state=rng.standard_normal(8).astype(np.float32),
        wrench=rng.standard_normal(6).astype(np.float32),
        wrench_window=np.zeros((20, 6), np.float32),
        subtask_id=0,
        t=0.0,
    )


def _count_prefix_calls(policy) -> list:
    calls = []
    orig = policy.ema.prefix

    def spy(rgb_s, rgb_w, text, wrench=None):
        calls.append(text.detach().clone())
        return orig(rgb_s, rgb_w, text, wrench)

    policy.ema.prefix = spy  # instance attribute shadows the bound method
    return calls


def test_cfg_velocity_hand_values():
    """The CFG combination v = v_blank + w (v_cond - v_blank), checked by hand."""
    v_c = torch.tensor([2.0, -1.0])
    v_b = torch.tensor([1.0, 1.0])
    assert torch.equal(cfg_velocity(v_c, v_b, 1.0), v_c)  # w = 1: the conditioned model
    assert torch.equal(cfg_velocity(v_c, v_b, 0.0), v_b)  # w = 0: the blank model
    assert torch.equal(cfg_velocity(v_c, v_b, 3.0), torch.tensor([4.0, -5.0]))  # extrapolates
    assert torch.equal(cfg_velocity(v_c, v_b, 0.5), torch.tensor([1.5, 0.0]))  # interpolates


def test_cfg_w_one_skips_the_blank_branch():
    """At the default w = 1.0, act() encodes exactly ONE prefix - the conditioned one."""
    torch.manual_seed(0)
    policy = build(_small_cfg()).eval()
    calls = _count_prefix_calls(policy)
    policy.act(_fake_obs(), "press the red button", generator=torch.Generator().manual_seed(0))
    assert len(calls) == 1
    assert torch.equal(calls[0][0], torch.from_numpy(tokenize("press the red button")))


def test_cfg_w_encodes_blank_before_cond_and_moves_the_action():
    """w != 1 encodes the blank prefix FIRST (aux stays the cond pass's) and changes dp."""
    torch.manual_seed(0)
    p1 = build(_small_cfg()).eval()
    torch.manual_seed(0)  # identical weights: only the CFG combination differs
    p3 = build(_small_cfg(cfg_w=3.0)).eval()
    obs = _fake_obs()
    a1 = p1.act(obs, "press the red button", generator=torch.Generator().manual_seed(5))
    calls = _count_prefix_calls(p3)
    a3 = p3.act(obs, "press the red button", generator=torch.Generator().manual_seed(5))
    assert len(calls) == 2
    blank = torch.from_numpy(tokenize(BLANK_INSTRUCTION))
    assert torch.equal(calls[0][0], blank)  # blank first ...
    assert torch.equal(calls[1][0], torch.from_numpy(tokenize("press the red button")))  # cond
    assert not np.array_equal(a1["dp"], a3["dp"])  # the guidance moved the sampled chunk


def test_cfg_on_the_blank_instruction_is_the_identity():
    """Conditioning ON the blank text makes v_cond == v_blank, so any w is a bit-exact no-op
    (lesson 16's arm C under CFG - the documented no-op by construction)."""
    torch.manual_seed(0)
    p1 = build(_small_cfg()).eval()
    torch.manual_seed(0)
    p3 = build(_small_cfg(cfg_w=3.0)).eval()
    obs = _fake_obs()
    a1 = p1.act(obs, BLANK_INSTRUCTION, generator=torch.Generator().manual_seed(5))
    a3 = p3.act(obs, BLANK_INSTRUCTION, generator=torch.Generator().manual_seed(5))
    assert np.array_equal(a1["dp"], a3["dp"]) and np.array_equal(a1["gripper"], a3["gripper"])


def test_text_dropout_zero_draws_no_rng(policy):
    """p = 0 never enters the dropout branch: loss() works with torch.rand forbidden, so the
    default training consumes exactly the pre-round-2 RNG stream."""
    batch = random_batch(policy.cfg)  # built before torch.rand is forbidden

    def boom(*a, **k):
        raise AssertionError("text_dropout=0 must not draw from torch.rand")

    orig = torch.rand
    torch.rand = boom
    try:
        loss, _ = policy.loss(batch)
    finally:
        torch.rand = orig
    assert torch.isfinite(loss)


def test_text_dropout_swaps_the_drawn_rows_for_the_blank_text():
    """With the draw pinned, exactly the rows below p train on the blank ids."""
    torch.manual_seed(0)
    policy = build(_small_cfg(text_dropout=0.5))
    batch = random_batch(policy.cfg, b=4)
    seen = {}
    orig_prefix = policy.net.prefix

    def spy(rgb_s, rgb_w, text, wrench=None):
        seen["text"] = text.detach().clone()
        return orig_prefix(rgb_s, rgb_w, text, wrench)

    policy.net.prefix = spy
    orig_rand = torch.rand
    torch.rand = lambda *a, **k: torch.tensor([0.0, 0.9, 0.1, 0.9])  # rows 0 and 2 drop
    try:
        policy.loss(batch)
    finally:
        torch.rand = orig_rand
    blank = torch.from_numpy(tokenize(BLANK_INSTRUCTION, policy.cfg.max_text))
    assert torch.equal(seen["text"][0], blank) and torch.equal(seen["text"][2], blank)
    assert torch.equal(seen["text"][1], batch["text"][1])
    assert torch.equal(seen["text"][3], batch["text"][3])


def test_blank_text_buffer_is_constant_and_absent_from_checkpoints(policy):
    """The blank ids live on the module (device-safe) but never in a state_dict - round-1
    checkpoints load into the round-2 model strictly, and vice versa."""
    assert torch.equal(policy.net.blank_text[0], torch.from_numpy(tokenize(BLANK_INSTRUCTION)))
    assert "blank_text" not in policy.net.state_dict()
    fresh = build(ModelConfig())
    fresh.net.load_state_dict(policy.net.state_dict())  # strict=True by default


def test_build_validates_round2_flags():
    with pytest.raises(ValueError, match="text_dropout"):
        build(ModelConfig(text_dropout=1.0))
    with pytest.raises(ValueError, match="text_dropout"):
        build(ModelConfig(text_dropout=-0.1))
    with pytest.raises(ValueError, match="cfg_w"):
        build(ModelConfig(cfg_w=-1.0))
    with pytest.raises(ValueError, match="cfg_w"):
        build(ModelConfig(cfg_w=float("inf")))
