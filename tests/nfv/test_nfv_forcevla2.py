"""ForceVLA2 modules (lesson 18): fusion + 3-expert MoE, hybrid head, force-prompt machine.

CPU-only and fast: the fusion runs on a fake prefix (the trunk is not needed to test it);
the one physics test records a single t0 demo for the dataset's derived ``s_sub`` label.
"""

import numpy as np
import pytest
import torch

from nfv.model.expert import ActionExpert
from nfv.model.flow import FlowMatching
from nfv.model.forcevla2 import (
    F_SCALE,
    HYBRID_DIM,
    TORQUE_SCALE,
    ForceVLA2Fusion,
    HybridHead,
    hybrid_from_model,
    hybrid_to_model,
)
from nfv.sim.controllers import DP_MAX
from nfv.sim.subtasks import (
    ADVANCE_AT,
    REARM_BELOW,
    SubtaskMachine,
    hindsight_progress,
    task_subtasks,
)

D, NH, D_STATE, NV, T = 256, 4, 8, 72, 12
S = NV + T  # prefix tokens


def _inputs(b=2, seed=0, n_words=4):
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(b, S, D, generator=g)  # fake prefix E [B, S, D]
    mask = torch.ones(b, S, dtype=torch.bool)
    mask[:, NV + n_words :] = False  # text <pad> after n_words words
    state = torch.randn(b, D_STATE, generator=g)
    wrench = torch.randn(b, 6, generator=g) * 5.0  # N / Nm, raw
    return e, mask, state, wrench


def _fusion(expert_inputs="all", seed=0):
    torch.manual_seed(seed)
    return ForceVLA2Fusion(D, D_STATE, NH, expert_inputs=expert_inputs, n_vision=NV)


def test_fusion_shapes_and_aux():
    fus = _fusion()
    e, mask, state, wrench = _inputs()
    cond, cond_mask, aux = fus(e, mask, state, wrench)
    assert cond.shape == (2, S + 3, D)
    assert cond_mask.shape == (2, S + 3)
    assert cond_mask[:, :S].equal(mask) and cond_mask[:, S:].all()  # 3 new tokens are real
    assert aux["gate"].shape == (2, S + 3, 3)
    assert aux["cross_attn"].shape == (2, NH, 2, S)
    assert set(aux["gate_groups"]) == {"vision", "text", "state", "force"}
    for g in aux["gate_groups"].values():
        assert g.shape == (2, 3)
        assert torch.allclose(g.sum(-1), torch.ones(2), atol=1e-5)


def test_gate_weights_sum_to_one_per_token():
    _, _, aux = _fusion()(*_inputs())
    w = aux["gate"]
    assert (w >= 0).all()
    assert torch.allclose(w.sum(-1), torch.ones(w.shape[:2]), atol=1e-6)


def test_cross_attention_ignores_pad_keys():
    fus = _fusion()
    e, mask, state, wrench = _inputs()
    _, _, aux1 = fus(e, mask, state, wrench)
    e2 = e.clone()
    e2[~mask] = 1e3  # garbage in the masked <pad> positions
    _, _, aux2 = fus(e2, mask, state, wrench)
    assert torch.allclose(aux1["cross_attn"], aux2["cross_attn"])
    assert aux1["cross_attn"][..., NV + 4 :].abs().max() == 0  # no weight on <pad> keys


def test_bypass_carries_force_when_cross_attention_is_zeroed():
    """Zeroed CrossAttn: state has no path out (no residual), force still flows via E_F."""
    fus = _fusion()
    with torch.no_grad():
        for p in fus.cross_attn.parameters():
            p.zero_()
    e, mask, state, wrench = _inputs()
    state.requires_grad_(True)
    wrench.requires_grad_(True)
    cond, _, _ = fus(e, mask, state, wrench)
    cond.square().sum().backward()
    assert state.grad.abs().max() == 0, "state leaked around the cross-attention"
    assert wrench.grad.abs().min() > 0, "the bypass did not carry every wrench axis"
    assert fus.phi_f.weight.grad.abs().sum() > 0
    assert fus.gate.weight.grad.abs().sum() > 0
    for ex in fus.experts.values():
        assert ex[0].weight.grad.abs().sum() > 0


def test_state_reaches_output_through_cross_attention():
    fus = _fusion()
    e, mask, state, wrench = _inputs()
    state.requires_grad_(True)
    cond, _, _ = fus(e, mask, state, wrench)
    cond.square().sum().backward()
    assert state.grad.abs().sum() > 0


def _captured_expert_input(fus, *inputs):
    seen = {}

    def hook(_mod, args):
        seen["x"] = args[0].detach().clone()

    h = fus.experts["force"].register_forward_pre_hook(hook)
    out = fus(*inputs)
    h.remove()
    return seen["x"], out


def test_expert_inputs_variants():
    inputs = _inputs()
    x_all, _ = _captured_expert_input(_fusion("all"), *inputs)
    x_vis, _ = _captured_expert_input(_fusion("vision_only"), *inputs)
    x_none, _ = _captured_expert_input(_fusion("none"), *inputs)
    e = inputs[0]
    # all: the full E_cond = [E ; E'_state ; E_F]
    assert x_all.shape == (2, S + 3, D)
    assert torch.equal(x_all[:, :S], e)
    assert x_all[:, S:].abs().sum(-1).min() > 0
    # vision_only: the prefix only, state + force tokens zeroed
    assert torch.equal(x_vis[:, :S], e)
    assert x_vis[:, S:].abs().max() == 0
    # none: only the two E'_state tokens survive (same seed => same E'_state as "all")
    assert x_none[:, :S].abs().max() == 0 and x_none[:, S + 2].abs().max() == 0
    assert torch.allclose(x_none[:, S : S + 2], x_all[:, S : S + 2])
    with pytest.raises(ValueError):
        ForceVLA2Fusion(D, D_STATE, NH, expert_inputs="force_only")


def test_vision_only_force_reaches_output_only_via_gate():
    """vision_only: the experts never see E_F, so a wrench change moves cond only through w."""
    fus = _fusion("vision_only")
    e, mask, state, wrench = _inputs()
    x1, (c1, _, a1) = _captured_expert_input(fus, e, mask, state, wrench)
    x2, (c2, _, a2) = _captured_expert_input(fus, e, mask, state, wrench * 3)
    assert torch.equal(x1, x2)  # identical expert input
    assert not torch.allclose(a1["gate"], a2["gate"])  # the gate saw the force
    assert not torch.allclose(c1, c2)


def test_hybrid_head_shapes_and_roundtrip():
    b, h = 3, 16
    head = HybridHead(D)
    assert head(torch.randn(b, h, D)).shape == (b, h, HYBRID_DIM) == (b, h, 11)
    g = torch.Generator().manual_seed(1)
    dp = (torch.rand(b, h, 3, generator=g) * 2 - 1) * DP_MAX
    grip = torch.rand(b, h, generator=g)
    ft = torch.cat(
        [
            (torch.rand(b, h, 3, generator=g) * 2 - 1) * F_SCALE,
            (torch.rand(b, h, 3, generator=g) * 2 - 1) * TORQUE_SCALE,
        ],
        -1,
    )
    s = torch.rand(b, h, generator=g)
    x = hybrid_to_model(dp, grip, ft, s)
    assert x.shape == (b, h, 11)
    assert x.abs().max() <= 1.0 + 1e-6  # every dim in [-1, 1] in model units
    back = hybrid_from_model(x)
    for k, ref in (("dp", dp), ("gripper", grip), ("f_target", ft), ("s", s)):
        assert torch.allclose(back[k], ref, atol=1e-5), k
    # the scales the docstring promises: -5 N on z -> -0.2, s = 1 -> +1
    one = hybrid_to_model(
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1),
        torch.tensor([[[0.0, 0.0, -5.0, 0.0, 0.0, 0.0]]]),
        torch.ones(1, 1),
    )
    assert torch.allclose(one[0, 0, 6], torch.tensor(-0.2)) and one[0, 0, 10] == 1.0
    # decode clamps to what the env/controller accept
    wild = hybrid_from_model(torch.full((1, 1, 11), 7.0))
    assert wild["dp"].max() == DP_MAX and wild["gripper"].max() == 1.0
    assert wild["f_target"][..., :3].max() == F_SCALE and wild["s"].max() == 1.0


def test_hybrid_flow_end_to_end_with_expert():
    """Fusion cond -> ActionExpert(act_dim=11) memory -> HybridHead -> flow loss -> grads."""
    b, h = 2, 16
    fus = _fusion()
    expert = ActionExpert(D, NH, 768, 2, HYBRID_DIM, h)
    head = HybridHead(D)
    state_in = torch.nn.Linear(D_STATE, D)
    e, mask, state, wrench = _inputs(b)
    cond, cond_mask, _ = fus(e, mask, state, wrench)
    x0 = hybrid_to_model(
        torch.zeros(b, h, 3), torch.zeros(b, h), torch.zeros(b, h, 6), torch.zeros(b, h)
    )

    def velocity(x_tau, tau):
        feats, _, cross = expert(x_tau, tau, state_in(state), cond, cond_mask)
        assert cross.shape == (b, NH, 1 + h, S + 3)  # the expert reads all of E_cond
        return head(feats)

    per, aux = FlowMatching().loss(velocity, x0)
    assert per.shape == (b,) and aux["per_dim"].shape == (HYBRID_DIM,)
    per.mean().backward()
    assert fus.phi_f.weight.grad.abs().sum() > 0
    assert fus.out_proj.weight.grad.abs().sum() > 0


def test_param_count(capsys):
    """Plan §4: +ForceVLA2 ≈ +1.7 M over nanoVLA; we print the real figure (report it)."""
    fus = _fusion()
    n_fus = sum(p.numel() for p in fus.parameters())
    # the hybrid head / 11-D act_in replace the 4-D ones: delta = 7 * (D + 1) + 7 * D
    n_head_delta = 7 * (D + 1) + 7 * D
    with capsys.disabled():
        print(f"\nForceVLA2Fusion params: {n_fus:,}  (+ head/act_in delta {n_head_delta:,})")
        for name, mod in fus.named_children():
            print(f"  {name:12s} {sum(p.numel() for p in mod.parameters()):>9,}")
    assert 1.0e6 < n_fus < 1.3e6  # d -> 2d -> d experts (see the report vs the 1.7 M estimate)


def test_subtask_machine_hysteresis_sequence():
    sm = SubtaskMachine(("a", "b", "c"))
    assert (ADVANCE_AT, REARM_BELOW) == (0.8, 0.3)
    seq = [0.1, 0.5, 0.85, 0.9, 0.5, 0.29, 0.79, 0.8, 0.95, 0.1, 0.99, 0.2, 0.9]
    want = ["a", "a", "b", "b", "b", "b", "b", "c", "c", "c", "c", "c", "c"]
    adv = [False, False, True, False, False, False, False, True, False, False, False, False, False]
    got, got_adv = [], []
    for s in seq:
        got_adv.append(sm.update(s))
        got.append(sm.current)
    assert got == want and got_adv == adv
    assert sm.is_last and sm.prompt("press the red button") == "press the red button c"
    sm.reset()
    assert sm.current == "a" and sm.armed
    with pytest.raises(ValueError):
        SubtaskMachine(("a",), advance_at=0.3, rearm_below=0.5)


def test_task_subtasks_are_the_labelled_ones_and_in_vocab():
    from nfv.model.text import MAX_TEXT, UNK, tokenize
    from nfv.sim.tasks import TASKS

    assert task_subtasks("t0") == ("reach", "grasp", "lift")
    assert task_subtasks("t1") == ("approach", "press", "done")
    assert task_subtasks("t2") == ("approach", "wipe")
    assert task_subtasks("t3") == ("approach", "descend", "search", "insert")
    for name in ("t0", "t1", "t2", "t3"):
        for instr in TASKS[name].INSTRUCTIONS:
            for sub in task_subtasks(name):
                prompt = f"{instr} {sub}"
                ids = tokenize(prompt)
                assert UNK not in ids, prompt
                assert len(prompt.split()) <= MAX_TEXT


def test_hindsight_progress_and_machine_tracking():
    ids = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    s = hindsight_progress(ids)
    assert np.allclose(s[:6], np.linspace(0, 1, 6)) and s[6] == 0 and s[10] == 1
    assert np.allclose(hindsight_progress(np.array([3])), [1.0])  # a 1-step run
    sm = SubtaskMachine(("a", "b", "c"))
    seen = []
    for v in s:
        seen.append(sm.index)
        sm.update(v)
    # switches at >= 80 % of each subtask, never late
    assert seen == [0] * 5 + [1] * 6 + [2] * 6  # expert: a x6, b x5, c x6


def test_dataset_serves_s_sub(tmp_path):
    from nfv.data.dataset import DemoDataset
    from nfv.data.record import record_demos

    record_demos("t0", n=1, out_dir=tmp_path, noise_mm=0.5)
    ds = DemoDataset(tmp_path, "t0")
    ep = ds.episodes[0]
    assert np.allclose(ep["s_sub"], hindsight_progress(ep["subtask"]))
    b = ds.sample_batch(4, np.random.default_rng(0), chunk=16)
    assert b["s_sub"].shape == (4, 16) and b["s_sub"].dtype == torch.float32
    assert (b["s_sub"] >= 0).all() and (b["s_sub"] <= 1).all()


# --- wiring into the nano Policy (build.py / config.py / train.py) -----------------------


def _tiny_cfg(**kw):
    from nfv.model.config import ModelConfig

    base = dict(trunk_layers=1, expert_layers=1, ffn=128, chunk=8, n_exec=4, euler_steps=2)
    base.update(kw)
    return ModelConfig(**base)


@pytest.fixture(scope="module")
def t2_demos(tmp_path_factory):
    from nfv.data.dataset import DemoDataset
    from nfv.data.record import record_demos

    root = tmp_path_factory.mktemp("t2demos")
    record_demos("t2", n=2, out_dir=root, noise_mm=0.5)
    return DemoDataset(root, "t2")


def test_build_validates_forcevla2_flags():
    from nfv.model.config import build

    with pytest.raises(ValueError, match="11-D"):
        build(_tiny_cfg(force_in="forcevla2", head="hybrid"))
    with pytest.raises(ValueError, match="expert_inputs"):
        build(_tiny_cfg(force_in="fvlmoe", expert_inputs="vision_only"))
    pol = build(_tiny_cfg(force_in="forcevla2", head="hybrid", act_dim=11, force_prompt=True))
    assert pol.net.fv2 is not None and pol.net.head.out_features == 11


@pytest.mark.parametrize(
    "kw",
    [
        dict(force_in="forcevla2", head="hybrid", act_dim=11, force_prompt=True),  # arm B
        dict(force_in="forcevla2", head="position", force_prompt=True),  # arm C
        dict(force_in="forcevla2", head="hybrid", act_dim=11, expert_inputs="none"),
    ],
)
def test_fit_and_act_end_to_end(kw, t2_demos, render_env):
    from nfv.model.build import param_table
    from nfv.train import fit

    cfg = _tiny_cfg(**kw)
    policy, hist = fit(cfg, t2_demos, steps=4, batch=4, log_every=2, device="cpu")
    last = hist[-1]
    assert np.isfinite(last["loss"])
    assert "gate_force_vision" in last and "gate_force_force" in last
    assert ("loss_f_target" in last) == (cfg.head == "hybrid")
    assert param_table(policy)["fusion"] > 0
    obs = render_env.reset(0, task="t2")
    out = policy.act(obs, "wipe the plate gently approach")
    assert out["dp"].shape == (8, 3) and out["gripper"].shape == (8,)
    assert set(out["aux"]["gate_groups"]) == {"vision", "text", "state", "force"}
    if cfg.head == "hybrid":
        assert out["f_target"].shape == (8, 6) and out["s"].shape == (8,)
        assert np.abs(out["f_target"][:, :3]).max() <= F_SCALE
        assert (out["s"] >= 0).all() and (out["s"] <= 1).all()
    else:
        assert out["f_target"] is None and out["s"] is None


def test_force_prompt_text_follows_chunk_subtask(t2_demos):
    from nfv.model.text import tokenize
    from nfv.sim.tasks import TASKS
    from nfv.train import episode_prompt_ids, sample_batch_with_text

    ids = episode_prompt_ids(t2_demos, 12)
    subs = TASKS["t2"].SUBTASKS
    assert ids.shape == (len(t2_demos), len(subs), 12)
    b = sample_batch_with_text(t2_demos, ids, 16, np.random.default_rng(0), 8)
    instr = [ep["instruction"] for ep in t2_demos.index["episodes"]]
    texts = {tuple(tokenize(f"{i} {s}")) for i in instr for s in subs}
    for row, sub in zip(b["text"].tolist(), b["subtask"].tolist(), strict=True):
        assert tuple(row) in texts
        assert any(tuple(row) == tuple(tokenize(f"{i} {subs[sub]}")) for i in instr)
    assert "s_sub" in b and b["s_sub"].shape == (16, 8)
