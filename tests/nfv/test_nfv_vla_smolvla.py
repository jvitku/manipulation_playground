"""nfv.vla_smolvla - the parts that need no lerobot (lab python, CPU): LR schedule, demo
normalisation, the instruction-aware sampler, the bf16 lost-update measurement, the spectrum
record and the regexes that pick the LoRA targets / heads out of SmolVLA's module names."""

import re

import numpy as np
import pytest
import torch
from torch import nn

from nfv import vla_smolvla as V


def test_lr_schedule_warmup_peak_and_floor():
    peak, steps, warm = 1e-4, 1000, 100
    assert V.lr_at(0, steps, peak, warm) == pytest.approx(peak / warm)
    assert V.lr_at(warm - 1, steps, peak, warm) == pytest.approx(peak)
    assert V.lr_at(warm, steps, peak, warm) == pytest.approx(peak)
    assert V.lr_at(steps, steps, peak, warm) == pytest.approx(peak * V.DECAY_FRAC)
    lrs = [V.lr_at(s, steps, peak, warm) for s in range(warm, steps)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:], strict=False))  # cosine decays


class _DS:
    """The two attributes Sampler / norm_stats read from a DemoDataset."""

    def __init__(self, lengths=(5, 3)):
        rng = np.random.default_rng(0)
        self.episodes = []
        for n in lengths:
            st = rng.normal(size=(n, 8)).astype(np.float32)
            st[:, 3] = 0.0  # a constant yaw: std 0 -> 1
            self.episodes.append({
                "rgb_static": np.zeros((n, 96, 96, 3), np.uint8),
                "rgb_wrist": np.ones((n, 96, 96, 3), np.uint8),
                "state": st, "dp": rng.normal(size=(n, 3)).astype(np.float32) * 1e-3,
                "gripper": np.linspace(0, 1, n).astype(np.float32),
            })  # fmt: skip
        self.index = {"episodes": [{"instruction": f"task {i}"} for i in range(len(lengths))]}


def test_norm_stats_shapes_and_constant_dims():
    s = V.norm_stats(_DS())
    assert len(s["state_mean"]) == 8 and len(s["action_mean"]) == 4
    assert s["state_std"][3] == 1.0  # constant yaw is not divided by ~0
    assert 1e-4 < s["action_std"][0] < 1e-2  # dp in m/step keeps its real scale


def test_sampler_pads_past_the_episode_end_and_keeps_the_text():
    ds = _DS(lengths=(2,))
    smp = V.Sampler(ds, chunk=4)
    out = smp.sample(16, np.random.default_rng(1))
    assert out["action"].shape == (16, 4, 4) and out["pad"].shape == (16, 4)
    assert out["rgb_static"].shape == (16, 96, 96, 3)
    for a, p in zip(out["action"], out["pad"], strict=True):
        assert p[2:].all()  # a 2-step episode: at most 2 real steps
        assert np.all(a[p] == 0)
    assert set(out["text"]) == {"task 0"}


def test_bf16_lost_fraction_counts_vanishing_updates():
    m = nn.Linear(64, 64)
    with torch.no_grad():
        m.weight.fill_(1.0)
    before = {"weight": m.weight.detach().clone()}
    with torch.no_grad():
        m.weight[:32] += 1e-4  # far below half a bf16 ulp at 1.0 (2^-8): lost
        m.weight[32:] += 0.1  # survives
    assert V.bf16_lost_fraction(before, m) == pytest.approx(0.5)


def test_spectrum_record():
    dw = torch.zeros(10, 6)
    dw[0, 0], dw[1, 1] = 3.0, 1.0
    s = V.spectrum(dw, r=1)
    assert s["shape"] == [10, 6] and s["n_sv"] == 6
    assert s["energy_top_r"] == pytest.approx(0.9)
    assert s["energy_top_16"] == pytest.approx(1.0)


NAMES = [  # real qualified names from smolvla_base (introspected in the /opt/vla venv)
    "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj",
    "model.vlm_with_expert.vlm.model.text_model.layers.15.self_attn.o_proj",
    "model.vlm_with_expert.vlm.model.text_model.layers.3.mlp.gate_proj",
    "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.q_proj",
    "model.vlm_with_expert.lm_expert.layers.7.self_attn.v_proj",
    "model.vlm_with_expert.lm_expert.layers.7.mlp.up_proj",
    "model.vlm_with_expert.vlm.lm_head",
    "model.action_time_mlp_in",
]


def test_lora_targets_hit_only_text_and_expert_attention():
    hit = [n for n in NAMES if any(re.fullmatch(p, n) for p in V.LORA_TARGETS)]
    assert hit == [NAMES[0], NAMES[1], NAMES[4]]  # never the vision tower, MLPs or lm_head


def test_head_patterns_hit_the_projections_only():
    params = ["model.state_proj.weight", "model.action_in_proj.bias",
              "model.action_out_proj.weight", "model.action_time_mlp_out.bias",
              "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight"]  # fmt: skip
    hit = [n for n in params if any(re.fullmatch(p, n) for p in V.HEAD_PATTERNS)]
    assert hit == params[:4]
    assert all(n.startswith("model.vlm_with_expert.lm_expert.") for n in V.SPECTRUM_LAYERS)
