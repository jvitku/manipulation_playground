"""nfv.lora - LoRA from scratch (lesson 20). Pure torch, CPU, runs in the lab's own python.

Pins the properties the lesson leans on: the wrapped layer is bit-identical to the base at
step 0 (B = 0), the merge equals the adapter forward, only A/B (+ explicitly unfrozen heads)
train, the parameter count is r (in + out) per layer, and a LoRA update has rank <= r.
"""

import copy

import pytest
import torch
from torch import nn

from nfv.lora import (
    LoRALinear,
    energy_top_k,
    inject_lora,
    lora_layers,
    merge_lora,
    set_trainable,
    singular_values,
    state_bytes,
    trainable_report,
    trainable_state_dict,
)


class _Attn(nn.Module):
    def __init__(self, d: int, kv: int):
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(d, d), nn.Linear(d, kv)
        self.v_proj, self.o_proj = nn.Linear(d, kv), nn.Linear(d, d, bias=False)

    def forward(self, x):  # not real attention - a shape-correct stand-in that uses all four
        return self.o_proj(self.q_proj(x)) + torch.cat([self.k_proj(x), self.v_proj(x)], -1)[
            ..., : x.shape[-1]
        ]  # fmt: skip


class _Block(nn.Module):
    def __init__(self, d: int = 32, kv: int = 16):
        super().__init__()
        self.self_attn = _Attn(d, kv)
        self.mlp = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, d))

    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp(x)


class _Tiny(nn.Module):
    """Two blocks + a task head: the shapes and names the SmolVLA targets look like."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return self.head(x)


ATTN = r"layers\.\d+\.self_attn\.(q|k|v|o)_proj"


def _randomise_B(model: nn.Module, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in lora_layers(model).values():
            m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=g) * 0.1)


# ------------------------------------------------------------------------------- LoRALinear
def test_step0_is_bit_identical_to_the_base():
    torch.manual_seed(0)
    base = nn.Linear(48, 24)
    x = torch.randn(5, 7, 48)
    want = base(x)
    lora = LoRALinear(base, r=8, alpha=16, dropout=0.1)
    lora.train()  # dropout active on the adapter path: still exact, B = 0
    assert torch.equal(lora(x), want)
    assert torch.count_nonzero(lora.lora_B) == 0
    assert torch.count_nonzero(lora.lora_A) == lora.lora_A.numel()  # kaiming-uniform


def test_merge_equals_the_adapter_forward_fp32():
    torch.manual_seed(1)
    lora = LoRALinear(nn.Linear(64, 40), r=4, alpha=8)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.2)
    x = torch.randn(16, 64)
    merged = lora.merge()
    assert type(merged) is nn.Linear
    assert merged.weight.shape == (40, 64)
    assert (lora(x) - merged(x)).abs().max().item() < 1e-5
    # the merge really changed W (otherwise the check above would be vacuous)
    assert (merged.weight - lora.base.weight).abs().max().item() > 1e-3


def test_only_A_B_require_grad_and_param_count():
    base = nn.Linear(30, 20)
    lora = LoRALinear(base, r=6, alpha=12)
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A", "lora_B"}
    assert lora.lora_A.shape == (6, 30) and lora.lora_B.shape == (20, 6)
    assert lora.n_adapter_params() == 6 * (30 + 20)
    assert sum(p.numel() for p in lora.parameters() if p.requires_grad) == 6 * (30 + 20)


def test_step0_gradients_flow_into_B_first():
    """dL/dB != 0 at step 0 (A is random), dL/dA == 0 while B == 0 - why A must not be 0."""
    torch.manual_seed(2)
    lora = LoRALinear(nn.Linear(12, 10), r=3, alpha=6)
    lora(torch.randn(8, 12)).pow(2).sum().backward()
    assert lora.lora_B.grad.abs().sum() > 0
    assert torch.count_nonzero(lora.lora_A.grad) == 0
    assert lora.base.weight.grad is None  # frozen


def test_host_view_reads_through_to_the_base():
    base = nn.Linear(9, 5).to(torch.bfloat16)
    lora = LoRALinear(base, r=2)
    assert lora.weight is base.weight and lora.bias is base.bias
    assert lora.weight.dtype == torch.bfloat16  # SmolVLA reads q_proj.weight.dtype
    assert lora.lora_A.dtype == torch.float32  # the adapter stays fp32
    assert (lora.in_features, lora.out_features) == (9, 5)
    y = lora(torch.randn(3, 9, dtype=torch.bfloat16))  # mixed dtype without autocast
    assert y.dtype == torch.bfloat16


def test_delta_has_rank_at_most_r():
    torch.manual_seed(3)
    lora = LoRALinear(nn.Linear(50, 40), r=5, alpha=10)
    with torch.no_grad():
        lora.lora_B.normal_()
    s = singular_values(lora.delta_weight())
    assert s.shape == (40,)
    assert torch.all(s[5:] < 1e-4 * s[0])
    assert energy_top_k(s, 5) == pytest.approx(1.0, abs=1e-8)
    assert energy_top_k(torch.ones(10), 3) == pytest.approx(0.3)


def test_bad_rank_and_bad_module_are_refused():
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), r=0)
    with pytest.raises(TypeError):
        LoRALinear(nn.Conv1d(4, 4, 1))


# ------------------------------------------------------------------------------- model level
def test_inject_freezes_everything_else_and_reports_the_count():
    torch.manual_seed(4)
    model = _Tiny()
    rep = inject_lora(model, [ATTN], r=4, alpha=8)
    assert rep["n_layers"] == 8  # 2 blocks x q/k/v/o
    assert sorted(rep["targets"])[0] == "layers.0.self_attn.k_proj"
    # r (in + out) per layer: q 32->32, k 32->16, v 32->16, o 32->32
    per_block = 4 * ((32 + 32) + (32 + 16) + (32 + 16) + (32 + 32))
    assert rep["lora_params"] == 2 * per_block
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names and all(n.endswith(("lora_A", "lora_B")) for n in names)
    rpt = trainable_report(model)
    assert rpt["trainable"] == rep["lora_params"]


def test_explicitly_unfrozen_head_trains_too():
    model = _Tiny()
    inject_lora(model, [ATTN], r=2, alpha=4)
    hit = set_trainable(model, [r"head\..*"])
    assert set(hit) == {"head.weight", "head.bias"}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert {"head.weight", "head.bias"} <= trainable
    assert not any("mlp" in n for n in trainable)
    groups = trainable_report(model, {"lora": r"lora_", "head": r"^head\.", "body": r"."})
    assert groups["groups"]["head"]["trainable"] == 32 * 4 + 4
    assert groups["groups"]["body"]["trainable"] == 0
    assert groups["trainable"] == groups["groups"]["lora"]["trainable"] + 32 * 4 + 4
    with pytest.raises(ValueError):
        set_trainable(model, [r"nhead\..*"])  # a typo must not silently leave a head frozen


def test_model_step0_identity_and_merge_equivalence():
    torch.manual_seed(5)
    model = _Tiny().eval()
    x = torch.randn(3, 6, 32)
    want = model(x)
    inject_lora(model, [ATTN, r"layers\.\d+\.mlp\.\d+"], r=4, alpha=8)
    assert torch.equal(model(x), want)  # bit-identical at step 0
    _randomise_B(model)
    adapted = model(x)
    assert (adapted - want).abs().max() > 1e-3
    assert merge_lora(model) == 12
    assert not lora_layers(model)
    assert all(type(m) is nn.Linear for n, m in model.named_modules() if n.endswith("_proj"))
    assert (model(x) - adapted).abs().max().item() < 1e-5


def test_no_double_wrap_and_typo_patterns_are_refused():
    model = _Tiny()
    inject_lora(model, [ATTN], r=2)
    with pytest.raises(ValueError):
        inject_lora(model, [ATTN], r=2)  # every match is now a LoRALinear or its .base
    with pytest.raises(ValueError):
        inject_lora(_Tiny(), [r"layers\.\d+\.attn\.q_proj"], r=2)  # matches nothing


def test_adapter_state_dict_holds_only_the_trainable_tensors():
    model = _Tiny()
    pretrained = copy.deepcopy(model.state_dict())  # in the real run: smolvla_base, HF cache
    rep = inject_lora(model, [ATTN], r=4, alpha=8)
    set_trainable(model, [r"head\..*"])
    sd = trainable_state_dict(model)
    assert all(k.endswith(("lora_A", "lora_B")) or k.startswith("head.") for k in sd)
    n = sum(t.numel() for t in sd.values())
    assert n == rep["lora_params"] + 32 * 4 + 4
    assert state_bytes(sd.values()) == 4 * n  # fp32
    # round trip into a fresh, identically injected model reproduces the outputs
    _randomise_B(model, seed=7)
    sd = trainable_state_dict(model)
    fresh = _Tiny()
    fresh.load_state_dict(pretrained)
    inject_lora(fresh, [ATTN], r=4, alpha=8)
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    assert not unexpected
    assert all(k.endswith(".weight") or k.endswith(".bias") for k in missing)  # frozen base
    x = torch.randn(2, 32)
    assert torch.allclose(fresh(x), model(x), atol=1e-6)
