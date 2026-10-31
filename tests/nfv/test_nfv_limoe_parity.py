"""Parity of the PyTorch LIMoE port against the reference JAX LIMoEBlock, by fixture.

The fixture (tests/nfv/fixtures/limoe_parity.npz) was produced by running the REAL reference
code — third_party/forcevla @ 79de88f (limoe_simple.py + its vendored flaxformer router) — on
host CPU via tests/nfv/gen_limoe_fixture.py, in deterministic eval mode, with the router /
expert dtypes pinned to float32 (the generator's documented deviation from the reference's
TPU bfloat16). This test loads the dumped input + weights + output, copies the weights into
the port, and requires max |diff| < 1e-5 — so the port's encoder math, grouping, top-1
batch-prioritized routing, capacity cap and output projection all match the reference.

Name mapping (flax param path -> port parameter; flax Dense kernels are [in, out] and
transpose into torch's [out, in]; flax attention kernels are [d, heads, head_dim] and
reshape+transpose; the vmapped experts carry a leading expert axis):

    encoderblock/encoder_in_norm/{scale,bias}         encoderblock.encoder_in_norm.{weight,bias}
    encoderblock/MultiHeadDotProductAttention_0/
        {query,key,value}/kernel [D,H,dh]             encoderblock.attention.{query,key,value}
                                                        .weight = kernel.reshape(D, H*dh).T
        {query,key,value}/bias [H,dh]                   .bias   = bias.reshape(-1)
        out/kernel [H,dh,D]                           encoderblock.attention.out
                                                        .weight = kernel.reshape(H*dh, D).T
    encoderblock/encoder_out_norm/{scale,bias}        encoderblock.encoder_out_norm.{weight,bias}
    encoderblock/MlpBlock_0/Dense_{0,1}/{kernel,bias} encoderblock.mlp.dense_{0,1}
    RouterWeights_0/w/{kernel,bias}                   moe_layer.router (the "w" DenseGeneral)
    MlpBlock_0/{wi,wo}/kernel [E,in,out]              moe_layer.experts[e].{wi,wo}.weight
    MlpBlock_0/{wi,wo}/bias [E,out]                     (= kernel[e].T / bias[e])
    Dense_0/{kernel,bias}                             out_proj (the block's final projection)
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from nfv.model.limoe_port import LIMoEBlock

FIXTURE = Path(__file__).parent / "fixtures" / "limoe_parity.npz"


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a, dtype=np.float32))


def load_reference_weights(block: LIMoEBlock, fx: dict[str, np.ndarray]) -> None:
    """Copy the fixture's flax params into the port, by the mapping in the module docstring."""

    def p(key: str) -> torch.Tensor:
        return _t(fx[f"param:{key}"])

    with torch.no_grad():
        enc = block.encoderblock
        enc.encoder_in_norm.weight.copy_(p("encoderblock/encoder_in_norm/scale"))
        enc.encoder_in_norm.bias.copy_(p("encoderblock/encoder_in_norm/bias"))
        enc.encoder_out_norm.weight.copy_(p("encoderblock/encoder_out_norm/scale"))
        enc.encoder_out_norm.bias.copy_(p("encoderblock/encoder_out_norm/bias"))
        attn = "encoderblock/MultiHeadDotProductAttention_0"
        d = enc.encoder_in_norm.weight.shape[0]
        for name in ("query", "key", "value"):
            lin = getattr(enc.attention, name)
            lin.weight.copy_(p(f"{attn}/{name}/kernel").reshape(d, -1).T)
            lin.bias.copy_(p(f"{attn}/{name}/bias").reshape(-1))
        enc.attention.out.weight.copy_(p(f"{attn}/out/kernel").reshape(-1, d).T)
        enc.attention.out.bias.copy_(p(f"{attn}/out/bias"))
        for i in (0, 1):
            lin = getattr(enc.mlp, f"dense_{i}")
            lin.weight.copy_(p(f"encoderblock/MlpBlock_0/Dense_{i}/kernel").T)
            lin.bias.copy_(p(f"encoderblock/MlpBlock_0/Dense_{i}/bias"))
        block.moe_layer.router.weight.copy_(p("RouterWeights_0/w/kernel").T)
        block.moe_layer.router.bias.copy_(p("RouterWeights_0/w/bias"))
        for e, expert in enumerate(block.moe_layer.experts):
            for name in ("wi", "wo"):
                lin = getattr(expert, name)
                lin.weight.copy_(p(f"MlpBlock_0/{name}/kernel")[e].T)
                lin.bias.copy_(p(f"MlpBlock_0/{name}/bias")[e])
        block.out_proj.weight.copy_(p("Dense_0/kernel").T)
        block.out_proj.bias.copy_(p("Dense_0/bias"))


@pytest.fixture(scope="module")
def fixture() -> dict[str, np.ndarray]:
    assert FIXTURE.exists(), (
        f"missing {FIXTURE} — generate it on the host: uv run --no-project "
        '--with "jax==0.4.38" --with "flax==0.10.2" --with numpy --with absl-py '
        "python tests/nfv/gen_limoe_fixture.py"
    )
    with np.load(FIXTURE) as z:
        return {k: z[k] for k in z.files}


def test_port_matches_reference_fixture(fixture):
    """Eval-mode forward on the fixture input matches the reference output to < 1e-5."""
    block = LIMoEBlock(
        d_model=32, mlp_dim=64, num_heads=2, out_dim=24, num_experts=4, expert_dim=2048
    ).eval()
    load_reference_weights(block, fixture)
    x = _t(fixture["in:x"])
    with torch.no_grad():
        y, aux = block(x)
    ref = _t(fixture["out:y"])
    assert y.shape == tuple(ref.shape)
    diff = float((y - ref).abs().max())
    print(f"limoe parity: max |port - reference| = {diff:.3e}")
    print(f"  expert_counts = {aux['expert_counts'].tolist()}")
    print(f"  fraction_dropped = {float(aux['fraction_dropped']):.4f}")
    assert diff < 1e-5


def test_every_fixture_weight_is_consumed(fixture):
    """The mapping covers the full reference param tree (no silently ignored weights)."""
    block = LIMoEBlock(
        d_model=32, mlp_dim=64, num_heads=2, out_dim=24, num_experts=4, expert_dim=2048
    )
    n_port = sum(p.numel() for p in block.parameters())
    n_ref = sum(int(np.prod(fixture[k].shape)) for k in fixture if k.startswith("param:"))
    assert n_port == n_ref, f"port has {n_port} params, the reference fixture {n_ref}"
