"""Shared mini-transformer: shapes, attention maths, masking, snapshots, and that it trains."""

import torch
from torch import nn

from lab.tasks.minitf import MHA, Block, CrossBlock, attn_snapshot, causal_mask, rope_rotate


def test_mha_shapes():
    mha = MHA(32, 4)
    out, attn = mha(torch.randn(2, 5, 32))
    assert out.shape == (2, 5, 32)
    assert attn.shape == (2, 4, 5, 5)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 4, 5), atol=1e-5)


def test_mha_cross_shapes():
    mha = MHA(32, 4)
    out, attn = mha(torch.randn(2, 3, 32), torch.randn(2, 7, 32))
    assert out.shape == (2, 3, 32)
    assert attn.shape == (2, 4, 3, 7)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 4, 3), atol=1e-5)


def test_causal_mask_blocks_future():
    m = causal_mask(6)
    assert m.dtype == torch.bool and m.shape == (6, 6)
    _, attn = MHA(16, 2)(torch.randn(2, 6, 16), mask=m)
    future = ~m
    assert attn[:, :, future].abs().max() == 0
    assert torch.allclose(attn.sum(-1), torch.ones(2, 2, 6), atol=1e-5)


def test_block_and_crossblock_shapes():
    x, mem = torch.randn(2, 5, 32), torch.randn(2, 9, 32)
    y, attn = Block(32, 4)(x)
    assert y.shape == x.shape and attn.shape == (2, 4, 5, 5)
    y, sa, ca = CrossBlock(32, 4)(x, mem, self_mask=causal_mask(5))
    assert y.shape == x.shape
    assert sa.shape == (2, 4, 5, 5) and ca.shape == (2, 4, 5, 9)


def test_attn_snapshot_downsamples():
    attn = torch.softmax(torch.randn(1, 2, 32, 32), -1)
    snap = attn_snapshot(attn)
    assert len(snap) == 2 and len(snap[0]) == 16 and len(snap[0][0]) == 16
    assert snap[0][0][0] == round(snap[0][0][0], 3)
    small = attn_snapshot(torch.softmax(torch.randn(3, 2, 4, 6), -1))
    assert len(small[0]) == 4 and len(small[0][0]) == 6


def test_rope_dot_depends_on_offset_only():
    # The defining property: after rotation, dot(q_m, k_n) is a function of m - n alone.
    torch.manual_seed(0)
    dh, t = 16, 24
    q = rope_rotate(torch.randn(dh).expand(1, 1, t, dh).clone())[0, 0]  # one vector, T positions
    k = rope_rotate(torch.randn(dh).expand(1, 1, t, dh).clone())[0, 0]
    same_offset = [((0, 3), (2, 5), (9, 12), (20, 23)), ((4, 0), (10, 6), (23, 19)),
                   ((0, 0), (7, 7))]  # fmt: skip
    for pairs in same_offset:
        dots = [float(q[m] @ k[n]) for m, n in pairs]  # equal offset -> equal dot
        assert max(dots) - min(dots) < 1e-5, pairs
    assert abs(float(q[0] @ k[0]) - float(q[0] @ k[3])) > 1e-3  # different offsets really differ


def test_rope_off_is_the_default_path():
    # rope=False (and the bare ctor) must be byte-identical to the pre-rope behavior.
    x, mem = torch.randn(2, 6, 32), torch.randn(2, 9, 32)
    torch.manual_seed(0)
    base = MHA(32, 4)
    torch.manual_seed(0)
    off = MHA(32, 4, rope=False)
    torch.manual_seed(0)
    on = MHA(32, 4, rope=True)
    for a, b in zip(base(x), off(x), strict=True):
        assert torch.equal(a, b)
    assert not torch.equal(base(x)[0], on(x)[0])  # rope changes self-attention...
    for a, b in zip(base(x, mem), on(x, mem), strict=True):
        assert torch.equal(a, b)  # ...but never a separate-x_kv (cross) call: no shared axis
    torch.manual_seed(1)
    b0 = Block(32, 4)
    torch.manual_seed(1)
    b1 = Block(32, 4, rope=False)
    assert torch.equal(b0(x)[0], b1(x)[0])


def test_rope_shapes_and_masking():
    assert rope_rotate(torch.randn(2, 4, 7, 16)).shape == (2, 4, 7, 16)
    v = torch.randn(1, 2, 5, 8)
    assert torch.allclose(rope_rotate(v).norm(dim=-1), v.norm(dim=-1), atol=1e-5)  # pure rotation
    out, attn = MHA(32, 4, rope=True)(torch.randn(2, 5, 32))
    assert out.shape == (2, 5, 32) and attn.shape == (2, 4, 5, 5)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 4, 5), atol=1e-5)
    m = causal_mask(6)
    y, attn = Block(32, 4, rope=True)(torch.randn(2, 6, 32), mask=m)
    assert y.shape == (2, 6, 32) and attn[:, :, ~m].abs().max() == 0
    y, sa, ca = CrossBlock(32, 4, rope=True)(
        torch.randn(2, 5, 32), torch.randn(2, 9, 32), self_mask=causal_mask(5)
    )
    assert y.shape == (2, 5, 32) and sa.shape == (2, 4, 5, 5) and ca.shape == (2, 4, 5, 9)


def test_mha_learns_lookup():
    # Fused-token lookup: the answer for each position is the id stored at another position.
    torch.manual_seed(0)
    emb, mha, head = nn.Embedding(8, 32), Block(32, 4), nn.Linear(32, 8)
    params = [*emb.parameters(), *mha.parameters(), *head.parameters()]
    opt = torch.optim.Adam(params, lr=3e-3)
    tok = torch.randint(0, 8, (16, 6))
    target = tok.flip(1)  # position i must output the token from position 5-i
    pos = nn.Parameter(torch.randn(6, 32) * 0.1)
    opt.add_param_group({"params": [pos]})
    for _ in range(200):
        y, _ = mha(emb(tok) + pos)
        loss = nn.functional.cross_entropy(head(y).reshape(-1, 8), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
