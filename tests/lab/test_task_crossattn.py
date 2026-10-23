"""Task "cross-attn" (ladder 8): date-format translation with an encoder-decoder."""

import json

import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task_crossattn as tc

SHORT = {"d_model": 64, "heads": 4, "steps": 500, "lr": 2e-3, "seed": 0}


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "cross-attn", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("xattn") / "run", SHORT)
    tc.train(ctx)
    return ctx


def test_spec_shape():
    spec = registry.get("cross-attn")
    assert spec.order == 8 and spec.live == "attnlive" and spec.vecgrids is True
    assert spec.viewer == "xattn"
    names = [p.name for p in spec.params + spec.eval_params]
    assert len(names) == len(set(names))
    assert {"d_model", "heads", "lr", "steps", "seed"} == {p.name for p in spec.params}
    assert {"eval_seed", "n_examples"} == {p.name for p in spec.eval_params}
    assert [c["id"] for c in spec.charts] == ["loss", "exact"] and spec.charts[0]["log_y"]
    assert spec.estimate({p.name: p.default for p in spec.params}) > 0


def test_generator_formats_and_range():
    pairs = tc.date_pairs("train", 600, 0)
    shapes = set()
    for src, tgt, _ in pairs:
        assert len(tgt) == 10 and tgt[4] == tgt[7] == "-" and 1970 <= int(tgt[:4]) <= 2070
        shapes.add("/" if "/" in src else "," if "," in src else " ")
    assert shapes == {"/", ",", " "}
    assert tc.format_pair(tc.PROBE_DATE, 0) == ("23 Oct 2026", "2026-10-23")
    assert tc.format_pair(tc.PROBE_DATE, 1)[0] == "Oct 23, 2026"
    assert tc.format_pair(tc.PROBE_DATE, 2)[0] == "23/10/2026"


def test_generator_split_disjoint():
    train, held = map(set, tc.split_days())
    assert train and held and not (train & held)
    assert len(train | held) == tc.LAST - tc.FIRST + 1
    assert 0.07 < len(held) / (len(train) + len(held)) < 0.13
    assert {d for _, _, d in tc.date_pairs("train", 800, 1)} <= train
    assert {d for _, _, d in tc.date_pairs("heldout", 800, 1)} <= held
    # the training tables the model sees contain no held-out day
    gen = torch.Generator().manual_seed(0)
    src, tgt = tc.make_batch("train", 512, gen, "cpu")
    heldout_targets = {tc.format_pair(d, 0)[1] for d in held}
    assert not any(tc.decode_ids(t.tolist()) in heldout_targets for t in tgt)


def test_greedy_decode_autoregressive(monkeypatch):
    torch.manual_seed(0)
    model = tc.build({"d_model": 32, "heads": 2}).eval()
    seen, forced = [], []
    real = model.decode

    def spy(dec_in, memory):
        seen.append(dec_in.clone())
        logits, sa, ca = real(dec_in, memory)
        nxt = logits[:, -1].argmax(-1)
        forced.append(nxt.clone())
        return logits, sa, ca

    monkeypatch.setattr(model, "decode", spy)
    out = tc.greedy_decode(model, "23 Oct 2026")  # called with the source string only
    assert isinstance(out, str) and seen
    for t, inp in enumerate(seen):
        assert inp.shape == (1, t + 1) and int(inp[0, 0]) == tc.BOS
        # every later input is the model's own previous argmax, never a target char
        assert [int(x) for x in inp[0, 1:]] == [int(f[0]) for f in forced[:t]]
    # untrained model: it stops either at EOS or at the length cap, never beyond
    assert len(seen) <= tc.MAX_OUT
    if tc.EOS in [int(f[0]) for f in forced]:
        assert len(seen) == [int(f[0]) for f in forced].index(tc.EOS) + 1


def test_train_reaches_target(trained):
    assert read_status(trained.dir)["state"] == "done"
    rows = [r for r in read_metrics(trained.dir) if "exact" in r]
    assert 10 <= len(rows) <= 25
    assert rows[-1]["exact"] >= 0.9 and rows[0]["exact"] < rows[-1]["exact"]
    lines = (trained.dir / "metrics.jsonl").read_text().splitlines()
    assert all(len(line) < 8192 for line in lines)
    snaps = [json.loads(line) for line in lines if '"attn"' in line]
    assert 40 <= len(snaps) <= 80
    a = snaps[0]["attn"]
    assert len(a) == 4 and len(a[0]) == 11 and len(a[0][0]) == tc.SRC_LEN
    assert len(snaps[0]["qkv"]["labels"]) == tc.SRC_LEN


def test_eval_roundtrip(tmp_path, trained):
    ck = torch.load(trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert {"d_model", "heads"} <= set(ck["meta"])
    ev = make_ctx(
        tmp_path / "eval",
        {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "eval_seed": 3, "n_examples": 5},
    )
    tc.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["heads"] == 4 and res["exact"] >= 0.9 and len(res["examples"]) == 5
    for e in res["examples"]:
        assert len(e["target"]) == 10 and e["correct"] == (e["decoded"] == e["target"])
        L = len(e["out_labels"])
        assert len(e["src_labels"]) == tc.SRC_LEN and len(e["in_labels"]) == L
        assert e["in_labels"][0] == "<bos>"
        if e["correct"]:
            assert L == 11 and e["out_labels"][-1] == "<eos>"
        assert len(e["cross_attn"]) == 4 and len(e["cross_attn"][0]) == L
        assert len(e["cross_attn"][0][0]) == tc.SRC_LEN
        assert len(e["dec_attn"][0]) == L and len(e["dec_attn"][0][0]) == L
        assert len(e["enc_attn"][0]) == tc.SRC_LEN
        dk = tc.build(ck["meta"]).dec.cross_attn.dh
        vq, vk, vv = (e["vec"][n] for n in "qkv")
        assert len(vq) == 4 and len(vq[0]) == L and len(vq[0][0]) == dk
        assert len(vk) == 4 and len(vk[0]) == tc.SRC_LEN and len(vk[0][0]) == dk
        assert len(vv[0]) == tc.SRC_LEN
        for row in e["cross_attn"][0]:
            assert abs(sum(row) - 1.0) < 0.02
        for i, row in enumerate(e["dec_attn"][0]):
            assert all(v == 0 for v in row[i + 1 :])  # causal
    # the shown examples are the held-out days drawn with the documented seed
    shown = {t for _, t, _ in tc.date_pairs("heldout", 5, 5003)}
    assert {e["target"] for e in res["examples"]} == shown
