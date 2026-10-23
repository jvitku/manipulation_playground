"""Task "vit-mnist" (ladder 7): a mini ViT on MNIST with live CLS attention overlays."""

import json

import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task_vit as tv

# shortened config: 7x7 patches, 1 layer, 2 heads, 1200 steps on the 10000-image subset
SHORT = {"patch": 7, "depth": 1, "heads": 2, "d_model": 64, "lr": 2e-3, "steps": 1200,
         "train_n": 10000, "seed": 0}  # fmt: skip
BIGGEST = {"patch": 7, "depth": 2, "heads": 4, "d_model": 64, "lr": 2e-3, "steps": 20,
           "train_n": 2000, "seed": 0}  # fmt: skip


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "vit-mnist", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("vit") / "run", SHORT)
    tv.train(ctx)
    return ctx


def test_spec_shape():
    spec = registry.get("vit-mnist")
    assert spec.order == 7 and spec.live == "attnlive" and spec.vecgrids is True
    assert spec.viewer == "vitattn"
    names = [p.name for p in spec.params + spec.eval_params]
    assert len(names) == len(set(names))
    assert {"patch", "depth", "heads", "d_model", "lr", "steps", "train_n", "seed"} == {
        p.name for p in spec.params
    }
    assert {"eval_seed", "n_examples"} == {p.name for p in spec.eval_params}
    d = {p.name: p.default for p in spec.params}
    assert d["depth"] == 2 and d["heads"] == 2 and d["d_model"] == 64 and d["train_n"] == 10000
    assert spec.estimate({p.name: p.default for p in spec.params}) > 0


def test_patchify_roundtrip():
    imgs = torch.rand(3, 28, 28)
    for p in (7, 14):
        tok = tv.patchify(imgs, p)
        g = 28 // p
        assert tok.shape == (3, g * g, p * p)
        back = tok.reshape(3, g, g, p, p).permute(0, 1, 3, 2, 4).reshape(3, 28, 28)
        assert torch.equal(back, imgs)
        # the first token is the top-left patch, the second the one to its right
        assert torch.equal(tok[:, 0].reshape(3, p, p), imgs[:, :p, :p])
        assert torch.equal(tok[:, 1].reshape(3, p, p), imgs[:, :p, p : 2 * p])


def test_train_reaches_target(trained):
    assert read_status(trained.dir)["state"] == "done"
    rows = [r for r in read_metrics(trained.dir) if "val_acc" in r]
    assert 10 <= len(rows) <= 20
    assert rows[-1]["val_acc"] >= 0.85
    assert rows[0]["val_acc"] < rows[-1]["val_acc"]


def test_metrics_rows_small(trained):
    lines = (trained.dir / "metrics.jsonl").read_text().splitlines()
    assert all(len(line) < 8192 for line in lines)
    snaps = [json.loads(line) for line in lines if '"attn"' in line]
    assert 40 <= len(snaps) <= 80
    s = snaps[0]
    assert len(s["attn"]) == SHORT["depth"] * SHORT["heads"] and s["grid"] == [4, 4]
    assert len(s["attn"][0][0]) == 4 * 4 + 1  # CLS + patches
    assert len(s["img"]) == 28 and len(s["img"][0]) == 28


def test_row_cap_at_largest_config(tmp_path):
    ctx = make_ctx(tmp_path / "run", BIGGEST)
    tv.train(ctx)
    lines = (ctx.dir / "metrics.jsonl").read_text().splitlines()
    assert lines and max(len(line) for line in lines) < 8192
    assert any('"attn"' in line for line in lines)


def test_eval_roundtrip(tmp_path, trained):
    ck = torch.load(trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert {"patch", "depth", "heads", "d_model"} <= set(ck["meta"])
    ev = make_ctx(
        tmp_path / "eval",
        {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "eval_seed": 3, "n_examples": 5},
    )
    tv.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["patch"] == 7 and res["grid"] == [4, 4] and 0.85 <= res["acc"] <= 1.0
    assert res["depth"] == 1 and res["heads"] == 2
    assert len(res["examples"]) == 5
    for e in res["examples"]:
        assert len(e["pixels"]) == 28 and len(e["pixels"][0]) == 28
        assert e["label"] in range(10) and e["pred"] in range(10)
        assert 0.0 <= e["conf"] <= 1.0
        a = e["attn"]
        assert len(a) == 1 and len(a[0]) == 2 and len(a[0][0]) == 4 and len(a[0][0][0]) == 4
        assert 0.0 < sum(sum(sum(r) for r in hd) for hd in a[0]) <= 2.0 + 1e-3
