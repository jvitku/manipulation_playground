"""Task "find-digit" (ladder 9): a query token reads a two-digit canvas by cross-attention."""

import json

import numpy as np
import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry, task_vit
from lab.tasks import task_finddigit as fd
from lab.tasks.mnist_data import load_mnist

# shortened config: 1 block, 2 heads, 1500 steps on the 10000-image subset
SHORT = {"depth": 1, "heads": 2, "d_model": 64, "lr": 2e-3, "steps": 1500,
         "train_n": 10000, "seed": 0}  # fmt: skip
BIGGEST = {"depth": 2, "heads": 4, "d_model": 64, "lr": 2e-3, "steps": 20,
           "train_n": 2000, "seed": 0}  # fmt: skip


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "find-digit", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("fd") / "run", SHORT)
    fd.train(ctx)
    return ctx


def test_generator_invariants():
    x, y = load_mnist("test")
    rng = np.random.default_rng(0)
    canvas, query, truth, classes = fd.make_canvases(x, y, 2000, rng)
    assert canvas.shape == (2000, 28, 56) and canvas.dtype == np.uint8
    assert (classes[:, 0] != classes[:, 1]).all()  # the two digits are of distinct classes
    left, right = classes[:, 0], classes[:, 1]
    want = np.where(query == left, 0, np.where(query == right, 1, 2))
    assert (truth == want).all()  # the label is the construction
    assert 0.25 <= (truth == 2).mean() <= 0.40
    assert set(np.unique(query)) <= set(range(10))
    # the canvas really is the two chosen digits side by side: left half has the left digit's ink
    assert (canvas[:, :, :28].sum((1, 2)) > 0).all() and (canvas[:, :, 28:].sum((1, 2)) > 0).all()
    # a fixed query and forced truths are honoured
    cv, q, t, cl = fd.make_canvases(x, y, 300, rng, query=4, truth=np.arange(300) % 3)
    assert (q == 4).all() and (t == np.arange(300) % 3).all()
    assert (cl[t == 0, 0] == 4).all() and (cl[t == 1, 1] == 4).all()
    assert (cl[t == 2] != 4).all() and (cl[:, 0] != cl[:, 1]).all()


def test_noise_param_applies():
    x, y = load_mnist("test")
    canvas = fd.make_canvases(x, y, 50, np.random.default_rng(1))[0]
    same = fd.add_noise(canvas, 0.0, np.random.default_rng(2))
    assert np.array_equal(same, canvas)
    noisy = fd.add_noise(canvas, 0.3, np.random.default_rng(2))
    assert noisy.dtype == np.uint8 and noisy.shape == canvas.shape
    assert (noisy != canvas).mean() > 0.5 and noisy.min() >= 0 and noisy.max() <= 255
    assert abs(float(noisy.astype(float).mean()) - float(canvas.astype(float).mean())) < 80


def test_patchify_non_square():
    imgs = torch.rand(2, 28, 56)
    tok = task_vit.patchify(imgs, 7)
    assert tok.shape == (2, 32, 49)
    assert torch.equal(tok[:, 7].reshape(2, 7, 7), imgs[:, :7, 49:56])  # row 0, column 7
    assert torch.equal(tok[:, 8].reshape(2, 7, 7), imgs[:, 7:14, :7])  # row 1, column 0


def test_spec_shape():
    spec = registry.get("find-digit")
    assert spec.order == 9 and spec.live == "fdlive" and spec.vecgrids is True
    assert spec.viewer == "finddigit"
    names = [p.name for p in spec.params + spec.eval_params]
    assert len(names) == len(set(names))
    assert {"depth", "heads", "d_model", "lr", "steps", "train_n", "seed"} == {
        p.name for p in spec.params
    }
    assert {"eval_seed", "n_examples", "noise", "query"} == {p.name for p in spec.eval_params}
    d = {p.name: p.default for p in spec.params}
    assert d["depth"] == 2 and d["heads"] == 2 and d["d_model"] == 64
    ev = {p.name: p for p in spec.eval_params}
    assert ev["noise"].default == 0 and ev["noise"].max == 0.5 and ev["n_examples"].default == 8
    assert ev["query"].default == "random" and "random" in ev["query"].choices
    assert [c for c in ev["query"].choices if c != "random"] == [str(i) for i in range(10)]
    assert {c["id"] for c in spec.charts} == {"loss", "val_acc", "absent_acc"}
    assert spec.estimate({p.name: p.default for p in spec.params}) > 0


def test_train_reaches_target(trained):
    assert read_status(trained.dir)["state"] == "done"
    rows = [r for r in read_metrics(trained.dir) if "val_acc" in r]
    assert 10 <= len(rows) <= 20
    assert all("absent_acc" in r for r in rows)
    assert rows[-1]["val_acc"] >= 0.8
    assert rows[0]["val_acc"] < rows[-1]["val_acc"]


def test_metrics_rows_small(trained):
    lines = (trained.dir / "metrics.jsonl").read_text().splitlines()
    assert all(len(line) < 8192 for line in lines)
    snaps = [json.loads(line) for line in lines if '"attn"' in line]
    assert 40 <= len(snaps) <= 80
    s = snaps[0]
    assert len(s["attn"]) == SHORT["heads"] and len(s["attn"][0]) == 1
    assert len(s["attn"][0][0]) == 32 and s["grid"] == [4, 8]  # no CLS slot: patches only
    assert len(s["img"]) == 28 and len(s["img"][0]) == 56 and 0 <= s["query"] <= 9


def test_row_cap_at_largest_config(tmp_path):
    ctx = make_ctx(tmp_path / "run", BIGGEST)
    fd.train(ctx)
    lines = (ctx.dir / "metrics.jsonl").read_text().splitlines()
    assert lines and max(len(line) for line in lines) < 8192
    assert any('"attn"' in line for line in lines)


def test_eval_roundtrip(tmp_path, trained):
    ck = torch.load(trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert {"depth", "heads", "d_model"} <= set(ck["meta"])
    ev = make_ctx(
        tmp_path / "eval",
        {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "eval_seed": 3, "n_examples": 6,
         "noise": 0, "query": "random"},
    )  # fmt: skip
    fd.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["grid"] == [4, 8] and res["heads"] == 2 and res["noise"] == 0
    assert 0.8 <= res["acc"] <= 1.0 and 0.5 <= res["absent_acc"] <= 1.0
    assert len(res["examples"]) == 6
    assert any(e["truth"] == 2 for e in res["examples"])  # an absent case is among them
    for e in res["examples"]:
        assert len(e["pixels"]) == 28 and len(e["pixels"][0]) == 56
        assert e["query"] in range(10) and e["truth"] in range(3) and e["pred"] in range(3)
        assert 0.0 <= e["conf"] <= 1.0
        a = e["attn"]
        assert len(a) == 2 and len(a[0]) == 4 and len(a[0][0]) == 8
        assert abs(sum(sum(r) for r in a[0]) - 1.0) < 1e-2  # a softmax row over the 32 patches
        assert len(e["vec"]["query"][0]) == 64 and len(e["vec"]["keys"]) == len(e["vec"]["patches"])
    # a fixed query digit and noise are honoured
    ev2 = make_ctx(
        tmp_path / "eval2",
        {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "eval_seed": 3, "n_examples": 6,
         "noise": 0.3, "query": "7"},
    )  # fmt: skip
    fd.evaluate(ev2)
    res2 = json.loads((ev2.dir / "result.json").read_text())
    assert all(e["query"] == 7 for e in res2["examples"]) and res2["noise"] == 0.3
    assert res2["acc"] <= res["acc"] + 0.02  # noise the model never saw does not help
