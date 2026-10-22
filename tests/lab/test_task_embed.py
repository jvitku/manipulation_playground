"""Task C3 (embeddings): four conditioning encodings trained sequentially in one run."""

import json

import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task_embed as te

TINY = {
    "demos": 40, "steps": 60, "lr": 1e-3, "batch": 32, "horizon": 16, "n_exec": 8,
    "d_embed": 8, "encodings": "all", "sample_steps": 2, "eval_every": 60,
}  # fmt: skip
ARMS = ("onehot", "binary", "text", "cont")


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "embeddings", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("emb") / "run", TINY)
    te.train(ctx)
    return ctx


def test_embed_registered_and_estimate():
    spec = registry.get("embeddings")
    assert spec.order == 3 and spec.viewer == "embeddings"
    assert registry.get("flow-bc").order == 4
    p = registry.validate_params(spec, {}, which="train")
    one = registry.validate_params(spec, {"encodings": "binary"}, which="train")
    assert spec.estimate(p) > spec.estimate(one) > 0


def test_embed_train_tiny(trained):
    assert read_status(trained.dir)["state"] == "done"
    rows = read_metrics(trained.dir)
    keys = {k for r in rows for k in r}
    for a in ARMS:
        assert {f"{a}_loss", f"{a}_success"} <= keys
    steps = [r["step"] for r in rows]
    assert steps == sorted(steps)  # one monotonic stream across arms
    ck = torch.load(trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert ck["arms"] == list(ARMS) and ck["gen"] == "flow"
    for a in ARMS:
        assert {"model", "ema", "norm", "arch", "gen", "encoding"} <= set(ck[a])
        assert ck[a]["encoding"] == a
    assert any(k.startswith("emb.") for k in ck["text"]["ema"])


def test_embed_single_encoding(tmp_path):
    ctx = make_ctx(tmp_path / "run", {**TINY, "encodings": "binary"})
    te.train(ctx)
    keys = {k for r in read_metrics(ctx.dir) for k in r}
    assert "binary_loss" in keys and not any(k.startswith("onehot_") for k in keys)
    ck = torch.load(ctx.dir / "ckpt" / "last.pt", weights_only=False)
    assert ck["arms"] == ["binary"]


def test_embed_eval_vectors_present(tmp_path, trained):
    ev = make_ctx(
        tmp_path / "eval",
        {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "n_episodes": 24, "eval_steps": 2,
         "encoding": "text"},
    )  # fmt: skip
    te.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    assert res["encoding"] == "text" and sum(res["counts"].values()) == 24
    assert len(res["commands"]) == 4 and len(res["wilson95"]) == 2
    enc = res["encodings"]
    assert enc["arms"] == list(ARMS) and len(enc["commands"]) == 4
    for a in ARMS:
        assert len(enc["vectors"][a]) == 4
        assert len(enc["vectors"][a][0]) == len(enc["labels"][a])
    assert len(enc["vectors"]["onehot"][0]) == 4 and len(enc["vectors"]["text"][0]) == 8
    for e in res["episodes"]:
        assert isinstance(e["task"], str) and len(e["enc"]) == 8
        assert len(e["act"]) == len(e["pos"]) - 1 and len(e["holes"]) == 2


def test_embed_eval_unknown_encoding_and_filter(tmp_path):
    tr = make_ctx(tmp_path / "run", {**TINY, "encodings": "cont"})
    te.train(tr)
    ck = str(tr.dir / "ckpt" / "last.pt")
    bad = make_ctx(tmp_path / "bad", {"ckpt": ck, "encoding": "onehot", "n_episodes": 8})
    te.evaluate(bad)
    st = read_status(bad.dir)
    assert st["state"] == "failed" and "onehot" in st["msg"]
    ok = make_ctx(tmp_path / "ok", {"ckpt": ck, "tasks": "press", "n_episodes": 12,
                                    "eval_steps": 2})  # fmt: skip
    te.evaluate(ok)
    res = json.loads((ok.dir / "result.json").read_text())
    assert res["encoding"] == "cont"  # "best" resolves to the only arm
    assert all(e["task"].startswith("press") for e in res["episodes"])
