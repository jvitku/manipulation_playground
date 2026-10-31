"""Task "tokenize" (ladder 14): three action interfaces (cont / bins / ar) on one tiny problem."""

import json

import pytest
import torch

from lab.core.jobs import read_metrics, read_status
from lab.core.runctx import RunContext
from lab.tasks import registry, task_crossattn
from lab.tasks import task_token as tt

TINY = {"steps": 800, "lr": 1e-3, "batch": 128, "n_bins": 9, "d_model": 16, "seed": 0}


def make_ctx(d, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": "tokenize", "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("token") / "run", TINY)
    tt.train(ctx)
    return ctx


@pytest.fixture(scope="module")
def result(trained, tmp_path_factory):
    ev = make_ctx(
        tmp_path_factory.mktemp("token_eval") / "eval",
        {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "eval_seed": 0, "eval_n": 1500,
         "temperature": 1.0},
    )  # fmt: skip
    tt.evaluate(ev)
    return json.loads((ev.dir / "result.json").read_text())


def test_spec_shape():
    spec = registry.get("tokenize")
    assert spec.order == 14 and spec.viewer == "tokens" and spec.vecgrids is True
    assert spec.lesson == "tokenization.html"
    names = [p.name for p in spec.params + spec.eval_params]
    assert len(names) == len(set(names))
    assert {"steps", "lr", "batch", "n_bins", "d_model", "seed"} == {p.name for p in spec.params}
    assert {"eval_seed", "eval_n", "temperature"} == {p.name for p in spec.eval_params}
    assert [c["id"] for c in spec.charts] == ["loss"]
    assert {s["key"] for s in spec.charts[0]["series"]} == {"cont_loss", "bins_loss", "ar_loss"}
    assert spec.charts[0]["log_y"] is True


def test_train_three_heads(trained):
    assert read_status(trained.dir)["state"] == "done"
    rows = read_metrics(trained.dir)
    steps = [r["step"] for r in rows]
    assert steps == sorted(steps)  # one monotonic stream across the three heads
    for h in tt.HEADS:
        own = [r[f"{h}_loss"] for r in rows if f"{h}_loss" in r]
        assert own, f"no {h}_loss rows"
        assert own[-1] < 3.0  # sane: CE starts near ln(n_bins)=2.2, MSE near 0.3
    # cont cannot beat the between-mode variance floor (~0.16); the token heads' CE keeps the
    # irreducible side bit (~ln 2 / 2) - sanity-check both floors held
    cont = [r["cont_loss"] for r in rows if "cont_loss" in r]
    assert 0.05 < cont[-1] < 0.6


def test_checkpoint_payload(trained):
    ck = torch.load(trained.dir / "ckpt" / "last.pt", weights_only=False)
    assert {"cont", "bins", "ar", "arch", "params"} <= set(ck)
    for h in tt.HEADS:
        assert {"state", "loss"} <= set(ck[h])
    assert ck["arch"]["n_bins"] == TINY["n_bins"] and ck["arch"]["d_model"] == TINY["d_model"]
    assert ck["params"]["steps"] == TINY["steps"]
    # evaluate rebuilds from arch alone
    m = tt._load_head(ck, "ar", "cpu")
    assert m.emb.weight.shape == (TINY["n_bins"], TINY["d_model"])
    index = json.loads((trained.dir / "ckpt" / "index.json").read_text())
    assert index[-1]["label"] == "final" and "ar_loss" in index[-1]["metrics"]


def test_eval_result_schema(result):
    n_bins, d_model = TINY["n_bins"], TINY["d_model"]
    assert {"n_bins", "d_model", "temperature", "bin_edges", "bin_centers", "heads",
            "probe_samples", "embedding", "expert_example", "text_demo"} == set(result)  # fmt: skip
    assert result["n_bins"] == n_bins and result["d_model"] == d_model
    assert len(result["bin_edges"]) == n_bins + 1 and len(result["bin_centers"]) == n_bins
    assert result["bin_edges"][0] == -1.0 and result["bin_edges"][-1] == 1.0
    assert set(result["heads"]) == set(tt.HEADS)
    for h in tt.HEADS:
        d = result["heads"][h]
        for k in ("collision", "pair_valid", "left_frac"):
            assert 0.0 <= d[k] <= 1.0
        assert d["mode_err"] >= 0.0
        assert 0 < len(d["points"]) <= 500
        assert all(len(p) == 3 for p in d["points"])
        assert all(abs(p[1]) <= 1.0 and abs(p[2]) <= 1.0 for p in d["points"])
    assert "probe" not in result["heads"]["cont"]


def test_eval_probes(result):
    n_bins = TINY["n_bins"]
    pb = result["heads"]["bins"]["probe"]
    assert pb["c"] == 0.0
    for key in ("px", "pz"):
        assert len(pb[key]) == n_bins and abs(sum(pb[key]) - 1.0) < 1e-2
    pa = result["heads"]["ar"]["probe"]
    assert len(pa["px"]) == n_bins and abs(sum(pa["px"]) - 1.0) < 1e-2
    for side in ("left", "right"):
        assert len(pa["pz_given"][side]) == n_bins
        assert abs(sum(pa["pz_given"][side]) - 1.0) < 1e-2
    assert pa["tok_left"] in range(n_bins) and pa["tok_right"] in range(n_bins)
    assert pa["tok_left"] < pa["tok_right"]  # left lobe is left of the right lobe
    assert set(result["probe_samples"]) == set(tt.HEADS)
    for h in tt.HEADS:
        s = result["probe_samples"][h]
        assert 0 < len(s) <= 200 and all(len(p) == 2 for p in s)


def test_eval_extras(result):
    n_bins, d_model = TINY["n_bins"], TINY["d_model"]
    emb = result["embedding"]
    assert emb["d"] == d_model and len(emb["table"]) == n_bins
    assert all(len(row) == d_model for row in emb["table"])
    ex = result["expert_example"]
    assert len(ex["a"]) == 2 and len(ex["tok"]) == 2
    assert all(isinstance(t, int) and 0 <= t < n_bins for t in ex["tok"])
    td = result["text_demo"]
    assert td["sample"] == "23 Oct 2026"
    assert td["vocab"] == "".join(task_crossattn.CHARS) and len(td["vocab"]) > 10


def test_heads_tell_the_story(result):
    """cont averages the modes (collision), bins mix the sides, ar keeps the pair valid."""
    cont, bins, ar = (result["heads"][h] for h in tt.HEADS)
    assert cont["collision"] >= 0.8  # the regression head drives into the obstacle
    assert ar["collision"] <= 0.3
    assert ar["pair_valid"] >= 0.7
    assert bins["pair_valid"] <= 0.85  # independent dims mix the sides about half the time
    assert ar["mode_err"] < cont["mode_err"]


def test_argmax_temperature(trained, tmp_path):
    ev = make_ctx(
        tmp_path / "eval",
        {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "eval_seed": 1, "eval_n": 600,
         "temperature": 0.0},
    )  # fmt: skip
    tt.evaluate(ev)
    res = json.loads((ev.dir / "result.json").read_text())
    # argmax decoding is deterministic: every probe sample of a token head is the same action
    for h in ("bins", "ar"):
        rows = {tuple(p) for p in res["probe_samples"][h]}
        assert len(rows) == 1


def test_stop_is_stopped(tmp_path):
    ctx = make_ctx(tmp_path / "run", {**TINY, "steps": 500})
    ctx._stop = True
    tt.train(ctx)
    assert read_status(ctx.dir)["state"] == "stopped"


def test_estimate_uses_calib(monkeypatch):
    spec = registry.get("tokenize")
    p = registry.validate_params(spec, {}, which="train")
    assert spec.estimate(p) > 0
    monkeypatch.setattr(
        registry, "calib", lambda tid: {"steps_per_s": 1.0, "overhead_s": 0.0}, raising=True
    )
    assert spec.estimate({**p, "steps": 100}) == pytest.approx(300.0)  # three heads
