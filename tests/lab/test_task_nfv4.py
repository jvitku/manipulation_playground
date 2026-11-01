"""Task "force-rl" (ladder 19): parallel residual-RL seeds on a frozen nano base, the shared-seed
evaluation, M4 claims. Tiny CPU budget: a random-init 1-layer base, 2 seeds x 300 env steps."""

import json
from dataclasses import asdict

import pytest
import torch

from lab.core.jobs import read_metrics
from lab.core.runctx import RunContext
from lab.tasks import registry
from lab.tasks import task_nfv4 as tn
from nfv.claims import M4_CLAIMS, NOT_MEASURED, VERDICTS
from nfv.model import ModelConfig
from nfv.model.config import build

ALLOWED = {*VERDICTS, NOT_MEASURED}
TINY = {"task": "t3", "seeds": 2, "seed": 0, "env_steps": 300, "bc_floor": 1.0,
        "peak_force_penalty": 0.0, "dp_res_mm": 2.0, "residual_force": False,
        "eval_every": 150, "n_eval": 1}  # fmt: skip


def make_ctx(d, task, params):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"task": task, "params": params}))
    return RunContext(d)


@pytest.fixture(scope="module")
def base_ckpt(tmp_path_factory):
    torch.manual_seed(0)
    cfg = ModelConfig(trunk_layers=1, expert_layers=1, ffn=256, euler_steps=2)
    pol = build(cfg)
    path = tmp_path_factory.mktemp("nfv4_base") / "last.pt"
    torch.save({"model": pol.net.state_dict(), "ema": pol.ema.state_dict(), "norm": {},
                "arch": asdict(cfg), "params": {"task": "t3"}, "gen": "flow"}, path)  # fmt: skip
    return path


@pytest.fixture(scope="module")
def trained(base_ckpt, tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("nfv4_run") / "run", "force-rl",
                   {**TINY, "base_ckpt": str(base_ckpt)})  # fmt: skip
    tn.train(ctx)
    return ctx


@pytest.fixture(scope="module")
def result(trained, tmp_path_factory):
    ev = make_ctx(tmp_path_factory.mktemp("nfv4_eval") / "eval", "force-rl",
                  {"ckpt": str(trained.dir / "ckpt" / "last.pt"), "episodes": 2,
                   "eval_seed": 0, "video": True})  # fmt: skip
    tn.evaluate(ev)
    return ev, json.loads((ev.dir / "result.json").read_text())


def test_spec_shape():
    s = registry.get("force-rl")
    assert s.order == 19 and s.viewer == "fvrollout" and s.lesson == "force-rl.html"
    names = {p.name for p in s.params}
    assert {"base_ckpt", "task", "seeds", "env_steps", "bc_floor", "peak_force_penalty",
            "residual_force"} <= names  # fmt: skip
    bc = next(p for p in s.params if p.name == "base_ckpt")
    assert bc.kind == "ckpt" and bc.ckpt_task == "nano-vla"
    assert next(p for p in s.params if p.name == "task").default == "t3"
    with pytest.raises(ValueError):
        s.check({**TINY, "bc_floor": 0.0})
    assert s.estimate({**TINY, "env_steps": 20000, "seeds": 5}) > 60


def test_train_parallel_seeds_one_checkpoint(trained):
    ck = torch.load(trained.dir / "ckpt" / "last.pt", map_location="cpu", weights_only=False)
    assert ck["kind"] == "nfv-residual-suite" and ck["seeds"] == [0, 1] and ck["task"] == "t3"
    assert set(ck["residuals"]) == {"0", "1"} and "actor" in ck["residuals"]["0"]
    assert "base" not in ck["residuals"]["0"]  # the base is stored once, at the top
    assert {"model", "ema", "arch"} <= set(ck["base"])
    for s in ("0", "1"):
        assert ck["train_curves"][s]["env_step"] == [0, 150, 300]


def test_train_metrics_per_seed_and_median(trained):
    rows = read_metrics(trained.dir)
    keys = {k for r in rows for k in r}
    assert {"s0_eval_success", "s1_eval_success", "eval_success_median"} <= keys
    # 300 steps < the 1000-step warm-up: no updates yet, but the residual magnitude streams
    assert {"s0_res_mm_ma", "s1_bc_weight", "res_mm_free_mean"} <= keys
    med_steps = sorted(r["step"] for r in rows if "eval_success_median" in r)
    assert med_steps == [0, 150, 300]


def test_eval_result_schema(result):
    ev, r = result
    assert r["task"] == "t3" and r["episodes"] == 2
    assert r["base"]["n"] == 2 and [x["seed"] for x in r["seeds"]] == [0, 1]
    for row in r["seeds"]:
        assert {"success", "wilson95", "delta", "peak_force_mean", "res_mm_contact",
                "res_mm_free", "success_cond"} <= set(row)  # fmt: skip
    assert r["median"]["seed"] in (0, 1)
    assert set(r["residual_trace"]) >= {"contact_mm", "free_mm", "trace"}
    assert "frozen base" in r["outcomes"]["per_task"]
    assert r["video"] == "rollout.mp4" and (ev.dir / "rollout.mp4").stat().st_size > 1000
    tr = r["episodes_traces"]
    assert len(tr) >= 2 and tr[0]["group"] == "frozen base" and tr[0]["seed"] == tr[1]["seed"]
    for t in tr:
        assert len(t["t"]) == len(t["fz"]) == len(t["mode"]) == len(t["res_mm"]) > 0


def test_eval_claims_rows(result):
    _, r = result
    ids = [c["id"] for c in r["claims"]]
    assert ids == [c.id for c in M4_CLAIMS]
    assert all(c["verdict"] in ALLOWED for c in r["claims"])
    spread = next(c for c in r["claims"] if c["id"] == "m4.seed_spread_reported")
    assert spread["verdict"] == "NOT CONFIRMED" and "s0" in spread["measured"]  # 2 < 5 seeds
    imp = next(c for c in r["claims"] if c["id"] == "m4.residual_improves_t3")
    assert imp["verdict"] != NOT_MEASURED


# ---- the simulated deployment (interventions=expert + deploy_mode) --------------------------
def test_new_claims_not_measured_without_operator(result):
    _, r = result
    rows = {c["id"]: c for c in r["claims"]}
    for cid in ("m4.corrections_unlock_learning", "m4.autonomy_rises"):
        assert rows[cid]["verdict"] == NOT_MEASURED
    assert "operator" not in r


def test_deploy_spec_params_and_check():
    s = registry.get("force-rl")
    p = {x.name: x for x in s.params}
    assert p["interventions"].default == "off" and p["interventions"].choices == ["off", "expert"]
    assert p["intervention_budget"].default == 0.2 and p["intervention_decay"].default == 0.0
    assert p["deploy_mode"].default is False and p["swap_every"].default == 500
    charts = {c["id"] for c in s.charts}
    assert {"operator", "robot_time"} <= charts
    with pytest.raises(ValueError, match="t3 only"):
        s.check({**TINY, "task": "t1", "interventions": "expert"})
    s.check({**TINY, "interventions": "expert"})


def test_operator_cost_pooling():
    dep = {"0": {"robot_min": 1.0, "operator_steps": 60, "operator_min": 0.1,
                 "operator_share": 0.1, "first20_rate": 0.4, "last20_rate": 0.1,
                 "n_segments": 3, "triggers": {"stall": 2, "force": 1}},
           "1": {"robot_min": 1.0, "operator_steps": 40, "operator_min": 0.067,
                 "operator_share": 0.067, "first20_rate": 0.2, "last20_rate": 0.2,
                 "n_segments": 2, "triggers": {"stall": 2, "force": 0}}}  # fmt: skip
    op = tn.operator_cost(dep)
    assert op["operator_steps"] == 100 and op["robot_min"] == pytest.approx(2.0)
    assert op["first20_rate"] == pytest.approx(0.3) and op["last20_rate"] == pytest.approx(0.15)
    assert op["autonomy_ratio"] == pytest.approx(0.5)
    assert tn.operator_cost({}) is None and tn.operator_cost(None) is None


def test_deploy_train_and_eval(base_ckpt, tmp_path_factory):
    params = {**TINY, "base_ckpt": str(base_ckpt), "interventions": "expert",
              "intervention_budget": 0.3, "deploy_mode": True, "swap_every": 50}  # fmt: skip
    run = make_ctx(tmp_path_factory.mktemp("nfv4_dep") / "run", "force-rl", params)
    tn.train(run)
    ck = torch.load(run.dir / "ckpt" / "last.pt", map_location="cpu", weights_only=False)
    assert set(ck["deploy"]) == {"0", "1"}
    for d in ck["deploy"].values():
        assert {"operator_steps", "operator_min", "robot_min", "first20_rate", "last20_rate",
                "curve", "swaps"} <= set(d)  # fmt: skip
        assert "episodes" not in d  # the per-episode log stays in the seed's result.json
    keys = {k for row in read_metrics(run.dir) for k in row}
    assert {"s0_robot_min", "s0_operator_min", "s1_operator_steps"} <= keys
    ev = make_ctx(tmp_path_factory.mktemp("nfv4_dep_eval") / "eval", "force-rl",
                  {"ckpt": str(run.dir / "ckpt" / "last.pt"), "episodes": 2, "eval_seed": 0,
                   "video": False})  # fmt: skip
    tn.evaluate(ev)
    r = json.loads((ev.dir / "result.json").read_text())
    assert r["operator"]["operator_steps"] == sum(d["operator_steps"]
                                                  for d in ck["deploy"].values())  # fmt: skip
    rows = {c["id"]: c for c in r["claims"]}
    assert [c["id"] for c in r["claims"]] == [c.id for c in M4_CLAIMS]
    unlock = rows["m4.corrections_unlock_learning"]
    assert unlock["verdict"] in VERDICTS and "operator" in unlock["measured"]
    assert rows["m4.autonomy_rises"]["verdict"] in ALLOWED
