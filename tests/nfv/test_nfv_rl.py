"""Lesson 19 core (nfv.rl): residual TD3+BC on a frozen nano VLA — CPU, tiny, random-init base.

* step-0 equivalence: residual == 0 -> the executed actions ARE the base's, bit for bit, on a
  seeded T1 episode (compared against a plain base rollout, task_nfv1's executor pattern);
* the zero-start init (last layer x0.01) keeps the residual within 5 % of its bound;
* bounds: a saturated actor (+ exploration noise) never moves dp more than DP_RES from the base;
* BC schedule: 2.5 -> 1.0, never below the floor, floors < 1.0 refused;
* a 300-env-step smoke run completes, logs curves/train rows through a ctx, and splits the
  residual magnitude by contact / free space.
"""

import numpy as np
import pytest
import torch

from nfv.model import ModelConfig
from nfv.model.config import build
from nfv.rl import loop as rl_loop
from nfv.rl import residual as rr
from nfv.sim.env import Action, Env

N_CMP = 24  # steps compared in the equivalence test (3 chunks at n_exec 8)


def tiny_base(seed: int = 0):
    torch.manual_seed(seed)
    cfg = ModelConfig(trunk_layers=1, expert_layers=1, ffn=256, euler_steps=2)
    return build(cfg).eval()


@pytest.fixture(scope="module")
def base():
    return tiny_base()


def _base_rollout(policy, env, seed):
    """The base alone, executed like task_nfv1._episode; -> list of executed dp [3]."""
    obs = env.reset(seed, task="t1")
    gen = torch.Generator()
    gen.manual_seed(seed)
    out, k, done = [], 0, False
    while k < N_CMP and not done:
        chunk = policy.act(obs, env.instruction, generator=gen)
        for j in range(policy.cfg.n_exec):
            dp = chunk["dp"][j].astype(np.float64)
            obs, info = env.step(Action(dp=dp, gripper=float(chunk["gripper"][j])))
            out.append(dp.copy())
            k += 1
            done = info["done"]
            if done or k >= N_CMP:
                break
    return out


def _residual_rollout(policy, env, seed, agent, off=False):
    actor = rl_loop.ResidualActor(policy, agent)
    slot = rl_loop._Slot(env=env, task="t1")
    executed, residuals = [], []
    step = env.step

    def rec(a):
        executed.append(np.asarray(a.dp, np.float64).copy())
        return step(a)

    env.step = rec
    try:
        actor.start(slot, seed)
        info = {"done": False}
        while len(executed) < N_CMP and not info["done"]:
            r = actor.residual(slot.feat, 0.0, None, off=off)
            residuals.append(r)
            _, info, _ = actor.step(slot, r)
            if not info["done"]:
                slot.feat = actor._features(slot)
    finally:
        del env.step
    return executed, residuals


def test_step0_equivalence_bit_exact(base):
    env = Env(task="t1", render=True)
    seed = 50_123
    ref = _base_rollout(base, env, seed)
    zero = rr.make_agent(zero_last_layer=True, hidden=32)
    got, res = _residual_rollout(base, env, seed, zero)
    assert all(np.all(r == 0) for r in res)
    assert len(got) == len(ref) > 0
    for a, b in zip(got, ref, strict=True):
        assert np.array_equal(a, b)  # bit-exact, not allclose
    got_off, _ = _residual_rollout(base, env, seed, rr.make_agent(hidden=32), off=True)
    for a, b in zip(got_off, ref, strict=True):
        assert np.array_equal(a, b)
    # the default zero-start (x0.01) residual is tiny but not exactly zero
    _, res_init = _residual_rollout(base, env, seed, rr.make_agent(hidden=256))
    peak = max(float(np.abs(r).max()) for r in res_init)
    assert peak < 0.05, peak  # < 5 % of DP_RES = 0.1 mm per axis
    env.close()


def test_bounds_respected():
    torch.manual_seed(0)
    agent = rr.make_agent(hidden=32)
    with torch.no_grad():  # saturate: big last layer -> tanh at +-1
        agent.actor.net[-1].weight.normal_(0, 50.0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        f = rng.normal(0, 3, rr.feature_dim()).astype(np.float32)
        r = agent.act(f)
        r = np.clip(r + rng.normal(0, 0.5, r.shape), -1, 1)
        base_dp = rng.uniform(-0.05, 0.05, 3)
        dp, g, ft = rr.compose(base_dp, 0.3, None, r)
        assert np.all(np.abs(dp - base_dp) <= rr.DP_RES + 1e-12)
        assert g == 0.3 and ft is None
    # force residual: bounded by F_RES on xyz, torques untouched
    ft0 = np.array([0, 0, -5.0, 0.1, 0.2, 0.3])
    _, _, ft = rr.compose(np.zeros(3), 0.0, ft0, np.array([0, 0, 0, 1.0, -1.0, 1.0]), True)
    assert np.allclose(ft[:3] - ft0[:3], [rr.F_RES, -rr.F_RES, rr.F_RES])
    assert np.array_equal(ft[3:], ft0[3:])
    with pytest.raises(ValueError):
        rr.compose(np.zeros(3), 0.0, None, np.zeros(6), True)


def test_residual_force_refuses_position_base(base):
    env = Env(task="t1", render=True)
    actor = rl_loop.ResidualActor(base, rr.make_agent(True, hidden=32), residual_force=True)
    with pytest.raises(ValueError, match="hybrid-head"):
        actor.start(rl_loop._Slot(env=env, task="t1"), 50_000)
    env.close()


def test_bc_schedule_floor():
    decay = 1000
    ws = [rr.bc_weight_at(t, decay) for t in range(0, 5000, 7)]
    assert ws[0] == pytest.approx(2.5)
    assert min(ws) >= 1.0 and ws[-1] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(ws, ws[1:], strict=False))  # only decays
    with pytest.raises(ValueError):
        rr.bc_weight_at(0, decay, floor=0.0)
    with pytest.raises(ValueError):
        rr.bc_weight_at(0, decay, start=0.5, floor=1.0)


def test_reward_terms():
    r, term = rr.reward({"done": True, "success": True, "outcome": "success", "peak_force": 2})
    assert r == pytest.approx(0.99) and term
    r, term = rr.reward({"done": True, "success": False, "outcome": "over_force",
                         "force_abort": False, "peak_force": 9})  # fmt: skip
    assert r == pytest.approx(-1.01) and term
    r, term = rr.reward({"done": True, "success": False, "outcome": "timeout", "peak_force": 0})
    assert r == pytest.approx(-0.01) and not term  # timeout bootstraps
    r, _ = rr.reward({"done": False, "peak_force": 12.5}, 0.0, peak_force_penalty=1.0)
    assert r == pytest.approx(-0.01 - 0.5)


class _Ctx:
    def __init__(self):
        self.logs, self.status, self.ckpts = [], [], []
        self.params = {"note": "test"}

    def should_continue(self):
        return True

    def log(self, step, **series):
        self.logs.append((step, series))

    def set_status(self, **kw):
        self.status.append(kw)

    def save_ckpt(self, step, payload, metrics=None):
        self.ckpts.append((step, sorted(payload), metrics))


def test_smoke_300_steps(base):
    ctx = _Ctx()
    out = rl_loop.run_residual_rl(
        base, task="t1", seed=0, env_steps=300, ctx=ctx, n_envs=2, warmup=100,
        eval_every=150, n_eval=1, hidden=32, batch=32, log_every=50,
    )  # fmt: skip
    c = out["curves"]
    assert c["env_step"] == [0, 150, 300]
    assert len(c["success"]) == 3 and all(0 <= s <= 1 for s in c["success"])
    assert len(c["success_cond"]) == 3 and all(0 <= s <= 1 for s in c["success_cond"])
    assert out["config"]["torch_threads"] == 1
    assert out["env_steps"] == 300 and out["updates"] == pytest.approx(100, abs=2)  # UTD 0.5
    assert len(out["train"]) == 6 and all(r["bc_weight"] >= 1.0 for r in out["train"])
    assert "critic_loss" in out["train"][-1] and "bc_loss" in out["train"][-1]
    fe = out["final_eval"]
    assert fe["n_steps_contact"] + fe["n_steps_free"] == len(fe["trace"]["t"])
    assert set(out["residual_trace"]) == {"contact_mm", "free_mm", "trace"}
    assert out["base_eval"]["res_mm_free"] in (0.0, None)  # residual OFF at point 0
    assert any("eval_success" in s for _, s in ctx.logs)
    assert any("critic_loss" in s for _, s in ctx.logs)
    assert ctx.ckpts and "base" in ctx.ckpts[-1][1] and "actor" in ctx.ckpts[-1][1]
    print(f"smoke: {out['env_steps_per_s']} env steps/s (train, tiny base), wall {out['wall_s']} s")


def test_load_base_resolves_suite_arm(base):
    from dataclasses import asdict

    pl = {"model": base.net.state_dict(), "ema": base.ema.state_dict(),
          "arch": asdict(base.cfg), "task": "t1"}  # fmt: skip
    suite = {"arms": ["a", "b", "blank"], "a": pl, "b": {**pl, "task": "t0"},
             "blank": {**pl, "blank_text": True}}  # fmt: skip
    p, info = rl_loop.load_base(suite, "t1")
    assert info["arm"] == "a" and not any(q.requires_grad for q in p.parameters())
    assert rl_loop.load_base(suite, "t0", arm="b")[1]["arm"] == "b"
    with pytest.raises(ValueError):
        rl_loop.load_base(suite, "t1", arm="zz")
    with pytest.raises(ValueError):
        rl_loop.load_base(suite, "t1", arm="blank")
    with pytest.raises(ValueError, match="pick the base arm"):
        rl_loop.load_base({**suite, "arms": ["a", "c"], "c": pl}, "t1")
