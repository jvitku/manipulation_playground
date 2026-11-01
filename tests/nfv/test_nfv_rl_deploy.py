"""Lesson 19's simulated deployment (nfv.rl.deploy) — CPU, tiny random-init base.

* off flags: deploy_mode with swap_every=1 IS the lesson-19 loop bit for bit (the snapshot is
  refreshed after every update, i.e. it always equals the learner), and explicit off flags
  equal the defaults;
* the correction lives in residual space and within the bound;
* the critics never see an intervened transition, and the autonomous transition that led into
  a take-over is cut (hand-built mini buffer + an audit of a real run's buffer);
* the BC batch: corr_frac rows target the operator's correction, the rest target 0;
* the snapshot only changes when refreshed, and refreshes land on multiples of swap_every;
* a tiny end-to-end run with the operator on T3.
"""

import numpy as np
import pytest
import torch

from fvb.policy.td3 import TD3, TD3Config
from nfv.model import ModelConfig
from nfv.model.config import build
from nfv.rl import deploy as dp
from nfv.rl import loop as rl_loop
from nfv.rl import residual as rr
from nfv.sim.env import Env


@pytest.fixture(scope="module")
def base():
    torch.manual_seed(0)
    return build(ModelConfig(trunk_layers=1, expert_layers=1, ffn=256, euler_steps=2)).eval()


TINY = dict(seed=0, env_steps=240, n_envs=2, warmup=80, eval_every=0, n_eval=1, hidden=32,
            batch=32, log_every=60)  # fmt: skip


def _strip(out: dict) -> dict:
    agent = out.pop("agent")
    out.pop("wall_s"), out.pop("env_steps_per_s")
    for r in out["train"]:
        r.pop("env_steps_per_s", None)
    w = [v.clone() for v in (*agent.actor.state_dict().values(),
                             *agent.critic.state_dict().values())]  # fmt: skip
    return out, w


def _same(a, b):
    (oa, wa), (ob, wb) = a, b
    assert all(torch.equal(x, y) for x, y in zip(wa, wb, strict=True))
    assert oa["curves"] == ob["curves"] and oa["final_eval"] == ob["final_eval"]
    assert oa["updates"] == ob["updates"] and oa["base_eval"] == ob["base_eval"]


def test_off_flags_bit_identical(base):
    ref = _strip(rl_loop.run_residual_rl(base, task="t3", **TINY))
    explicit = _strip(rl_loop.run_residual_rl(base, task="t3", **TINY, interventions="off",
                                              deploy_mode=False))  # fmt: skip
    _same(ref, explicit)
    assert ref[0]["train"] == explicit[0]["train"] and "deploy" not in ref[0]
    # a snapshot refreshed after EVERY update always equals the learner: same run, bit for bit
    snap1 = rl_loop.run_residual_rl(base, task="t3", **TINY, deploy_mode=True, swap_every=1)
    assert snap1["deploy"]["swaps"] == list(range(1, snap1["updates"] + 1))
    snap1.pop("deploy")
    s1 = _strip(snap1)
    _same(ref, s1)
    for a, b in zip(ref[0]["train"], s1[0]["train"], strict=True):
        assert {k: v for k, v in b.items() if k not in ("robot_min", "swaps")} == a


def test_correction_in_residual_space_and_bounded():
    rng = np.random.default_rng(0)
    for _ in range(500):
        op = rng.uniform(-0.004, 0.004, 3)
        base_dp = rng.uniform(-0.01, 0.01, 3)
        c = dp.correction(op, base_dp, rr.DP_RES)
        assert c.shape == (3,) and c.dtype == np.float32 and np.all(np.abs(c) <= 1.0)
        exe, _, _ = rr.compose(base_dp, 0.0, None, c)  # what the env executes
        assert np.all(np.abs(exe - base_dp) <= rr.DP_RES + 1e-9)  # never beyond the bound
        inside = np.abs(op - base_dp) <= rr.DP_RES
        assert np.allclose(exe[inside], op[inside], atol=1e-9)  # reachable -> exactly the op's
        assert np.all(np.sign(exe - base_dp)[~inside] == np.sign(op - base_dp)[~inside])
    # the operator on a real T3 episode: aims at the TRUE hole, moves at most 2 mm per axis
    env = Env(task="t3", render=False)
    env.reset(50_001)
    o = dp.PegOperator(env.controller.kp)
    step = o.operator_dp(env)
    e = env.task.hole_xy - env.task.peg_tip(env)[:2]
    assert np.all(np.abs(step) <= dp.OP_LAT_MAX + 1e-12)
    assert np.all(np.sign(step[:2]) == np.sign(e)) and step[2] == dp.OP_CREEP_DP  # free space
    env.close()


def _mini_buffer():
    """Two interleaved episodes, as the loop writes them (o[0] = the transition's own index).

    ep 0: t0 auto, t1 auto (-> boundary), t2 OP, t3 OP, t4 auto, t5 auto
    ep 1: t0 auto (-> boundary), t1 OP, t2 auto
    """
    b = dp.DeployBuffer(4, 3, 32)
    plan = [
        (0, False),
        (1, False),
        (0, False),
        (1, True),
        (0, True),
        (0, True),
        (1, False),
        (0, False),
        (0, False),
    ]  # (episode, operator acts) in the loop's round-robin order
    last = {0: -1, 1: -1}
    ep_t = {0: 0, 1: 0}
    for ep, iv in plan:
        if iv and (b.n == 0 or not b.intervened[last[ep]]):
            b.mark_boundary(last[ep])  # the loop does this when a take-over begins
        i = b.i
        o = np.array([i, ep, ep_t[ep], float(iv)], np.float32)
        last[ep] = b.add(o, np.full(3, 0.5 if iv else 0.0), -0.01, o, False, iv, ep, ep_t[ep])
        ep_t[ep] += 1
    return b


def test_critic_mask_on_hand_built_buffer():
    b = _mini_buffer()
    # index: 0 e0t0, 1 e1t0, 2 e0t1, 3 e1t1 OP, 4 e0t2 OP, 5 e0t3 OP, 6 e1t2, 7 e0t4, 8 e0t5
    assert list(b.intervened[: b.n]) == [0, 0, 0, 1, 1, 1, 0, 0, 0]
    # boundaries: e1t0 (index 1, led into e1t1) and e0t1 (index 2, led into e0t2) are cut
    assert list(b.critic_ok[: b.n]) == [1, 0, 0, 0, 0, 0, 1, 1, 1]
    assert list(b.critic_idx()) == [0, 6, 7, 8]
    view = dp.CriticView(b)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        o, a, r, o2, d, _, _ = view.sample(16, rng, "cpu")
        idx = o[:, 0].long().tolist()
        seen.update(idx)
        assert not (o[:, 3] > 0).any()  # never an operator step
        # never bootstrap into a take-over: no sampled o2 is followed by an operator step
        for i in idx:
            ep, t = int(b.ep_id[i]), int(b.ep_t[i])
            nxt = np.flatnonzero((b.ep_id[: b.n] == ep) & (b.ep_t[: b.n] == t + 1))
            assert len(nxt) == 0 or not b.intervened[nxt[0]]
    assert seen == {0, 6, 7, 8}
    # a TD3 update runs on the view (the twin critics see only these four transitions)
    agent = TD3(4, 3, TD3Config(hidden=16, batch=8))
    out = agent.update(view, rng, demo=dp.CorrectionBCView(b), bc_weight=1.0)
    assert np.isfinite(out["critic_loss"])


def test_bc_batch_targets():
    b = _mini_buffer()
    v = dp.CorrectionBCView(b, corr_frac=0.25)
    o, target, *_ = v.sample(16, np.random.default_rng(1), "cpu")
    k = v.last_n_corr
    assert k == 4
    assert (o[:k, 3] == 1).all() and torch.allclose(target[:k], torch.full((k, 3), 0.5))
    assert (o[k:, 3] == 0).all() and (target[k:] == 0).all()
    # before any intervention the batch is all autonomous (BC toward 0 = ZeroActionView)
    b0 = dp.DeployBuffer(4, 3, 8)
    for _ in range(3):
        b0.add(np.zeros(4), np.ones(3), 0.0, np.zeros(4), False)
    o, target, *_ = dp.CorrectionBCView(b0).sample(8, np.random.default_rng(0), "cpu")
    assert (target == 0).all()


def test_snapshot_only_changes_on_refresh():
    torch.manual_seed(0)
    agent = rr.make_agent(hidden=16)
    snap = dp.PolicySnapshot(agent)
    f = np.random.default_rng(0).normal(0, 1, rr.feature_dim()).astype(np.float32)
    a0 = snap.act(f)
    assert np.array_equal(a0, agent.act(f))
    with torch.no_grad():
        agent.actor.net[-1].weight.add_(1.0)  # the learner moves (an update)
    assert np.array_equal(snap.act(f), a0) and not np.array_equal(agent.act(f), a0)
    snap.refresh()
    assert np.array_equal(snap.act(f), agent.act(f)) and snap.swaps == 1


def test_budget_and_minutes():
    assert dp.robot_minutes(20_000) == pytest.approx(20_000 / 10 / 60)
    assert dp.budget_at(0, 100, 0.2, 1.0) == pytest.approx(0.2)
    assert dp.budget_at(50, 100, 0.2, 1.0) == pytest.approx(0.1)
    assert dp.budget_at(100, 100, 0.2, 1.0) == 0.0
    assert dp.budget_at(100, 100, 0.2, 0.0) == pytest.approx(0.2)


def test_operator_trigger_rule():
    class _T:
        def __init__(self):
            self.d = 0.0

        def depth_mm(self, env):
            return self.d

    class _E:
        def __init__(self):
            self.task, self.wrench = _T(), np.zeros(6)

    env, o = _E(), dp.PegOperator(800.0)
    env.wrench[2] = -3.0  # in contact, no depth progress
    for _ in range(dp.STALL_STEPS - 1):
        o.observe(env)
        assert o.wants_in(env) == ""
    o.observe(env)
    assert o.wants_in(env) == "stall"
    env.task.d = 2.0  # depth progress resets the stall
    o.observe(env)
    assert o.stall == 0 and o.wants_in(env) == ""
    env.wrench[2] = -13.0
    assert o.wants_in(env) == "force"
    o.begin("force")
    for _ in range(dp.SEGMENT_STEPS):
        assert o.active
        o.tick()
    assert not o.active and o.stall == 0


class _Capture(dp.DeployBuffer):
    last = None

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _Capture.last = self


def test_e2e_interventions_tiny(base, monkeypatch):
    monkeypatch.setattr(rl_loop, "DeployBuffer", _Capture)
    monkeypatch.setattr(dp, "STALL_STEPS", 3)  # the random base must trigger in 240 steps
    monkeypatch.setattr(dp, "CONTACT_N", -1.0)  # (any contact, even none, counts as a stall)
    out = rl_loop.run_residual_rl(base, task="t3", **TINY, interventions="expert",
                                  intervention_budget=0.3, deploy_mode=True,
                                  swap_every=7)  # fmt: skip
    d = out["deploy"]
    assert d["operator_steps"] > 0 and d["n_segments"] > 0
    assert d["operator_steps"] <= 0.3 * out["env_steps"] + 2 * dp.SEGMENT_STEPS
    assert d["robot_min"] == pytest.approx(out["env_steps"] / 600, abs=1e-3)
    assert len(d["curve"]["rate"]) == rl_loop.DEPLOY_BINS
    assert d["swaps"] and all(s % 7 == 0 for s in d["swaps"])
    row = out["train"][-1]
    assert {"intervention_rate_ma", "operator_steps", "operator_min", "robot_min",
            "corr_share"} <= set(row)  # fmt: skip
    # audit the real buffer: the operator's steps never train the critics, and neither does
    # an autonomous step whose successor in its episode was the operator's
    b = _Capture.last
    assert b.intervened[: b.n].sum() == d["operator_steps"]
    for i in range(b.n):
        nxt = np.flatnonzero((b.ep_id[: b.n] == b.ep_id[i]) & (b.ep_t[: b.n] == b.ep_t[i] + 1))
        into_op = len(nxt) > 0 and b.intervened[nxt[0]]
        assert b.critic_ok[i] == (not b.intervened[i] and not into_op), i
    # stored actions of operator steps are the corrections (bounded residuals)
    assert np.all(np.abs(b.a[: b.n][b.intervened[: b.n]]) <= 1.0)
    with pytest.raises(ValueError, match="T3 only"):
        rl_loop.run_residual_rl(base, task="t1", **TINY, interventions="expert")
