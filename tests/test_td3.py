"""Stage 1 RL: TD3 agent mechanics and the arm RL environment wrapper."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fvb.policy.td3 import TD3, ReplayBuffer, TD3Config  # noqa: E402


def test_td3_learns_a_one_step_bandit():
    """Reward = -|a - 0.5|: the actor must move to 0.5 (smoke test of critic + delayed actor)."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    agent = TD3(2, 1, TD3Config(hidden=64, batch=64))
    buf = ReplayBuffer(2, 1, 5000)
    o = np.zeros(2, np.float32)
    for _ in range(2000):
        a = rng.uniform(-1, 1, 1).astype(np.float32)
        buf.add(o, a, -abs(float(a[0]) - 0.5), o, True)
    for _ in range(1500):
        out = agent.update(buf, rng)
    assert np.isfinite(out["critic_loss"])
    assert abs(float(agent.act(o)[0]) - 0.5) < 0.15


def test_arm_rl_env_zeroes_force_and_rewards_progress():
    from fvb.policy.arm_task import ArmTaskParams, ArmWorld
    from fvb.policy.data import Norm
    from fvb.policy.rl_env import ArmRLEnv
    from fvb.policy.spec import ARM

    p = ArmTaskParams()
    world = ArmWorld(p, seed=0)
    norm = Norm(np.zeros(ARM.obs_dim), np.ones(ARM.obs_dim), np.zeros(3), np.ones(3))
    try:
        env = ArmRLEnv(p, world, norm, use_force=False, history=3)
        o = env.reset(5)
        assert o.shape == (3 * ARM.obs_dim,)
        down = np.array([0.0, 0.0, -1.0])
        rewards = []
        for _ in range(4):  # free-space descent: depth increases -> positive progress reward
            o, r, term, trunc, info = env.step(down)
            rewards.append(r)
        frames = o.reshape(3, ARM.obs_dim)
        assert np.all(frames[:, ARM.force_idx] == 0.0)
        assert np.mean(rewards) > 0 and not term and not trunc
    finally:
        world.close()


def test_td3_policy_matches_the_env_pipeline(tmp_path):
    """TD3Policy on raw ArmTask observations == the actor on ArmRLEnv observations."""
    import json

    from fvb.policy.arm_task import ArmTaskParams, ArmWorld
    from fvb.policy.data import Norm
    from fvb.policy.rl_env import ArmRLEnv
    from fvb.policy.spec import ARM
    from fvb.policy.td3 import TD3Policy

    rng = np.random.default_rng(1)
    norm = Norm(
        rng.normal(size=ARM.obs_dim), rng.uniform(0.5, 2, ARM.obs_dim), np.zeros(3), np.ones(3)
    )
    (tmp_path / "norm.json").write_text(json.dumps(norm.to_json()))
    p = ArmTaskParams()
    world = ArmWorld(p, seed=0)
    try:
        env = ArmRLEnv(p, world, norm, use_force=False, history=3)
        agent = TD3(env.obs_dim, 3, TD3Config(hidden=32))
        meta = {
            "algo": "td3",
            "use_force": False,
            "history": 3,
            "norm": str(tmp_path / "norm.json"),
            "xy_scale": float(env.scale[0]),
            "z_scale": float(env.scale[2]),
        }
        torch.save(agent.state(meta), tmp_path / "ck.pt")
        pol = TD3Policy(str(tmp_path / "ck.pt"))
        o = env.reset(4)
        pol.reset()
        for _ in range(3):
            a_env = np.clip(agent.act(o), -1, 1) * env.scale
            a_pol = pol.act(env.task.observe())
            assert np.allclose(a_env, a_pol, atol=1e-5)
            o, *_ = env.step(agent.act(o))
    finally:
        world.close()


def test_td3_bc_term_pulls_actor_to_demonstrations():
    """With only the BC term informative (constant reward), TD3+BC imitates the demo action."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    agent = TD3(2, 1, TD3Config(hidden=64, batch=64))
    buf, demo = ReplayBuffer(2, 1, 1000), ReplayBuffer(2, 1, 1000)
    o = np.zeros(2, np.float32)
    for _ in range(500):
        buf.add(o, rng.uniform(-1, 1, 1).astype(np.float32), 0.0, o, True)
        demo.add(o, np.array([-0.7], np.float32), 0.0, o, True)
    for _ in range(800):
        out = agent.update(buf, rng, demo, bc_weight=5.0)
    assert "bc_loss" in out
    assert abs(float(agent.act(o)[0]) + 0.7) < 0.1


def test_asymmetric_critic_learns_from_privileged_input_only():
    """Reward depends on a hidden sign the actor cannot see; with the privileged input the
    critic still fits it (TD error drops), and the actor stays privilege-free."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    agent = TD3(2, 1, TD3Config(hidden=64, batch=64), priv_dim=1)
    buf = ReplayBuffer(2, 1, 4000, priv_dim=1)
    o = np.zeros(2, np.float32)
    for _ in range(3000):
        h = rng.choice([-1.0, 1.0])
        a = rng.uniform(-1, 1, 1).astype(np.float32)
        buf.add(
            o, a, float(h * a[0]), o, True, np.array([h], np.float32), np.array([h], np.float32)
        )
    first = np.mean([agent.update(buf, rng)["critic_loss"] for _ in range(50)])
    for _ in range(1500):
        agent.update(buf, rng)
    last = np.mean([agent.update(buf, rng)["critic_loss"] for _ in range(50)])
    assert last < 0.2 * first
    assert agent.actor.net[0].in_features == 2  # actor input = observation only
    assert agent.critic.q1[0].in_features == 2 + 1 + 1  # obs + priv + action


def test_expert_demos_through_the_env_in_xy_abs_mode():
    """Demos encoded as xy_abs actions (lateral target) still insert, and the lateral labels are
    persistent steps, not one-step pulses."""
    from fvb.policy.arm_task import ArmTaskParams, ArmWorld
    from fvb.policy.data import Norm
    from fvb.policy.rl_env import ArmRLEnv
    from fvb.policy.spec import ARM

    p = ArmTaskParams(action_mode="xy_abs")
    world = ArmWorld(p, seed=0)
    norm = Norm(np.zeros(ARM.obs_dim), np.ones(ARM.obs_dim), np.zeros(3), np.ones(3))
    try:
        env = ArmRLEnv(p, world, norm, history=2, xy_scale=0.005)
        env.reset(5)
        env.world.place_hole(np.array([0.0015, 0.0]))  # beyond the clearance: needs a correction
        env.task.hole_xy = np.array([0.0015, 0.0])
        acts, done, info = [], False, {}
        while not done:
            a = np.clip(env.task.encode(env.task.expert_action()) / env.scale, -1, 1)
            acts.append(a)
            _, _, term, trunc, info = env.step(a)
            done = term or trunc
        assert info["reason"] == "success"
        ax = np.array(acts)[:, 0]
        assert np.count_nonzero(ax) > 0.3 * len(ax)  # a held lateral target, not a pulse
    finally:
        world.close()
