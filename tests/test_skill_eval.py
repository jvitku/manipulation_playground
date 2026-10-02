"""V9 skill-level closed loop: policy_obs in, skill-unit actions out (fvb.vla.skill_eval)."""

import numpy as np
import pytest

pytest.importorskip("robosuite")

from fvb.policy.sort_skill_env import SortInsertEnv  # noqa: E402
from fvb.vla.skill_eval import run_insert_episode  # noqa: E402


class ExpertPolicy:
    """Replays the skill env's own demonstrator through the policy interface."""

    def __init__(self, sk):
        self.sk, self.seen = sk, []

    def reset(self):
        pass

    def act(self, obs):
        self.seen.append(obs)
        return np.r_[self.sk.expert_action(), np.zeros(4)]


@pytest.fixture(scope="module")
def sk():
    return SortInsertEnv(use_force=True, offscreen=True)


def test_expert_through_the_policy_interface_inserts(sk):
    pol = ExpertPolicy(sk)
    r = run_insert_episode(sk, pol, seed=3, task="t", max_steps=150, size=64)
    assert r["outcome"] == "success", r
    o = pol.seen[0]
    assert o["observation.images.front"].shape == (64, 64, 3)
    assert o["observation.images.wrist"].shape == (64, 64, 3)
    assert o["observation.torque_hist"].shape == (150,)


def test_zero_policy_times_out(sk):
    class Zero:
        def reset(self):
            pass

        def act(self, obs):
            return np.zeros(7)

    r = run_insert_episode(sk, Zero(), seed=3, task="t", max_steps=5, size=32)
    assert r["outcome"] == "timeout" and r["steps"] == 5
