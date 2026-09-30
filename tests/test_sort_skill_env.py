"""V6: the SortBoltsNuts insertion skill env (PLAN §13 V6)."""

import numpy as np
import pytest

pytest.importorskip("robosuite")

from fvb.policy.sort_skill_env import SortInsertEnv  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = SortInsertEnv(use_force=False, noise=False, max_steps=60)
    yield e


def test_reset_hovers_a_bolt_above_the_hole(env):
    o = env.reset(1)
    assert o.shape == (env.obs_dim,) and np.all(np.isfinite(o))
    assert env.expert._held(env.part)
    assert 0.01 < -env.depth() < 0.04  # tip ~3 cm above the fixture top
    assert np.all(o.reshape(env.history, -1)[:, 4:] == 0)  # force channels zeroed


def test_scripted_descent_over_the_true_hole_succeeds(env):
    env.reset(1)
    reason, total = None, 0.0
    for _ in range(60):
        e = env.privileged()  # hole - tip (xy / 5 mm), depth / 25 mm
        a = np.r_[np.clip(e[:2] * 5.0, -1, 1), -1.0 if np.hypot(*e[:2]) * 5 < 0.6 else 0.0]
        _, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            reason = info["reason"]
            break
    assert reason == "success", info
    assert total > 0


def test_force_channels_present_with_force(env):
    e2 = SortInsertEnv(use_force=True, noise=False, max_steps=5)
    o = e2.reset(1)
    assert np.any(o.reshape(e2.history, -1)[:, 4:11] != 0)  # tau_ext carries the bolt's weight
