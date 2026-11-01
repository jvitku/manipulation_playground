"""The clock ladder: 2 ms physics, 200 Hz controller, 10 Hz policy — asserted, not assumed."""

import numpy as np

from nfv.sim.controllers import (
    CTRL_DT,
    CTRL_PER_POLICY,
    PHYS_DT,
    PHYS_PER_CTRL,
    PHYS_PER_POLICY,
    POLICY_DT,
)


def test_ratios():
    assert PHYS_DT == 0.002  # 2 ms physics (500 Hz)
    assert CTRL_DT == 0.005  # 200 Hz controller
    assert POLICY_DT == 0.1  # 10 Hz policy
    assert CTRL_PER_POLICY == 20
    assert PHYS_PER_POLICY == 50
    assert abs(CTRL_PER_POLICY * CTRL_DT - POLICY_DT) < 1e-12
    assert abs(PHYS_PER_POLICY * PHYS_DT - POLICY_DT) < 1e-12
    # 200 Hz on a 500 Hz physics clock = 2.5 steps/tick, realised as alternating 2/3 substeps
    per_tick = [PHYS_PER_CTRL[i % 2] for i in range(CTRL_PER_POLICY)]
    assert sum(per_tick) == PHYS_PER_POLICY
    assert np.mean(per_tick) * PHYS_DT == CTRL_DT


def test_env_clock(sim_env):
    from nfv.sim.env import Action

    obs = sim_env.reset(seed=0, task="t0")
    assert sim_env.model.opt.timestep == PHYS_DT
    assert obs.wrench_window.shape == (CTRL_PER_POLICY, 6)
    assert obs.t == 0.0
    t0 = float(sim_env.data.time)
    obs, _ = sim_env.step(Action(dp=np.zeros(3)))
    assert abs(float(sim_env.data.time) - t0 - POLICY_DT) < 1e-9  # one step = exactly 100 ms
    assert abs(obs.t - POLICY_DT) < 1e-9
