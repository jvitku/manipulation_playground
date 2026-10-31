"""Controllers: position tracking, hybrid selection + hysteresis, the 5 N hold, the 25 N abort.

The 5 N hold at three hidden surface heights is the M0 gate's controller half, run here with a
small step budget; tests/nfv/measure_m0_gate.py runs the full version and prints the numbers.
"""

import numpy as np
import pytest

from nfv.seeds import eval_seeds
from nfv.sim.env import Action

FT_5N = np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0])  # press down with 5 N (up-push = negative)


def test_position_mode_reaches_target(sim_env):
    sim_env.reset(seed=1, task="t0")
    p0 = sim_env.qpos_xyz
    dp = np.array([0.02, -0.015, 0.0])
    sim_env.step(Action(dp=dp))
    for _ in range(3):  # let the PD settle (overdamped, tau ~ 75 ms)
        sim_env.step(Action(dp=np.zeros(3)))
    err = sim_env.qpos_xyz - (p0 + dp)
    assert abs(err[0]) < 0.0015 and abs(err[1]) < 0.0015  # x/y: no standing load
    # the commanded target is exact (droop under gravity is the PD's, not the interpolator's)
    assert np.allclose(sim_env.controller.p_cmd[:2], (p0 + dp)[:2], atol=1e-9)


def test_hybrid_selection_hysteresis(sim_env):
    sim_env.reset(seed=2, task="t0")
    ctl = sim_env.controller
    ctl.begin_policy_step(np.zeros(3), 0.0, np.array([0.0, 0.0, -0.45, 0, 0, 0]))
    assert not ctl.selected[2]  # below the 0.5 N select threshold
    ctl.begin_policy_step(np.zeros(3), 0.0, np.array([0.0, 0.0, -0.60, 0, 0, 0]))
    assert ctl.selected[2]  # selected
    ctl.begin_policy_step(np.zeros(3), 0.0, np.array([0.0, 0.0, -0.45, 0, 0, 0]))
    assert ctl.selected[2]  # 0.1 N hysteresis: stays selected above 0.4 N
    ctl.begin_policy_step(np.zeros(3), 0.0, np.array([0.0, 0.0, -0.35, 0, 0, 0]))
    assert not ctl.selected[2]  # dropped below 0.4 N -> position mode again
    ctl.begin_policy_step(np.zeros(3), 0.0, None)
    assert not ctl.selected.any()


@pytest.mark.parametrize("dz_mm", [-8.0, 0.0, 8.0])
def test_hybrid_holds_5n_on_unknown_height(sim_env, dz_mm):
    """M0 gate (small budget): hold 5 N within ±1 N on a flat plate at three hidden heights."""
    seed = eval_seeds("t2", 1)[0]
    sim_env.reset(
        seed=seed,
        task="t2",
        hidden_overrides={"dz": dz_mm * 1e-3, "tilt_x": 0.0, "tilt_y": 0.0},
    )
    f_n = []
    for i in range(30):  # 3 s: ~1.5 s descend + build, then the hold
        obs, info = sim_env.step(Action(dp=np.zeros(3), f_target=FT_5N))
        if i >= 20:
            f_n.extend(-obs.wrench_window[:, 2])  # F_n = -F_z at 200 Hz over the last second
        assert not info["force_abort"]
    mean = float(np.mean(f_n))
    assert abs(mean - 5.0) < 1.0, f"held {mean:.2f} N at dz={dz_mm} mm, want 5 +- 1"


def test_force_abort_at_25n(sim_env):
    """Commanding 30 N drives the admittance through the 25 N safety -> force_abort."""
    seed = eval_seeds("t2", 1)[0]
    sim_env.reset(seed=seed, task="t2", hidden_overrides={"dz": 0.0})
    ft = np.array([0.0, 0.0, -30.0, 0.0, 0.0, 0.0])
    info = {}
    for _ in range(40):
        _, info = sim_env.step(Action(dp=np.zeros(3), f_target=ft))
        if info["done"]:
            break
    assert info["force_abort"] and info["outcome"] == "force_abort"
    assert info["peak_force"] > 25.0
