"""Tasks T0-T3: expert success on held-out seeds, and the T1 threshold-invisibility guard.

The expert bar here is >= 18/20 held-out seeds per task (fast tier). The full M0 gate — 95 %
over 100 seeds per task — runs as a script: tests/nfv/measure_m0_gate.py.
"""

import numpy as np
import pytest

from nfv.seeds import eval_seeds
from nfv.sim.experts import rollout_expert

N_SEEDS = 20
MIN_PASS = 18


@pytest.mark.parametrize("task", ["t0", "t1", "t2", "t3"])
def test_expert_succeeds(sim_env, task):
    fails = []
    for seed in eval_seeds(task, N_SEEDS):
        info = rollout_expert(sim_env, seed, task=task)
        if not info["success"]:
            fails.append((seed, info["outcome"], round(info["peak_force"], 2)))
    assert len(fails) <= N_SEEDS - MIN_PASS, f"{task}: {len(fails)} failures {fails}"


def test_t1_threshold_is_invisible(render_env):
    """The course's core premise: two episodes differing ONLY in the hidden click threshold
    render PIXEL-IDENTICAL at reset — the camera cannot carry the deciding quantity."""
    seed = eval_seeds("t1", 1)[0]
    a = render_env.reset(seed=seed, task="t1", hidden_overrides={"threshold": 3.0})
    img_static_a, img_wrist_a = a.rgb_static.copy(), a.rgb_wrist.copy()
    thr_a = dict(render_env.task.threshold)
    b = render_env.reset(seed=seed, task="t1", hidden_overrides={"threshold": 9.0})
    assert render_env.task.threshold["red"] == 9.0 and thr_a["red"] == 3.0  # overrides landed
    assert np.array_equal(img_static_a, b.rgb_static)
    assert np.array_equal(img_wrist_a, b.rgb_wrist)


def test_t1_wrong_button_fails(sim_env):
    """Pressing the other button ends the episode as a failure, not a success."""
    from nfv.sim.env import Action

    seed = eval_seeds("t1", 1)[0]
    sim_env.reset(seed=seed, task="t1")
    task = sim_env.task
    wrong = [c for c in task.COLORS if c != task.target][0]  # the non-instructed button
    bxy = task.button_pos(sim_env, wrong)[:2]
    info = {}
    for _ in range(int(task.TIME_LIMIT_S / 0.1)):
        err = bxy - sim_env.qpos_xyz[:2]
        if np.linalg.norm(err) > 0.003:
            act = Action(dp=np.array([*np.clip(err, -0.03, 0.03), 0.0]))
        else:
            act = Action(dp=np.zeros(3), f_target=np.array([0, 0, -12.0, 0, 0, 0]))
        _, info = sim_env.step(act)
        if info["done"]:
            break
    assert info["done"] and info["outcome"] == "wrong_button" and not info["success"]


def test_t2_position_mode_cannot_hold_force(sim_env):
    """Teaching check: pure position mode on T2 either digs in or floats — the tracked
    fraction stays low, which is why the hybrid controller exists."""
    from nfv.sim.env import Action

    seed = eval_seeds("t2", 2)[1]
    sim_env.reset(seed=seed, task="t2", instruction="wipe the plate gently")
    # descend blind to a fixed depth, then sweep like the expert would, position-only
    info = {}
    for _ in range(140):
        z_err = -0.36 - sim_env.qpos_xyz[2]
        dp = np.array([0.008, 0.0, np.clip(z_err, -0.01, 0.0)])
        _, info = sim_env.step(Action(dp=dp))
        if info["done"]:
            break
    assert not info["success"]
