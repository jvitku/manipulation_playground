"""Stage 1 G1: the arm hidden-hole task (PegInHole on the Panda). One env, reused."""

import numpy as np
import pytest

from fvb.policy.arm_task import ArmTask, ArmTaskParams, ArmWorld
from fvb.policy.spec import ARM


@pytest.fixture(scope="module")
def world():
    w = ArmWorld(ArmTaskParams(), seed=0)
    yield w
    w.close()


def _expert(task):
    done, reason, obs = False, None, []
    while not done:
        obs.append(task.observe())
        done, reason = task.step(task.expert_action())
    return reason, np.stack(obs)


def test_observation_hides_the_hole(world):
    p = ArmTaskParams()
    a = ArmTask(p, 7, world, hole_offset=np.array([0.002, -0.001]))
    oa = a.observe()
    b = ArmTask(p, 7, world, hole_offset=np.array([-0.0015, 0.0025]))
    ob = b.observe()
    assert oa.shape == (ARM.obs_dim,)
    assert np.allclose(oa, ob, atol=1e-9)  # same seed, different hole: identical first obs
    assert not np.allclose(a.hole_xy, b.hole_xy)


def test_expert_recovers_from_jams(world):
    p = ArmTaskParams()
    for off in ([0.0015, 0.0], [-0.001, 0.002]):  # beyond the 0.5 mm clearance
        task = ArmTask(p, 3, world, hole_offset=np.array(off))
        reason, obs = _expert(task)
        assert reason == "success", (off, reason)
        assert task.depth() >= p.success_depth
        # the jam was felt through the force channels (world-frame compensated wrench)
        assert np.abs(obs[:, ARM.force_idx]).max() > p.jam_force_N


def test_hole_moves_with_the_offset(world):
    p = ArmTaskParams()
    task = ArmTask(p, 5, world, hole_offset=np.array([0.003, -0.002]))
    c = task.env.hole_center_world[:2] - world.hole_pos0[:2]
    assert np.allclose(c, [0.003, -0.002], atol=1e-12)
