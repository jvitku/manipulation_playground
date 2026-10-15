"""PLAN §14: the scripted expert completes its seeds (rack presentation).

Regression tests, not measurements: the noise-free seeds are three the first V4 gate run solved
(outputs/v4_expert_ab/rack_clean50d.jsonl); the realistic-noise seed is a development seed the
across-the-hex nut grasp fixed (two on-edge nuts 25 mm apart). The gate itself is
scripts/43_expert_gate.py on seeds 9000-9049. A change that breaks any of these should be
explained before the gate is re-measured.
"""

import math

import mujoco
import numpy as np
import pytest

pytest.importorskip("robosuite")

import robosuite as suite  # noqa: E402

import fvb.envs  # noqa: E402, F401
from fvb.envs.sort_bolts_nuts import BIN_SIZE, SortTaskParams  # noqa: E402
from fvb.policy.sort_expert import NO_NOISE, NOISE_PRESETS, SortExpert  # noqa: E402


def _make(seed):
    return suite.make(
        "SortBoltsNuts",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        seed=seed,
        horizon=3600,
        task=SortTaskParams(bolt_presentation="rack"),
    )


def _run(env, ex):
    info = {}
    for _ in range(3600):
        _, _, done, info = env.step(ex.act())
        if done:
            break
    return info


@pytest.mark.parametrize("seed", [7010, 7025, 7045])
def test_noise_free_expert_sorts_every_part(seed):
    env = suite.make(
        "SortBoltsNuts",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        seed=seed,
        horizon=3600,
        task=SortTaskParams(bolt_presentation="rack"),
    )
    env.reset()
    ex = SortExpert(NO_NOISE, seed)
    ex.reset(env)
    info = {}
    for _ in range(3600):
        _, _, done, info = env.step(ex.act())
        if done:
            break
    env.close()
    assert info["outcome"] == "success", info
    assert info["n_sorted"] == len(env.parts) == 6


def test_realistic_expert_sorts_two_close_on_edge_nuts():
    env = _make(7102)
    env.reset()
    ex = SortExpert(NOISE_PRESETS["realistic"], 7102)
    ex.reset(env)
    info = _run(env, ex)
    env.close()
    assert info["outcome"] == "success", (info, ex.fail_log)


def test_realistic_noise_relooks_at_the_hole_per_attempt():
    """A fresh estimate per approach / miss around (true hole + fixed bias), not one per episode."""
    env = _make(0)
    env.reset()
    ex = SortExpert(NOISE_PRESETS["realistic"], 0)
    ex.reset(env)
    ests = []
    for _ in range(400):
        ex._relook()
        ests.append(ex.hole_est[:2] - env.hole_top[:2])
    env.close()
    ests = np.array(ests)
    noise = NOISE_PRESETS["realistic"]
    assert np.allclose(ests.mean(axis=0), ex.hole_bias, atol=3 * noise.hole_attempt_std / 20)
    assert ests.std(axis=0) == pytest.approx([noise.hole_attempt_std] * 2, rel=0.15)


def test_on_edge_nut_against_the_wall_is_grasped_across_the_hex():
    """Face grasp would put a finger into the tray wall: the planner takes the hex fallback."""
    env = _make(0)
    env.reset()
    ex = SortExpert(NO_NOISE, 0)
    ex.reset(env)
    i = next(k for k, p in enumerate(env.parts) if p.kind == "nut")
    m, d = env.sim.model._model, env.sim.data._data
    # on its edge with the axis along y, 24 mm from the -y wall (as in dev seed 7139)
    pos = env.bin_floor + [0.0, -(BIN_SIZE[1] / 2 - 0.042), 0.017]
    q = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)  # nut axis z -> -y
    qa, da = env.part_qadr[i], env.part_dadr[i]
    d.qpos[qa : qa + 3] = pos
    d.qpos[qa + 3 : qa + 7] = q
    d.qvel[da : da + 6] = 0
    mujoco.mj_forward(m, d)
    axis = env.part_pose(i)[1][:, 2]
    _, R = ex._grasp_plan(i)
    env.close()
    assert abs(axis[1]) > 0.99
    assert abs(R[:, 0] @ axis) < 0.05  # closing axis across the hex, not along the nut axis
