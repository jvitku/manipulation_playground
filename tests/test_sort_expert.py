"""PLAN §14: the scripted expert completes 3 seeds (rack presentation, noise-free).

A regression test, not a measurement: the seeds are three the V4 gate run solved
(outputs/v4_expert_ab/rack_clean50d.jsonl, 47/50 unseen seeds). A change to the env or the expert
that breaks any of them should be explained before the gate is re-measured.
"""

import pytest

pytest.importorskip("robosuite")

import robosuite as suite  # noqa: E402

import fvb.envs  # noqa: E402, F401
from fvb.envs.sort_bolts_nuts import SortTaskParams  # noqa: E402
from fvb.policy.sort_expert import NO_NOISE, SortExpert  # noqa: E402


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
