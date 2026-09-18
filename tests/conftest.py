import os

import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")


@pytest.fixture(scope="session")
def wipe_env():
    """One robosuite Wipe env shared across the session (construction is slow)."""
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    cfg = load_composite_controller_config(controller="BASIC")
    env = suite.make(
        "Wipe",
        robots="Panda",
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    yield env
    env.close()
