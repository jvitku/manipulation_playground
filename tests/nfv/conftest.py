"""Shared fixtures for the nfv (force-VLA school) test suite — CPU-only, osmesa rendering."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

# make `import nfv` work from a host checkout too (the fvb-lab image has src/ on sys.path)
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@pytest.fixture(scope="session")
def sim_env():
    """One physics-only Env shared by the suite (switch tasks via reset(seed, task=...))."""
    from nfv.sim.env import Env

    env = Env(task="t0", render=False)
    yield env
    env.close()


@pytest.fixture(scope="session")
def render_env():
    """One rendering Env (96x96 osmesa) shared by the suite."""
    from nfv.sim.env import Env

    env = Env(task="t1", render=True)
    yield env
    env.close()
