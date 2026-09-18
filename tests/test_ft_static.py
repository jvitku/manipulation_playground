import numpy as np

from fvb.control.gantry import Gantry
from fvb.scenes.builder import SceneParams, build_model


def _rest_ft(mass: float) -> np.ndarray:
    p = SceneParams(peg_mass=mass)
    m, d = build_model(p)
    g = Gantry(m, d, p)
    g.settle(2.0)
    assert d.ncon == 0, "peg must hang free"
    return g.ft.ft_raw()


def test_static_reads_mg_within_1pct():
    ft = _rest_ft(0.1)
    mg = 0.1 * 9.81
    assert abs(abs(ft[2]) - mg) / mg < 0.01
    assert np.allclose(ft[[0, 1, 3, 4, 5]], 0.0, atol=1e-6)


def test_doubling_mass_doubles_fz():
    f1 = _rest_ft(0.1)[2]
    f2 = _rest_ft(0.2)[2]
    assert abs(f2 / f1 - 2.0) < 0.01
    assert np.sign(f1) == np.sign(f2)
