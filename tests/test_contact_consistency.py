import numpy as np

from fvb.control.gantry import Gantry
from fvb.ft.compensate import LoadParams
from fvb.ft.frames import world_to_site
from fvb.scenes.builder import SceneParams, build_model


def test_static_rim_press_ft_comp_equals_contact_sum():
    """PLAN M2: static pressed-against-rim case, ft_comp force equals summed contact force
    within 2 %. Sign: ft_comp is the wrench the load applies to the world, i.e. minus the
    contact wrench acting on the peg."""
    p = SceneParams()
    load = LoadParams(mass=p.peg_mass, com=np.array([0, 0, -p.peg_len / 2]))
    m, d = build_model(p)
    g = Gantry(m, d, p, 20.0, load=load)
    g.set_target([2e-3, 0, 0])  # 2 mm offset -> lands on the rim
    g.settle(1.0)
    z_rim = -(p.start_height - 0.01 - p.peg_len - p.hole_depth)
    for z in np.linspace(0, z_rim - 0.01, 40):
        g.step([2e-3, 0, z])
    for _ in range(40):
        g.step([2e-3, 0, z_rim - 0.01])
    cw, n = g.contact_wrench()
    assert n > 0
    comp = g.ft_comp_last()
    cw_site = world_to_site(cw, g.ee_rot())
    assert np.linalg.norm(cw_site[:3]) > 5.0
    err = np.linalg.norm(comp[:3] + cw_site[:3]) / np.linalg.norm(cw_site[:3])
    assert err < 0.02
    assert np.allclose(comp[3:], -cw_site[3:], atol=2e-2 * max(np.linalg.norm(cw_site[3:]), 1e-3))
