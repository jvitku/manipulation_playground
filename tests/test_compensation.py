import numpy as np

from fvb.control.gantry import Gantry
from fvb.ft.compensate import LoadParams, identify_mass_com
from fvb.scenes.builder import SceneParams, build_model


def _rot_xyzw(q):
    import mujoco

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.array([q[3], q[0], q[1], q[2]]))
    return R.reshape(3, 3)


def test_identify_mass_com_from_tilt_poses():
    p = SceneParams(tilt_axis=(0, 1, 0))
    m, d = build_model(p)
    g = Gantry(m, d, p, 20.0)
    g.settle(1.0)
    fts, Rs = [], []
    for ang in np.deg2rad([0, 30, 60, 90]):
        g.set_tilt(ang)
        g.settle(2.0)
        fts.append(g.ft.ft_raw())
        Rs.append(g.ee_rot())
    load = identify_mass_com(np.array(fts), np.array(Rs))
    assert abs(load.mass - p.peg_mass) / p.peg_mass < 0.01
    assert np.allclose(load.com, [0, 0, -p.peg_len / 2], atol=1e-4)


def test_free_space_residual_2hz_below_5pct():
    """PLAN M2: residual after compensation < 5 % of m·g RMS for the 2 Hz motion.

    Compensation runs at physics rate inside Gantry.step (finite-difference acceleration),
    averaged per control step. Control-rate compensation does NOT pass this (M2 findings).
    """
    p = SceneParams()
    load = LoadParams(mass=p.peg_mass, com=np.array([0, 0, -p.peg_len / 2]))
    m, d = build_model(p)
    g = Gantry(m, d, p, 20.0, load=load)
    g.settle(2.0)
    from fvb.logging.episode import EpisodeLogger

    log = EpisodeLogger(config={}, seed=0)
    t = np.arange(0, 3.0, 1 / 20.0)
    for z in 0.01 * np.sin(2 * np.pi * 2.0 * t):
        g.step([0, 0, z], log)
    a = log.arrays()
    mg = p.peg_mass * 9.81
    raw_res = np.sqrt(np.mean(np.sum((a["ft_raw"][:, :3] - [0, 0, mg]) ** 2, axis=1)))
    comp_res = np.sqrt(np.mean(np.sum(a["ft_comp"][:, :3] ** 2, axis=1)))
    assert raw_res / mg > 0.1  # the motion really does contaminate the raw signal
    assert comp_res / mg < 0.05
