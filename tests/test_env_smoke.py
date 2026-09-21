import mujoco
import numpy as np

from fvb.ft.read import MujocoFT, RobosuiteFT
from fvb.scenes import seed_xml_text


def test_raw_scene_loads_and_steps():
    m = mujoco.MjModel.from_xml_string(seed_xml_text())
    d = mujoco.MjData(m)
    ft = MujocoFT(m, d)
    for _ in range(100):
        mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos))
    assert np.all(np.isfinite(d.qvel))
    assert ft.force().shape == (3,)
    assert ft.torque().shape == (3,)
    assert ft.ft_raw().shape == (6,)
    assert np.all(np.isfinite(ft.ft_raw()))


def test_robosuite_wipe_smoke(wipe_env):
    env = wipe_env
    obs = env.reset()
    assert "robot0_eef_force" not in obs  # PLAN §2: F/T is not an observation
    ft = RobosuiteFT(env)
    zero = np.zeros(env.action_dim)
    for _ in range(10):
        obs, _r, _d, _i = env.step(zero)
    assert np.asarray(env.robots[0].ee_force["right"]).shape == (3,)
    assert np.asarray(env.robots[0].ee_torque["right"]).shape == (3,)
    assert ft.ft_raw().shape == (6,)
    assert ft.ft_robosuite().shape == (6,)
    assert np.all(np.isfinite(ft.ft_raw()))


def test_custom_peg_in_hole_env():
    """M6 stretch: fvb.envs.PegInHole makes, resets, steps; flange F/T reads m*g at rest."""
    from fvb.control.osc_scripts import ArmRig, make_env

    env = make_env("PegInHole", seed=0, clearance=0.0005)
    rig = ArmRig(env)
    rig.reset(0)
    assert env.action_dim == 6
    for _ in range(15):
        rig.step(np.zeros(6))
    assert rig.d.ncon == 0
    f = rig.ft.ft_raw()
    assert f.shape == (6,) and np.all(np.isfinite(f))
    mg = float(rig.m.body_subtreemass[rig.sensor_body]) * 9.81
    assert abs(np.linalg.norm(f[:3]) - mg) / mg < 0.02
    assert env.hole_half_opening == env.peg_radius + 0.0005
    env.close()
