"""V2 gate: joint-torque sensing on the Panda (PLAN §13, §14 test_joint_torque)."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

pytest.importorskip("robosuite")

from fvb.sensors.joint_torque import PANDA_ARM_JOINTS, JointTorqueSensor  # noqa: E402
from fvb.sensors.tactile import TactilePandaGripper  # noqa: E402

G = 9.81


def _panda(load_kg: float = 0.0) -> mujoco.MjModel:
    from robosuite.models.robots import Panda
    from robosuite.models.world import MujocoWorldBase

    world = MujocoWorldBase()
    robot = Panda()
    robot.add_gripper(TactilePandaGripper(idn="0_right"))
    world.merge(robot)
    if load_kg:
        body = ET.SubElement(world.worldbody, "body", name="load", pos="0.5 0 1.0")
        ET.SubElement(body, "freejoint", name="load_free")
        ET.SubElement(
            body,
            "geom",
            type="box",
            size="0.02 0.02 0.02",
            mass=f"{load_kg:g}",
            contype="0",
            conaffinity="0",
        )
        eq = world.root.find("equality")
        if eq is None:
            eq = ET.SubElement(world.root, "equality")
        # the load's frame coincides with the end-effector frame (grip site)
        ET.SubElement(eq, "weld", body1="load", body2="gripper0_right_eef", relpose="0 0 0 1 0 0 0")
    model = world.get_model(mode="mujoco")
    model.opt.timestep = 0.001
    return model


def _hold(model, data, steps=3000, kp=300.0, kd=40.0, xfrc=None, reset=True):
    """Joint PD + model gravity/Coriolis compensation around the initial arm pose."""
    arm = np.array([model.jnt_dofadr[model.joint(j).id] for j in PANDA_ARM_JOINTS])
    qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in PANDA_ARM_JOINTS])
    q0 = np.array([0.0, np.pi / 16, 0.0, -np.pi / 2 - np.pi / 3, 0.0, np.pi - 0.2, np.pi / 4])
    if reset:
        data.qpos[qadr] = q0
        mujoco.mj_forward(model, data)
    hand = model.body("robot0_right_hand").id
    for _ in range(steps):
        if xfrc is not None:
            data.xfrc_applied[hand] = xfrc
        tau = data.qfrc_bias[arm] - kp * (data.qpos[qadr] - q0) - kd * data.qvel[arm]
        data.ctrl[:7] = tau
        mujoco.mj_step(model, data)
    return arm


def test_free_motion_tau_ext_is_zero():
    model = _panda()
    data = mujoco.MjData(model)
    sensor = JointTorqueSensor(model, noise_std=0.0, bias_std=0.0)
    _hold(model, data, steps=500)
    _, tau_ext = sensor.read(data)
    assert np.abs(tau_ext).max() < 1e-6


def test_applied_wrench_is_exact():
    """tau_ext = J^T F for a wrench applied at the hand's centre of mass (any motion)."""
    model = _panda()
    data = mujoco.MjData(model)
    sensor = JointTorqueSensor(model, noise_std=0.0, bias_std=0.0)
    f = np.array([3.0, -5.0, 8.0, 0.2, 0.1, -0.3])  # force on the robot
    arm = _hold(model, data, steps=300, xfrc=f)
    _, tau_ext = sensor.true(data)
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    mujoco.mj_jacBodyCom(model, data, jp, jr, model.body("robot0_right_hand").id)
    expected = -(jp[:, arm].T @ f[:3] + jr[:, arm].T @ f[3:])
    np.testing.assert_allclose(tau_ext, expected, atol=1e-6)


def test_static_1kg_load_matches_jt_mg():
    """PLAN §13 V2 gate: holding a 1 kg load, tau_ext ~ J^T m g within 2 %."""
    model = _panda(load_kg=1.0)
    data = mujoco.MjData(model)
    sensor = JointTorqueSensor(model, noise_std=0.0, bias_std=0.0)
    _hold(model, data, steps=0)  # arm at its pose; start the load where the weld wants it
    lid = model.body("load").id
    adr = model.jnt_qposadr[model.joint("load_free").id]
    eef = model.body("gripper0_right_eef").id
    data.qpos[adr : adr + 3] = data.xpos[eef]
    mujoco.mju_mat2Quat(data.qpos[adr + 3 : adr + 7], data.xmat[eef])
    mujoco.mj_forward(model, data)
    _hold(model, data, steps=4000, reset=False)
    assert np.abs(data.qvel[:7]).max() < 1e-3
    _, tau_ext = sensor.true(data)
    jp = np.zeros((3, model.nv))
    # arm Jacobian of the point where the load hangs (the load itself is a free body)
    mujoco.mj_jac(model, data, jp, None, data.xipos[lid], eef)
    expected = jp[:, :7].T @ np.array([0.0, 0.0, 1.0 * G])
    big = np.abs(expected) > 0.1 * np.abs(expected).max()
    rel = np.abs(tau_ext - expected)[big] / np.abs(expected)[big]
    assert rel.max() < 0.02, (tau_ext, expected)
    # Cartesian estimate at the load: the robot pushes the load up with m g
    w = sensor.wrench(data, tau_ext, "gripper0_right_grip_site")
    assert abs(w[2] - G) < 0.02 * G
    assert np.abs(w[:2]).max() < 0.02 * G


def test_noise_statistics():
    model = _panda()
    data = mujoco.MjData(model)
    sensor = JointTorqueSensor(model, noise_std=0.05, bias_std=0.1, seed=3)
    _hold(model, data, steps=50)
    sensor.reset()
    _, true_ext = sensor.true(data)
    reads = np.array([sensor.read(data)[1] for _ in range(4000)]) - true_ext
    np.testing.assert_allclose(reads.mean(axis=0), sensor.bias, atol=0.005)
    np.testing.assert_allclose(reads.std(axis=0), 0.05, rtol=0.1)
    assert np.abs(sensor.bias).max() > 0  # a bias was drawn
