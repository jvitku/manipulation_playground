"""V2 gate: Franka Hand taxel pads (PLAN §13, §14 test_tactile)."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

pytest.importorskip("robosuite")

from fvb.envs.fasteners import STEEL_SOLIMP, STEEL_SOLREF  # noqa: E402
from fvb.sensors.tactile import (  # noqa: E402
    DEFAULT_TACTILE,
    TactilePads,
    TactilePandaGripper,
)

P = "gripper0_right_"


def _scene(probe: str = "box"):
    """Fixed, fully open hand in zero gravity plus a free steel probe near the left pad.

    The probes have part-like mass and the steel contact parameters of the V1 parts (MuJoCo mixes
    contact parameters from both geoms, and its contact stiffness scales with the effective mass,
    so a 10 g probe pressed with 5 N sinks through a soft pad).
    """
    from robosuite.models.world import MujocoWorldBase

    world = MujocoWorldBase()
    world.merge(TactilePandaGripper(idn="0_right"))
    body = ET.SubElement(world.worldbody, "body", name="probe", pos="0 0 0.5")
    ET.SubElement(body, "freejoint", name="probe_free")
    if probe == "box":  # covers the whole 16 x 16 mm pad
        ET.SubElement(
            body,
            "geom",
            name="probe_g",
            type="box",
            size="0.012 0.012 0.004",
            mass="0.1",
            friction="1 0.005 0.0001",
            solref=STEEL_SOLREF,
            solimp=STEEL_SOLIMP,
        )
    else:  # small sphere: a point contact
        ET.SubElement(
            body,
            "geom",
            name="probe_g",
            type="sphere",
            size="0.003",
            mass="0.1",
            friction="1 0.005 0.0001",
            solref=STEEL_SOLREF,
            solimp=STEEL_SOLIMP,
        )
    model = world.get_model(mode="mujoco")
    model.opt.timestep = 0.001
    model.opt.gravity[:] = 0
    data = mujoco.MjData(model)
    j1 = model.joint(P + "finger_joint1").qposadr[0]
    j2 = model.joint(P + "finger_joint2").qposadr[0]
    data.qpos[[j1, j2]] = [0.04, -0.04]
    data.ctrl[:] = [0.04, -0.04]
    mujoco.mj_forward(model, data)
    return model, data


def _place(model, data, xy, gap=0.002, half_thick=0.004):
    """Put the probe on the left pad at pad-frame (x, y), ``gap`` above its surface."""
    sid = model.site(P + "left_pad").id
    c, R = data.site_xpos[sid], data.site_xmat[sid].reshape(3, 3)
    n = R[:, 2]
    pos = c + R[:, 0] * xy[0] + R[:, 1] * xy[1] + n * (0.004 + half_thick + gap)
    adr = model.joint("probe_free").qposadr[0]
    data.qpos[adr : adr + 3] = pos
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())  # probe z (its thin axis) along the pad normal
    data.qpos[adr + 3 : adr + 7] = quat
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return R


def _press(model, data, force_world, steps=800):
    pid = model.body("probe").id
    for _ in range(steps):
        data.xfrc_applied[pid, :3] = force_world
        mujoco.mj_step(model, data)


def test_force_limit_raised_to_70N():
    model, _ = _scene()
    for i in range(model.nu):
        np.testing.assert_allclose(model.actuator_forcerange[i], [-70, 70])


def test_no_contact_all_zero():
    model, data = _scene()
    pads = TactilePads(model, prefix=P)
    _place(model, data, (0, 0), gap=0.003)
    _press(model, data, np.zeros(3), steps=50)
    assert np.all(pads.taxels(data) == 0)
    assert np.all(pads.summary(data) == 0)


@pytest.mark.parametrize("f", [5.0, 20.0])
def test_taxel_sum_equals_applied_force(f):
    """PLAN §13 V2 gate: taxel sum ~ applied normal force within 5 %."""
    model, data = _scene("box")
    pads = TactilePads(model, prefix=P)
    R = _place(model, data, (0, 0))
    _press(model, data, -f * R[:, 2])
    t = pads.taxels(data)
    assert abs(t[0].sum() - f) < 0.05 * f, t[0]
    assert t[1].sum() == 0
    assert (t[0] > 0).sum() >= 12  # the flat probe loads (nearly) every taxel
    s = pads.summary(data)
    assert np.abs(s[0, 1:3]).max() < 0.001  # centred probe -> CoP near the pad centre


def test_cop_follows_contact_point():
    pitch = DEFAULT_TACTILE.pitch
    cops = []
    for x, y in [
        (-1.5 * pitch, 0.5 * pitch),
        (0.5 * pitch, -1.5 * pitch),
        (1.5 * pitch, 1.5 * pitch),
    ]:
        model, data = _scene("sphere")
        pads = TactilePads(model, prefix=P)
        R = _place(model, data, (x, y), gap=0.0002, half_thick=0.003)
        _press(model, data, -5.0 * R[:, 2])
        s = pads.summary(data)
        depth = (
            data.xpos[model.body("probe").id] - data.site_xpos[model.site(P + "left_pad").id]
        ) @ R[:, 2]
        assert depth > 0.004 + 0.003 - 0.001  # the probe rests on the pad, < 1 mm into it
        assert abs(s[0, 0] - 5.0) < 0.25
        cops.append(s[0, 1:3])
        np.testing.assert_allclose(s[0, 1:3], [x, y], atol=pitch / 2)
    assert len({tuple(np.round(c, 4)) for c in cops}) == 3


def test_pad_shear_sign():
    """Dragging the probe along pad +x shows up as tangential pad force along x."""
    model, data = _scene("box")
    pads = TactilePads(model, prefix=P)
    readings = {}
    for sgn in (1.0, -1.0):
        model, data = _scene("box")
        pads = TactilePads(model, prefix=P)
        R = _place(model, data, (0, 0))
        _press(model, data, -10.0 * R[:, 2] + sgn * 3.0 * R[:, 0])
        readings[sgn] = pads.pad_force(data)[0]
        assert abs(pads.summary(data)[0, 3] - 3.0) < 0.3
    assert np.sign(readings[1.0][0]) == -np.sign(readings[-1.0][0]) != 0
