"""Wrist sensor: sign convention and gravity compensation, pinned with known masses.

The course convention (sensors.py): the compensated wrench is the external contact wrench,
contact pushing UP on the tool = NEGATIVE F_z, a mass resting ON the tool = POSITIVE
F_z = +m*9.81. Each test settles a static scene and checks the analytic expectation.
"""

import mujoco
import numpy as np

from nfv.sim.controllers import GantryController
from nfv.sim.gantry_xml import build_model
from nfv.sim.sensors import WrenchSensor

G = 9.81
BLOCK_MASS = 0.2  # kg, the known mass for the resting test
BLOCK_XML = f"""
<body name="block" pos="0 0 0.43">
  <freejoint name="block"/>
  <geom name="block" type="box" size="0.012 0.012 0.012" mass="{BLOCK_MASS}"
        rgba="0.3 0.6 0.3 1"/>
</body>
"""


def _settle(model, data, steps=600):
    ctl = GantryController(model, data)  # holds the start pose
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return ctl


def test_free_space_compensates_to_zero():
    """Hanging free: raw shows the tool weight, compensated is ~zero (the analytic check)."""
    m, d = build_model()
    _settle(m, d)
    sensor = WrenchSensor(m, d)
    raw = sensor.raw()
    comp = sensor.compensated()
    assert abs(raw[2] - sensor.load_mass * G) < 0.05  # raw = +m_tool*g (parent holds the load)
    assert np.linalg.norm(comp[:3]) < 0.05  # compensated force ~ 0
    assert np.linalg.norm(comp[3:]) < 0.01  # compensated torque ~ 0


def test_known_mass_resting_on_tool_reads_positive_mg():
    """A 0.2 kg block resting on the palm pushes DOWN: F_z = +m*g = +1.962 N."""
    m, d = build_model(worldbody_extra=BLOCK_XML)
    _settle(m, d, steps=1000)
    comp = WrenchSensor(m, d).compensated()
    assert abs(comp[2] - BLOCK_MASS * G) < 0.1
    assert abs(comp[0]) < 0.1 and abs(comp[1]) < 0.1


def test_pressing_down_reads_negative_fz():
    """Pressing the fingers into the floor: the floor pushes UP, so F_z is negative."""
    m, d = build_model()
    _settle(m, d)
    # command the z axis 15 mm past floor contact (fingers touch the floor at jz = -0.34)
    az = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "az")
    for jz_t in np.linspace(0.0, -0.355, 200):  # ramp down over 0.4 s, then settle
        d.ctrl[az] = jz_t
        mujoco.mj_step(m, d)
    for _ in range(500):
        mujoco.mj_step(m, d)
    comp = WrenchSensor(m, d).compensated()
    assert comp[2] < -5.0  # up-push reads negative; ~ -K_p * overshoot


def test_filtered_sample_converges_to_compensated():
    """The 200 Hz low-pass seeds on its first sample and tracks a static wrench exactly."""
    m, d = build_model(worldbody_extra=BLOCK_XML)
    _settle(m, d, steps=1000)
    sensor = WrenchSensor(m, d)
    first = sensor.sample()
    for _ in range(50):
        filt = sensor.sample()
    assert np.allclose(first, sensor.compensated(), atol=1e-6)  # seeded, no startup transient
    assert np.allclose(filt, sensor.compensated(), atol=1e-3)
