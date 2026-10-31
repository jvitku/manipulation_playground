"""Parametrised MJCF builder for the nfv Cartesian gantry (style of fvb/scenes/builder.py).

The robot is a rigid gantry: three prismatic joints (jx, jy, jz, world-aligned), a yaw hinge
(jyaw) and a parallel gripper (jgl, jgr; two opposed slide joints driven by one scalar command).
Between the z-carriage and the tool sits the wrist: site ``wrist`` on the ``tool`` body carries a
``force`` + ``torque`` sensor pair, so the sensor sees everything distal to it — palm, fingers
and whatever task geometry rides on the tool. A grasped object is *not* part of that subtree;
its weight arrives as a contact wrench, which is exactly what a real F/T sensor would report.

Frames and units: world z up, gravity -9.81 m/s^2; metres, newtons, seconds. The prismatic
axes equal the world axes (yaw is position-held at 0 in M0), so a Cartesian step dp is exactly
a joint-space step — the reason the course uses a gantry at all.

Cameras (both render at 96x96, the offscreen buffer is sized to match):
  * ``static`` — fixed in the world, framing the tabletop workspace from the front.
  * ``wrist``  — on the tool, looking down at the tool tip.

Task geometry is inserted through the three hook strings of :func:`build_xml`:
``worldbody_extra`` (objects on the floor), ``tool_extra`` (geometry riding on the tool, e.g.
the press tip or the peg) and ``assets_extra`` (extra textures/materials). ``contact_extra``
takes ``<contact>`` children (e.g. collision excludes for button caps).

Key fixed dimensions (used by tasks and experts):
  * ``START_HEIGHT`` = 0.40 — world z of the tool origin at jz = 0.
  * ``FINGER_TIP_Z`` = -0.060 — lowest finger point relative to the tool origin.
  * fingers open to a 72 mm gap at grip = 0 and close to 12 mm at grip = 1
    (``GRIP_TRAVEL`` = 0.03 per finger).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

START_HEIGHT = 0.40  # m, world z of the tool origin at jz = 0
FINGER_TIP_Z = -0.060  # m, lowest finger point relative to the tool origin
GRIP_TRAVEL = 0.03  # m, inward slide range of each finger (grip command 0..1 maps onto this)


@dataclass
class GantryParams:
    """Everything the gantry XML hard-codes, as fields (fvb SceneParams pattern)."""

    # controller (position actuators = springs; this is the exposed K_p of the position mode)
    kp: float = 800.0  # N/m per prismatic axis
    joint_damping: float = 60.0  # N·s/m, the D term of the PD
    yaw_kp: float = 5.0  # N·m/rad
    yaw_damping: float = 0.2
    grip_kp: float = 500.0  # N/m per finger
    grip_damping: float = 5.0
    grip_force_max: float = 20.0  # N, per finger (actuator forcerange)
    # masses
    link_mass: float = 0.5  # kg, each gantry carriage (inertia only, no geom)
    tool_mass: float = 0.15  # kg, the palm plate
    finger_mass: float = 0.02  # kg each
    # joint ranges
    x_range: tuple[float, float] = (-0.30, 0.30)
    y_range: tuple[float, float] = (-0.30, 0.30)
    z_range: tuple[float, float] = (-0.45, 0.05)
    # solver / contact (fvb Track-A values; proven for the 0.5 mm-clearance insert)
    timestep: float = 0.002
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    solref: tuple[float, float] = (0.005, 1.0)
    solimp: tuple[float, float, float] = (0.9, 0.95, 0.001)
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    condim: int = 4
    solver_iterations: int = 100
    # offscreen render targets (both cameras render at this size)
    offwidth: int = 96
    offheight: int = 96


def _fmt(v) -> str:
    if isinstance(v, (tuple, list)):
        return " ".join(f"{x:g}" for x in v)
    return f"{v:g}"


def build_xml(
    p: GantryParams | None = None,
    worldbody_extra: str = "",
    tool_extra: str = "",
    assets_extra: str = "",
    contact_extra: str = "",
) -> str:
    """Return the MJCF string for the gantry plus the task geometry passed in the hooks."""
    p = p or GantryParams()
    return f"""<mujoco model="nfv_gantry">
  <option timestep="{p.timestep:g}" integrator="{p.integrator}" cone="{p.cone}"
          iterations="{p.solver_iterations}"/>
  <visual>
    <global offwidth="{p.offwidth}" offheight="{p.offheight}"/>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.72 0.80 0.90" rgb2="0.35 0.40 0.50"
             width="32" height="32"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.75 0.75 0.78" rgb2="0.62 0.62 0.66"
             width="64" height="64"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0"/>
    {assets_extra}
  </asset>
  <default>
    <geom condim="{p.condim}" friction="{_fmt(p.friction)}" solref="{_fmt(p.solref)}"
          solimp="{_fmt(p.solimp)}"/>
  </default>
  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.25 0.25 -1" diffuse="0.6 0.6 0.6"/>
    <light pos="-0.4 0.4 1.0" dir="0.3 -0.3 -1" diffuse="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="1 1 0.1" material="grid"/>
    <camera name="static" pos="0 -0.55 0.5" xyaxes="1 0 0 0 0.63324 0.77397" fovy="45"/>
    <body name="gantry_x" pos="0 0 {START_HEIGHT:g}">
      <joint name="jx" type="slide" axis="1 0 0" range="{_fmt(p.x_range)}"
             damping="{p.joint_damping:g}"/>
      <inertial pos="0 0 0" mass="{p.link_mass:g}" diaginertia="1e-3 1e-3 1e-3"/>
      <body name="gantry_y">
        <joint name="jy" type="slide" axis="0 1 0" range="{_fmt(p.y_range)}"
               damping="{p.joint_damping:g}"/>
        <inertial pos="0 0 0" mass="{p.link_mass:g}" diaginertia="1e-3 1e-3 1e-3"/>
        <body name="gantry_z">
          <joint name="jz" type="slide" axis="0 0 1" range="{_fmt(p.z_range)}"
                 damping="{p.joint_damping:g}"/>
          <inertial pos="0 0 0" mass="{p.link_mass:g}" diaginertia="1e-3 1e-3 1e-3"/>
          <body name="tool">
            <joint name="jyaw" type="hinge" axis="0 0 1" damping="{p.yaw_damping:g}"/>
            <site name="wrist" pos="0 0 0" size="0.005" rgba="1 0.2 0.2 1"/>
            <camera name="wrist" pos="0 0.07 0" xyaxes="1 0 0 0 0.75257 -0.65850" fovy="60"/>
            <geom name="palm" type="box" size="0.05 0.025 0.01" mass="{p.tool_mass:g}"
                  rgba="0.30 0.30 0.36 1"/>
            <body name="finger_l" pos="-0.04 0 -0.01">
              <joint name="jgl" type="slide" axis="1 0 0" range="0 {GRIP_TRAVEL:g}"
                     damping="{p.grip_damping:g}"/>
              <geom name="finger_l" type="box" size="0.004 0.012 0.025" pos="0 0 -0.025"
                    mass="{p.finger_mass:g}" rgba="0.22 0.22 0.27 1"/>
            </body>
            <body name="finger_r" pos="0.04 0 -0.01">
              <joint name="jgr" type="slide" axis="-1 0 0" range="0 {GRIP_TRAVEL:g}"
                     damping="{p.grip_damping:g}"/>
              <geom name="finger_r" type="box" size="0.004 0.012 0.025" pos="0 0 -0.025"
                    mass="{p.finger_mass:g}" rgba="0.22 0.22 0.27 1"/>
            </body>
            {tool_extra}
          </body>
        </body>
      </body>
    </body>
    {worldbody_extra}
  </worldbody>
  <contact>
    {contact_extra}
  </contact>
  <actuator>
    <position name="ax" joint="jx" kp="{p.kp:g}" ctrlrange="{_fmt(p.x_range)}"/>
    <position name="ay" joint="jy" kp="{p.kp:g}" ctrlrange="{_fmt(p.y_range)}"/>
    <position name="az" joint="jz" kp="{p.kp:g}" ctrlrange="{_fmt(p.z_range)}"/>
    <position name="ayaw" joint="jyaw" kp="{p.yaw_kp:g}" ctrlrange="-3.2 3.2"/>
    <position name="agl" joint="jgl" kp="{p.grip_kp:g}" ctrlrange="0 {GRIP_TRAVEL:g}"
              forcerange="-{p.grip_force_max:g} {p.grip_force_max:g}"/>
    <position name="agr" joint="jgr" kp="{p.grip_kp:g}" ctrlrange="0 {GRIP_TRAVEL:g}"
              forcerange="-{p.grip_force_max:g} {p.grip_force_max:g}"/>
  </actuator>
  <sensor>
    <force name="wrist_force" site="wrist"/>
    <torque name="wrist_torque" site="wrist"/>
  </sensor>
</mujoco>
"""


def build_model(
    p: GantryParams | None = None,
    worldbody_extra: str = "",
    tool_extra: str = "",
    assets_extra: str = "",
    contact_extra: str = "",
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the gantry + extras into a model/data pair."""
    xml = build_xml(p, worldbody_extra, tool_extra, assets_extra, contact_extra)
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)
