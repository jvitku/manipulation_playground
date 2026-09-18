"""Parametrised Track A peg-in-hole MJCF builder.

Everything the seed scene (PLAN.md §6) hard-codes is a field of :class:`SceneParams`.
``build_xml`` returns the MJCF string; ``build_model`` compiles it.

Geometry: square hole of half-width ``hole_half`` = peg_half + clearance, depth 40 mm,
no chamfer. The peg body hangs under ``wrist`` with the F/T site at the body origin, so
the sensor sees everything distal to the wrist (peg weight, inertia, contact).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import mujoco

INTEGRATORS = ("Euler", "RK4", "implicit", "implicitfast")
CONES = ("pyramidal", "elliptic")


@dataclass
class SceneParams:
    # geometry
    peg_half: float = 0.010  # m, half-width of square peg
    peg_len: float = 0.060  # m, full length
    clearance: float = 0.0005  # m per side
    hole_depth: float = 0.040  # m
    peg_mass: float = 0.1  # kg
    start_height: float = 0.15  # m, wrist origin above floor
    # controller (position actuators = springs)
    kp: float = 800.0
    joint_damping: float = 20.0
    link_mass: float = 0.5
    # solver / contact
    timestep: float = 0.002
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    solref: tuple[float, float] = (0.005, 1.0)
    solimp: tuple[float, float, float] = (0.9, 0.95, 0.001)
    friction: tuple[float, float, float] = (0.6, 0.005, 0.0001)
    noslip_iterations: int = 0
    solver_iterations: int = 100
    condim: int = 4
    # wrist tilt joint (M2 tilt experiment). None = no hinge.
    tilt_axis: tuple[float, float, float] | None = None
    tilt_kp: float = 50.0
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _fmt(v) -> str:
    if isinstance(v, (tuple, list)):
        return " ".join(f"{x:g}" for x in v)
    return f"{v:g}"


def build_xml(p: SceneParams | None = None) -> str:
    p = p or SceneParams()
    assert p.integrator in INTEGRATORS, p.integrator
    assert p.cone in CONES, p.cone
    hh = p.peg_half + p.clearance  # hole half-opening
    wall = 0.02  # wall thickness (half-size along thin dimension)
    hz = p.hole_depth / 2
    px_pos = hh + wall
    py_len = hh  # y-walls span the opening width only
    tilt_joint = ""
    if p.tilt_axis is not None:
        tilt_joint = f'<joint name="jtilt" type="hinge" axis="{_fmt(p.tilt_axis)}" damping="0.5"/>'
    tilt_act = f'<position name="atilt" joint="jtilt" kp="{p.tilt_kp:g}"/>' if p.tilt_axis else ""
    return f"""<mujoco model="peg_in_hole">
  <option timestep="{p.timestep:g}" integrator="{p.integrator}" cone="{p.cone}"
          iterations="{p.solver_iterations}" noslip_iterations="{p.noslip_iterations}"/>
  <default>
    <geom condim="{p.condim}" friction="{_fmt(p.friction)}" solref="{_fmt(p.solref)}"
          solimp="{_fmt(p.solimp)}"/>
  </default>
  <worldbody>
    <light pos="0 0 1" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 .1" rgba="0.8 0.8 0.8 1"/>
    <body name="hole" pos="0 0 {hz:g}">
      <geom name="w_px" type="box" size="{wall:g} {hh + 2 * wall:g} {hz:g}" pos=" {px_pos:g} 0 0"/>
      <geom name="w_nx" type="box" size="{wall:g} {hh + 2 * wall:g} {hz:g}" pos="-{px_pos:g} 0 0"/>
      <geom name="w_py" type="box" size="{py_len:g} {wall:g} {hz:g}" pos="0  {px_pos:g} 0"/>
      <geom name="w_ny" type="box" size="{py_len:g} {wall:g} {hz:g}" pos="0 -{px_pos:g} 0"/>
    </body>
    <body name="gantry_x" pos="0 0 {p.start_height:g}">
      <joint name="jx" type="slide" axis="1 0 0" damping="{p.joint_damping:g}"/>
      <inertial pos="0 0 0" mass="{p.link_mass:g}" diaginertia="1e-3 1e-3 1e-3"/>
      <body name="gantry_y">
        <joint name="jy" type="slide" axis="0 1 0" damping="{p.joint_damping:g}"/>
        <inertial pos="0 0 0" mass="{p.link_mass:g}" diaginertia="1e-3 1e-3 1e-3"/>
        <body name="wrist">
          <joint name="jz" type="slide" axis="0 0 1" damping="{p.joint_damping:g}"/>
          {tilt_joint}
          <inertial pos="0 0 0" mass="{p.link_mass:g}" diaginertia="1e-3 1e-3 1e-3"/>
          <body name="peg" pos="0 0 -0.01">
            <site name="ft" pos="0 0 0" size="0.004" rgba="1 0 0 1"/>
            <geom name="peg" type="box" size="{p.peg_half:g} {p.peg_half:g} {p.peg_len / 2:g}"
                  pos="0 0 -{p.peg_len / 2:g}" mass="{p.peg_mass:g}" rgba="0.2 0.4 0.9 1"/>
          </body>
        </body>
      </body>
    </body>
    <camera name="side" pos="0.35 -0.35 0.2" xyaxes="0.707 0.707 0 -0.3 0.3 0.9"/>
  </worldbody>
  <actuator>
    <position name="ax" joint="jx" kp="{p.kp:g}"/>
    <position name="ay" joint="jy" kp="{p.kp:g}"/>
    <position name="az" joint="jz" kp="{p.kp:g}"/>
    {tilt_act}
  </actuator>
  <sensor>
    <force name="ft_force" site="ft"/>
    <torque name="ft_torque" site="ft"/>
  </sensor>
</mujoco>
"""


def build_model(p: SceneParams | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = mujoco.MjModel.from_xml_string(build_xml(p))
    return m, mujoco.MjData(m)


def peg_tip_height(p: SceneParams, jz: float) -> float:
    """World z of the peg's bottom face for a given jz (wrist slide joint value)."""
    return p.start_height + jz - 0.01 - p.peg_len


def insertion_depth(p: SceneParams, jz: float) -> float:
    """How far the peg tip is below the hole rim (positive = inside the hole)."""
    return p.hole_depth - peg_tip_height(p, jz)
