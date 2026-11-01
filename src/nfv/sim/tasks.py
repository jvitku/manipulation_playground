"""Tasks T0-T3 of the force-VLA school: geometry, hidden quantities, success criteria.

Design rule (plan §5): every task hides the quantity that decides success from the 96x96
camera, so a policy that ignores force is measurably worse. The hidden quantities live either
purely in Python (T1's click threshold — the rendered scene is PIXEL-IDENTICAL across
thresholds, unit-tested) or as sub-pixel geometry shifts (T2's ±8 mm height / ±3° tilt, T3's
±4 mm hole offset; the static camera covers ~0.6 m with 96 px, so 1 px ≈ 6 mm).

Each task provides:
  * MJCF hook strings (``WORLD_XML``, ``TOOL_XML``, ``CONTACT_XML``) for the gantry builder —
    the geometry is compiled ONCE per task; per-episode randomisation only mutates
    ``model.body_pos`` / ``body_quat`` / free-joint qpos, never recompiles;
  * ``randomize(model, data, rng, overrides)`` — draws placements + hidden params from the
    seed-derived rng and returns the sampled instruction. ``overrides`` force specific hidden
    values (used by tests, e.g. the T1 pixel-identity guard and the 3-height hybrid gate);
    they are applied AFTER all rng draws so they never change the stream;
  * 200 Hz hook ``on_ctrl_sample`` (click latching, wipe coverage/tracking counters),
    ``success`` / ``failure`` checked once per 10 Hz policy step, ``step_info`` extras,
    and ``infer_subtask`` — the heuristic subtask label the env puts in Obs (demo records
    store the expert's own FSM phase instead, which is the cleaner label).

Task table (success criteria from the plan, with this file's concrete numbers):

  T0 reach-and-grasp  "pick up the cube"; nothing hidden; success = cube lifted 5 cm; 10 s.
  T1 press-until-click  two buttons (red/blue); hidden per-button click threshold
      U(3, 9) N; the cap rides a spring (K_SPRING = 1000 N/m), the click fires when the
      spring force k·d crosses the threshold — detected in Python from the joint depression,
      so nothing about the threshold exists in the compiled scene. Success = instructed
      button clicked AND episode peak |F| < 1.5x threshold; clicking the other button fails
      the episode; 10 s.
  T2 wipe  strip on a plate with hidden height ±8 mm and tilt ±3°; "wipe the plate gently"
      = 3 N / "firmly" = 8 N normal-force target. F_n := −F_z,meas (the up-push, positive
      when pressing; tilt ≤ 3° makes the plate normal ≈ z within cos 3° = 0.9986).
      20 coverage cells of 1 cm along the strip; a cell counts when the tip is over it while
      in contact (F_n > 0.5 N). Success = ≥ 90 % cells covered AND |F_n − target| < 1.5 N for
      ≥ 80 % of contact time AND ≥ 1.5 s of contact AND no abort, within 15 s.
  T3 peg-insert  visible marker disk; the TRUE hole centre is marker + hidden offset
      U(±4 mm) per axis (≈ 0.6 px — the camera cannot resolve it); square peg, 0.5 mm
      clearance per side. Success = peg tip seated (depth ≥ 28 mm of the 30 mm walls) inside
      the hole within 20 s, no abort; peak force is logged either way.

Shapes/units: positions m, forces N; wrenches (6,) in the sensors.py sign convention.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from nfv.sim.gantry_xml import FINGER_TIP_Z, START_HEIGHT

START_CLEAR_Z = 0.08  # m, world z of the LOWEST tool point at the episode start

# Shared press/wipe tip (tasks T1, T2): a small sphere under the palm, below the finger tips,
# so pressing never lands the fingers on the surface (18 mm of clearance > the 8 mm hidden
# height + tilt).  Friction 0.4 keeps the wipe drag moderate (contact friction is the
# elementwise max of the two geoms, the plate carries 0.4 too).  The tip's contact is made
# compliant (solref 0.02 -> ~2 mm give at 5 N): with the repo-default stiff contact, sliding
# under force control chatters — the raw impact spikes pass the 20 Hz filter late and the
# admittance answers by hopping (measured: in-band force time 30 % stiff vs ~90 % compliant
# at the worst hidden tilt). A real wipe pad is compliant; this is that, written down.
TIP_TIP_Z = -0.078  # m, lowest tip point relative to the tool origin
TIP_XML = """
<geom name="tip" type="sphere" size="0.008" pos="0 0 -0.07" mass="0.01"
      friction="0.4 0.005 0.0001" solref="0.02 1" rgba="0.85 0.75 0.2 1"/>
"""

PEG_TIP_Z = -0.110  # m, peg tip relative to the tool origin (task T3)
PEG_HALF = 0.008  # m, square peg half-width
PEG_XML = f"""
<geom name="peg" type="box" size="{PEG_HALF:g} {PEG_HALF:g} 0.025" pos="0 0 -0.085" mass="0.05"
      rgba="0.2 0.4 0.9 1"/>
"""


def _quat_tilt(rx: float, ry: float) -> np.ndarray:
    """Quaternion (w, x, y, z) composing small tilts rx about x and ry about y.

    Hamilton product of (cx, sx, 0, 0) and (cy, 0, sy, 0); at ±3° the order is immaterial.
    """
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    return np.array([cx * cy, sx * cy, cx * sy, sx * sy])


def _body_id(model: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise KeyError(f"body {name!r} not in model")
    return i


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if i < 0:
        raise KeyError(f"geom {name!r} not in model")
    return i


def _joint_qposadr(model: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if i < 0:
        raise KeyError(f"joint {name!r} not in model")
    return int(model.jnt_qposadr[i])


class Task:
    """Base class; subclasses override the class attributes and the hooks they need."""

    NAME = ""
    TIME_LIMIT_S = 10.0
    SUBTASKS: tuple[str, ...] = ("approach",)
    INSTRUCTIONS: tuple[str, ...] = ()
    WORLD_XML = ""
    TOOL_XML = ""
    CONTACT_XML = ""
    LOWEST_TOOL_Z = FINGER_TIP_Z  # lowest tool point relative to the tool origin

    def __init__(self) -> None:
        self.instruction = ""

    @property
    def start_jz(self) -> float:
        """jz putting the lowest tool point at START_CLEAR_Z above the floor."""
        return START_CLEAR_Z - self.LOWEST_TOOL_Z - START_HEIGHT

    # --- per-episode lifecycle (called by Env.reset in this order) ---
    def bind(self, model: mujoco.MjModel) -> None:
        """Cache ids on the compiled model (once per Env/task)."""

    def randomize(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
        overrides: dict | None = None,
    ) -> str:
        """Place objects, draw hidden params, return the sampled instruction."""
        raise NotImplementedError

    def set_instruction(self, instruction: str) -> None:
        """Override the sampled instruction (must be one of INSTRUCTIONS)."""
        if instruction not in self.INSTRUCTIONS:
            raise ValueError(f"{instruction!r} not in {self.INSTRUCTIONS}")
        self.instruction = instruction

    def after_settle(self, env) -> None:
        """Called once after the reset settle phase (capture rest poses etc.)."""

    # --- per-step hooks ---
    def on_ctrl_sample(self, env, wrench: np.ndarray) -> None:
        """200 Hz hook with the latest filtered wrench (click latching, wipe counters)."""

    def success(self, env) -> bool:
        raise NotImplementedError

    def failure(self, env) -> str | None:
        """A task-specific failure outcome name, or None (abort/timeout live in the env)."""
        return None

    def step_info(self, env) -> dict:
        return {}

    def infer_subtask(self, env) -> int:
        return 0


class ReachGrasp(Task):
    """T0: pick up the cube and lift it 5 cm. Nothing hidden — the no-regression check."""

    NAME = "t0"
    TIME_LIMIT_S = 10.0
    SUBTASKS = ("reach", "grasp", "lift")
    INSTRUCTIONS = ("pick up the cube",)
    WORLD_XML = """
    <body name="cube" pos="0 0.08 0.016">
      <freejoint name="cube"/>
      <geom name="cube" type="box" size="0.015 0.015 0.015" mass="0.05" rgba="0.85 0.35 0.10 1"/>
    </body>
    """

    def bind(self, model: mujoco.MjModel) -> None:
        self._cube_qadr = _joint_qposadr(model, "cube")
        self._cube_bid = _body_id(model, "cube")

    def randomize(self, model, data, rng, overrides=None):
        xy = rng.uniform(-0.10, 0.10, size=2)
        adr = self._cube_qadr
        data.qpos[adr : adr + 3] = (xy[0], xy[1], 0.016)
        data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.cube_z0 = 0.015  # refined after settle
        self.instruction = self.INSTRUCTIONS[0]
        return self.instruction

    def after_settle(self, env) -> None:
        self.cube_z0 = float(env.data.xpos[self._cube_bid][2])

    def cube_pos(self, env) -> np.ndarray:
        """Privileged: the cube body position (3,) in the world (experts only)."""
        return np.asarray(env.data.xpos[self._cube_bid], dtype=np.float64)

    def success(self, env) -> bool:
        return float(env.data.xpos[self._cube_bid][2]) > self.cube_z0 + 0.05

    def step_info(self, env) -> dict:
        return {"cube_lift_m": float(env.data.xpos[self._cube_bid][2]) - self.cube_z0}

    def infer_subtask(self, env) -> int:
        cube = self.cube_pos(env)
        if cube[2] > self.cube_z0 + 0.005 and env.grip_norm > 0.3:
            return 2  # lift
        tool = env.tool_pos
        if np.linalg.norm(cube[:2] - tool[:2]) < 0.02 and tool[2] < 0.12:
            return 1  # grasp
        return 0  # reach


class PressButton(Task):
    """T1: press the instructed button until it clicks; the threshold is invisible.

    The cap rides a vertical slide joint with XML spring stiffness K_SPRING; the click fires
    when the spring force K_SPRING * depression crosses the hidden threshold — checked at
    200 Hz in Python and latched, so two episodes with different thresholds compile and render
    identically (the course's core premise; unit-tested). The quasi-static press force the
    tool feels at the click is threshold + cap weight (~0.2 N).
    """

    NAME = "t1"
    TIME_LIMIT_S = 10.0
    SUBTASKS = ("approach", "press", "done")
    INSTRUCTIONS = ("press the red button", "press the blue button")
    COLORS = ("red", "blue")
    LOWEST_TOOL_Z = TIP_TIP_Z
    TOOL_XML = TIP_XML
    K_SPRING = 1000.0  # N/m, must match the cap joints' stiffness below
    CAP_TOP_Z = 0.048  # m, world z of the cap top at rest
    WORLD_XML = """
    <body name="button_red" pos="-0.08 0 0">
      <geom name="base_red" type="cylinder" size="0.030 0.015" pos="0 0 0.015"
            rgba="0.50 0.50 0.55 1"/>
      <body name="cap_red_body" pos="0 0 0.040">
        <joint name="jcap_red" type="slide" axis="0 0 1" range="-0.012 0"
               stiffness="1000" damping="10"/>
        <geom name="cap_red" type="cylinder" size="0.022 0.008" mass="0.02"
              rgba="0.85 0.10 0.10 1"/>
      </body>
    </body>
    <body name="button_blue" pos="0.08 0 0">
      <geom name="base_blue" type="cylinder" size="0.030 0.015" pos="0 0 0.015"
            rgba="0.50 0.50 0.55 1"/>
      <body name="cap_blue_body" pos="0 0 0.040">
        <joint name="jcap_blue" type="slide" axis="0 0 1" range="-0.012 0"
               stiffness="1000" damping="10"/>
        <geom name="cap_blue" type="cylinder" size="0.022 0.008" mass="0.02"
              rgba="0.10 0.20 0.85 1"/>
      </body>
    </body>
    """
    # the cap sinks past the base top when pressed; they must never collide with each other
    CONTACT_XML = """
    <exclude body1="button_red" body2="cap_red_body"/>
    <exclude body1="button_blue" body2="cap_blue_body"/>
    """

    def bind(self, model: mujoco.MjModel) -> None:
        self._bid = {c: _body_id(model, f"button_{c}") for c in self.COLORS}
        self._cap_qadr = {c: _joint_qposadr(model, f"jcap_{c}") for c in self.COLORS}

    def randomize(self, model, data, rng, overrides=None):
        # placements: one button left, one right, ±jitter; sides swap with p = 0.5
        xs = 0.07 + rng.uniform(0.0, 0.04, size=2)
        ys = rng.uniform(-0.05, 0.05, size=2)
        swap = bool(rng.random() < 0.5)
        self.threshold = {c: float(rng.uniform(3.0, 9.0)) for c in self.COLORS}
        self.target = self.COLORS[int(rng.integers(2))]
        sides = (-1.0, 1.0) if not swap else (1.0, -1.0)
        for c, side, x, y in zip(self.COLORS, sides, xs, ys, strict=True):
            model.body_pos[self._bid[c]] = (side * x, y, 0.0)
        ov = overrides or {}
        if "threshold" in ov:  # one value for every button (the pixel-identity test)
            self.threshold = {c: float(ov["threshold"]) for c in self.COLORS}
        for c in self.COLORS:
            if f"threshold_{c}" in ov:
                self.threshold[c] = float(ov[f"threshold_{c}"])
        if "target" in ov:
            self.target = str(ov["target"])
        self.clicked = {c: False for c in self.COLORS}
        self.instruction = self.INSTRUCTIONS[self.COLORS.index(self.target)]
        return self.instruction

    def set_instruction(self, instruction: str) -> None:
        super().set_instruction(instruction)
        self.target = self.COLORS[self.INSTRUCTIONS.index(instruction)]

    def button_pos(self, env, color: str) -> np.ndarray:
        """Privileged: a button's base position (3,) in the world (experts only)."""
        return np.asarray(env.model.body_pos[self._bid[color]], dtype=np.float64)

    def depression(self, env, color: str) -> float:
        """Current cap depression d >= 0 (m); spring force = K_SPRING * d."""
        return max(0.0, -float(env.data.qpos[self._cap_qadr[color]]))

    def on_ctrl_sample(self, env, wrench: np.ndarray) -> None:
        # 200 Hz hook; wrench (6,) is the filtered stream (unused: the click is kinematic)
        for c in self.COLORS:
            if not self.clicked[c]:
                # click depression (m): the spring force K_SPRING * d crosses the threshold
                d_click = self.threshold[c] / self.K_SPRING
                if self.depression(env, c) >= d_click:
                    self.clicked[c] = True  # latched

    def success(self, env) -> bool:
        return (
            self.clicked[self.target]
            and env.peak_force < 1.5 * self.threshold[self.target]
            and self.failure(env) is None
        )

    def failure(self, env) -> str | None:
        other = self.COLORS[1 - self.COLORS.index(self.target)]
        if self.clicked[other]:
            return "wrong_button"
        if self.clicked[self.target] and env.peak_force >= 1.5 * self.threshold[self.target]:
            return "over_force"
        return None

    def step_info(self, env) -> dict:
        return {
            "click": self.clicked[self.target],
            "threshold_n": self.threshold[self.target],
            "depression_mm": self.depression(env, self.target) * 1e3,
        }

    def infer_subtask(self, env) -> int:
        if self.clicked[self.target]:
            return 2  # done
        if np.linalg.norm(env.wrench[:3]) > 0.5:
            return 1  # press
        return 0  # approach


class Wipe(Task):
    """T2: wipe the strip at the commanded normal force on a surface of hidden height/tilt."""

    NAME = "t2"
    TIME_LIMIT_S = 15.0
    SUBTASKS = ("approach", "wipe", "done")
    INSTRUCTIONS = ("wipe the plate gently", "wipe the plate firmly")
    FORCE_TARGETS = (3.0, 8.0)  # N, aligned with INSTRUCTIONS
    LOWEST_TOOL_Z = TIP_TIP_Z
    TOOL_XML = TIP_XML
    PLATE_Z0 = 0.012  # m, nominal plate body height (top at PLATE_Z0 + 0.010)
    HIDE_DZ = 0.008  # m, hidden height range ±
    HIDE_TILT = math.radians(3.0)  # rad, hidden tilt range ± about x and y
    N_CELLS = 20  # coverage cells along the strip
    CELL_W = 0.01  # m
    STRIP_HALF_Y = 0.02  # m, lateral tolerance for a cell hit
    CONTACT_N = 0.5  # N, F_n above this counts as contact
    TRACK_TOL_N = 1.5  # N, |F_n − target| below this counts as tracked
    TRACK_FRAC = 0.8  # required tracked / contact sample ratio
    COVER_FRAC = 0.9  # required covered-cell fraction
    MIN_CONTACT_S = 1.5  # s of contact before success can latch
    WORLD_XML = """
    <body name="plate" pos="0 0 0.012">
      <geom name="plate" type="box" size="0.14 0.09 0.010" rgba="0.80 0.80 0.85 1"
            friction="0.4 0.005 0.0001"/>
      <geom name="strip" type="box" size="0.10 0.015 0.0008" pos="0 0 0.0112"
            rgba="0.95 0.80 0.20 1" contype="0" conaffinity="0"/>
    </body>
    """

    def bind(self, model: mujoco.MjModel) -> None:
        self._plate_bid = _body_id(model, "plate")
        self._tip_gid = _geom_id(model, "tip")

    def randomize(self, model, data, rng, overrides=None):
        dz = float(rng.uniform(-self.HIDE_DZ, self.HIDE_DZ))
        rx = float(rng.uniform(-self.HIDE_TILT, self.HIDE_TILT))
        ry = float(rng.uniform(-self.HIDE_TILT, self.HIDE_TILT))
        pick = int(rng.integers(2))
        ov = overrides or {}
        dz = float(ov.get("dz", dz))
        rx = float(ov.get("tilt_x", rx))
        ry = float(ov.get("tilt_y", ry))
        if "instruction" in ov:
            pick = self.INSTRUCTIONS.index(ov["instruction"])
        self.hidden = {"dz": dz, "tilt_x": rx, "tilt_y": ry}
        model.body_pos[self._plate_bid] = (0.0, 0.0, self.PLATE_Z0 + dz)
        model.body_quat[self._plate_bid] = _quat_tilt(rx, ry)
        self.f_target_n = self.FORCE_TARGETS[pick]
        self.covered = np.zeros(self.N_CELLS, dtype=bool)
        self.contact_n = 0  # 200 Hz samples in contact
        self.track_n = 0  # of those, samples inside the tracking band
        self.instruction = self.INSTRUCTIONS[pick]
        return self.instruction

    def set_instruction(self, instruction: str) -> None:
        super().set_instruction(instruction)
        self.f_target_n = self.FORCE_TARGETS[self.INSTRUCTIONS.index(instruction)]

    def normal_force(self, wrench: np.ndarray) -> float:
        """F_n = −F_z,meas: positive when the tool presses on the plate (tilt ≤ 3° ≈ z)."""
        return -float(wrench[2])

    def on_ctrl_sample(self, env, wrench: np.ndarray) -> None:
        f_n = self.normal_force(wrench)
        if f_n <= self.CONTACT_N:
            return
        self.contact_n += 1
        if abs(f_n - self.f_target_n) < self.TRACK_TOL_N:
            self.track_n += 1
        # coverage: tip position in the plate's (hidden, tilted) frame
        tip_w = env.data.geom_xpos[self._tip_gid]
        r = np.asarray(env.data.xmat[self._plate_bid]).reshape(3, 3)
        p = r.T @ (tip_w - env.data.xpos[self._plate_bid])
        if abs(p[1]) <= self.STRIP_HALF_Y and abs(p[0]) <= 0.10:
            cell = int(np.clip((p[0] + 0.095) / self.CELL_W + 0.5, 0, self.N_CELLS - 1))
            self.covered[cell] = True

    @property
    def coverage(self) -> float:
        return float(np.mean(self.covered))

    @property
    def track_frac(self) -> float:
        return self.track_n / self.contact_n if self.contact_n else 0.0

    def plate_top_at(self, env, x: float, y: float) -> float:
        """Privileged: world z of the plate top surface under (x, y) (experts only)."""
        r = np.asarray(env.data.xmat[self._plate_bid]).reshape(3, 3)
        pos = np.asarray(env.data.xpos[self._plate_bid])
        # top surface plane: point pos + R @ (0,0,0.010), normal R[:,2]
        c = pos + r @ np.array([0.0, 0.0, 0.010])
        n = r[:, 2]
        return float(c[2] - (n[0] * (x - c[0]) + n[1] * (y - c[1])) / n[2])

    def success(self, env) -> bool:
        from nfv.sim.controllers import CTRL_DT

        return (
            self.coverage >= self.COVER_FRAC
            and self.contact_n * CTRL_DT >= self.MIN_CONTACT_S
            and self.track_frac >= self.TRACK_FRAC
        )

    def step_info(self, env) -> dict:
        return {
            "coverage": self.coverage,
            "track_frac": self.track_frac,
            "f_n": self.normal_force(env.wrench),
            "f_target_n": self.f_target_n,
        }

    def infer_subtask(self, env) -> int:
        if self.coverage >= self.COVER_FRAC:
            return 2  # done
        if self.normal_force(env.wrench) > self.CONTACT_N:
            return 1  # wipe
        return 0  # approach


class PegInsert(Task):
    """T3: insert the peg; the hole is offset ±4 mm from the visible marker (sub-pixel)."""

    NAME = "t3"
    TIME_LIMIT_S = 20.0
    SUBTASKS = ("approach", "descend", "search", "insert", "done")
    INSTRUCTIONS = ("insert the peg",)
    LOWEST_TOOL_Z = PEG_TIP_Z
    TOOL_XML = PEG_XML
    HOLE_HALF = PEG_HALF + 0.0005  # 0.5 mm clearance per side
    HOLE_TOP_Z = 0.040  # m, walls span z 0.010 .. 0.040
    SEAT_TIP_Z = 0.012  # m, success when the peg tip is at/below this inside the hole
    OFFSET_MM = 4.0  # hidden |offset| bound per axis
    # walls after fvb/scenes/builder.py: wall thickness 0.02, x-walls span the full width,
    # y-walls span the opening only; hh = HOLE_HALF = 0.0085 -> px_pos = 0.0285
    WORLD_XML = """
    <body name="hole" pos="0 0.05 0.025">
      <geom name="h_px" type="box" size="0.02 0.0485 0.015" pos="0.0285 0 0"
            rgba="0.55 0.55 0.60 1"/>
      <geom name="h_nx" type="box" size="0.02 0.0485 0.015" pos="-0.0285 0 0"
            rgba="0.55 0.55 0.60 1"/>
      <geom name="h_py" type="box" size="0.0085 0.02 0.015" pos="0 0.0285 0"
            rgba="0.55 0.55 0.60 1"/>
      <geom name="h_ny" type="box" size="0.0085 0.02 0.015" pos="0 -0.0285 0"
            rgba="0.55 0.55 0.60 1"/>
    </body>
    <body name="marker" pos="0 0.05 0">
      <geom name="marker" type="cylinder" size="0.012 0.0005" pos="0 0 0.0405"
            rgba="0.10 0.90 0.30 1" contype="0" conaffinity="0"/>
    </body>
    """

    def bind(self, model: mujoco.MjModel) -> None:
        self._hole_bid = _body_id(model, "hole")
        self._marker_bid = _body_id(model, "marker")
        self._peg_gid = _geom_id(model, "peg")

    def randomize(self, model, data, rng, overrides=None):
        mx, my = rng.uniform(-0.08, 0.08, size=2)
        ox, oy = rng.uniform(-self.OFFSET_MM, self.OFFSET_MM, size=2) * 1e-3
        ov = overrides or {}
        ox = float(ov.get("offset_x", ox))
        oy = float(ov.get("offset_y", oy))
        self.marker_xy = np.array([mx, my])
        self.offset = np.array([ox, oy])  # hidden
        self.hole_xy = self.marker_xy + self.offset
        model.body_pos[self._marker_bid] = (mx, my, 0.0)
        model.body_pos[self._hole_bid] = (self.hole_xy[0], self.hole_xy[1], 0.025)
        self.instruction = self.INSTRUCTIONS[0]
        return self.instruction

    def peg_tip(self, env) -> np.ndarray:
        """World position (3,) of the peg tip (bottom-face centre)."""
        tip = np.asarray(env.data.geom_xpos[self._peg_gid], dtype=np.float64).copy()
        tip[2] -= 0.025  # half the peg geom height (yaw is held at 0, the peg stays upright)
        return tip

    def depth_mm(self, env) -> float:
        """Insertion depth of the tip below the hole top (mm, clipped at 0)."""
        return max(0.0, (self.HOLE_TOP_Z - self.peg_tip(env)[2]) * 1e3)

    def success(self, env) -> bool:
        tip = self.peg_tip(env)
        in_hole = np.linalg.norm(tip[:2] - self.hole_xy) < self.HOLE_HALF
        return in_hole and tip[2] <= self.SEAT_TIP_Z

    def step_info(self, env) -> dict:
        return {"depth_mm": self.depth_mm(env), "peak_force_n": env.peak_force}

    def infer_subtask(self, env) -> int:
        if self.success(env):
            return 4  # done
        depth = self.depth_mm(env)
        if depth > 5.0:
            return 3  # insert
        contact = np.linalg.norm(env.wrench[:3]) > 1.0
        if contact and depth <= 5.0:
            return 2  # search
        tip = self.peg_tip(env)
        if np.linalg.norm(tip[:2] - self.marker_xy) < 0.02:
            return 1  # descend
        return 0  # approach


TASKS: dict[str, type[Task]] = {
    "t0": ReachGrasp,
    "t1": PressButton,
    "t2": Wipe,
    "t3": PegInsert,
}


def make(name: str) -> Task:
    """Instantiate a task by name ('t0'..'t3')."""
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}, expected one of {sorted(TASKS)}")
    return TASKS[name]()
