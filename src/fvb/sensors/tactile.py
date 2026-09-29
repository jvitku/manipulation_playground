"""Franka Hand with 4 x 4 taxel finger pads (PLAN §13, V2).

``tactile_gripper_xml`` rewrites robosuite's ``panda_gripper.xml``:

* the finger force limit is raised from robosuite's 20 N to the real Franka Hand's 70 N
  (continuous grasp force);
* each 16 x 8 x 16 mm pad box is replaced by ``rows x cols`` small box geoms ("taxels") with a
  rubber-like friction. MuJoCo makes only a few contact points between two boxes, so a touch
  sensor over one pad would give a blocky, mostly empty map; a geom per taxel gives every taxel
  its own contacts and a true per-taxel normal force;
* the pad stands ``protrude`` (1.5 mm) further out of the finger than robosuite's box: the finger
  mesh's convex hull reaches to 0.37 mm below the original pad face, so a pressed object touched
  the finger hull too and part of the grip force bypassed the pad (measured in the V2 tests);
* one ``touch`` sensor per taxel. A touch sensor sums the normal force of contacts on its site's
  *body* that lie inside the site volume, and all taxels share the fingertip body, so each taxel's
  site covers exactly its own cell of the pad (cells partition the pad face);
* one ``force`` sensor per pad (fingertip-to-finger interaction force, 3-D) for shear;
* a ``{side}_pad`` site per pad whose z axis is the pad's inward normal (towards the object);
  taxel (i, j) sits at pad-frame x = (j - (cols-1)/2) * pitch, y = (i - (rows-1)/2) * pitch.

``TactilePandaGripper`` is the robosuite gripper built from that XML (registered in robosuite's
``GRIPPER_MAPPING`` on import) and ``TactilePads`` reads the sensors from a compiled model.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

SIDES = ("left", "right")
# finger tip body -> (pad sign along tip y, quat of the pad site: site z = inward normal)
_FINGERS = {
    "left": ("finger_joint1_tip", "finger1_pad_collision", -1, "0.7071068 0.7071068 0 0"),
    "right": ("finger_joint2_tip", "finger2_pad_collision", 1, "0.7071068 -0.7071068 0 0"),
}
PAD_HALF = (0.008, 0.004, 0.008)  # robosuite pad box half-extents (x, thickness y, z)
PAD_Z = -0.015  # pad centre along the tip body z


@dataclass(frozen=True)
class TactileConfig:
    rows: int = 4
    cols: int = 4
    pitch: float = 0.004
    gap: float = 0.0002  # between neighbouring taxel geoms
    friction: str = "1.0 0.005 0.0001"  # rubber-like pad
    solref: str = "0.01 1"
    solimp: str = "0.9 0.95 0.001"
    force_limit: float = 70.0  # N, Franka Hand continuous grasp force
    # finger position gain. robosuite's 1000 gives ~12 N on a 24 mm part (the target is fully
    # closed); a held bolt then pivoted about the closing axis at the first chamfer contact (V4).
    # 4000 gives ~45 N, inside the Franka Hand's 70 N continuous force.
    kp: float = 4000.0
    protrude: float = 0.0015  # m, pad moved inwards past the finger mesh hull


DEFAULT_TACTILE = TactileConfig()


def _panda_gripper_path() -> str:
    import robosuite

    return os.path.join(
        os.path.dirname(robosuite.__file__), "models", "assets", "grippers", "panda_gripper.xml"
    )


def tactile_gripper_xml(cfg: TactileConfig = DEFAULT_TACTILE) -> str:
    """Return the modified gripper MJCF as a string (mesh paths made absolute)."""
    src = _panda_gripper_path()
    root = ET.parse(src).getroot()
    folder = os.path.dirname(src)
    for node in root.find("asset").findall("./*[@file]"):
        node.set("file", os.path.join(folder, node.get("file")))
    for act in root.find("actuator"):
        act.set("forcerange", f"{-cfg.force_limit:g} {cfg.force_limit:g}")
        act.set("kp", f"{cfg.kp:g}")

    sensors = root.find("sensor")
    for side in SIDES:
        tip_name, pad_name, sign, site_quat = _FINGERS[side]
        tip = root.find(f".//body[@name='{tip_name}']")
        pad = tip.find(f"geom[@name='{pad_name}']")
        tip.remove(pad)
        # pad centre along tip y: robosuite places it at -/+5 mm; move it inwards by `protrude`
        y = sign * (0.005 + cfg.protrude)
        ET.SubElement(
            tip,
            "site",
            name=f"{side}_pad",
            pos=f"0 {y:g} {PAD_Z:g}",
            quat=site_quat,
            size="0.001",
            rgba="0 0 1 0",
            group="1",
        )
        hx = cfg.pitch / 2 - cfg.gap / 2
        for i in range(cfg.rows):
            for j in range(cfg.cols):
                # pad-frame (x, y) -> tip frame: x_tip = x_pad; the pad site's y axis maps to
                # tip +z for the left pad (Rx(+90)) and -z for the right pad (Rx(-90)).
                px = (j - (cfg.cols - 1) / 2) * cfg.pitch
                py = (i - (cfg.rows - 1) / 2) * cfg.pitch
                pz = PAD_Z - sign * py
                pos = f"{px:g} {y:g} {pz:g}"
                name = f"{side}_tx{i}{j}"
                ET.SubElement(
                    tip,
                    "geom",
                    name=name,
                    type="box",
                    pos=pos,
                    size=f"{hx:g} {PAD_HALF[1]:g} {hx:g}",
                    friction=cfg.friction,
                    solref=cfg.solref,
                    solimp=cfg.solimp,
                    condim="4",
                    group="0",
                    contype="1",
                    conaffinity="1",
                    rgba="0.15 0.15 0.15 1",
                )
                # site: exactly the taxel's cell, thicker than the geom to hold contact points
                ET.SubElement(
                    tip,
                    "site",
                    name=f"{name}_site",
                    type="box",
                    pos=pos,
                    size=f"{cfg.pitch / 2:g} {PAD_HALF[1] + 0.0015:g} {cfg.pitch / 2:g}",
                    rgba="0 1 0 0",
                    group="1",
                )
                ET.SubElement(sensors, "touch", name=f"{name}_touch", site=f"{name}_site")
        ET.SubElement(sensors, "force", name=f"{side}_padF", site=f"{side}_pad")
    return ET.tostring(root, encoding="unicode")


def taxel_names(side: str, cfg: TactileConfig = DEFAULT_TACTILE) -> list[str]:
    return [f"{side}_tx{i}{j}" for i in range(cfg.rows) for j in range(cfg.cols)]


def _write_xml(cfg: TactileConfig) -> str:
    folder = tempfile.mkdtemp(prefix="fvb_tactile_")
    path = os.path.join(folder, "panda_gripper_tactile.xml")
    with open(path, "w") as f:
        f.write(tactile_gripper_xml(cfg))
    return path


def _make_gripper_class():
    from robosuite.models.grippers import GRIPPER_MAPPING, PandaGripper
    from robosuite.models.grippers.gripper_model import GripperModel

    class TactilePandaGripper(PandaGripper):
        """robosuite's PandaGripper (binary open/close) with taxel pads and a 70 N force limit."""

        tactile_cfg = DEFAULT_TACTILE
        _xml_cache: dict = {}

        def __init__(self, idn=0):
            cfg = self.tactile_cfg
            if cfg not in self._xml_cache:
                self._xml_cache[cfg] = _write_xml(cfg)
            GripperModel.__init__(self, self._xml_cache[cfg], idn=idn)

        @property
        def _important_geoms(self):
            left = taxel_names("left", self.tactile_cfg)
            right = taxel_names("right", self.tactile_cfg)
            return {
                "left_finger": ["finger1_collision", *left],
                "right_finger": ["finger2_collision", *right],
                "left_fingerpad": left,
                "right_fingerpad": right,
            }

    GRIPPER_MAPPING["TactilePandaGripper"] = TactilePandaGripper
    return TactilePandaGripper


TactilePandaGripper = _make_gripper_class()


class TactilePads:
    """Reads the taxel grids and pad forces from a compiled model.

    ``prefix`` is the robosuite naming prefix of the gripper (e.g. ``"gripper0_right_"``).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        prefix: str = "gripper0_right_",
        cfg: TactileConfig = DEFAULT_TACTILE,
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.model, self.cfg, self.prefix = model, cfg, prefix
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        def adr(name: str) -> int:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, prefix + name)
            if sid < 0:
                raise ValueError(f"no sensor {prefix + name}")
            return int(model.sensor_adr[sid])

        self.touch_adr = np.array(
            [[adr(f"{n}_touch") for n in taxel_names(s, cfg)] for s in SIDES]
        )  # (2, rows*cols)
        self.force_adr = np.array([adr(f"{s}_padF") for s in SIDES])
        self.pad_site = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{s}_pad") for s in SIDES
        ]
        r, c = cfg.rows, cfg.cols
        self.x = (np.arange(c) - (c - 1) / 2) * cfg.pitch  # pad-frame x per column
        self.y = (np.arange(r) - (r - 1) / 2) * cfg.pitch  # pad-frame y per row

    def taxels(self, data: mujoco.MjData) -> np.ndarray:
        """(2, rows, cols) normal force per taxel, N (left, right pad)."""
        t = data.sensordata[self.touch_adr].reshape(2, self.cfg.rows, self.cfg.cols).copy()
        if self.noise_std > 0:
            t = np.maximum(t + self.rng.normal(0.0, self.noise_std, t.shape), 0.0)
        return t

    def pad_force(self, data: mujoco.MjData) -> np.ndarray:
        """(2, 3) fingertip interaction force in each pad site's frame (z = pad normal)."""
        return np.stack([data.sensordata[a : a + 3] for a in self.force_adr])

    def summary(self, data: mujoco.MjData, taxels: np.ndarray | None = None) -> np.ndarray:
        """(2, 4) per pad: [normal force, CoP x, CoP y, shear magnitude] (N, m, m, N).

        CoP is the taxel-force-weighted centre in the pad frame, 0 when there is no contact.
        """
        t = self.taxels(data) if taxels is None else taxels
        f = self.pad_force(data)
        out = np.zeros((2, 4))
        for k in range(2):
            n = t[k].sum()
            out[k, 0] = n
            if n > 1e-6:
                out[k, 1] = (t[k].sum(axis=0) * self.x).sum() / n
                out[k, 2] = (t[k].sum(axis=1) * self.y).sum() / n
            out[k, 3] = np.hypot(f[k, 0], f[k, 1]) if n > 1e-6 else 0.0
        return out
