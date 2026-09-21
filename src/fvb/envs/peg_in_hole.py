"""Single-arm PegInHole for robosuite 1.5.2 (PLAN §7 M6 stretch).

* The peg (``CylinderObject``, as in ``TwoArmPegInHole``) is welded to the Panda's flange in place
  of the gripper (``gripper_types=None``).
* The hole is a four-box body fixed to the table with a parametrised square opening, mirroring
  Track A, because ``PlateWithHoleObject`` has a 100 mm square opening — 2–4 cm of clearance
  around any sensible peg, which is not a peg-in-hole problem.
* A ``force`` + ``torque`` sensor pair (``peg_force`` / ``peg_torque``) is added on a site at
  the flange, so the sensor sees exactly what a wrist F/T sees: peg weight, inertia, contact.

Registered with robosuite on import, so ``suite.make("PegInHole", robots="Panda", ...)`` works.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.base import register_env
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, find_elements
from robosuite.utils.observables import Observable, sensor

FT_SITE = "peg_ft"
FT_FORCE = "peg_force"
FT_TORQUE = "peg_torque"


class PegInHole(ManipulationEnv):
    """Panda with a cylindrical peg on the flange; square hole fixed to the table.

    Args (beyond ManipulationEnv):
        peg_radius: m. peg_half_length: m (cylinder half-height).
        clearance: m per side between the peg radius and the square opening's half-width.
        hole_depth: m (through the plate; the table is the floor of the hole).
        hole_xy: hole centre in the table frame (table centre = (0, 0)).
        wall: half-thickness of the four wall boxes, m.
    """

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types=None,
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(0.6, 0.005, 0.0001),
        peg_radius=0.0095,
        peg_half_length=0.05,
        clearance=0.0005,
        hole_depth=0.04,
        hole_xy=(0.0, 0.0),
        wall=0.02,
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        **kwargs,
    ):
        assert gripper_types is None, "PegInHole mounts the peg in place of the gripper"
        self.table_full_size = tuple(table_full_size)
        self.table_friction = tuple(table_friction)
        self.table_offset = np.array((0, 0, 0.8))
        self.peg_radius = float(peg_radius)
        self.peg_half_length = float(peg_half_length)
        self.clearance = float(clearance)
        self.hole_depth = float(hole_depth)
        self.hole_xy = np.asarray(hole_xy, dtype=float)
        self.wall = float(wall)
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            **kwargs,
        )

    # -- geometry helpers ---------------------------------------------------------------
    @property
    def hole_half_opening(self) -> float:
        return self.peg_radius + self.clearance

    @property
    def hole_center_world(self) -> np.ndarray:
        return np.array(
            [
                self.table_offset[0] + self.hole_xy[0],
                self.table_offset[1] + self.hole_xy[1],
                self.table_offset[2],
            ]
        )

    @property
    def hole_top_z(self) -> float:
        return float(self.table_offset[2] + self.hole_depth)

    # -- model --------------------------------------------------------------------------
    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        # hole: four boxes on the table, square opening of half-width hh, depth hole_depth
        hh, w, hz = self.hole_half_opening, self.wall, self.hole_depth / 2
        c = self.hole_center_world
        hole = ET.SubElement(
            arena.worldbody, "body", name="hole", pos=array_to_string([c[0], c[1], c[2] + hz])
        )
        walls = [
            ("hole_w_px", [w, hh + 2 * w, hz], [hh + w, 0, 0]),
            ("hole_w_nx", [w, hh + 2 * w, hz], [-(hh + w), 0, 0]),
            ("hole_w_py", [hh, w, hz], [0, hh + w, 0]),
            ("hole_w_ny", [hh, w, hz], [0, -(hh + w), 0]),
        ]
        for name, size, pos in walls:
            ET.SubElement(
                hole,
                "geom",
                name=name,
                type="box",
                size=array_to_string(size),
                pos=array_to_string(pos),
                rgba="0.55 0.35 0.2 1",
                group="0",
                friction=array_to_string(self.table_friction),
            )
        ET.SubElement(hole, "site", name="hole_site", pos="0 0 0", size="0.003", rgba="0 0 1 0.5")

        # peg on the flange (as TwoArmPegInHole does), with a flange-mounted F/T site
        self.peg = CylinderObject(
            name="peg",
            size_min=(self.peg_radius, self.peg_half_length),
            size_max=(self.peg_radius, self.peg_half_length),
            rgba=[0.2, 0.4, 0.9, 1],
            joints=None,
            friction=self.table_friction,
            rng=self.rng,
        )
        peg_obj = self.peg.get_obj()
        peg_obj.set("pos", array_to_string([0, 0, self.peg_half_length]))
        # CylinderObject writes margin="0.001": a 1 mm collision skin that would eat a sub-mm
        # clearance (contact at 1 mm before touching). Remove it on the collision geom.
        for g in peg_obj.findall("geom"):
            if g.get("name") == "peg_g0":
                g.set("margin", "0")
        ET.SubElement(
            peg_obj,
            "site",
            name=FT_SITE,
            pos=array_to_string([0, 0, -self.peg_half_length]),
            size="0.004",
            rgba="1 0 0 1",
            group="0",
        )
        eef = self.robots[0].robot_model.eef_name["right"]
        body = find_elements(
            root=self.robots[0].robot_model.worldbody,
            tags="body",
            attribs={"name": eef},
            return_first=True,
        )
        body.append(peg_obj)

        self.model = ManipulationTask(
            mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots]
        )
        self.model.merge_assets(self.peg)
        # wrist F/T sensor pair on the flange site
        sens = self.model.root.find("sensor")
        if sens is None:
            sens = ET.SubElement(self.model.root, "sensor")
        ET.SubElement(sens, "force", name=FT_FORCE, site=FT_SITE)
        ET.SubElement(sens, "torque", name=FT_TORQUE, site=FT_SITE)

    def _setup_references(self):
        super()._setup_references()
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)
        self.peg_geom_id = self.sim.model.geom_name2id("peg_g0")
        self.ft_site_id = self.sim.model.site_name2id(FT_SITE)
        self.hole_site_id = self.sim.model.site_name2id("hole_site")
        self.hole_geom_ids = [
            self.sim.model.geom_name2id(n)
            for n in ("hole_w_px", "hole_w_nx", "hole_w_py", "hole_w_ny")
        ]

    # -- state ----------------------------------------------------------------------------
    def peg_tip_pos(self) -> np.ndarray:
        """World position of the peg's far end (the tip that enters the hole)."""
        R = np.array(self.sim.data.body_xmat[self.peg_body_id]).reshape(3, 3)
        return np.array(self.sim.data.body_xpos[self.peg_body_id]) + R @ np.array(
            [0, 0, self.peg_half_length]
        )

    def peg_axis_world(self) -> np.ndarray:
        R = np.array(self.sim.data.body_xmat[self.peg_body_id]).reshape(3, 3)
        return R[:, 2]

    def insertion_depth(self) -> float:
        """Peg-tip depth below the hole's top face (m); negative = above."""
        return self.hole_top_z - float(self.peg_tip_pos()[2])

    def tip_lateral_error(self) -> np.ndarray:
        return self.peg_tip_pos()[:2] - self.hole_center_world[:2]

    def _check_success(self):
        e = np.abs(self.tip_lateral_error())
        return bool(
            self.insertion_depth() > 0.5 * self.hole_depth and np.all(e < self.hole_half_opening)
        )

    def reward(self, action=None):
        if self._check_success():
            return 1.0 * self.reward_scale
        if not self.reward_shaping:
            return 0.0
        tip = self.peg_tip_pos()
        target = self.hole_center_world + np.array([0, 0, self.hole_depth])
        d = np.linalg.norm(tip - target)
        return float(self.reward_scale * (1 - np.tanh(10.0 * d)))

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def peg_tip_pos(obs_cache):
                return self.peg_tip_pos()

            @sensor(modality=modality)
            def peg_axis(obs_cache):
                return self.peg_axis_world()

            @sensor(modality=modality)
            def hole_pos(obs_cache):
                return self.hole_center_world + np.array([0, 0, self.hole_depth])

            @sensor(modality=modality)
            def tip_to_hole(obs_cache):
                return (
                    self.hole_center_world + np.array([0, 0, self.hole_depth])
                ) - self.peg_tip_pos()

            for s in (peg_tip_pos, peg_axis, hole_pos, tip_to_hole):
                observables[s.__name__] = Observable(
                    name=s.__name__, sensor=s, sampling_rate=self.control_freq
                )
        return observables

    def _reset_internal(self):
        super()._reset_internal()


register_env(PegInHole)
