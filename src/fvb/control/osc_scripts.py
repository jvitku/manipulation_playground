"""Track B: robosuite Panda + OSC_POSE scripted end-effector behaviours, with F/T logging.

``ArmRig`` wraps a robosuite single-arm env and
  * hooks every physics substep (robosuite calls ``_update_observables`` once per substep) to
    record physics-rate F/T and to run gravity + inertial compensation at physics rate,
    exactly as ``fvb.control.gantry.Gantry`` does for Track A;
  * logs one PLAN §5 row per control step;
  * exposes a simple Cartesian P-controller on top of OSC_POSE delta actions.

OSC_POSE (BASIC config) action semantics verified in M4: 6-D ``[dx, dy, dz, dax, day, daz]`` in
[-1, 1], scaled to ±0.05 m / ±0.5 rad per control step, *relative to the current EE pose*, in
the robot base frame. The controller is an impedance ``F = kp·Δx − kd·ẋ`` in task space.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import mujoco
import numpy as np

from fvb.contacts import geom_contact_wrench
from fvb.ft.compensate import LoadParams, compensate, identify_mass_com
from fvb.ft.frames import mat_to_quat_xyzw, site_to_world
from fvb.ft.read import RobosuiteFT
from fvb.logging.episode import EpisodeLogger


def make_controller_config(
    kp: float | None = None,
    damping_ratio: float | None = None,
    impedance_mode: str | None = None,
    kp_limits: tuple[float, float] | None = None,
) -> dict:
    from robosuite.controllers import load_composite_controller_config

    cfg = load_composite_controller_config(controller="BASIC")
    arm = cfg["body_parts"]["right"]
    if kp is not None:
        arm["kp"] = kp
    if damping_ratio is not None:
        arm["damping_ratio"] = damping_ratio
    if impedance_mode is not None:
        arm["impedance_mode"] = impedance_mode
    if kp_limits is not None:
        arm["kp_limits"] = list(kp_limits)
    return cfg


def make_env(
    task: str = "Wipe",
    controller_config: dict | None = None,
    offscreen: bool = False,
    control_freq: int = 20,
    seed: int | None = None,
    **kw,
):
    import robosuite as suite

    cfg = controller_config or make_controller_config()
    env = suite.make(
        task,
        robots="Panda",
        controller_configs=copy.deepcopy(cfg),
        has_renderer=False,
        has_offscreen_renderer=offscreen,
        use_camera_obs=False,
        control_freq=control_freq,
        ignore_done=True,
        hard_reset=False,
        **kw,
    )
    if seed is not None:
        np.random.seed(seed)
    return env


@dataclass
class ArmRig:
    env: object
    load: LoadParams | None = None
    tool_geoms: list[str] = field(default_factory=list)
    _hf_t: list = field(default_factory=list)
    _hf_ft: list = field(default_factory=list)
    _comp_acc: np.ndarray = field(default_factory=lambda: np.zeros(6))
    _comp_n: int = 0
    _prev_vel: np.ndarray | None = None
    _pending_action: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.ft = RobosuiteFT(self.env)
        self.m = self.env.sim.model._model
        self.d = self.env.sim.data._data
        self.robot = self.env.robots[0]
        if not self.tool_geoms:
            self.tool_geoms = list(self.robot.gripper["right"].contact_geoms)
        self.site_id = self.ft.raw.site_id
        self.sensor_body = int(self.m.site_bodyid[self.site_id])
        self.dt_phys = float(self.env.model_timestep)
        self.dt_ctrl = float(self.env.control_timestep)
        self.n_sub = int(round(self.dt_ctrl / self.dt_phys))
        orig = self.env._update_observables

        def hooked(force=False):
            orig(force)
            self._on_substep()

        self.env._update_observables = hooked
        self.obs = None

    # -- kinematics -----------------------------------------------------------------
    def ee_pos(self) -> np.ndarray:
        return np.array(self.d.site_xpos[self.site_id])

    def ee_rot(self) -> np.ndarray:
        return np.array(self.d.site_xmat[self.site_id]).reshape(3, 3)

    def ee_vel(self) -> np.ndarray:
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_SITE, self.site_id, v, 0)
        return np.concatenate([v[3:], v[:3]])

    def contact_wrench(self) -> tuple[np.ndarray, int]:
        return geom_contact_wrench(self.m, self.d, self.tool_geoms, about_point=self.ee_pos())

    def set_tool_mass(self, mass: float) -> None:
        """Override the mass of the body carrying the F/T site (inertia scaled with it).

        Used to emulate a heavier end-effector (e.g. a 0.9 kg parallel gripper) on the same
        arm; the model is edited in place, so make_env must use hard_reset=False.
        """
        b = self.sensor_body
        scale = mass / float(self.m.body_mass[b]) if self.m.body_mass[b] > 0 else 1.0
        self.m.body_mass[b] = mass
        self.m.body_inertia[b] *= scale
        mujoco.mj_setConst(self.m, self.d)

    def model_load(self) -> LoadParams:
        """Mass/COM distal to the sensor straight from the model (the 'cheating' reference)."""
        mass = float(self.m.body_subtreemass[self.sensor_body])
        # subtree COM in world -> sensor frame
        mujoco.mj_forward(self.m, self.d)
        com_w = np.array(self.d.subtree_com[self.sensor_body])
        com_s = self.ee_rot().T @ (com_w - self.ee_pos())
        return LoadParams(mass=mass, com=com_s)

    # -- substep hook ---------------------------------------------------------------
    def _on_substep(self) -> None:
        raw = self.ft.ft_raw()
        self._hf_t.append(float(self.d.time))
        self._hf_ft.append(raw)
        if self.load is not None:
            v = self.ee_vel()[:3]
            a = np.zeros(3) if self._prev_vel is None else (v - self._prev_vel) / self.dt_phys
            self._prev_vel = v
            self._comp_acc += compensate(raw, self.ee_rot(), self.load, a)
            self._comp_n += 1

    def ft_comp_last(self) -> np.ndarray:
        if self.load is None:
            return self.ft.ft_raw()
        return self._comp_acc / max(self._comp_n, 1)

    # -- stepping -------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        """Reset the env; ``seed`` makes the episode reproducible.

        robosuite 1.5.2 draws robot init noise and object placements from ``env.rng`` (a
        ``numpy.random.Generator`` created once at construction and shared by reference with
        the placement sampler), NOT from ``np.random``. Reseeding must therefore mutate that
        Generator in place. Verified: identical resets -> identical episodes.
        """
        if seed is not None:
            self.env.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
            np.random.seed(seed)
        self.obs = self.env.reset()
        self._prev_vel = None
        self._hf_t.clear()
        self._hf_ft.clear()
        return self.obs

    def step(self, action: np.ndarray, log: EpisodeLogger | None = None):
        self._hf_t.clear()
        self._hf_ft.clear()
        self._comp_acc[:] = 0.0
        self._comp_n = 0
        self.obs, r, done, info = self.env.step(np.asarray(action, dtype=np.float64))
        if log is not None:
            for t, f in zip(self._hf_t, self._hf_ft, strict=True):
                log.step_hf(t, f)
            raw = self.ft.ft_raw()
            R = self.ee_rot()
            cw, n = self.contact_wrench()
            log.step(
                t=self.d.time,
                ee_pos=self.ee_pos(),
                # NOTE: obs["robot0_eef_quat"] is the grip site, rotated 90 deg about z w.r.t.
                # the ft_frame site. Log the F/T site pose so ee_* and ft_* share one frame.
                ee_quat=mat_to_quat_xyzw(R),
                ee_vel=self.ee_vel(),
                ft_raw=raw,
                ft_world=site_to_world(raw, R),
                ft_comp=self.ft_comp_last(),
                contact_wrench=cw,
                n_contacts=n,
                action=np.asarray(action, dtype=np.float64),
                joint_torque=self.ft.joint_torques(),
            )
        return self.obs, r, done, info

    # -- simple Cartesian P-control over OSC deltas ---------------------------------
    def action_towards(
        self,
        target_pos: np.ndarray,
        gain: float = 1.0,
        dz_override: float | None = None,
        rot: np.ndarray | None = None,
    ):
        """Unit-scaled delta action moving the EE toward ``target_pos`` (world/base frame).

        OSC output limit is 0.05 m per step, so err/0.05 saturates at |err| >= 5 cm.
        """
        err = np.asarray(target_pos) - self.ee_pos()
        a = np.zeros(6)
        a[:3] = np.clip(gain * err / 0.05, -1, 1)
        if dz_override is not None:
            a[2] = dz_override
        if rot is not None:
            a[3:] = rot
        return a


def identify_load(
    rig: ArmRig, angles_deg=(0.0, 20.0, -20.0), hold_steps: int = 40, settle_steps: int = 20
) -> tuple[LoadParams, dict]:
    """Rotate the EE about its x and y axes, hold, and least-squares the sensor load.

    The same procedure as Track A's tilt experiment, done with OSC rotation deltas. The arm
    never touches anything. Returns the identified load and a diagnostic dict.
    """
    ft_s, R_s = [], []
    poses = [(0.0, 0.0)]
    poses += [(a, 0.0) for a in angles_deg if a] + [(0.0, a) for a in angles_deg if a]
    p0 = rig.ee_pos().copy()
    for ax, ay in poses:
        # rotate toward the pose over a few steps, then hold still and sample
        rot = np.array([np.deg2rad(ax), np.deg2rad(ay), 0.0])
        for _ in range(settle_steps):
            rig.step(rig.action_towards(p0, rot=np.clip(rot / 0.5, -1, 1)))
            rot = np.zeros(3)  # deltas are relative: apply once, then hold
        for k in range(hold_steps):
            rig.step(rig.action_towards(p0))
            if k >= hold_steps // 2 and np.linalg.norm(rig.ee_vel()) < 2e-3:
                ft_s.append(rig.ft.ft_raw())
                R_s.append(rig.ee_rot())
        # undo the rotation
        for _ in range(settle_steps):
            rig.step(
                rig.action_towards(
                    p0, rot=np.clip(-np.array([np.deg2rad(ax), np.deg2rad(ay), 0.0]) / 0.5, -1, 1)
                )
            )
            ax, ay = 0.0, 0.0
    load = identify_mass_com(np.array(ft_s), np.array(R_s))
    ref = rig.model_load()
    diag = {
        "n_samples": len(ft_s),
        "mass_est_kg": load.mass,
        "mass_model_kg": ref.mass,
        "com_est_m": load.com.tolist(),
        "com_model_m": ref.com.tolist(),
        "force_residual_rms_N": load.residual_force_rms,
        "torque_residual_rms_Nm": load.residual_torque_rms,
    }
    return load, diag


@dataclass
class PressSlideParams:
    hover_height: float = 0.08  # m above table
    descend_speed: float = -0.4  # unit action (x 0.05 m per step)
    contact_threshold_N: float = 1.0
    press_depth: float = 0.01  # m commanded below the contact height (sets force via kp)
    slide_speed: float = 0.2  # unit action along +y (x 0.05 m per step commanded)
    slide_steps: int = 40
    max_descend_steps: int = 120
    use_compensated: bool = True  # detector uses ft_comp (True) or ft_raw (False)


def press_and_slide(
    rig: ArmRig,
    target_xy: np.ndarray,
    table_z: float,
    prm: PressSlideParams,
    log: EpisodeLogger | None = None,
) -> dict:
    """Move above target, descend until |F| > threshold, then slide in +y at fixed press depth.

    Returns phase boundaries (control-step indices) and detector statistics: when the F/T
    detector fired vs when ground-truth contact (any tool contact) began.
    """
    hover = np.array([target_xy[0], target_xy[1], table_z + prm.hover_height])
    events = {"phase_idx": {}, "false_triggers": 0, "t_detect": None, "t_contact_gt": None}
    k = 0
    # approach
    for _ in range(80):
        rig.step(rig.action_towards(hover), log)
        k += 1
        if np.linalg.norm(rig.ee_pos() - hover) < 3e-3 and np.linalg.norm(rig.ee_vel()) < 1e-2:
            break
    events["phase_idx"]["descend"] = k
    # descend with the detector armed
    base = rig.ft_comp_last() if prm.use_compensated else rig.ft.ft_raw()
    contact_height = None
    for _ in range(prm.max_descend_steps):
        rig.step(rig.action_towards(hover, dz_override=prm.descend_speed), log)
        k += 1
        cw, n = rig.contact_wrench()
        f = rig.ft_comp_last() if prm.use_compensated else rig.ft.ft_raw()
        fired = (
            np.linalg.norm(f[:3] - (0 if prm.use_compensated else base[:3]))
            > prm.contact_threshold_N
        )
        if n > 0 and events["t_contact_gt"] is None:
            events["t_contact_gt"] = float(rig.d.time)
        if fired and events["t_detect"] is None:
            events["t_detect"] = float(rig.d.time)
            if n == 0:
                events["false_triggers"] += 1
                events["t_detect"] = None  # keep looking for the real one
                continue
        if fired and n > 0:
            contact_height = rig.ee_pos()[2]
            break
    events["phase_idx"]["slide"] = k
    events["contact_height"] = None if contact_height is None else float(contact_height)
    if contact_height is None:
        return events
    # slide: hold z target press_depth below contact height, move +y
    for _ in range(prm.slide_steps):
        tgt = rig.ee_pos().copy()
        tgt[2] = contact_height - prm.press_depth
        a = rig.action_towards(tgt)
        a[1] = prm.slide_speed
        a[0] = 0.0
        rig.step(a, log)
        k += 1
    events["phase_idx"]["end"] = k
    return events


def press_and_slide_variable_kp(
    rig: ArmRig,
    target_xy: np.ndarray,
    table_z: float,
    prm: PressSlideParams,
    kp_free: float,
    kp_contact: float,
    log: EpisodeLogger | None = None,
    switch_delay_steps: int = 0,
) -> dict:
    """Same behaviour as :func:`press_and_slide` on a ``variable_kp`` controller.

    The 12-D action is ``[kp(6) in raw units, delta(6)]``. ``kp_free`` is used until the F/T
    detector fires, then ``kp_contact`` (after ``switch_delay_steps`` more control steps — a
    detection/actuation latency knob). The detector runs on the compensated wrench.
    """
    hover = np.array([target_xy[0], target_xy[1], table_z + prm.hover_height])
    events = {"phase_idx": {}, "t_detect": None, "t_contact_gt": None, "t_switch": None}
    kp = float(kp_free)

    def act(a6):
        return np.concatenate([np.full(6, kp), a6])

    k = 0
    for _ in range(80):
        rig.step(act(rig.action_towards(hover)), log)
        k += 1
        if np.linalg.norm(rig.ee_pos() - hover) < 3e-3 and np.linalg.norm(rig.ee_vel()) < 1e-2:
            break
    events["phase_idx"]["descend"] = k
    contact_height = None
    delay = None
    for _ in range(prm.max_descend_steps):
        rig.step(act(rig.action_towards(hover, dz_override=prm.descend_speed)), log)
        k += 1
        _, n = rig.contact_wrench()
        fired = np.linalg.norm(rig.ft_comp_last()[:3]) > prm.contact_threshold_N
        if n > 0 and events["t_contact_gt"] is None:
            events["t_contact_gt"] = float(rig.d.time)
        if fired and n > 0 and events["t_detect"] is None:
            events["t_detect"] = float(rig.d.time)
            contact_height = rig.ee_pos()[2]
            delay = switch_delay_steps
        if delay is not None:
            if delay == 0:
                kp = float(kp_contact)
                events["t_switch"] = float(rig.d.time)
                break
            delay -= 1
    events["phase_idx"]["slide"] = k
    events["contact_height"] = None if contact_height is None else float(contact_height)
    if contact_height is None:
        return events
    for _ in range(prm.slide_steps):
        tgt = rig.ee_pos().copy()
        tgt[2] = contact_height - prm.press_depth
        a = rig.action_towards(tgt)
        a[1] = prm.slide_speed
        a[0] = 0.0
        rig.step(act(a), log)
        k += 1
    events["phase_idx"]["end"] = k
    return events
