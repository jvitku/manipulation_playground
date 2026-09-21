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

    import fvb.envs  # noqa: F401  (registers PegInHole)

    cfg = controller_config or make_controller_config()
    if task == "PegInHole":
        kw.setdefault("gripper_types", None)
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
    default_grip: float = -1.0  # appended when a 6-D arm action is given to a 7-D env (open)
    ft_sensors: tuple[str, str] | None = None  # (force, torque) sensor names if not the gripper's

    def __post_init__(self) -> None:
        if self.ft_sensors is None and hasattr(self.env, "peg") and hasattr(self.env, "ft_site_id"):
            from fvb.envs.peg_in_hole import FT_FORCE, FT_TORQUE

            self.ft_sensors = (FT_FORCE, FT_TORQUE)
            if not self.tool_geoms:
                self.tool_geoms = ["peg_g0"]
        self.ft = RobosuiteFT(
            self.env,
            force_sensor=self.ft_sensors[0] if self.ft_sensors else None,
            torque_sensor=self.ft_sensors[1] if self.ft_sensors else None,
        )
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
        action = np.asarray(action, dtype=np.float64)
        if len(action) == self.env.action_dim - 1:  # arm-only action on a gripper env
            action = np.append(action, self.default_grip)
        self.obs, r, done, info = self.env.step(action)
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


# ----------------------------------------------------------------------------------------------
# NutAssemblyRound scripted grasp -> transport -> mate (M6)
# ----------------------------------------------------------------------------------------------
@dataclass
class NutMateParams:
    hover: float = 0.10  # m above the handle before descending
    grasp_dz: float = 0.006  # eef site height above the (settled) handle centre when closing
    lift_z: float = 1.04  # world z to carry the nut (peg top is at 0.95)
    settle_steps: int = 8  # the nut is spawned ~6 cm above the table and must drop first
    carry_gain: float = 0.4  # P-gain (of the 0.05 m/step saturation) while carrying the nut
    lower_speed: float = -0.3  # unit action during mating descent
    contact_threshold_N: float = 2.0
    mate_steps: int = 40  # steps of lowering after contact is detected
    release: bool = True


def _yaw_action(rig: ArmRig, yaw_target_world: float) -> float:
    """Unit yaw delta (rotation about world z) moving the finger closing axis to a target yaw."""
    R = rig.ee_rot()
    # finger closing axis is the site's x axis in world (see M6 probe: fingers along world y at
    # yaw 0, site x = world y) -> current closing yaw
    ax = R[:, 0]
    cur = float(np.arctan2(ax[1], ax[0]))
    err = (yaw_target_world - cur + np.pi) % (2 * np.pi) - np.pi
    return float(np.clip(err / 0.5, -1, 1))


def nut_grasp_and_mate(
    rig: ArmRig,
    prm: NutMateParams,
    log: EpisodeLogger | None = None,
    nut_geoms: list[str] | None = None,
) -> dict:
    """Scripted NutAssemblyRound: yaw-align, descend on the handle, grasp, lift, carry over
    peg2, lower until the F/T detector fires, keep lowering, release. Uses privileged sim
    state (nut/handle/peg poses) — this is a data-collection script, not a policy.

    Returns phase indices (control steps), event times and the env's own success flag.
    """
    env, m, d = rig.env, rig.m, rig.d
    sid_h = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "RoundNut_handle_site")
    sid_c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "RoundNut_center_site")
    bid_peg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "peg2")
    ev: dict = {"phase_idx": {}, "t_contact_gt": None, "t_detect": None, "grasped": False}
    k = 0

    def step(a6, grip, phase=None):
        nonlocal k
        rig.step(np.concatenate([a6, [grip]]), log)
        k += 1

    def eef():
        return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])

    def towards(target, gain=1.0, dz=None, yaw=0.0):
        err = np.asarray(target) - eef()
        a = np.zeros(6)
        a[:3] = np.clip(gain * err / 0.05, -1, 1)
        if dz is not None:
            a[2] = dz
        a[5] = yaw
        return a

    # 0. let the nut drop onto the table (spawned above it), then read its pose
    ev["phase_idx"]["settle"] = k
    base_geoms = list(rig.tool_geoms)
    for _ in range(prm.settle_steps):
        step(np.zeros(6), -1.0)
    handle = np.array(d.site_xpos[sid_h])
    center = np.array(d.site_xpos[sid_c])
    radial = handle[:2] - center[:2]
    yaw_close = float(np.arctan2(radial[1], radial[0]) + np.pi / 2)  # close across the handle
    ev["handle_z_settled"] = float(handle[2])
    # 1. open + yaw align + hover above handle
    ev["phase_idx"]["approach"] = k
    hover = handle + np.array([0, 0, prm.hover])
    for _ in range(60):
        step(towards(hover, yaw=_yaw_action(rig, yaw_close)), -1.0)
        if np.linalg.norm(eef() - hover) < 4e-3 and abs(_yaw_action(rig, yaw_close)) < 0.02:
            break
    # 2. descend to grasp height
    ev["phase_idx"]["descend"] = k
    g = np.array([handle[0], handle[1], handle[2] + prm.grasp_dz])
    for _ in range(60):
        step(towards(g, yaw=_yaw_action(rig, yaw_close)), -1.0)
        if np.linalg.norm(eef() - g) < 3e-3:
            break
    # 3. close
    ev["phase_idx"]["grasp"] = k
    for _ in range(12):
        step(towards(g), +1.0)
    # 4. lift; from here the nut is part of the tool for the ground-truth contact wrench
    ev["phase_idx"]["lift"] = k
    if nut_geoms:
        rig.tool_geoms = base_geoms + list(nut_geoms)
    lift = np.array([eef()[0], eef()[1], prm.lift_z])
    for _ in range(50):
        step(towards(lift, gain=prm.carry_gain), +1.0)
        if abs(eef()[2] - prm.lift_z) < 4e-3:
            break
    ev["grasped"] = bool(d.site_xpos[sid_c][2] > 0.95)
    ev["failure_mode"] = None if ev["grasped"] else "grasp_missed"
    # 5. transport: bring the nut centre over peg2 (closed loop on the nut's actual pose)
    ev["phase_idx"]["transport"] = k
    peg = np.array(d.xpos[bid_peg])
    for _ in range(80):
        if not ev["grasped"]:
            break
        if d.site_xpos[sid_c][2] < 0.95:
            ev["grasped"] = False
            ev["failure_mode"] = "dropped_in_transport"
            break
        off = peg[:2] - np.array(d.site_xpos[sid_c])[:2]
        tgt = np.array([eef()[0] + off[0], eef()[1] + off[1], prm.lift_z])
        step(towards(tgt, gain=prm.carry_gain), +1.0)
        if np.linalg.norm(off) < 3e-3 and np.linalg.norm(rig.ee_vel()[:3]) < 1e-2:
            break
    ev["xy_error_before_mate_mm"] = 1e3 * float(
        np.linalg.norm(peg[:2] - np.array(d.site_xpos[sid_c])[:2])
    )
    # 6. lower until the detector fires, then keep lowering (mating)
    ev["phase_idx"]["lower"] = k
    detected = False
    for _ in range(80 if ev["grasped"] else 0):
        off = peg[:2] - np.array(d.site_xpos[sid_c])[:2]
        tgt = np.array([eef()[0] + off[0], eef()[1] + off[1], prm.lift_z])
        step(towards(tgt, dz=prm.lower_speed), +1.0)
        _, n = rig.contact_wrench()
        if n > 0 and ev["t_contact_gt"] is None:
            ev["t_contact_gt"] = float(d.time)
        if np.linalg.norm(rig.ft_comp_last()[:3]) > prm.contact_threshold_N and not detected:
            ev["t_detect"] = float(d.time)
            detected = True
            break
        if eef()[2] < 0.86:
            break
    ev["phase_idx"]["mate"] = k
    for _ in range(prm.mate_steps if ev["grasped"] else 0):
        off = peg[:2] - np.array(d.site_xpos[sid_c])[:2]
        tgt = np.array([eef()[0] + off[0], eef()[1] + off[1], prm.lift_z])
        step(towards(tgt, dz=prm.lower_speed * 0.5), +1.0)
        if eef()[2] < 0.86:
            break
    # 7. release and back off
    ev["phase_idx"]["release"] = k
    if prm.release:
        for _ in range(8):
            step(np.zeros(6), -1.0)
        for _ in range(15):
            step(towards(eef() + np.array([0, 0, 0.08])), -1.0)
    ev["phase_idx"]["end"] = k
    ev["nut_center_final"] = np.array(d.site_xpos[sid_c]).tolist()
    ev["peg_xy"] = peg[:2].tolist()
    ev["nut_on_peg"] = bool(env.on_peg(np.array(d.xpos[env.obj_body_id["RoundNut"]]), 1))
    ev["env_success"] = bool(env._check_success())
    if ev["failure_mode"] is None and not ev["env_success"]:
        ev["failure_mode"] = "mate_failed" if ev["grasped"] else "unknown"
    rig.tool_geoms = base_geoms
    return ev


# ----------------------------------------------------------------------------------------------
# fvb.envs.PegInHole scripted insertion (M6 stretch)
# ----------------------------------------------------------------------------------------------
@dataclass
class PegInsertParams:
    hover: float = 0.03  # tip height above the hole top before descending, m
    hover_tol: float = 0.0002  # m; the OSC has no integral action, so the approach adds one
    tilt_tol_deg: float = 0.2  # peg axis must be this close to vertical before descending
    rot_gain: float = 1.0
    descend_action: float = -0.06  # unit z action (x 0.05 m/step commanded = 3 mm/step)
    target_depth: float = 0.03  # stop when the tip is this deep (m)
    max_force_N: float = 40.0  # or when |F_comp| exceeds this (jam)
    hold_push: float = 0.002  # m of downward target offset held after stopping
    hold_steps: int = 20
    max_descend_steps: int = 200


def peg_insert(
    rig: ArmRig, offset_xy: np.ndarray, prm: PegInsertParams, log: EpisodeLogger | None = None
) -> dict:
    """Hover the peg tip above the hole with a lateral offset, descend until inserted or
    jammed (force limit), then hold a small downward push. Closed loop on the peg tip
    (privileged sim state). The approach integrates the residual tip error because the OSC
    (a PD in task space with model-based gravity compensation) settles ~1 mm off the goal."""
    env = rig.env
    hole_top = env.hole_center_world + np.array([0, 0, env.hole_depth])
    ev: dict = {"phase_idx": {}, "t_contact_gt": None, "t_detect": None, "stop_reason": None}
    k = 0
    bias = np.zeros(3)

    def tilt_rotvec():
        """World-frame rotation vector taking the peg axis to straight down (rad)."""
        a = env.peg_axis_world()
        d = np.array([0.0, 0.0, -1.0])
        r = np.cross(a, d)
        ang = float(np.arctan2(np.linalg.norm(r), np.dot(a, d)))
        return r / max(np.linalg.norm(r), 1e-9) * ang

    def tip_target_action(tip_target, dz=None, integrate=False, xy_only=False):
        nonlocal bias
        tip = env.peg_tip_pos()
        if integrate:
            err = np.asarray(tip_target) - tip
            if xy_only:
                err = err * np.array([1, 1, 0])
            bias = np.clip(bias + 0.5 * err, -0.01, 0.01)
        site_target = np.asarray(tip_target) + bias + (rig.ee_pos() - tip)
        # OSC delta rotations are world-frame rotation vectors (x0.5 rad/step); zero means
        # "hold the current orientation", which drifts under contact torque, so always servo
        # the peg axis to vertical (the Panda's home pose already has the flange 8 deg off).
        rot = np.clip(prm.rot_gain * tilt_rotvec() / 0.5, -1, 1)
        return rig.action_towards(site_target, dz_override=dz, rot=rot)

    hover = hole_top + np.array([offset_xy[0], offset_xy[1], prm.hover])
    ev["phase_idx"]["approach"] = k
    for _ in range(150):
        rig.step(tip_target_action(hover, integrate=True), log)
        k += 1
        if (
            np.linalg.norm(env.peg_tip_pos() - hover) < prm.hover_tol
            and np.linalg.norm(rig.ee_vel()[:3]) < 2e-3
        ):
            break
    ev["tip_error_at_hover_mm"] = (1e3 * (env.peg_tip_pos()[:2] - hover[:2])).tolist()
    ev["phase_idx"]["descend"] = k
    tgt = hover.copy()
    tgt[2] = hole_top[2] - prm.target_depth
    tilt_max = 0.0
    for _ in range(prm.max_descend_steps):
        # keep servoing xy on the tip (z is the commanded descent)
        rig.step(tip_target_action(tgt, dz=prm.descend_action, integrate=True, xy_only=True), log)
        k += 1
        ax = env.peg_axis_world()
        tilt_max = max(tilt_max, float(np.degrees(np.arccos(np.clip(-ax[2], -1, 1)))))
        _, n = rig.contact_wrench()
        f = np.linalg.norm(rig.ft_comp_last()[:3])
        if n > 0 and ev["t_contact_gt"] is None:
            ev["t_contact_gt"] = float(rig.d.time)
        if f > 1.0 and ev["t_detect"] is None:
            ev["t_detect"] = float(rig.d.time)
        if env.insertion_depth() >= prm.target_depth:
            ev["stop_reason"] = "inserted"
            break
        if f > prm.max_force_N:
            ev["stop_reason"] = "force_limit"
            break
    else:
        ev["stop_reason"] = "timeout"
    ev["tilt_max_deg_descent"] = tilt_max
    ev["tip_lateral_error_at_stop_mm"] = (1e3 * env.tip_lateral_error()).tolist()
    ev["phase_idx"]["hold"] = k
    for _ in range(prm.hold_steps):
        hold = env.peg_tip_pos() + np.array([0, 0, -prm.hold_push])
        hold[:2] = tgt[:2]
        rig.step(tip_target_action(hold), log)
        k += 1
    ev["phase_idx"]["end"] = k
    ev["depth_mm"] = 1e3 * float(env.insertion_depth())
    ev["tip_lateral_error_mm"] = (1e3 * env.tip_lateral_error()).tolist()
    ev["success"] = bool(env._check_success())
    return ev
