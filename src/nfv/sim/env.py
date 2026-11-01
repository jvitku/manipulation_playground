"""The M0 environment: 10 Hz policy steps over the gantry, tasks T0-T3, 96x96 osmesa rendering.

API (fixed by the plan; everything later plugs into these two calls):

    env = Env(task="t0")                      # or render=False for physics-only rollouts
    obs = env.reset(seed, task=None, instruction=None, hidden_overrides=None)
    obs, info = env.step(Action(dp, gripper, f_target, s))

Obs (one 10 Hz policy sample; shapes fixed):
    rgb_static     uint8 [96, 96, 3]   fixed workspace camera (None when render=False)
    rgb_wrist      uint8 [96, 96, 3]   tool-mounted camera (None when render=False)
    state          float32 [8]         [x, y, z, yaw, grip, vx, vy, vz]: the four joint
                                       positions (m, rad), the gripper opening normalised to
                                       0..1 (0 = open, 1 = closed) and the prismatic
                                       velocities (m/s)
    wrench         float32 [6]         latest compensated+filtered wrench (sensors.py sign:
                                       pushing up on the tool = negative F_z)
    wrench_window  float32 [20, 6]     the policy step's 20 samples of the 200 Hz stream
                                       (oldest first; at reset all rows hold the settled value)
    subtask_id     int                 index into the task's SUBTASKS (heuristic label)
    t              float               seconds since reset

Action (one 10 Hz chunk element):
    dp        float [3]   Cartesian step in metres, clamped to ±0.05 per axis
    gripper   float       0 = open .. 1 = closed
    f_target  float [6] | None   commanded contact wrench; None = pure position mode.
                                 The controller mode is decided HERE, per axis, from
                                 |f_target,i| (controllers.py hybrid law); torques ignored.
    s         float | None       predicted progress — carried for the later models, the
                                 physics ignores it (it is echoed into info).

One policy step = 20 controller ticks = 50 physics steps (2 ms); the exact clock, including
the alternating 2/3-substep realisation of 200 Hz on a 500 Hz physics clock, lives in
controllers.py and is asserted in tests/nfv/test_nfv_timing.py.

Episode end (reported in info, the env refuses to step a finished episode):
    info["done"]     True on success / force_abort / task failure / timeout
    info["outcome"]  "success" | "force_abort" | "timeout" | task-specific ("wrong_button", …)
    info["success"], info["force_abort"], info["peak_force"], info["subtask"],
    plus the task's own step_info (coverage, click, depth_mm, …).

Rendering uses mujoco.Renderer at 96x96; the fvb-lab image sets MUJOCO_GL=osmesa, so both the
physics and the renderer are CPU-only by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from nfv.sim import controllers, gantry_xml, tasks
from nfv.sim.controllers import (
    CTRL_PER_POLICY,
    F_ABORT,
    PHYS_PER_CTRL,
    GantryController,
)
from nfv.sim.sensors import WrenchSensor

RENDER_HW = (96, 96)  # (height, width) of both camera streams
SETTLE_STEPS = 150  # physics steps after reset (0.3 s) to let objects/springs come to rest


@dataclass
class Obs:
    """One 10 Hz observation; see the module docstring for the shape legend."""

    rgb_static: np.ndarray | None
    rgb_wrist: np.ndarray | None
    state: np.ndarray  # float32 [8]
    wrench: np.ndarray  # float32 [6]
    wrench_window: np.ndarray  # float32 [20, 6]
    subtask_id: int
    t: float


@dataclass
class Action:
    """One 10 Hz action; see the module docstring. dp is required, the rest default off."""

    dp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gripper: float = 0.0
    f_target: np.ndarray | None = None
    s: float | None = None


class Env:
    """Gantry environment over one task at a time (models are cached per task)."""

    def __init__(
        self,
        task: str = "t0",
        render: bool = True,
        gantry: gantry_xml.GantryParams | None = None,
    ):
        self.render_enabled = render
        self.gantry = gantry or gantry_xml.GantryParams()
        self._cache: dict[str, tuple] = {}  # task name -> (model, data, task, sensor, ctl)
        self._renderer: mujoco.Renderer | None = None
        self._renderer_task: str | None = None
        self._done = True
        self._switch_task(task)

    # --- lifecycle ---
    def _switch_task(self, name: str) -> None:
        if name not in self._cache:
            task = tasks.make(name)
            model, data = gantry_xml.build_model(
                self.gantry,
                worldbody_extra=task.WORLD_XML,
                tool_extra=task.TOOL_XML,
                contact_extra=task.CONTACT_XML,
            )
            assert model.opt.timestep == controllers.PHYS_DT, "XML timestep != PHYS_DT"
            task.bind(model)
            sensor = WrenchSensor(model, data)
            ctl = GantryController(model, data)
            self._cache[name] = (model, data, task, sensor, ctl)
        self.task_name = name
        self.model, self.data, self.task, self.sensor, self.controller = self._cache[name]

        def jid(n: str) -> int:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)

        joints = ("jx", "jy", "jz", "jyaw", "jgl", "jgr")
        self._q = {n: int(self.model.jnt_qposadr[jid(n)]) for n in joints}  # joint qposadr
        self._v = {n: int(self.model.jnt_dofadr[jid(n)]) for n in ("jx", "jy", "jz")}  # dofadr
        self._tool_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tool")

    def reset(
        self,
        seed: int,
        task: str | None = None,
        instruction: str | None = None,
        hidden_overrides: dict | None = None,
    ) -> Obs:
        """Start an episode: place + randomise from ``seed``, settle, prime the sensor.

        ``instruction`` overrides the sampled instruction (must be one the task offers);
        ``hidden_overrides`` forces hidden quantities (test/teaching hook — overrides are
        applied after all rng draws, so they never shift the random stream).
        """
        if task is not None and task != self.task_name:
            self._switch_task(task)
        m, d = self.model, self.data
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(m, d)
        d.qpos[self._q["jz"]] = self.task.start_jz
        self.instruction = self.task.randomize(m, d, rng, hidden_overrides)
        if instruction is not None:
            self.task.set_instruction(instruction)
            self.instruction = instruction
        mujoco.mj_forward(m, d)
        self.controller.reset()  # anchors the position command on the start pose
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(m, d)
        self.sensor.reset()
        w = self.sensor.sample()
        self._last_wrench = w
        self._window = np.tile(w, (CTRL_PER_POLICY, 1))  # [20, 6]
        self.task.after_settle(self)
        self._t0 = float(d.time)
        self.peak_force = 0.0
        self.force_abort = False
        self._done = False
        self._outcome: str | None = None
        self._last_s: float | None = None
        return self._obs()

    @property
    def t(self) -> float:
        """Seconds since the end of the reset settle phase."""
        return float(self.data.time) - self._t0

    @property
    def wrench(self) -> np.ndarray:
        """Latest filtered 200 Hz wrench sample (6,)."""
        return self._last_wrench

    @property
    def grip_norm(self) -> float:
        """Gripper closure 0..1 (mean of the two finger joints over their travel)."""
        g = 0.5 * (self.data.qpos[self._q["jgl"]] + self.data.qpos[self._q["jgr"]])
        return float(g / self.controller.grip_travel)

    @property
    def tool_pos(self) -> np.ndarray:
        """World position (3,) of the tool origin (= the wrist)."""
        return np.asarray(self.data.xpos[self._tool_bid], dtype=np.float64).copy()

    @property
    def qpos_xyz(self) -> np.ndarray:
        """The three prismatic joint positions (3,) — the controller's coordinate system."""
        return np.array([self.data.qpos[self._q[n]] for n in ("jx", "jy", "jz")])

    def step(self, action: Action) -> tuple[Obs, dict]:
        """Run one 10 Hz policy step; see the module docstring for Obs/info."""
        if self._done:
            raise RuntimeError("episode is done; call reset()")
        m, d = self.model, self.data
        self.controller.begin_policy_step(
            np.asarray(action.dp, dtype=np.float64), action.gripper, action.f_target
        )
        self._last_s = action.s
        aborted = False
        for i in range(CTRL_PER_POLICY):
            self.controller.tick(self._last_wrench)
            for _ in range(PHYS_PER_CTRL[i % 2]):
                mujoco.mj_step(m, d)
            w = self.sensor.sample()
            self._window[i] = w
            self._last_wrench = w
            self.peak_force = max(self.peak_force, float(np.linalg.norm(w[:3])))
            self.task.on_ctrl_sample(self, w)
            if GantryController.abort(w):
                aborted = True
                self._window[i + 1 :] = w  # pad the cut-short window with the last sample
                break
        # episode verdict (priority: abort > task failure > success > timeout)
        fail = self.task.failure(self)
        if aborted:
            self.force_abort, self._done, self._outcome = True, True, "force_abort"
        elif fail is not None:
            self._done, self._outcome = True, fail
        elif self.task.success(self):
            self._done, self._outcome = True, "success"
        elif self.t >= self.task.TIME_LIMIT_S - 1e-9:
            self._done, self._outcome = True, "timeout"
        obs = self._obs()
        info = {
            "done": self._done,
            "outcome": self._outcome,
            "success": self._outcome == "success",
            "force_abort": self.force_abort,
            "peak_force": self.peak_force,
            "t": self.t,
            "instruction": self.instruction,
            "subtask": self.task.SUBTASKS[obs.subtask_id],
            "s": self._last_s,
        }
        info.update(self.task.step_info(self))
        return obs, info

    # --- observation building ---
    def _state(self) -> np.ndarray:
        q, v = self._q, self._v
        d = self.data
        return np.array(
            [
                d.qpos[q["jx"]],
                d.qpos[q["jy"]],
                d.qpos[q["jz"]],
                d.qpos[q["jyaw"]],
                self.grip_norm,
                d.qvel[v["jx"]],
                d.qvel[v["jy"]],
                d.qvel[v["jz"]],
            ],
            dtype=np.float32,
        )

    def _render_pair(self) -> tuple[np.ndarray, np.ndarray]:
        if self._renderer is None or self._renderer_task != self.task_name:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=RENDER_HW[0], width=RENDER_HW[1])
            self._renderer_task = self.task_name
        out = []
        for cam in ("static", "wrist"):
            self._renderer.update_scene(self.data, camera=cam)
            out.append(self._renderer.render().copy())
        return out[0], out[1]

    def _obs(self) -> Obs:
        rgb_s = rgb_w = None
        if self.render_enabled:
            rgb_s, rgb_w = self._render_pair()
        return Obs(
            rgb_static=rgb_s,
            rgb_wrist=rgb_w,
            state=self._state(),
            wrench=self._last_wrench.astype(np.float32),
            wrench_window=self._window.astype(np.float32),
            subtask_id=self.task.infer_subtask(self),
            t=self.t,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


ABORT_FORCE_N = F_ABORT  # re-export: the 25 N safety is part of the env's contract
