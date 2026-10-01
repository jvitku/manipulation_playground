"""V6: the insertion skill of SortBoltsNuts as an RL environment (same API as ``rl_env.ArmRLEnv``).

Set-up per episode (not learned): the scripted expert (rack presentation, human-like noise)
picks a bolt by its head, carries it over its *estimate* of the hole and lowers it until the tip
is ~3 cm above the fixture. The grasp noise
leaves an unknown in-hand offset and the estimate is off by ~1.5 mm, so the skill has to find
and enter the hole the way Stage 1's peg did. The episode then belongs to the agent.

* Observation per frame (``history`` frames stacked, oldest first): pad point minus the hole
  estimate [cm] (3), gripper width [cm] (1), and the force channels - tau_ext [N m / 10] (7) and
  the tactile summary per pad [N / 10, cm, cm, N / 10] (8) - zeroed unless ``use_force``.
* Action: 3 numbers in [-1, 1] -> pad target delta of +-xy_scale / +-z_scale (m); the hand
  orientation is held pointing down.
* Reward (privileged: uses the true hole and tip): depth progress [mm] + potential-based lateral
  shaping - time cost - force penalty + success bonus / - abort penalty, as in Stage 1.
* ``terminated`` on success (tip >= 8 mm into the hole, inside it laterally) or force abort,
  ``truncated`` after ``max_steps``.
"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np

from fvb.policy.rl_env import RewardParams

ENTER_DEPTH = 0.008  # m, the expert releases here too


class SortInsertEnv:
    def __init__(
        self,
        use_force: bool = True,
        history: int = 4,
        xy_scale: float = 0.001,
        z_scale: float = 0.003,
        max_steps: int = 150,
        rp: RewardParams | None = None,
        noise: bool = True,
        cache: int = 0,
        grasp_rand: bool = False,
    ):
        """``cache`` > 0: training seeds map onto ``cache`` set-ups, each run by the expert once
        and then restored from a physics snapshot (one expert set-up takes 300-600 steps, the
        skill episode ~60). With the rack every seed compiles the same model, so any set-up's
        state restores into the one env instance."""
        self.use_force, self.history = use_force, history
        self.scale = np.array([xy_scale, xy_scale, z_scale])
        self.max_steps = max_steps
        self.rp = rp or RewardParams()
        self.noise = noise
        self.frame_dim = 4 + 15
        self.obs_dim = self.frame_dim * history
        self.act_dim = 3
        self.env = None
        self.grasp_rand = grasp_rand
        self.cache = cache
        self._snap: dict = {}
        self._frames: deque = deque(maxlen=history)

    # -- set-up ------------------------------------------------------------------------------
    def _make(self, seed: int):
        import robosuite as suite

        import fvb.envs  # noqa: F401
        from fvb.envs.sort_bolts_nuts import SortTaskParams

        if self.env is not None:
            self.env.close()
        self.env = suite.make(
            "SortBoltsNuts",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            seed=seed,
            ignore_done=True,
            task=SortTaskParams(bolt_presentation="rack"),
        )

    def _setup(self, seed: int, max_setup_steps: int = 700, start_h: float = 0.03) -> bool:
        """Run the expert until the held bolt's tip is ``start_h`` above the fixture during its
        hover. False if it never got there."""
        from fvb.policy.sort_expert import NO_NOISE, ExpertNoise, SortExpert

        self.env.reset()
        ex = SortExpert(ExpertNoise() if self.noise else NO_NOISE, seed)
        ex.reset(self.env)
        if self.grasp_rand:
            # in-hand poses like the learned pick leaves them (3-7 mm off centre, tilted):
            # sideways +-7 mm along the pad, 0 ... +4 mm higher on the head
            r = np.random.default_rng(seed + 12345)
            ex.grasp_dy = float(r.uniform(-0.007, 0.007))
            ex.grasp_dz = float(r.uniform(0.0, 0.004))
        self.expert = ex
        ready = False
        for _ in range(max_setup_steps):
            if ex.phase == "hover":
                self.part = ex.part
                if -self.depth() < start_h:
                    ready = True
                    break
            self.obs, _, _, self.info = self.env.step(ex.act())
        if not ready:
            return False
        self.part = ex.part
        self.R = ex.R_insert
        return True

    # -- quantities ---------------------------------------------------------------------------
    def _tip(self) -> np.ndarray:
        from fvb.envs.fasteners import DEFAULT_BOLT

        pos, Rp = self.env.part_pose(self.part)
        return pos - Rp[:, 2] * DEFAULT_BOLT.length

    def depth(self) -> float:
        return float(self.env.hole_top[2] - self._tip()[2])

    def _lat_err_mm(self) -> float:
        return 1e3 * float(np.linalg.norm(self._tip()[:2] - self.env.hole_top[:2]))

    def privileged(self) -> np.ndarray:
        e = 1e3 * (self.env.hole_top[:2] - self._tip()[:2]) / 5.0
        return np.array([*e, 1e3 * self.depth() / 25.0], np.float32)

    def _phi(self) -> float:
        return -self.rp.lat_coef * self._lat_err_mm()

    def _frame(self) -> np.ndarray:
        ex, env = self.expert, self.env
        rel = (ex.pad_point() - ex.hole_est) * 100.0
        width = np.array([np.sum(np.abs(self.obs["robot0_gripper_qpos"])) * 100.0])
        force = np.zeros(15)
        if self.use_force:
            force[:7] = self.obs["tau_ext"] / 10.0
            if env.pads is not None:
                tx = env.taxel_buf.last(1)[0]
                pf = env.pad_force_buf.last(1)[0]
                s = env.tactile_summary(tx, pf)
                s[:, 0] /= 10.0
                s[:, 1:3] *= 100.0
                s[:, 3] /= 10.0
                force[7:] = s.ravel()
        return np.concatenate([rel, width, force]).astype(np.float32)

    def _obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames))

    # -- snapshots --------------------------------------------------------------------------------
    def _save(self, key: int) -> None:
        import mujoco

        m, d = self.env.sim.model._model, self.env.sim.data._data
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, d, state, spec)
        env_ref, self.expert.env = self.expert.env, None
        ex = copy.deepcopy(self.expert)
        self.expert.env = env_ref
        status = [(p.status, p.seated_since) for p in self.env.parts]
        self._snap[key] = (state, ex, self.part, self.R.copy(), status, copy.deepcopy(self.obs))

    def _restore(self, key: int) -> None:
        import mujoco

        state, ex, part, R, status, obs = self._snap[key]
        m, d = self.env.sim.model._model, self.env.sim.data._data
        mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(m, d)
        for p, (st, since) in zip(self.env.parts, status, strict=True):
            p.status, p.seated_since = st, since
        self.env.outcome = "running"
        for b in (
            self.env.tau_ext_buf,
            self.env.tau_j_buf,
            self.env.taxel_buf,
            self.env.pad_force_buf,
        ):
            b.clear()
        self.expert = copy.deepcopy(ex)
        self.expert.env = self.env
        self.part, self.R, self.obs = part, R.copy(), copy.deepcopy(obs)

    def expert_action(self) -> np.ndarray:
        """Privileged demonstrator for TD3+BC: centre the true tip over the true hole, then go
        down (Stage 1's expert did the same with the true hole)."""
        e = 1e3 * (self.env.hole_top[:2] - self._tip()[:2])  # mm
        a = np.zeros(3)
        a[:2] = np.clip(e / (1e3 * self.scale[:2]), -1, 1)
        a[2] = -1.0 if np.hypot(*e) < 0.6 else 0.0
        return a

    # -- gym-like API ----------------------------------------------------------------------------
    def reset(self, seed: int) -> np.ndarray:
        key = seed % self.cache if self.cache and seed < 50_000 else seed
        if key in self._snap:
            self._restore(key)
        else:
            for k in range(10):  # a set-up the expert could not finish: the next sub-seed
                if self.env is None or not self.cache:
                    self._make(key * 100 + k)
                else:
                    self.env.seed = key * 100 + k
                    self.env.rng = np.random.default_rng(key * 100 + k)
                if self._setup(key * 100 + k):
                    break
            else:
                raise RuntimeError(f"expert set-up failed for seed {seed}")
            if self.cache:
                self._save(key)
        self.t = 0
        f = self._frame()
        self._frames.clear()
        for _ in range(self.history):
            self._frames.append(f)
        self._depth = self.depth()
        self._phi_prev = self._phi()
        return self._obs()

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ex = self.expert
        ex.gripper = 1.0  # keep holding
        target = ex.pad_point() + a * self.scale
        act, _, _ = ex.track(target, self.R, gain=1.0)
        self.obs, _, _, self.info = self.env.step(act)
        self.t += 1
        rp = self.rp
        depth = self.depth()
        phi = self._phi()
        f = float(np.linalg.norm(self.env.f_ext_hat()[:3]))
        success = depth >= ENTER_DEPTH and self._lat_err_mm() < 8.5
        abort = self.env.outcome == "force_abort" or f > self.env.task.force_abort_N
        dropped = not ex._held(self.part)
        r = 1e3 * (depth - self._depth)
        r += rp.gamma * phi - self._phi_prev
        r -= rp.time_cost + rp.force_coef * max(0.0, f - rp.force_free_N)
        reason = None
        if success:
            r += rp.success_bonus
            reason = "success"
        elif abort or dropped:
            r -= rp.abort_penalty
            reason = "force_abort" if abort else "dropped"
        elif self.t >= self.max_steps:
            reason = "timeout"
        self._depth, self._phi_prev = depth, phi
        self._frames.append(self._frame())
        terminated = reason in ("success", "force_abort", "dropped")
        truncated = reason == "timeout"
        info = {
            "reason": reason,
            "force_N": f,
            "depth_mm": 1e3 * depth,
            "priv": self.privileged(),
        }
        return self._obs(), rp.reward_scale * r, terminated, truncated, info


class SortPickEnv(SortInsertEnv):
    """V6: the pick skill (grasp one given part and lift it), same API as ``SortInsertEnv``.

    Set-up: the expert (rack presentation, noise) selects the nearest part and brings the open,
    yawed hand ~16 cm above it (its "above" phase); the agent takes over. It sees the pad point
    relative to a noisy estimate of the grasp point (+-3 mm, a vision estimate), the gripper
    width and the force channels; it acts with a pad delta (+-xy_scale, +-z_scale) and a gripper
    command (> 0.5 close, < -0.5 open, else hold). Success: the part is held 5 cm above the
    start height for 10 steps. Reward: progress of the pad towards the true grasp point [mm],
    a lift bonus, time cost, force penalty, success bonus / abort penalty.
    """

    def __init__(self, *args, z_scale: float = 0.006, xy_scale: float = 0.003, **kw):
        super().__init__(*args, z_scale=z_scale, xy_scale=xy_scale, **kw)
        self.act_dim = 4
        self.max_steps = kw.get("max_steps", 200)

    def _setup(self, seed: int, max_setup_steps: int = 700, start_h: float = 0.0) -> bool:
        from fvb.policy.sort_expert import NO_NOISE, ExpertNoise, SortExpert

        self.env.reset()
        ex = SortExpert(ExpertNoise() if self.noise else NO_NOISE, seed)
        ex.reset(self.env)
        self.expert = ex
        for _ in range(max_setup_steps):
            if ex.phase == "descend":
                break
            self.obs, _, _, self.info = self.env.step(ex.act())
        if ex.phase != "descend":
            return False
        self.part = ex.part
        self.grasp_point, self.R = ex.plan[0].copy(), ex.plan[1].copy()
        rng = np.random.default_rng(seed)
        self.grasp_est = self.grasp_point + np.r_[rng.normal(0, 0.003, 2), 0.0]
        self.start_z = float(ex.pad_point()[2])
        return True

    def _save(self, key: int) -> None:
        super()._save(key)
        self._snap[key] = (
            *self._snap[key],
            self.grasp_point.copy(),
            self.grasp_est.copy(),
            self.start_z,
        )

    def _restore(self, key: int) -> None:
        *base, gp, ge, sz = self._snap[key]
        saved = self._snap[key]
        self._snap[key] = tuple(base)
        super()._restore(key)
        self._snap[key] = saved
        self.grasp_point, self.grasp_est, self.start_z = gp.copy(), ge.copy(), sz

    def _dist_mm(self) -> float:
        return 1e3 * float(np.linalg.norm(self.expert.pad_point() - self.grasp_point))

    def privileged(self) -> np.ndarray:
        e = 1e3 * (self.grasp_point - self.expert.pad_point()) / 20.0
        return e.astype(np.float32)

    def _phi(self) -> float:
        return -0.5 * self._dist_mm()

    def _frame(self) -> np.ndarray:
        f = super()._frame()
        f[:3] = (self.expert.pad_point() - self.grasp_est) * 100.0
        return f

    def depth(self) -> float:  # used by the base reset; the lift height for this skill
        return float(self.expert.pad_point()[2] - self.start_z)

    def expert_action(self) -> np.ndarray:
        """Privileged demonstrator: to the true grasp point, close, lift."""
        ex = self.expert
        a = np.zeros(4)
        held = ex._held(self.part) and ex.gripper > 0
        if not held and ex.gripper <= 0:
            e = (self.grasp_point - ex.pad_point()) / self.scale
            a[:3] = np.clip(e, -1, 1)
            a[3] = 1.0 if self._dist_mm() < 4.0 else -1.0
        else:
            a[2] = 1.0
            a[3] = 1.0
        return a

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ex = self.expert
        if a[3] > 0.5:
            ex.gripper = 1.0
        elif a[3] < -0.5:
            ex.gripper = -0.2  # pre-grasp opening level
        target = ex.pad_point() + a[:3] * self.scale
        act, _, _ = ex.track(target, self.R, gain=1.0)
        self.obs, _, _, self.info = self.env.step(act)
        self.t += 1
        rp = self.rp
        phi = self._phi()
        f = float(np.linalg.norm(self.env.f_ext_hat()[:3]))
        held = ex._held(self.part) and self.env.parts[self.part].status == "grasped"
        lift = self.depth()
        self.held_up = getattr(self, "held_up", 0) + 1 if held and lift > 0.05 else 0
        abort = self.env.outcome == "force_abort" or f > self.env.task.force_abort_N
        r = 0.0 if held else rp.gamma * phi - self._phi_prev
        r += 20.0 * max(0.0, lift - self._depth) * 1e2 if held else 0.0
        r -= rp.time_cost + rp.force_coef * max(0.0, f - rp.force_free_N)
        reason = None
        if self.held_up >= 10:
            r += rp.success_bonus
            reason = "success"
        elif abort:
            r -= rp.abort_penalty
            reason = "force_abort"
        elif self.t >= self.max_steps:
            reason = "timeout"
        self._depth, self._phi_prev = lift, phi
        self._frames.append(self._frame())
        terminated = reason in ("success", "force_abort")
        truncated = reason == "timeout"
        info = {
            "reason": reason,
            "force_N": f,
            "depth_mm": 1e3 * lift,
            "priv": self.privileged(),
        }
        return self._obs(), rp.reward_scale * r, terminated, truncated, info

    def reset(self, seed: int) -> np.ndarray:
        self.held_up = 0
        return super().reset(seed)


def _attach_common(skill, env, expert, obs) -> None:
    skill.env, skill.expert, skill.obs = env, expert, obs
    skill.part = expert.part
    skill.t = 0


def attach_insert(skill: SortInsertEnv, env, expert, obs) -> np.ndarray:
    """Hand a live episode to the insert skill (expert hovering a bolt <= 3 cm above the hole)."""
    _attach_common(skill, env, expert, obs)
    skill.R = expert.R_insert
    f = skill._frame()
    skill._frames.clear()
    for _ in range(skill.history):
        skill._frames.append(f)
    skill._depth, skill._phi_prev = skill.depth(), skill._phi()
    return skill._obs()


def attach_pick(skill: SortPickEnv, env, expert, obs, rng) -> np.ndarray:
    """Hand a live episode to the pick skill (expert about to descend onto its planned grasp)."""
    _attach_common(skill, env, expert, obs)
    skill.held_up = 0
    skill.grasp_point, skill.R = expert.plan[0].copy(), expert.plan[1].copy()
    skill.grasp_est = skill.grasp_point + np.r_[rng.normal(0, 0.003, 2), 0.0]
    skill.start_z = float(expert.pad_point()[2])
    f = skill._frame()
    skill._frames.clear()
    for _ in range(skill.history):
        skill._frames.append(f)
    skill._depth, skill._phi_prev = skill.depth(), skill._phi()
    return skill._obs()
