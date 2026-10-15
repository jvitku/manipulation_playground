"""T1 and T2: tasks without force, to learn what a generative action head buys you.

T1 - two modes. Observation c in [-0.4, 0.4] (where an obstacle is); the expert's action is a
2-D waypoint that passes the obstacle on the LEFT or on the RIGHT with equal probability:
``a = (c -/+ 0.5, 0.5) + noise`` - inside [-1, 1], the range DDPM's sampler clips to (an
unnormalised target is silently clipped and looks like a worse model). A regression head
predicts the mean ``(c, 0.5)``: straight into the obstacle. A flow / diffusion head samples
both modes.

T2 - point mass around an obstacle (closed loop, action chunks). The mass starts near (0, -1)
and must reach a goal near (0, 1) without touching a disc of radius 0.3 near the origin. The
expert picks a side at random per episode and holds it. Observation (6-D): position, goal minus
position, obstacle centre minus position. Action: a velocity in [-1, 1]^2 (step 0.04).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- T1
T1_HALF_GAP = 0.5
T1_OBSTACLE = 0.25  # half-width: |a_x - c| < this is a collision


def t1_data(n: int, rng: np.random.Generator, noise: float = 0.05):
    c = rng.uniform(-0.4, 0.4, n)
    side = rng.choice([-1.0, 1.0], n)
    a = np.stack([c + side * T1_HALF_GAP, np.full(n, 0.5)], 1) + rng.normal(0, noise, (n, 2))
    return c[:, None].astype(np.float32), a.astype(np.float32)


def t1_metrics(c: np.ndarray, a: np.ndarray) -> dict:
    """``c`` (n,), ``a`` (n, 2) sampled actions: collision rate, mode balance and the distance to
    the nearest expert mode (0 for a perfect sampler, ~0.5 for the regression mean)."""
    dx = a[:, 0] - c
    left = dx < 0
    modes = np.stack([c - T1_HALF_GAP, c + T1_HALF_GAP], 1)
    err = np.min(np.abs(a[:, :1] - modes), 1)
    return {
        "collision": float(np.mean(np.abs(dx) < T1_OBSTACLE)),
        "left_frac": float(left.mean()),
        "mode_err": float(np.mean(np.hypot(err, a[:, 1] - 0.5))),
    }


# ---------------------------------------------------------------- T2
class ObstacleReach:
    """Vectorised over ``n`` episodes. ``obs()`` -> (n, 6); ``step(a)`` with a in [-1, 1]^2."""

    SPEED, RADIUS, GOAL_TOL, HORIZON = 0.04, 0.3, 0.06, 100

    def __init__(self, n: int, seed: int):
        r = np.random.default_rng(seed)
        self.n = n
        self.p = np.stack([r.uniform(-0.1, 0.1, n), -1 + r.uniform(-0.05, 0.05, n)], 1)
        self.g = np.stack([r.uniform(-0.1, 0.1, n), np.ones(n)], 1)
        self.o = np.stack([r.uniform(-0.15, 0.15, n), np.zeros(n)], 1)
        self.done = np.zeros(n, bool)
        self.success = np.zeros(n, bool)
        self.collided = np.zeros(n, bool)
        self.t = 0

    def obs(self) -> np.ndarray:
        return np.concatenate([self.p, self.g - self.p, self.o - self.p], 1).astype(np.float32)

    def step(self, a: np.ndarray) -> None:
        a = np.clip(a, -1, 1)
        live = ~self.done
        self.p[live] += self.SPEED * a[live]
        hit = live & (np.linalg.norm(self.p - self.o, axis=1) < self.RADIUS)
        reach = live & ~hit & (np.linalg.norm(self.p - self.g, axis=1) < self.GOAL_TOL)
        self.collided |= hit
        self.success |= reach
        self.t += 1
        self.done |= hit | reach | (self.t >= self.HORIZON)


class ObstacleExpert:
    """Commits to a random side per episode; waypoint beside the obstacle, then the goal."""

    def __init__(self, env: ObstacleReach, seed: int, noise: float = 0.1):
        r = np.random.default_rng(seed + 1)
        self.side = r.choice([-1.0, 1.0], env.n)
        self.past = np.zeros(env.n, bool)
        self.noise, self.r = noise, r

    def act(self, env: ObstacleReach) -> np.ndarray:
        w = env.o + np.stack([self.side * (env.RADIUS + 0.15), np.zeros(env.n)], 1)
        self.past |= (np.linalg.norm(env.p - w, axis=1) < 0.08) | (env.p[:, 1] > env.o[:, 1])
        tgt = np.where(self.past[:, None], env.g, w)
        d = tgt - env.p
        v = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        v = v * np.minimum(1.0, np.linalg.norm(d, axis=1, keepdims=True) / env.SPEED)
        return np.clip(v + self.r.normal(0, self.noise, v.shape), -1, 1).astype(np.float32)


def t2_collect(n: int, seed: int) -> list[dict]:
    env, ex = ObstacleReach(n, seed), None
    ex = ObstacleExpert(env, seed)
    obs, act, alive = [], [], []
    while not env.done.all():
        alive.append(~env.done)
        obs.append(env.obs())
        a = ex.act(env)
        act.append(a)
        env.step(a)
    obs, act, alive = np.stack(obs, 1), np.stack(act, 1), np.stack(alive, 1)
    return [
        {"obs": obs[i, alive[i]], "action": act[i, alive[i]], "success": bool(env.success[i])}
        for i in range(n)
    ]
