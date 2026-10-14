"""A 2-D compliant contact world for T3 (peg search) and T4 (mini VLA, four tasks).

The vertical plane (x right, z up). A point peg is driven by an impedance controller: the action
moves a *reference* point (``ref += 0.01 * a``, a in [-1, 1]^2); the peg follows the reference
until the surface blocks it, then rests on the surface and the controller's spring presses with

    F = k * (pos - ref)   (N; the force the surface exerts on the peg, sensed by a F/T sensor)

so pushing the reference further into the surface raises the force. The stiffness k is drawn per
episode from [200, 1000] N per unit, and is unobservable without the force sensor.

The surface is flat at height z0 (random per episode, also unobservable without contact) except
at each hole: a 45-degree chamfer of width WC leading to a hole of half-width WH and depth 0.06.
On the chamfer the reaction force tilts toward the hole centre, so Fx tells which way to slide.

Motion is resolved quasi-statically (``_move``): the peg moves toward the reference in 4
sub-steps; a sub-step that ends inside the solid is projected to the nearest surface point. That
gives sliding along the chamfer and dropping into a hole when the peg passes over it while
pressed. A force above 15 N aborts the episode (as a real arm's protective stop would).
"""

from __future__ import annotations

import numpy as np

WH, WC, DEPTH = 0.006, 0.012, 0.06  # hole half-width, chamfer width, hole depth (units ~ m)
STEP, F_ABORT, HIST = 0.01, 15.0, 10
INSERT_DEPTH = 0.02  # success: tip this far below the chamfer bottom, inside the target hole
TASKS = ("insert into the red hole", "insert into the blue hole", "press gently", "press firmly")
PRESS_N = {2: 2.0, 3: 6.0}  # target force of the press tasks
VOCAB = ["<pad>"] + sorted({w for t in TASKS for w in t.split()})
LANG_LEN = 5


def tokens(task: int) -> list[int]:
    ids = [VOCAB.index(w) for w in TASKS[task].split()]
    return ids + [0] * (LANG_LEN - len(ids))


class Contact2D:
    """``n`` episodes in parallel. ``holes``: (n, K) hole centres; ``target``: (n,) index of the
    hole to insert into (-1 for press tasks)."""

    def __init__(
        self, z0, holes, k, start, target, task=None, horizon=160, sensor_noise=0.05, seed=0
    ):
        self.n = len(z0)
        self.rng = np.random.default_rng(seed)
        self.z0, self.holes, self.k = (
            np.asarray(z0, float),
            np.atleast_2d(holes).astype(float),
            np.asarray(k, float),
        )
        self.target = np.asarray(target, int)
        self.task = np.full(self.n, -1) if task is None else np.asarray(task, int)
        self.pos = np.asarray(start, float).copy()
        self.ref = self.pos.copy()
        self.force = np.zeros((self.n, 2))
        self.hist = np.zeros((self.n, HIST, 2))
        self.noise, self.horizon, self.t = sensor_noise, horizon, 0
        self.done = np.zeros(self.n, bool)
        self.success = np.zeros(self.n, bool)
        self.outcome = np.array(["running"] * self.n, dtype=object)
        self.press_ok = np.zeros(self.n, int)
        self._build_polyline()

    # ---- geometry ----------------------------------------------------------------------------
    def _build_polyline(self):
        n, K = self.holes.shape
        h = np.sort(self.holes, 1)
        z0 = self.z0[:, None]
        zf, zb = z0 - WC, z0 - WC - DEPTH
        xs = np.stack([h - WH - WC, h - WH, h - WH, h + WH, h + WH, h + WH + WC], 2).reshape(
            n, 6 * K
        )
        zs = np.stack([np.broadcast_to(z0, h.shape), zf + 0 * h, zb + 0 * h, zb + 0 * h, zf + 0 * h,
                       np.broadcast_to(z0, h.shape)], 2).reshape(n, 6 * K)  # fmt: skip
        vx = np.concatenate([np.full((n, 1), -1.0), xs, np.full((n, 1), 1.0)], 1)
        vz = np.concatenate([self.z0[:, None], zs, self.z0[:, None]], 1)
        self.V = np.stack([vx, vz], 2)  # (n, 6K+2, 2) boundary vertices, left to right

    def surface(self, x, rows=slice(None)):
        """Surface height under x for the episodes ``rows`` (x has their shape)."""
        h, z0 = self.holes[rows], self.z0[rows]
        d = np.abs(x[..., None] - h)  # (..., K)
        zz = np.broadcast_to(z0[:, None], d.shape).copy()
        zz = np.where(d < WH + WC, z0[:, None] - (WH + WC - d), zz)
        zz = np.where(d < WH, (z0 - WC - DEPTH)[:, None], zz)
        return zz.min(-1)

    def _project(self, p, rows):
        """Nearest point on the boundary polyline (p: (m, 2), rows: indices of the episodes)."""
        A, B = self.V[rows, :-1], self.V[rows, 1:]
        AB = B - A
        t = np.clip(
            np.einsum("msd,msd->ms", p[:, None] - A, AB) / np.maximum((AB**2).sum(-1), 1e-12), 0, 1
        )
        q = A + t[..., None] * AB
        i = np.argmin(((q - p[:, None]) ** 2).sum(-1), 1)
        return q[np.arange(len(p)), i]

    def _move(self):
        pos = self.pos.copy()
        for i in range(4):
            cand = pos + (self.ref - pos) / (4 - i)
            solid = np.flatnonzero(cand[:, 1] < self.surface(cand[:, 0]) - 1e-9)
            if len(solid):
                cand[solid] = self._project(cand[solid], solid)
            pos = cand
        self.pos = pos

    # ---- stepping ----------------------------------------------------------------------------
    def sensed(self) -> np.ndarray:
        return self.force + self.rng.normal(0, self.noise, self.force.shape)

    def step(self, a: np.ndarray) -> None:
        live = ~self.done
        a = np.clip(np.asarray(a, float), -1, 1)
        self.ref[live] += STEP * a[live]
        old = self.pos.copy()
        self._move()
        self.pos[~live] = old[~live]
        self.force = self.k[:, None] * (self.pos - self.ref)
        f = self.sensed()
        self.hist = np.concatenate([self.hist[:, 1:], f[:, None]], 1)
        self.t += 1
        self._judge(live, f)

    def _judge(self, live, f):
        abort = live & (np.linalg.norm(self.force, axis=1) > F_ABORT)
        hole_x = np.take_along_axis(self.holes, np.maximum(self.target, 0)[:, None], 1)[:, 0]
        deep = self.pos[:, 1] < self.z0 - WC - INSERT_DEPTH
        ins = live & ~abort & (self.target >= 0) & deep
        good = ins & (np.abs(self.pos[:, 0] - hole_x) < WH + 1e-6)
        press = live & ~abort & (self.target < 0)
        tgt = np.array([PRESS_N.get(int(t), 0.0) for t in self.task])
        within = np.abs(f[:, 1] - tgt) < 0.25 * tgt
        self.press_ok = np.where(press & within, self.press_ok + 1, 0)
        pressed = press & (self.press_ok >= 8)
        self.success |= good | pressed
        out = np.where(
            abort,
            "force_abort",
            np.where(good | pressed, "success", np.where(ins, "wrong_hole", "")),
        )
        timeout = live & (self.t >= self.horizon) & (out == "")
        out = np.where(timeout, "timeout", out)
        fin = out != ""
        self.outcome[fin] = out[fin]
        self.done |= fin


# ---- scene generators ------------------------------------------------------------------------
def t3_scenes(n: int, seed: int, est_sigma: float = 0.03):
    """T3: one hole; the policy is told x relative to a noisy estimate of the hole (sigma 3 cm,
    the chamfer catches only |error| < 1.8 cm)."""
    r = np.random.default_rng(seed)
    h = r.uniform(-0.08, 0.08, n)
    est = h + r.normal(0, est_sigma, n)
    start = np.stack([est + r.uniform(-0.01, 0.01, n), np.full(n, 0.10)], 1)
    env = Contact2D(r.uniform(-0.03, 0.03, n), h[:, None], r.uniform(200, 1000, n), start,
                    np.zeros(n, int), seed=seed + 7)  # fmt: skip
    env.est = est
    return env


def t4_scenes(n: int, seed: int, tasks=(0, 1, 2, 3)):
    """T4: two holes (red = index 0, blue = index 1) at random places, a task per episode."""
    r = np.random.default_rng(seed)
    a = r.uniform(-0.12, 0.12, n)
    b = a + r.choice([-1, 1], n) * r.uniform(0.06, 0.12, n)
    b = np.where(np.abs(b) > 0.13, a - (b - a), b)
    task = r.choice(np.asarray(tasks), n)
    target = np.where(task <= 1, task, -1)
    start = np.stack([r.uniform(-0.12, 0.12, n), np.full(n, 0.12)], 1)
    env = Contact2D(r.uniform(-0.03, 0.03, n), np.stack([a, b], 1), r.uniform(200, 1000, n), start,
                    target, task=task, seed=seed + 7)  # fmt: skip
    return env


# ---- the experts (observable information + their own memory) --------------------------------
SWEEP = np.array([0.025, -0.035, 0.05, -0.065, 0.08, -0.095, 0.11])


class Expert:
    """Insert: go above the target estimate, descend until touch, press lightly (3 N); on the
    chamfer slide along Fx; on the flat sweep a widening zig-zag; once the peg drops, push in.
    Press: descend, then regulate Fz to the task's force."""

    def __init__(self, env: Contact2D, tx: np.ndarray, seed: int, noise: float = 0.05):
        self.env, self.tx = env, np.asarray(tx, float)
        self.phase = np.zeros(env.n, int)  # 0 approach, 1 contact, 2 insert
        self.sweep = np.zeros(env.n, int)
        self.cz = np.zeros(env.n)
        self.r, self.noise = np.random.default_rng(seed), noise

    def act(self) -> np.ndarray:
        e, n = self.env, self.env.n
        f = e.hist[:, -1]
        fx, fz = f[:, 0], f[:, 1]
        x, z = e.pos[:, 0], e.pos[:, 1]
        a = np.zeros((n, 2))
        press = e.target < 0
        # approach
        ph0 = self.phase == 0
        dx = self.tx - x
        a[:, 0] = np.where(ph0 & ~press, np.clip(dx / STEP, -1, 1), a[:, 0])
        a[:, 1] = np.where(ph0, np.where(press | (np.abs(dx) < 0.01), -0.6, 0.0), a[:, 1])
        touch = ph0 & (fz > 0.8)
        self.cz = np.where(touch, z, self.cz)
        self.phase = np.where(touch, 1, self.phase)
        # contact, insert tasks
        ph1 = (self.phase == 1) & ~press
        dropped = ph1 & (z < self.cz - 0.006)
        self.phase = np.where(dropped, 2, self.phase)
        ph1 &= ~dropped
        slope = np.abs(fx) > 0.35 * np.maximum(fz, 0.3)
        tgt = self.tx + SWEEP[np.minimum(self.sweep, len(SWEEP) - 1)]
        reached = ph1 & ~slope & (np.abs(tgt - x) < 0.004)
        self.sweep = np.where(reached, self.sweep + 1, self.sweep)
        tgt = self.tx + SWEEP[np.minimum(self.sweep, len(SWEEP) - 1)]
        vx = np.where(slope, 0.6 * np.sign(fx), 0.7 * np.sign(tgt - x))
        vz = np.clip(-(3.0 - fz) * 0.12, -0.5, 0.5)
        a[ph1] = np.stack([vx, vz], 1)[ph1]
        # insert
        ph2 = self.phase == 2
        a[ph2] = np.stack([np.clip(0.2 * fx, -0.3, 0.3), np.full(n, -0.6)], 1)[ph2]
        # press tasks
        php = (self.phase == 1) & press
        ft = np.array([PRESS_N.get(int(t), 0.0) for t in e.task])
        a[php] = np.stack([np.zeros(n), np.clip(-(ft - fz) * 0.12, -0.5, 0.5)], 1)[php]
        a += self.r.normal(0, self.noise, a.shape)
        return np.clip(a, -1, 1).astype(np.float32)


# ---- observations -----------------------------------------------------------------------------
def t3_obs(env: Contact2D) -> dict:
    return {
        "state": np.stack([env.pos[:, 0] - env.est, env.pos[:, 1]], 1).astype(np.float32),
        "torque": env.hist.reshape(env.n, -1).astype(np.float32),
    }


def t4_obs(env: Contact2D) -> dict:
    return {
        "state": env.pos.astype(np.float32),
        "torque": env.hist.reshape(env.n, -1).astype(np.float32),
        "scene": np.concatenate([env.z0[:, None], env.holes, env.pos], 1).astype(np.float32),
        "lang": np.array([tokens(int(t)) for t in env.task], np.int64),
    }


def render(scene, size: int = 32, noise: float = 0.03, generator=None):
    """Side-view RGB images (B, 3, size, size) in [0, 1] from ``scene`` = (z0, red x, blue x,
    peg x, peg z) per row (torch tensor). Field of view x in [-0.16, 0.16], z in [-0.12, 0.20]:
    one pixel = 1 cm, so the 1.2 cm hole is about one pixel wide - vision localises it to ~1 cm,
    the chamfer (1.8 cm) must do the rest."""
    import torch

    B, dev = scene.shape[0], scene.device
    px = (torch.arange(size, device=dev) + 0.5) / size
    xg = (-0.16 + 0.32 * px)[None, None, :].expand(B, size, size)
    zg = (0.20 - 0.32 * px)[None, :, None].expand(B, size, size)
    z0 = scene[:, 0, None, None]
    surf = z0.expand(B, size, size).clone()
    for j in (1, 2):
        d = (xg - scene[:, j, None, None]).abs()
        surf = torch.minimum(surf, torch.where(d < WH + WC, z0 - (WH + WC - d), z0))
        surf = torch.where(d < WH + 0.004, z0 - WC - DEPTH, surf)  # rendered slightly wider
    img = torch.empty(B, 3, size, size, device=dev)
    img[:] = torch.tensor([0.10, 0.10, 0.15], device=dev)[None, :, None, None]
    solid = (zg < surf)[:, None]
    img = torch.where(solid, torch.tensor([0.55, 0.55, 0.55], device=dev)[None, :, None, None], img)
    for j, col in ((1, (1.0, 0.2, 0.2)), (2, (0.2, 0.45, 1.0))):
        m = ((xg - scene[:, j, None, None]).abs() < 0.011) & ((zg - (z0 + 0.025)).abs() < 0.011)
        img = torch.where(m[:, None], torch.tensor(col, device=dev)[None, :, None, None], img)
    peg = ((xg - scene[:, 3, None, None]).abs() < 0.011) & (
        (zg - scene[:, 4, None, None]).abs() < 0.011
    )
    img = torch.where(peg[:, None], torch.ones(1, 3, 1, 1, device=dev), img)
    if noise:
        img = img + noise * torch.randn(img.shape, generator=generator, device=dev)
    return img.clamp(0, 1)


def collect(kind: str, n: int, seed: int, expert_noise: float = 0.05) -> list[dict]:
    """Run the expert on ``n`` fresh scenes; returns one dict of arrays per episode (successful
    or not; the caller filters). Keys: obs fields per step, action, force (the sensed force,
    for the auxiliary target), success, outcome."""
    if kind == "t3":
        env = t3_scenes(n, seed)
        ex, obs_fn = Expert(env, env.est, seed, expert_noise), t3_obs
    else:
        env = t4_scenes(n, seed)
        r = np.random.default_rng(seed + 3)
        hx = np.take_along_axis(env.holes, np.maximum(env.target, 0)[:, None], 1)[:, 0]
        tx = np.where(env.target >= 0, hx + r.normal(0, 0.004, n), env.pos[:, 0])
        ex, obs_fn = Expert(env, tx, seed, expert_noise), t4_obs
    steps, alive = [], []
    while not env.done.all():
        o = obs_fn(env)
        a = ex.act()
        alive.append(~env.done)
        env.step(a)
        o["action"] = a
        o["force"] = env.hist[:, -1].astype(np.float32).copy()  # the reading after this action
        steps.append(o)
    alive = np.stack(alive, 1)
    eps = []
    for i in range(n):
        m = alive[i]
        ep = {k: np.stack([s[k][i] for s in steps])[m] for k in steps[0]}
        ep["success"], ep["outcome"] = bool(env.success[i]), str(env.outcome[i])
        if kind == "t4":
            ep["task"] = int(env.task[i])
        eps.append(ep)
    return eps
