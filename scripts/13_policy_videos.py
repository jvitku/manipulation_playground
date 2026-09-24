#!/usr/bin/env python
"""Stage 1 report: render untrained vs trained policies on the same hidden-hole episodes.

`--task gantry` (default) or `--task arm` (Panda PegInHole; the arm env is built once).

For every seed, rolls out the privileged expert, an untrained ACT-lite (random init, same
normalisation), the trained no-force policy and the trained force policy. Writes per-policy
MP4s, a synchronised 2x2 comparison MP4 per seed, and a per-step JSON trace (positions,
velocities, compensated wrench, actions) for plotting.

Each frame: 3D close-up (hole walls translucent) + a to-scale top-down schematic in mm + a live
strip of |F| / Fz [N] and Tx / Ty [N·m] (compensated, sensor frame = world here).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import mujoco
import numpy as np

from fvb.ft.frames import site_to_world
from fvb.policy.rollout import TorchPolicy, UntrainedPolicy
from fvb.policy.task import GantryTask, TaskParams
from fvb.viz.plots import write_mp4

W, H = 320, 240  # one camera view
STRIP_H = 150
FPS = 20  # control rate -> real time

COLORS = {  # BGR-free: frames are RGB
    "fmag": (40, 40, 40),
    "fz": (30, 110, 220),
    "tx": (220, 90, 30),
    "ty": (40, 160, 70),
}


class View:
    """What the renderer needs from a task: model/data, walls, hole centre, tip, peg shape."""

    def __init__(self, task, kind: str):
        self.task, self.kind = task, kind
        if kind == "gantry":
            self.m, self.d = task.m, task.d
            self.walls = ("w_px", "w_nx", "w_py", "w_ny")
            self.hole_mm = 1e3 * np.asarray(task.hole_xy)
            self.start_mm = np.zeros(2)
            self.peg_half_mm, self.round_peg = 1e3 * task.scene.peg_half, False
            self.lookat = [*task.hole_xy, 0.035]
            self.cam = (0.26, 125.0, -28.0)
        else:
            self.m, self.d = task.rig.m, task.rig.d
            self.walls = ("hole_w_px", "hole_w_nx", "hole_w_py", "hole_w_ny")
            c = task.env.hole_center_world
            self.hole_mm = 1e3 * c[:2]
            self.start_mm = 1e3 * task.world.hole_pos0[:2]
            self.peg_half_mm, self.round_peg = 1e3 * task.env.peg_radius, True
            self.lookat = [c[0], c[1], c[2] + 0.06]
            self.cam = (0.55, 150.0, -22.0)
        self.clearance_mm = 1e3 * task.p.clearance

    def tip_mm(self) -> np.ndarray:
        t = self.task
        return 1e3 * (t.g.ee_pos()[:2] if self.kind == "gantry" else t.env.peg_tip_pos()[:2])

    def rel_pos(self) -> np.ndarray:
        t = self.task
        p = t.g.ee_pos() if self.kind == "gantry" else t.env.peg_tip_pos()
        return p - t.pos0

    def vel(self) -> np.ndarray:
        t = self.task
        return (t.g.ee_vel() if self.kind == "gantry" else t.rig.ee_vel())[:3]

    def wrench(self) -> np.ndarray:
        """Compensated wrench in world axes (the gantry sensor frame is the world frame)."""
        t = self.task
        if self.kind == "gantry":
            return t.g.ft_comp_last().copy()
        return site_to_world(t.rig.ft_comp_last(), t.rig.ee_rot())


def _cam(view: View):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = view.lookat
    cam.distance, cam.azimuth, cam.elevation = view.cam
    return cam


def _topdown(view: View, path: list) -> np.ndarray:
    """To-scale top-down schematic in mm: hole opening, peg outline now, path of the peg centre."""
    img = np.full((H, W, 3), 250, np.uint8)
    s = H / 30.0  # px per mm, +-15 mm window centred on the hole
    hx, hy = view.hole_mm
    peg_half_mm = view.peg_half_mm
    c0 = np.array([W / 2, H / 2 + 8])

    def px(x, y):  # world mm -> pixel (y up)
        return (int(c0[0] + (x - hx) * s), int(c0[1] - (y - hy) * s))

    hh = peg_half_mm + view.clearance_mm
    cv2.rectangle(
        img, px(hx - hh - 6, hy + hh + 6), px(hx + hh + 6, hy - hh - 6), (200, 196, 190), -1
    )
    cv2.rectangle(img, px(hx - hh, hy + hh), px(hx + hh, hy - hh), (250, 250, 250), -1)
    cv2.rectangle(img, px(hx - hh, hy + hh), px(hx + hh, hy - hh), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.drawMarker(img, px(hx, hy), (44, 125, 79), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
    if len(path) > 1:
        pts = np.array([px(x, y) for x, y in path], np.int32)
        cv2.polylines(img, [pts], False, (31, 103, 201), 1, cv2.LINE_AA)
    x, y = path[-1]
    if view.round_peg:
        cv2.circle(img, px(x, y), int(peg_half_mm * s), (31, 103, 201), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(
            img,
            px(x - peg_half_mm, y + peg_half_mm),
            px(x + peg_half_mm, y - peg_half_mm),
            (31, 103, 201),
            2,
            cv2.LINE_AA,
        )
    cv2.circle(img, px(x, y), 3, (31, 103, 201), -1, cv2.LINE_AA)
    cv2.circle(img, px(*view.start_mm), 3, (120, 120, 120), -1, cv2.LINE_AA)
    cv2.putText(
        img,
        "top-down, to scale [mm]",
        (6, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"hole ({hx - view.start_mm[0]:+.1f}, {hy - view.start_mm[1]:+.1f})  "
        f"peg ({x - view.start_mm[0]:+.1f}, {y - view.start_mm[1]:+.1f})",
        (6, H - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "grey: hole walls  green +: hole centre  blue: peg",
        (6, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (110, 110, 110),
        1,
        cv2.LINE_AA,
    )
    return img


def _strip(hist: list[np.ndarray], t_max: float, f_range=(-10.0, 15.0), t_range=0.06):
    img = np.full((STRIP_H, 2 * W, 3), 250, np.uint8)
    pad_l, pad_r, top, bot = 40, 8, 18, STRIP_H - 16
    panels = [
        (
            0,
            "force [N]",
            f_range,
            [("fmag", lambda f: np.linalg.norm(f[:3]), "|F|"), ("fz", lambda f: f[2], "Fz")],
        ),
        (
            W,
            "torque [N*m]",
            (-t_range, t_range),
            [("tx", lambda f: f[3], "Tx"), ("ty", lambda f: f[4], "Ty")],
        ),
    ]
    for x0, title, (lo, hi), series in panels:
        x_a, x_b = x0 + pad_l, x0 + W - pad_r

        def ymap(v, lo=lo, hi=hi):
            return int(bot - (np.clip(v, lo, hi) - lo) / (hi - lo) * (bot - top))

        cv2.rectangle(img, (x_a, top), (x_b, bot), (215, 215, 215), 1)
        cv2.line(img, (x_a, ymap(0.0)), (x_b, ymap(0.0)), (190, 190, 190), 1)
        for v in (lo, 0.0, hi):
            cv2.putText(
                img,
                f"{v:g}",
                (x0 + 2, ymap(v) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            img, title, (x_a, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA
        )
        lx = x_a + 95
        for key, fn, lab in series:
            if len(hist) > 1:
                ys = np.array([fn(f) for f in hist])
                xs = x_a + np.arange(len(ys)) / max(t_max * FPS, 1) * (x_b - x_a)
                pts = np.column_stack([xs, [ymap(v) for v in ys]]).astype(np.int32)
                cv2.polylines(img, [pts], False, COLORS[key], 1, cv2.LINE_AA)
            cv2.putText(
                img, lab, (lx, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS[key], 1, cv2.LINE_AA
            )
            lx += 34
        cv2.putText(
            img,
            f"time [s], 0-{t_max:g}",
            (x_b - 80, STRIP_H - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (110, 110, 110),
            1,
            cv2.LINE_AA,
        )
    return img


def run(policy, name: str, make_task, p, seed: int, kind: str, t_max: float):
    if policy is not None and getattr(policy, "task_overrides", None):
        p = replace(p, **policy.task_overrides)
    task = make_task(p, seed)
    view = View(task, kind)
    for g in view.walls:
        gid = mujoco.mj_name2id(view.m, mujoco.mjtObj.mjOBJ_GEOM, g)
        view.m.geom_rgba[gid] = [0.55, 0.5, 0.45, 0.35]
    rend = mujoco.Renderer(view.m, height=H, width=W)
    cam = _cam(view)
    strip_kw = {} if kind == "gantry" else {"f_range": (-40.0, 40.0), "t_range": 0.3}
    path: list = []
    if policy is not None:
        policy.reset()
    frames, hist, trace = [], [], []
    done, reason = False, None

    def frame(status: str) -> np.ndarray:
        rend.update_scene(view.d, camera=cam)
        path.append(tuple(float(v) for v in view.tip_mm()))
        views = [rend.render().copy(), _topdown(view, path)]
        img = np.vstack([np.hstack(views), _strip(hist, t_max, **strip_kw)])
        txt = (
            f"{name}   t {task.k / FPS:4.2f} s   depth {1e3 * task.depth():5.1f} mm   "
            f"|F| {task.force_norm():5.2f} N"
        )
        cv2.rectangle(img, (0, 0), (2 * W, 20), (255, 255, 255), -1)
        cv2.putText(img, txt, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        if status:
            col = (30, 140, 60) if status == "success" else (200, 50, 40)
            cv2.putText(
                img,
                status.upper(),
                (12, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                col,
                2,
                cv2.LINE_AA,
            )
        return img

    frames.append(frame(""))
    while not done:
        obs = task.observe()
        a = policy.act(obs) if policy is not None else task.encode(task.expert_action())
        done, reason = task.step(a)
        f = view.wrench()
        hist.append(f)
        trace.append(
            {
                "t": round(task.k / FPS, 3),
                "pos_mm": (1e3 * view.rel_pos()).round(3).tolist(),
                "vel_mmps": (1e3 * view.vel()).round(2).tolist(),
                "ft_comp": f.round(5).tolist(),
                "action_mm": (1e3 * np.asarray(a)).round(3).tolist(),
                "depth_mm": round(1e3 * task.depth(), 3),
            }
        )
        frames.append(frame(""))
    last = frame(reason)
    frames.extend([last] * FPS)  # hold the result for 1 s
    rend.close()
    meta = {
        "policy": name,
        "seed": seed,
        "hole_xy_mm": (1e3 * np.asarray(task.hole_xy)).round(3).tolist(),
        "reason": reason,
        "steps": task.k,
        "peak_F_N": float(max(np.linalg.norm(h[:3]) for h in hist)),
    }
    return frames, {"meta": meta, "trace": trace}


def grid(streams: list[list[np.ndarray]]) -> list[np.ndarray]:
    n = max(len(s) for s in streams)
    padded = [s + [s[-1]] * (n - len(s)) for s in streams]
    out = []
    for i in range(n):
        row0 = np.hstack([padded[0][i], padded[1][i]])
        row1 = np.hstack([padded[2][i], padded[3][i]])
        out.append(np.vstack([row0, row1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["gantry", "arm"], default="gantry")
    ap.add_argument("--ckpt-force", default="outputs/s1/force/best.pt")
    ap.add_argument("--ckpt-noforce", default="outputs/s1/noforce/best.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[50003, 50011, 50020])
    ap.add_argument("--offset-sigma-mm", type=float, default=1.5)
    ap.add_argument("--init-seed", type=int, default=0, help="seed of the untrained weights")
    ap.add_argument("--out", default="outputs/s1/videos")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    close = lambda: None  # noqa: E731
    if args.task == "gantry":
        p = TaskParams(offset_sigma=args.offset_sigma_mm * 1e-3)
        make_task, t_max = GantryTask, 8.0
    else:
        from fvb.policy.arm_task import ArmTask, ArmTaskParams, ArmWorld

        p = ArmTaskParams(offset_sigma=args.offset_sigma_mm * 1e-3)
        world = ArmWorld(p, seed=0)
        make_task, t_max, close = (lambda p_, s: ArmTask(p_, s, world)), 10.0, world.close
    force = TorchPolicy(args.ckpt_force)
    model = force.meta["model"].upper().replace("ACT", "ACT-lite")
    policies = [
        (f"untrained {model}", UntrainedPolicy(args.ckpt_force, args.init_seed)),
        ("trained, no force", TorchPolicy(args.ckpt_noforce)),
        (f"trained, with force ({model})", force),
        ("expert (privileged)", None),
    ]
    slugs = ["untrained", "noforce", "force", "expert"]
    summary = []
    for seed in args.seeds:
        streams, traces = [], []
        for (name, pol), slug in zip(policies, slugs, strict=True):
            frames, tr = run(pol, name, make_task, p, seed, args.task, t_max)
            write_mp4(frames, out / f"seed{seed}_{slug}.mp4", fps=FPS)
            streams.append(frames)
            traces.append(tr)
            summary.append(tr["meta"])
            print(json.dumps(tr["meta"]))
        write_mp4(grid(streams), out / f"seed{seed}_compare.mp4", fps=FPS)
        (out / f"seed{seed}_traces.json").write_text(json.dumps(traces))
    close()
    (out / "summary.json").write_text(json.dumps({"config": vars(args), "runs": summary}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
