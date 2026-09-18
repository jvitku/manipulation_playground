#!/usr/bin/env python
"""M6: generic episode recorder -> PLAN §5 .npz/.json files in data/.

  --track A  : Track A gantry insertion (fvb.control.gantry), lateral offset randomised
  --track B  : robosuite Panda; --task Wipe (press+slide) or NutAssemblyRound (grasp+mate)

Initial lateral offset noise is N(0, --noise-mm) per axis (Track A: peg x/y target; Track B
Wipe: the press target; Track B Nut: robosuite's own placement randomisation + nudge of the
transport target). --images 1 adds a (T,H,W,3) uint8 camera stream (off by default).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fvb.control.gantry import Gantry, linear_descent_targets
from fvb.ft.compensate import LoadParams
from fvb.logging.episode import EpisodeLogger, validate
from fvb.scenes.builder import SceneParams, build_model, insertion_depth


def record_track_a(i: int, seed: int, noise_mm: float, images: bool, out: Path, args) -> dict:
    import mujoco

    rng = np.random.default_rng(seed)
    p = SceneParams(kp=args.kp, timestep=args.timestep)
    load = LoadParams(mass=p.peg_mass, com=np.array([0, 0, -p.peg_len / 2]))
    m, d = build_model(p)
    g = Gantry(m, d, p, args.control_freq, load=load)
    off = rng.normal(0.0, noise_mm * 1e-3, size=2)
    g.set_target([off[0], off[1], 0.0])
    g.settle(1.0)
    cfg = {
        "track": "A",
        "scene": asdict(p),
        "offset_m": off.tolist(),
        "speed_mps": args.speed,
        "control_freq": args.control_freq,
        "noise_mm": noise_mm,
    }
    log = EpisodeLogger(config=cfg, seed=seed)
    dt = 1.0 / args.control_freq
    z_end = -(p.start_height - 0.01 - p.peg_len - p.hole_depth) - 0.045
    targets = linear_descent_targets(0.0, z_end, args.speed, dt, xy=off, hold=0.5)
    renderer = mujoco.Renderer(m, 96, 128) if images else None
    imgs = []
    for tgt in targets:
        g.step(tgt, log)
        if renderer is not None:
            renderer.update_scene(d, camera="side")
            imgs.append(renderer.render().copy())
    if renderer is not None:
        renderer.close()
        log._rows["image"] = [np.asarray(x) for x in imgs]
    depth = insertion_depth(p, g.q[2])
    log.success = bool(depth > 0.035)
    path = log.save(out / f"ep{i:04d}.npz")
    a = log.arrays()
    return {
        "path": str(path),
        "success": log.success,
        "offset_mm": (off * 1e3).tolist(),
        "peak_F_comp_N": float(np.max(np.linalg.norm(a["ft_comp"][:, :3], axis=1))),
        "peak_F_hf_N": float(np.max(np.linalg.norm(a["ft_raw_hf"][:, :3], axis=1))),
        "T": int(len(a["t"])),
    }


def record_track_b(
    rig, task: str, i: int, seed: int, noise_mm: float, images: bool, out: Path, nut_geoms, args
) -> dict:
    from fvb.control.osc_scripts import (
        NutMateParams,
        PressSlideParams,
        nut_grasp_and_mate,
        press_and_slide,
    )

    rng = np.random.default_rng(seed)
    rig.reset(seed)
    cfg = {
        "track": "B",
        "task": task,
        "noise_mm": noise_mm,
        "load": {"mass": rig.load.mass, "com": rig.load.com.tolist()},
    }
    log = EpisodeLogger(config=cfg, seed=seed)
    imgs = []
    renderer = None
    if images:
        import mujoco

        renderer = mujoco.Renderer(rig.m, 96, 128)
    orig = rig.step

    def step_img(a, lg=None, _o=orig, _r=renderer, _im=imgs):
        r = _o(a, lg)
        if _r is not None:
            _r.update_scene(rig.d, camera="frontview")
            _im.append(_r.render().copy())
        return r

    rig.step = step_img
    if task == "Wipe":
        prm = PressSlideParams(use_compensated=True)
        cfg["press_slide"] = prm.__dict__
        target = np.array(rig.obs["wipe_centroid"][:2]) + rng.normal(0, noise_mm * 1e-3, 2)
        ev = press_and_slide(rig, target, float(rig.env.table_offset[2]), prm, log)
        success = ev.get("contact_height") is not None
    else:
        prm = NutMateParams()
        cfg["nut_mate"] = prm.__dict__
        ev = nut_grasp_and_mate(rig, prm, log, nut_geoms=nut_geoms)
        success = bool(ev["env_success"])
    rig.step = orig
    if renderer is not None:
        renderer.close()
        log._rows["image"] = [np.asarray(x) for x in imgs]
    log.success = success
    path = log.save(out / f"ep{i:04d}.npz")
    a = log.arrays()
    return {
        "path": str(path),
        "success": success,
        "failure_mode": ev.get("failure_mode"),
        "peak_F_comp_N": float(np.max(np.linalg.norm(a["ft_comp"][:, :3], axis=1))),
        "peak_F_hf_N": float(np.max(np.linalg.norm(a["ft_raw_hf"][:, :3], axis=1))),
        "T": int(len(a["t"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", choices=["A", "B"], required=True)
    ap.add_argument("--task", default="Wipe", choices=["Wipe", "NutAssemblyRound"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--noise-mm", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--images", type=int, default=0)
    ap.add_argument("--out", default=None, help="default data/track{A,B}[_task]")
    # track A knobs
    ap.add_argument("--kp", type=float, default=800.0)
    ap.add_argument("--speed", type=float, default=0.05, help="m/s")
    ap.add_argument("--timestep", type=float, default=0.001, help="M3 recommended default")
    ap.add_argument("--control-freq", type=float, default=20.0)
    args = ap.parse_args()
    out = Path(
        args.out or (f"data/track{args.track}" + (f"_{args.task}" if args.track == "B" else ""))
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    t0 = time.perf_counter()
    rows = []
    if args.track == "A":
        for i in range(args.n):
            rows.append(
                record_track_a(i, args.seed + i, args.noise_mm, bool(args.images), out, args)
            )
            print(json.dumps(rows[-1]))
    else:
        import mujoco

        from fvb.control.osc_scripts import ArmRig, identify_load, make_env

        env = make_env(args.task, offscreen=bool(args.images), seed=args.seed)
        rig = ArmRig(env)
        rig.reset(args.seed)
        load, diag = identify_load(rig)
        ref = rig.model_load()
        if abs(load.mass - ref.mass) / ref.mass > 0.2:
            load = ref
        rig.load = load
        m = rig.m
        nut_geoms = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, k)
            for k in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, k) or "").startswith("RoundNut_g")
            and m.geom_contype[k]
        ]
        for i in range(args.n):
            rows.append(
                record_track_b(
                    rig,
                    args.task,
                    i,
                    args.seed + i,
                    args.noise_mm,
                    bool(args.images),
                    out,
                    nut_geoms,
                    args,
                )
            )
            print(json.dumps(rows[-1]))
        env.close()
    # validate everything we wrote
    from fvb.logging.episode import load as load_ep

    for r in rows:
        arrs, _meta = load_ep(r["path"])
        validate(arrs)
    summary = {
        "n": len(rows),
        "n_success": int(sum(r["success"] for r in rows)),
        "wall_s": time.perf_counter() - t0,
        "episodes": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    from fvb.viz.plots import plot_peak_force_distribution

    label = f"Track {args.track}" + (f" {args.task}" if args.track == "B" else "")
    plot_peak_force_distribution(
        {label: ([r["peak_F_comp_N"] for r in rows], [r["peak_F_hf_N"] for r in rows])},
        out / "peak_force_distribution.png",
        title=f"{label}: peak force per episode",
    )
    print(f"{summary['n_success']}/{summary['n']} success, {summary['wall_s']:.0f} s, wrote {out}")


if __name__ == "__main__":
    main()
