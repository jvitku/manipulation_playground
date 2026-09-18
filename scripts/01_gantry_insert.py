#!/usr/bin/env python
"""M1: Track A scripted linear descent into the hole at several lateral offsets.

Writes per-offset episodes (PLAN §5 .npz + .json), plots and (if GL works) an MP4 to --out.
Also runs the sign-convention experiment: hang the peg, then accelerate it up and down, and
report what the sensor does.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from fvb.control.gantry import Gantry, linear_descent_targets
from fvb.ft.filters import contact_onset_index, spike_metrics
from fvb.logging.episode import EpisodeLogger
from fvb.scenes.builder import SceneParams, build_model, insertion_depth
from fvb.viz.plots import (
    plot_fmag,
    plot_force_vs_depth,
    plot_ft_timeseries,
    plot_lateral_vs_offset,
    write_mp4,
)


def run_insert(
    p: SceneParams,
    x_offset: float,
    speed: float,
    control_freq: float,
    target_depth: float,
    push_mm: list[float],
    seed: int,
    render: bool,
) -> tuple[dict, EpisodeLogger, list[np.ndarray], list[dict]]:
    """Descend at ``speed`` until the peg tip is ``target_depth`` below the rim, hold, then
    (if inserted) push laterally to each x offset in ``push_mm`` and hold."""
    m, d = build_model(p)
    g = Gantry(m, d, p, control_freq)
    g.set_target([x_offset, 0.0, 0.0])
    g.settle(1.0)  # hang at rest at start height
    cfg = {
        "scene": asdict(p),
        "x_offset_m": x_offset,
        "speed_mps": speed,
        "control_freq": control_freq,
        "target_depth_m": target_depth,
        "push_mm": push_mm,
    }
    log = EpisodeLogger(config=cfg, seed=seed)
    dt = 1.0 / control_freq
    # jz such that the peg tip is target_depth below the rim
    z_end = -(p.start_height - 0.01 - p.peg_len - p.hole_depth) - target_depth
    targets = linear_descent_targets(0.0, z_end, speed, dt, xy=[x_offset, 0.0], hold=1.0)
    frames: list[np.ndarray] = []
    renderer = None
    if render:
        try:
            renderer = mujoco.Renderer(m, height=360, width=480)
        except Exception as e:  # noqa: BLE001
            print(f"[render] disabled: {e}")
    every = max(1, int(round(control_freq / 30)))
    i = 0

    def run(tgts):
        nonlocal i
        for tgt in tgts:
            g.step(tgt, log)
            if renderer is not None and i % every == 0:
                renderer.update_scene(d, camera="side")
                frames.append(renderer.render().copy())
            i += 1

    run(targets)
    depth = insertion_depth(p, g.q[2])
    inserted = bool(depth >= 0.5 * min(target_depth, p.hole_depth))
    log.success = inserted
    pushes: list[dict] = []
    if inserted:
        for pm in push_mm:
            n_hold = int(1.0 / dt)
            run(np.tile([pm * 1e-3, 0.0, z_end], (n_hold, 1)))
            a = log.arrays()
            tail = slice(-n_hold // 2, None)
            pushes.append(
                {
                    "push_mm": pm,
                    "x_actual_mm": float(np.median(a["ee_pos"][tail, 0])) * 1e3,
                    "Fx_sensor_N": float(np.median(a["ft_raw"][tail, 0])),
                    "Fy_sensor_N": float(np.median(a["ft_raw"][tail, 1])),
                    "Ty_sensor_Nm": float(np.median(a["ft_raw"][tail, 4])),
                    "contact_Fx_world_N": float(np.median(a["contact_wrench"][tail, 0])),
                    "kp_times_x_err_N": float(p.kp * (pm * 1e-3 - np.median(a["ee_pos"][tail, 0]))),
                    "n_contacts": int(a["n_contacts"][-1]),
                }
            )
    if renderer is not None:
        renderer.close()
    return cfg, log, frames, pushes


def z_end_of(p: SceneParams, target_depth: float) -> float:
    return -(p.start_height - 0.01 - p.peg_len - p.hole_depth) - target_depth


def sign_experiment(p: SceneParams, out: Path) -> dict:
    """Derive the sensor sign convention from free-space motion, not from documentation.

    1. Hang at rest: gravity pulls the peg down; the wrist holds it up.
    2. Command a step UP in the target: the wrist accelerates the peg upward.
    3. Command a step DOWN: wrist pulls it down (spring), peg accelerates downward.
    """
    m, d = build_model(p)
    g = Gantry(m, d, p)
    g.settle(1.0)
    rest = g.ft.ft_raw().copy()
    mg = p.peg_mass * abs(m.opt.gravity[2])

    def kick(dz: float) -> tuple[float, float]:
        g.set_target([0, 0, g.q[2] + dz])
        best_fz, best_az = rest[2], 0.0
        for _ in range(25):
            mujoco.mj_step(m, d)
            az = float(d.qacc[g.vadr[2]])
            fz = float(g.ft.force()[2])
            if abs(az) > abs(best_az):
                best_az, best_fz = az, fz
        g.settle(1.0)
        return best_fz, best_az

    fz_up, az_up = kick(+0.02)
    fz_dn, az_dn = kick(-0.02)
    res = {
        "rest_ft_raw": rest.tolist(),
        "m_g_N": mg,
        "kick_up": {"peak_peg_accel_mps2": az_up, "Fz_at_peak": fz_up},
        "kick_down": {"peak_peg_accel_mps2": az_dn, "Fz_at_peak": fz_dn},
    }
    # Sensor measures the force the child (peg) exerts on the parent (wrist)?  If the wrist pushes
    # the peg UP (a>0), Newton-3 says the peg pushes the wrist DOWN: force-on-parent Fz < 0 would
    # then read as *more negative*. We observe the opposite sign => the sensor reports the force the
    # PARENT exerts on the CHILD... unless rest reading is +m g. Let the numbers decide:
    # F_parent_on_child = m(a + g) upward; at rest = +m g upward => Fz_rest = +0.981.
    res["interpretation"] = (
        "Fz_rest = +m*g and Fz grows when the peg is accelerated upward, so the sensor reports "
        "the force applied BY the parent (wrist) ON the child (peg) — the constraint/support "
        "force that keeps the child in place — expressed in the site frame. A contact force "
        "pushing the peg UP (bottoming out) therefore reduces Fz, and can drive it negative."
        if (fz_up > rest[2] and fz_dn < rest[2])
        else "UNEXPECTED — inspect the numbers"
    )
    (out / "sign_convention.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/m1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--x-offset-mm", type=float, nargs="+", default=[0.0, 0.3, 1.0, 4.0])
    ap.add_argument("--speed-mmps", type=float, default=50.0)
    ap.add_argument("--control-freq", type=float, default=20.0)
    ap.add_argument(
        "--target-depth-mm",
        type=float,
        default=50.0,
        help="commanded tip depth below rim; > hole depth (40) bottoms out",
    )
    ap.add_argument(
        "--push-mm",
        type=float,
        nargs="+",
        default=[0.3, 1.0, 4.0],
        help="after insertion, lateral x targets to push against the wall",
    )
    ap.add_argument("--kp", type=float, default=800.0)
    ap.add_argument("--peg-mass", type=float, default=0.1)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    p = SceneParams(kp=args.kp, peg_mass=args.peg_mass)
    sign = sign_experiment(p, out)
    print("sign convention:", json.dumps(sign, indent=2))

    summary = []
    steady_fx, steady_fy = [], []
    for off_mm in args.x_offset_mm:
        cfg, log, frames, pushes = run_insert(
            p,
            off_mm * 1e-3,
            args.speed_mmps * 1e-3,
            args.control_freq,
            args.target_depth_mm * 1e-3,
            args.push_mm,
            args.seed,
            render=not args.no_render,
        )
        tag = f"offset_{off_mm:g}mm"
        path = log.save(out / f"{tag}.npz")
        a = log.arrays()
        fmag = np.linalg.norm(a["ft_raw"][:, :3], axis=1)
        fmag_hf = np.linalg.norm(a["ft_raw_hf"][:, :3], axis=1)
        # contact onset: first n_contacts > 0 (ground truth), and first |F| deviation (sensor)
        nc = np.nonzero(a["n_contacts"] > 0)[0]
        onset_gt = int(nc[0]) if len(nc) else None
        onset_ft = contact_onset_index(fmag, threshold=0.2)
        onset_t = float(a["t"][onset_gt]) if onset_gt is not None else None
        sm = spike_metrics(fmag, a["t"], onset_ft)
        onset_hf = contact_onset_index(fmag_hf, threshold=0.2, baseline_n=100)
        sm_hf = spike_metrics(fmag_hf, a["t_hf"], onset_hf)
        jz = a["ee_pos"][:, 2] - (p.start_height - 0.01)
        depth = np.array([insertion_depth(p, z) for z in jz])
        z_err = a["action"][:, 2] - jz  # commanded - actual (m)
        # steady state = last half-second of the hold after the descent
        t_hold_end = (
            a["t"][0]
            + (abs(z_end_of(p, args.target_depth_mm * 1e-3)) / (args.speed_mmps * 1e-3))
            + 1.0
        )
        tail = (a["t"] > t_hold_end - 0.5) & (a["t"] <= t_hold_end)
        row = {
            "x_offset_mm": off_mm,
            "success": bool(a["success"]),
            "final_depth_mm": float(depth[-1] * 1e3),
            "onset_t_gt": onset_t,
            "onset_t_ft": float(a["t"][onset_ft]) if onset_ft is not None else None,
            "peak_F_control_rate_N": sm["peak"],
            "peak_F_physics_rate_N": sm_hf["peak"],
            "spike_width_control_s": sm["spike_width_s"],
            "spike_width_physics_s": sm_hf["spike_width_s"],
            "steady_F_N": sm["steady"],
            "steady_Fz_sensor_N": float(np.median(a["ft_raw"][tail, 2])),
            "steady_Fx_sensor_N": float(np.median(a["ft_raw"][tail, 0])),
            "steady_Fy_sensor_N": float(np.median(a["ft_raw"][tail, 1])),
            "steady_contact_Fz_world_N": float(np.median(a["contact_wrench"][tail, 2])),
            "steady_z_err_mm": float(np.median(z_err[tail]) * 1e3),
            "kp_times_z_err_N": float(p.kp * np.median(z_err[tail])),
            "n_contacts_end": int(a["n_contacts"][tail][-1]),
            "pushes": pushes,
        }
        summary.append(row)
        steady_fx.append(row["steady_Fx_sensor_N"])
        steady_fy.append(row["steady_Fy_sensor_N"])
        print(json.dumps(row))
        plot_ft_timeseries(
            a["t"],
            a["ft_raw"],
            out / f"{tag}_ft.png",
            title=f"Track A insertion, x offset {off_mm:g} mm, raw sensor",
            onset_t=onset_t,
            t_hf=a["t_hf"],
            ft_hf=a["ft_raw_hf"],
        )
        plot_fmag(
            a["t"],
            a["ft_raw"],
            out / f"{tag}_fmag.png",
            title=f"|F| control rate vs physics rate, offset {off_mm:g} mm",
            onset_t=onset_t,
            t_hf=a["t_hf"],
            ft_hf=a["ft_raw_hf"],
        )
        plot_force_vs_depth(
            depth,
            a["ft_raw"],
            out / f"{tag}_fz_vs_depth.png",
            title=f"Fz vs insertion depth, offset {off_mm:g} mm",
        )
        if frames:
            write_mp4(frames, out / f"{tag}.mp4", fps=30)
        print(f"wrote {path}")

    plot_lateral_vs_offset(
        args.x_offset_mm,
        steady_fx,
        steady_fy,
        out / "lateral_vs_offset.png",
        title="Steady lateral force vs commanded x offset (descent phase)",
    )
    pushed = [r for r in summary if r["pushes"]]
    if pushed:
        pr = pushed[0]["pushes"]
        plot_lateral_vs_offset(
            [q["push_mm"] for q in pr],
            [q["Fx_sensor_N"] for q in pr],
            [q["Fy_sensor_N"] for q in pr],
            out / "lateral_vs_push.png",
            title=f"Inserted peg pushed sideways (offset {pushed[0]['x_offset_mm']:g} mm)",
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
