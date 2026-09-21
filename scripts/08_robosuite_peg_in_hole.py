#!/usr/bin/env python
"""M6 stretch: custom single-arm PegInHole (fvb.envs) — scripted insertion at lateral offsets.

Same experiment as M1 on the arm: offsets 0 / 0.3 / 1 / 4 mm into a 0.5 mm-clearance square
hole, at two OSC stiffnesses. Logs PLAN §5 episodes, plots, MP4 for the first run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from fvb.control.osc_scripts import (
    ArmRig,
    PegInsertParams,
    identify_load,
    make_controller_config,
    make_env,
    peg_insert,
)
from fvb.logging.episode import EpisodeLogger
from fvb.viz.plots import plot_fmag, plot_ft_timeseries, write_mp4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/m6_pih")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--x-offset-mm", type=float, nargs="+", default=[0.0, 0.3, 1.0, 4.0])
    ap.add_argument("--kp", type=float, nargs="+", default=[150.0, 400.0])
    ap.add_argument("--clearance-mm", type=float, default=0.5)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    render = not args.no_render
    results = []
    for kp in args.kp:
        env = make_env(
            "PegInHole",
            controller_config=make_controller_config(kp=kp),
            offscreen=render,
            seed=args.seed,
            clearance=args.clearance_mm * 1e-3,
        )
        rig = ArmRig(env)
        rig.reset(args.seed)
        load, diag = identify_load(rig)
        ref = rig.model_load()
        if abs(load.mass - ref.mass) / ref.mass > 0.2:
            load = ref
        rig.load = load
        if kp == args.kp[0]:
            print("identified peg load:", json.dumps(diag, indent=2))
        for off in args.x_offset_mm:
            rig.reset(args.seed)
            prm = PegInsertParams()
            log = EpisodeLogger(
                config={
                    "task": "PegInHole",
                    "kp": kp,
                    "x_offset_mm": off,
                    "clearance_mm": args.clearance_mm,
                    "insert": prm.__dict__,
                    "load": {"mass": load.mass, "com": load.com.tolist()},
                },
                seed=args.seed,
            )
            frames = []
            renderer = mujoco.Renderer(rig.m, 360, 480) if (render and off == 4.0) else None
            orig = rig.step

            def step_r(a, lg=None, _o=orig, _r=renderer, _f=frames, _rig=rig):
                r = _o(a, lg)
                if _r is not None:
                    _r.update_scene(_rig.d, camera="frontview")
                    _f.append(_r.render().copy())
                return r

            rig.step = step_r
            ev = peg_insert(rig, np.array([off * 1e-3, 0.0]), prm, log)
            rig.step = orig
            if renderer is not None:
                renderer.close()
            log.success = ev["success"]
            a = log.arrays()
            tag = f"kp{kp:g}_offset_{off:g}mm"
            log.save(out / f"{tag}.npz")
            if frames:
                write_mp4(frames, out / f"{tag}.mp4", fps=20)
            fm = np.linalg.norm(a["ft_comp"][:, :3], axis=1)
            fh = np.linalg.norm(a["ft_raw_hf"][:, :3], axis=1)
            hold = slice(ev["phase_idx"]["hold"] + 5, ev["phase_idx"]["end"])
            row = {
                "kp": kp,
                "x_offset_mm": off,
                **{k: v for k, v in ev.items() if k != "phase_idx"},
                "peak_F_control_N": float(fm.max()),
                "peak_F_physics_N": float(fh.max()),
                "hold_F_comp_N": a["ft_comp"][hold, :3].mean(axis=0).tolist(),
                "hold_contact_world_N": a["contact_wrench"][hold, :3].mean(axis=0).tolist(),
                "hold_Ty_Nm": float(a["ft_comp"][hold, 4].mean()),
                "n_contacts_end": int(a["n_contacts"][-1]),
            }
            results.append(row)
            print(json.dumps(row))
            plot_ft_timeseries(
                a["t"],
                a["ft_comp"],
                out / f"{tag}_comp.png",
                title=f"PegInHole kp {kp:g}, offset {off:g} mm: compensated F/T "
                f"({ev['stop_reason']}, depth {ev['depth_mm']:.1f} mm)",
                onset_t=ev["t_contact_gt"],
            )
            plot_fmag(
                a["t"],
                a["ft_comp"],
                out / f"{tag}_fmag.png",
                title=f"|F| kp {kp:g}, offset {off:g} mm",
                onset_t=ev["t_contact_gt"],
                t_hf=a["t_hf"],
                ft_hf=a["ft_raw_hf"],
            )
        env.close()
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
