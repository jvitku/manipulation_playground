#!/usr/bin/env python
"""M6: NutAssemblyRound with a scripted grasp -> transport -> lower sequence (7-D action).

Logs F/T through grasp, transport and mating; reports success and failure modes over a few
seeds. Expect partial success from a script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from fvb.control.osc_scripts import (
    ArmRig,
    NutMateParams,
    identify_load,
    make_env,
    nut_grasp_and_mate,
)
from fvb.logging.episode import EpisodeLogger
from fvb.viz.plots import plot_fmag, plot_ft_timeseries, write_mp4


def _seg(a, fm, fh, pi, name, nxt):
    """Force statistics over one phase [pi[name], pi[nxt])."""
    i0, i1 = pi.get(name), pi.get(nxt)
    if i0 is None or i1 is None or i1 <= i0:
        return None
    hf = (a["t_hf"] >= a["t"][i0]) & (a["t_hf"] < a["t"][i1 - 1]) if i1 - 1 > i0 else None
    return {
        "mean_F_comp_N": float(fm[i0:i1].mean()),
        "peak_F_comp_N": float(fm[i0:i1].max()),
        "peak_F_hf_N": float(fh[hf].max()) if hf is not None and hf.any() else float("nan"),
        "mean_ncon": float(a["n_contacts"][i0:i1].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/m6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=5, help="episodes (seeds seed..seed+n-1)")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    render = not args.no_render
    env = make_env("NutAssemblyRound", offscreen=render, seed=args.seed)
    rig = ArmRig(env)
    rig.reset(args.seed)
    # F/T load on this gripper (0.52 kg Panda gripper) — identify as in M4
    load, diag = identify_load(rig)
    ref = rig.model_load()
    print("identified load:", json.dumps(diag, indent=2))
    if abs(load.mass - ref.mass) / ref.mass > 0.2:
        load = ref
    rig.load = load
    # the nut geoms count as part of the "tool" once grasped, so finger<->nut contacts are
    # internal and only nut<->peg / nut<->table / gripper<->world show up in contact_wrench
    m = rig.m
    nut_geoms = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("RoundNut_g")
        and m.geom_contype[i]
    ]
    nut_mass = float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "RoundNut_main")])

    results = []
    prm = NutMateParams()
    for ep in range(args.n):
        seed = args.seed + ep
        rig.reset(seed)
        log = EpisodeLogger(
            config={
                "task": "NutAssemblyRound",
                "params": prm.__dict__,
                "load": {"mass": load.mass, "com": load.com.tolist()},
            },
            seed=seed,
        )
        frames = []
        renderer = mujoco.Renderer(m, height=360, width=480) if (render and ep == 0) else None
        orig = rig.step

        def step_r(a, lg=None, _o=orig, _r=renderer, _f=frames):
            r = _o(a, lg)
            if _r is not None:
                _r.update_scene(rig.d, camera="frontview")
                _f.append(_r.render().copy())
            return r

        rig.step = step_r
        ev = nut_grasp_and_mate(rig, prm, log, nut_geoms=nut_geoms)
        rig.step = orig
        if renderer is not None:
            renderer.close()
        log.success = ev["env_success"]
        a = log.arrays()
        log.save(out / f"nut_seed{seed}.npz")
        if frames:
            write_mp4(frames, out / f"nut_seed{seed}.mp4", fps=20)
        pi = ev["phase_idx"]
        fm = np.linalg.norm(a["ft_comp"][:, :3], axis=1)
        fh = np.linalg.norm(a["ft_raw_hf"][:, :3], axis=1)

        row = {
            "seed": seed,
            "success": ev["env_success"],
            "nut_on_peg": ev["nut_on_peg"],
            "grasped": ev["grasped"],
            "failure_mode": ev.get("failure_mode"),
            "xy_error_before_mate_mm": ev.get("xy_error_before_mate_mm"),
            "t_contact_gt": ev["t_contact_gt"],
            "t_detect": ev["t_detect"],
            "nut_center_final": ev["nut_center_final"],
            "peg_xy": ev["peg_xy"],
            "phases": {
                "grasp": _seg(a, fm, fh, pi, "grasp", "lift"),
                "lift_transport": _seg(a, fm, fh, pi, "lift", "lower"),
                "lower": _seg(a, fm, fh, pi, "lower", "mate"),
                "mate": _seg(a, fm, fh, pi, "mate", "release"),
                "release": _seg(a, fm, fh, pi, "release", "end"),
            },
            "comp_residual_while_carrying_N": float(np.median(fm[pi["transport"] : pi["lower"]]))
            if pi.get("lower", 0) > pi.get("transport", 0)
            else None,
            "nut_weight_N": nut_mass * 9.81,
        }
        results.append(row)
        print(json.dumps(row, default=str))
        marks = {n: float(a["t"][i]) for n, i in pi.items() if i < len(a["t"])}
        plot_ft_timeseries(
            a["t"],
            a["ft_comp"],
            out / f"nut_seed{seed}_comp.png",
            title=f"NutAssemblyRound seed {seed}: compensated F/T; "
            f"grasp@{marks.get('grasp', 0):.1f}s lift@{marks.get('lift', 0):.1f}s "
            f"lower@{marks.get('lower', 0):.1f}s release@{marks.get('release', 0):.1f}s",
            onset_t=ev["t_contact_gt"],
        )
        plot_fmag(
            a["t"],
            a["ft_comp"],
            out / f"nut_seed{seed}_fmag.png",
            title=f"|F| seed {seed} (success={ev['env_success']})",
            onset_t=ev["t_contact_gt"],
            t_hf=a["t_hf"],
            ft_hf=a["ft_raw_hf"],
        )
    (out / "results.json").write_text(
        json.dumps({"load": diag, "episodes": results}, indent=2, default=str)
    )
    n_ok = sum(r["success"] for r in results)
    print(f"success {n_ok}/{len(results)}; wrote {out / 'results.json'}")
    env.close()


if __name__ == "__main__":
    main()
