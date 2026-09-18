#!/usr/bin/env python
"""M5: OSC stiffness vs contact force on the Panda (`Wipe`).

Part 1 — fixed impedance: kp x damping_ratio grid, same press-and-slide, same seed. Per run:
free-space tracking error, effective Cartesian stiffness (steady force / achieved press
depth), onset peak (control and physics rate), steady force.
Part 2 — variable_kp: approach at kp_free, switch to kp_contact when the (compensated) F/T
detector fires, with 0/1/2 control steps of extra latency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fvb.control.osc_scripts import (
    ArmRig,
    PressSlideParams,
    make_controller_config,
    make_env,
    press_and_slide,
    press_and_slide_variable_kp,
)
from fvb.ft.compensate import LoadParams
from fvb.logging.episode import EpisodeLogger

KPS = [50.0, 150.0, 400.0, 1000.0]
ZETAS = [0.7, 1.0, 2.0]


def analyse(a: dict, ev: dict, dt: float, press_depth: float) -> dict:
    fmag_c = np.linalg.norm(a["ft_comp"][:, :3], axis=1)
    fmag_hf = np.linalg.norm(a["ft_raw_hf"][:, :3], axis=1)
    d0 = ev["phase_idx"]["descend"]
    sl = ev["phase_idx"].get("slide")
    end = ev["phase_idx"].get("end", len(a["t"]))
    # free-space tracking: commanded per-step displacement (action*0.05) vs achieved
    cmd = a["action"][:, -6:-3] * 0.05 if a["action"].shape[1] >= 6 else a["action"][:, :3] * 0.05
    ach = np.diff(a["ee_pos"], axis=0, prepend=a["ee_pos"][:1])
    free = slice(1, sl if sl is not None else len(a["t"]))
    track_err = np.linalg.norm(cmd[free] - ach[free], axis=1)
    # descent-phase lag only (constant commanded speed) is the cleanest stiffness proxy
    desc = slice(d0 + 2, (sl - 1) if sl is not None else len(a["t"]))
    lag = np.linalg.norm(cmd[desc] - ach[desc], axis=1)
    out = {
        "settle_steps_to_hover": int(d0),
        "track_err_rms_mm": 1e3 * float(np.sqrt(np.mean(track_err**2))),
        "descent_lag_mean_mm": 1e3 * float(np.mean(lag)) if lag.size else float("nan"),
        "descent_speed_mps": float(np.mean(np.linalg.norm(a["ee_vel"][desc, :3], axis=1)))
        if a["ee_vel"][desc].size
        else float("nan"),
        "found_contact": ev.get("contact_height") is not None,
    }
    if sl is None or ev.get("contact_height") is None:
        return out
    press = slice(sl + 10, end)
    steady = float(np.mean(fmag_c[press]))
    # The OSC goal is re-anchored to the current pose every control step, so during the press the
    # commanded goal sits `press_depth` below the (barely moving) tool: the steady force divided by
    # the commanded depth is the effective Cartesian stiffness Λ·kp the tool actually presents.
    z_press = ev["contact_height"] - float(np.mean(a["ee_pos"][press, 2]))
    t_on = ev["t_contact_gt"]
    win_hf = (a["t_hf"] >= t_on - 0.01) & (a["t_hf"] <= t_on + 0.3)
    win_c = (a["t"] >= t_on - 0.01) & (a["t"] <= t_on + 0.3)
    out.update(
        {
            "peak_F_physics_N": float(fmag_hf[win_hf].max()),
            "peak_F_control_N": float(fmag_c[win_c].max()),
            "steady_F_N": steady,
            "achieved_press_depth_mm": 1e3 * z_press,
            "commanded_press_depth_mm": 1e3 * press_depth,
            "effective_stiffness_N_per_m": steady / press_depth,
            "impact_speed_mps": float(np.linalg.norm(a["ee_vel"][int(round(t_on / dt)) - 1, :3])),
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/m5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--press-depth-mm", type=float, default=10.0)
    ap.add_argument("--descend-speed", type=float, default=-0.4)
    args = ap.parse_args()
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({**vars(args), "kps": KPS, "zetas": ZETAS}, indent=2)
    )
    load = LoadParams(mass=0.03, com=np.zeros(3))  # identified in M4 (outputs/m4/load_params.json)
    prm = PressSlideParams(
        press_depth=args.press_depth_mm * 1e-3,
        descend_speed=args.descend_speed,
        use_compensated=True,
    )

    # ---- part 1: fixed impedance grid ---------------------------------------------------
    rows = []
    for kp in KPS:
        for z in ZETAS:
            env = make_env(
                "Wipe",
                controller_config=make_controller_config(kp=kp, damping_ratio=z),
                seed=args.seed,
            )
            rig = ArmRig(env, load=load)
            rig.reset(args.seed)
            table_z = float(env.table_offset[2])
            log = EpisodeLogger(
                config={"kp": kp, "damping_ratio": z, "press": prm.__dict__}, seed=args.seed
            )
            ev = press_and_slide(rig, np.array(rig.obs["wipe_centroid"][:2]), table_z, prm, log)
            a = log.arrays()
            log.save(out / f"fixed_kp{kp:g}_z{z:g}.npz")
            row = {"kp": kp, "damping_ratio": z, **analyse(a, ev, rig.dt_ctrl, prm.press_depth)}
            rows.append(row)
            print(
                json.dumps(
                    {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}
                )
            )
            env.close()
    (out / "fixed_grid.json").write_text(json.dumps(rows, indent=2))

    # ---- part 2: variable_kp, switch on contact -------------------------------------------
    vrows = []
    cfg = make_controller_config(impedance_mode="variable_kp", kp_limits=(0.0, 1000.0))
    env = make_env("Wipe", controller_config=cfg, seed=args.seed)
    rig = ArmRig(env, load=load)
    table_z = float(env.table_offset[2])
    for kp_free in (400.0, 1000.0):
        for kp_contact in (kp_free, 150.0, 50.0):
            for delay in (0, 1, 2):
                rig.reset(args.seed)
                log = EpisodeLogger(
                    config={
                        "mode": "variable_kp",
                        "kp_free": kp_free,
                        "kp_contact": kp_contact,
                        "switch_delay": delay,
                        "press": prm.__dict__,
                    },
                    seed=args.seed,
                )
                ev = press_and_slide_variable_kp(
                    rig,
                    np.array(rig.obs["wipe_centroid"][:2]),
                    table_z,
                    prm,
                    kp_free,
                    kp_contact,
                    log,
                    switch_delay_steps=delay,
                )
                a = log.arrays()
                log.save(out / f"varkp_{kp_free:g}_to_{kp_contact:g}_d{delay}.npz")
                row = {
                    "kp_free": kp_free,
                    "kp_contact": kp_contact,
                    "switch_delay_steps": delay,
                    **analyse(a, ev, rig.dt_ctrl, prm.press_depth),
                }
                # peak after the switch (excludes the impact step itself)
                if ev.get("t_switch") is not None:
                    fm = np.linalg.norm(a["ft_comp"][:, :3], axis=1)
                    fh = np.linalg.norm(a["ft_raw_hf"][:, :3], axis=1)
                    row["peak_after_switch_control_N"] = float(fm[a["t"] > ev["t_switch"]].max())
                    row["peak_after_switch_physics_N"] = float(fh[a["t_hf"] > ev["t_switch"]].max())
                vrows.append(row)
                print(
                    json.dumps(
                        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}
                    )
                )
    (out / "variable_kp.json").write_text(json.dumps(vrows, indent=2))
    env.close()

    # ---- plots -------------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for z in ZETAS:
        sel = [r for r in rows if r["damping_ratio"] == z and r["found_contact"]]
        kps = [r["kp"] for r in sel]
        axes[0].plot(
            kps, [r["peak_F_physics_N"] for r in sel], "o-", label=f"peak (physics), ζ={z}"
        )
        axes[0].plot(kps, [r["steady_F_N"] for r in sel], "s--", label=f"steady, ζ={z}")
        axes[1].plot(kps, [r["descent_lag_mean_mm"] for r in sel], "o-", label=f"ζ={z}")
        axes[2].plot(kps, [r["effective_stiffness_N_per_m"] for r in sel], "o-", label=f"ζ={z}")
    axes[0].set_xlabel("OSC kp [1/s²]")
    axes[0].set_ylabel("|F| [N] (compensated, tool on table)")
    axes[0].set_title("Contact force vs kp (10 mm press)")
    axes[1].set_xlabel("OSC kp [1/s²]")
    axes[1].set_ylabel("free-space lag per step [mm]")
    axes[1].set_title("Tracking error vs kp (descent at −0.4)")
    axes[2].set_xlabel("OSC kp [1/s²]")
    axes[2].set_ylabel("steady F / commanded 10 mm [N/m]")
    axes[2].set_title("Effective Cartesian stiffness")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "stiffness_tradeoff.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for r in vrows:
        tag = f"{r['kp_free']:g}→{r['kp_contact']:g} d{r['switch_delay_steps']}"
        ax.bar(tag, r.get("peak_F_control_N", 0), color="C0", alpha=0.7)
        ax.bar(tag, r.get("steady_F_N", 0), color="C1", alpha=0.7, width=0.4)
    ax.set_ylabel("|F| [N]")
    ax.set_title("variable_kp: onset peak (control rate, blue) and steady (orange)")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "variable_kp.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
