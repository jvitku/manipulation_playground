#!/usr/bin/env python
"""M4: robosuite Panda + OSC_POSE on `Wipe`: press-and-slide with F/T logging.

Checks (PLAN §7 M4):
  * env.robots[0].ee_force == raw MuJoCo sensor `gripper0_right_force_ee`
  * wrench rotated to world through the `ft_frame` site: pressing down gives world −Z on the
    tool→table wrench (i.e. +Z reaction from the table)
  * free-space F/T on the arm vs Track A
  * does an F/T-threshold contact detector false-trigger during fast free-space motion, before
    and after compensation?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fvb.control.osc_scripts import (
    ArmRig,
    PressSlideParams,
    identify_load,
    make_env,
    press_and_slide,
)
from fvb.ft.compensate import LoadParams
from fvb.logging.episode import EpisodeLogger
from fvb.viz.plots import plot_fmag, plot_ft_timeseries, write_mp4


def rms(x):
    return float(np.sqrt(np.mean(np.sum(np.atleast_2d(x) ** 2, axis=-1))))


def free_space_sweep(
    rig: ArmRig, load: LoadParams, out: Path, seed: int, speeds=(0.2, 0.5, 1.0)
) -> dict:
    """Fast free-space moves (up/down/sideways) — how big is the raw vs compensated swing?"""
    res = {}
    for s in speeds:
        rig.reset(seed)
        rig.load = load
        log = EpisodeLogger(config={"experiment": f"free_space_{s}"}, seed=0)
        for _ in range(10):
            rig.step(np.zeros(6), log)
        for sign in (+1, -1, +1, -1):
            for _ in range(8):
                rig.step(np.array([0, 0, sign * s, 0, 0, 0]), log)
        for sign in (+1, -1, +1, -1):
            for _ in range(8):
                rig.step(np.array([sign * s, 0, 0, 0, 0, 0]), log)
        for _ in range(10):
            rig.step(np.zeros(6), log)
        a = log.arrays()
        assert a["n_contacts"].max() == 0, "free-space run touched something"
        log.save(out / f"free_space_{s}.npz")
        g = load.mass * 9.81
        raw_dev = a["ft_raw"][:, :3] - a["ft_raw"][:5, :3].mean(axis=0)
        res[str(s)] = {
            "peak_speed_mps": float(np.max(np.linalg.norm(a["ee_vel"][:, :3], axis=1))),
            "peak_accel_est_mps2": float(
                np.max(np.abs(np.diff(a["ee_vel"][:, :3], axis=0))) / rig.dt_ctrl
            ),
            "raw_dev_rms_N": rms(raw_dev),
            "raw_dev_peak_N": float(np.max(np.linalg.norm(raw_dev, axis=1))),
            "raw_hf_peak_dev_N": float(
                np.max(
                    np.linalg.norm(
                        a["ft_raw_hf"][:, :3] - a["ft_raw_hf"][:5, :3].mean(axis=0), axis=1
                    )
                )
            ),
            "comp_rms_N": rms(a["ft_comp"][:, :3]),
            "comp_peak_N": float(np.max(np.linalg.norm(a["ft_comp"][:, :3], axis=1))),
            "m_g_N": g,
            "joint_torque_rms_Nm": rms(a["joint_torque"]),
        }
        plot_ft_timeseries(
            a["t"],
            a["ft_raw"],
            out / f"free_space_{s}_raw.png",
            title=f"Panda free-space moves, action scale {s}: raw sensor",
            t_hf=a["t_hf"],
            ft_hf=a["ft_raw_hf"],
        )
        plot_ft_timeseries(
            a["t"],
            a["ft_comp"],
            out / f"free_space_{s}_comp.png",
            title=f"Panda free-space moves, action scale {s}: compensated",
        )
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/m4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--press-depth-mm", type=float, default=10.0)
    ap.add_argument("--descend-speed", type=float, default=-0.4, help="unit action, x0.05 m/step")
    ap.add_argument("--heavy-tool-kg", type=float, default=0.9)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    report: dict = {}

    render = not args.no_render
    env = make_env("Wipe", offscreen=render, seed=args.seed)
    rig = ArmRig(env)
    rig.reset(args.seed)

    # ---- 1. accessor agreement + sign/frame check at rest ---------------------------------
    for _ in range(10):
        rig.step(np.zeros(6))
    raw, rs = rig.ft.ft_raw(), rig.ft.ft_robosuite()
    report["accessor_agreement"] = {
        "ft_raw_sensor": raw.tolist(),
        "ft_robosuite": rs.tolist(),
        "max_abs_diff": float(np.max(np.abs(raw - rs))),
    }
    assert np.allclose(raw, rs), "ee_force does not match raw sensor"
    ref = rig.model_load()
    R = rig.ee_rot()
    pred = ref.mass * (R.T @ np.array([0, 0, +9.81]))  # m·Rᵀ(−g), Track A convention
    report["rest_check"] = {
        "site_z_axis_world": R[:, 2].tolist(),
        "tool_subtree_mass_kg": ref.mass,
        "tool_com_sensor_frame_m": ref.com.tolist(),
        "ft_raw_force_N": raw[:3].tolist(),
        "predicted_m_RT_minus_g_N": pred.tolist(),
        "grip_site_vs_ft_site_rotation_deg": float(
            np.rad2deg(
                np.arccos(
                    np.clip((np.trace(_rot_xyzw(rig.obs["robot0_eef_quat"]).T @ R) - 1) / 2, -1, 1)
                )
            )
        ),
        "same_sign_convention_as_track_A": bool(np.allclose(raw[:3], pred, atol=0.05)),
    }
    print("rest check:", json.dumps(report["rest_check"], indent=2))

    # ---- 2. identify the load from static poses ------------------------------------------
    load, diag = identify_load(rig)
    report["identification"] = diag
    print("identified:", json.dumps(diag, indent=2))
    if abs(load.mass - ref.mass) / ref.mass > 0.2:
        print("!! identification off by >20%, using model values for compensation")
        load = ref
    (out / "load_params.json").write_text(
        json.dumps({"mass": load.mass, "com": load.com.tolist()}, indent=2)
    )

    # ---- 3. free-space contamination on the arm ------------------------------------------
    report["free_space"] = free_space_sweep(rig, load, out, args.seed)
    print("free space:", json.dumps(report["free_space"], indent=2))

    # ---- 4. press and slide, detector on raw vs compensated -------------------------------
    table_z = float(env.table_offset[2])
    for tag, use_comp in [("raw", False), ("comp", True)]:
        rig.reset(args.seed)
        rig.load = load
        obs = rig.obs
        target_xy = np.array(obs["wipe_centroid"][:2])
        prm = PressSlideParams(
            contact_threshold_N=args.threshold,
            press_depth=args.press_depth_mm * 1e-3,
            descend_speed=args.descend_speed,
            use_compensated=use_comp,
        )
        log = EpisodeLogger(
            config={
                "task": "Wipe",
                "press_slide": prm.__dict__,
                "load": {"mass": load.mass, "com": load.com.tolist()},
            },
            seed=args.seed,
        )
        frames = []
        renderer = None
        if render and tag == "comp":
            try:
                import mujoco

                renderer = mujoco.Renderer(rig.m, height=360, width=480)
            except Exception as e:  # noqa: BLE001
                print("render disabled:", e)
        orig_step = rig.step
        rig.step = _with_render(orig_step, rig, renderer, frames)
        ev = press_and_slide(rig, target_xy, table_z, prm, log)
        rig.step = orig_step
        a = log.arrays()
        log.success = bool(a["n_contacts"][-1] > 0)
        log.save(out / f"press_slide_{tag}.npz")
        if frames:
            write_mp4(frames, out / f"press_slide_{tag}.mp4", fps=20)
        # world-Z check while pressing: ft_world of the compensated wrench should be -Z
        sl = ev["phase_idx"].get("slide")
        end = ev["phase_idx"].get("end", len(a["t"]))
        R_all = np.stack([_rot_xyzw(q) for q in a["ee_quat"]])
        comp_world = np.stack([R_all[i] @ a["ft_comp"][i, :3] for i in range(len(a["t"]))])
        press = slice(sl + 10, end) if sl is not None else slice(0, 0)
        summary = {
            "events": ev,
            "steady_press_ft_comp_world_N": comp_world[press].mean(axis=0).tolist()
            if press.stop > press.start
            else None,
            "steady_contact_wrench_world_N": a["contact_wrench"][press, :3].mean(axis=0).tolist()
            if press.stop > press.start
            else None,
            "steady_ft_raw_sensor_N": a["ft_raw"][press, :3].mean(axis=0).tolist()
            if press.stop > press.start
            else None,
            "peak_F_control_N": float(np.max(np.linalg.norm(a["ft_comp"][:, :3], axis=1))),
            "peak_F_physics_N": float(np.max(np.linalg.norm(a["ft_raw_hf"][:, :3], axis=1))),
            "joint_torque_change_on_contact_Nm": (
                a["joint_torque"][press].mean(axis=0) - a["joint_torque"][:5].mean(axis=0)
            ).tolist()
            if press.stop > press.start
            else None,
        }
        report[f"press_slide_{tag}"] = summary
        print(tag, json.dumps(summary, indent=2, default=str))
        onset_t = ev["t_contact_gt"]
        plot_ft_timeseries(
            a["t"],
            a["ft_raw"],
            out / f"press_slide_{tag}_raw.png",
            title=f"Wipe press+slide ({tag} detector): raw sensor",
            onset_t=onset_t,
            t_hf=a["t_hf"],
            ft_hf=a["ft_raw_hf"],
        )
        plot_ft_timeseries(
            a["t"],
            a["ft_comp"],
            out / f"press_slide_{tag}_comp.png",
            title=f"Wipe press+slide ({tag} detector): compensated",
            onset_t=onset_t,
        )
        plot_fmag(
            a["t"],
            a["ft_comp"],
            out / f"press_slide_{tag}_fmag.png",
            title=f"|F| compensated, {tag} detector",
            onset_t=onset_t,
            t_hf=a["t_hf"],
            ft_hf=a["ft_raw_hf"],
        )
        # joint torque view
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 3.5))
        for j in range(a["joint_torque"].shape[1]):
            ax.plot(a["t"], a["joint_torque"][:, j], lw=1, label=f"j{j + 1}")
        if onset_t:
            ax.axvline(onset_t, color="k", ls="--", lw=0.8)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("applied joint torque [N·m]")
        ax.set_title("Joint torques (sim.data.ctrl) through approach / contact / slide")
        ax.legend(ncol=7, fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"press_slide_{tag}_joint_torque.png", dpi=130)
        plt.close(fig)
        if renderer is not None:
            renderer.close()

    # ---- 5. detector false-trigger sweep: threshold x descend speed, raw vs compensated ----
    ft_sweep = []
    for thr in (0.1, 0.3, 1.0):
        for spd in (-0.4, -1.0):
            for use_comp in (False, True):
                rig.reset(args.seed)
                rig.load = load
                prm = PressSlideParams(
                    contact_threshold_N=thr,
                    descend_speed=spd,
                    use_compensated=use_comp,
                    slide_steps=0,
                )
                ev = press_and_slide(rig, np.array(rig.obs["wipe_centroid"][:2]), table_z, prm)
                lead = (
                    None
                    if ev["t_detect"] is None or ev["t_contact_gt"] is None
                    else ev["t_contact_gt"] - ev["t_detect"]
                )
                ft_sweep.append(
                    {
                        "threshold_N": thr,
                        "descend_action": spd,
                        "compensated": use_comp,
                        "false_triggers": ev["false_triggers"],
                        "detect_lead_s": lead,
                        "found_contact": ev["contact_height"] is not None,
                    }
                )
                print(ft_sweep[-1])
    report["detector_sweep"] = ft_sweep

    # ---- 6. same again with a heavy tool (0.9 kg ~ a parallel gripper) --------------------
    rig.set_tool_mass(args.heavy_tool_kg)
    rig.reset(args.seed)
    heavy_ref = rig.model_load()
    heavy_load, hdiag = identify_load(rig)
    report["heavy_tool"] = {"mass_model_kg": heavy_ref.mass, "identification": hdiag}
    if abs(heavy_load.mass - heavy_ref.mass) / heavy_ref.mass > 0.2:
        heavy_load = heavy_ref
    hout = out / "heavy_tool"
    hout.mkdir(exist_ok=True)
    report["heavy_tool"]["free_space"] = free_space_sweep(
        rig, heavy_load, hout, args.seed, speeds=(1.0,)
    )
    hsweep = []
    for thr in (0.3, 1.0, 3.0):
        for spd in (-0.4, -1.0):
            for use_comp in (False, True):
                rig.reset(args.seed)
                rig.load = heavy_load
                prm = PressSlideParams(
                    contact_threshold_N=thr,
                    descend_speed=spd,
                    use_compensated=use_comp,
                    slide_steps=0,
                )
                ev = press_and_slide(rig, np.array(rig.obs["wipe_centroid"][:2]), table_z, prm)
                hsweep.append(
                    {
                        "threshold_N": thr,
                        "descend_action": spd,
                        "compensated": use_comp,
                        "false_triggers": ev["false_triggers"],
                        "found_contact": ev["contact_height"] is not None,
                    }
                )
                print("heavy", hsweep[-1])
    report["heavy_tool"]["detector_sweep"] = hsweep
    print("heavy tool free space:", json.dumps(report["heavy_tool"]["free_space"], indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out / 'report.json'}")
    env.close()


def _with_render(step_fn, rig, renderer, frames):
    def step_and_render(a, lg=None):
        r = step_fn(a, lg)
        if renderer is not None:
            renderer.update_scene(rig.d, camera="frontview")
            frames.append(renderer.render().copy())
        return r

    return step_and_render


def _rot_xyzw(q):
    import mujoco

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.array([q[3], q[0], q[1], q[2]]))
    return R.reshape(3, 3)


if __name__ == "__main__":
    main()
