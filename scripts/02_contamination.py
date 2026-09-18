#!/usr/bin/env python
"""M2: gravity and inertial contamination of the wrist F/T in free space, and compensation.

Experiments (no contact anywhere):
  hold        - peg hanging still
  descent     - constant-velocity descent at 50 and 150 mm/s
  sine_*Hz    - sinusoidal z motion, 10 mm amplitude, at 0.5 / 2 / 5 Hz
  tilt        - wrist hinge about y, peg rotated 0..90 deg in steps, held at each (static poses)

Mass and COM are identified by least squares from the tilt static poses (as on a real robot),
then every episode is compensated with (a) gravity only, (b) gravity + finite-difference
acceleration from control-rate velocity, (c) gravity + MuJoCo's exact site acceleration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from fvb.control.gantry import Gantry
from fvb.ft.compensate import (
    LoadParams,
    compensate_series,
    estimate_accel,
    identify_mass_com,
)
from fvb.ft.frames import mat_to_quat_xyzw, site_to_world
from fvb.logging.episode import EpisodeLogger
from fvb.scenes.builder import SceneParams, build_model
from fvb.viz.plots import plot_ft_timeseries


def _rot_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.array([q[3], q[0], q[1], q[2]]))
    return R.reshape(3, 3)


class FreeSpaceRunner:
    """Runs free-space target trajectories on the gantry and logs episodes with a_world."""

    def __init__(
        self, p: SceneParams, control_freq: float, seed: int, load: LoadParams | None = None
    ):
        self.p, self.control_freq, self.seed = p, control_freq, seed
        self.m, self.d = build_model(p)
        self.g = Gantry(self.m, self.d, p, control_freq, load=load)
        self.g.set_target([0.0, 0.0, 0.0])
        self.g.settle(2.0)

    def run(self, name: str, targets: np.ndarray, tilts: np.ndarray | None = None):
        log = EpisodeLogger(
            config={"scene": asdict(self.p), "experiment": name, "control_freq": self.control_freq},
            seed=self.seed,
        )
        a_mj = []
        for i, tgt in enumerate(targets):
            if tilts is not None:
                self.g.set_tilt(float(tilts[i]))
            self.g.step(tgt, log)
            a_mj.append(self.g.ee_acc_world())
            assert self.d.ncon == 0, f"{name}: unexpected contact"
        return log, np.array(a_mj)


def make_targets(kind: str, dt: float, z0: float = 0.0) -> tuple[np.ndarray, np.ndarray | None]:
    if kind == "hold":
        n = int(3.0 / dt)
        return np.tile([0, 0, z0], (n, 1)), None
    if kind.startswith("descent_"):
        v = float(kind.split("_")[1]) * 1e-3
        dz = 0.03
        n = int(dz / v / dt)
        zs = np.concatenate(
            [
                np.full(int(0.5 / dt), z0),
                z0 - np.arange(n) * v * dt,
                np.full(int(1.0 / dt), z0 - n * v * dt),
            ]
        )
        return np.column_stack([np.zeros((len(zs), 2)), zs]), None
    if kind.startswith("sine_"):
        f = float(kind.split("_")[1])
        amp = 0.010
        t = np.arange(0, 4.0, dt)
        zs = z0 + amp * np.sin(2 * np.pi * f * t) - amp * 0  # start at z0
        return np.column_stack([np.zeros((len(zs), 2)), zs]), None
    if kind == "tilt":
        angles = np.deg2rad([0, 15, 30, 45, 60, 75, 90, 60, 30, 0])
        hold = int(2.0 / dt)
        tilts = np.repeat(angles, hold)
        return np.tile([0, 0, z0], (len(tilts), 1)), tilts
    raise ValueError(kind)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.atleast_2d(x) ** 2, axis=-1))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/m2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-freq", type=float, default=20.0)
    ap.add_argument("--kp", type=float, default=800.0)
    ap.add_argument("--peg-mass", type=float, default=0.1)
    args = ap.parse_args()
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    dt = 1.0 / args.control_freq

    p_tilt = SceneParams(kp=args.kp, peg_mass=args.peg_mass, tilt_axis=(0, 1, 0))
    p_plain = SceneParams(kp=args.kp, peg_mass=args.peg_mass)
    mg = args.peg_mass * 9.81
    report: dict = {"m_g_N": mg}

    # ---- 1. tilt experiment -> static poses -> identify mass and COM -----------------------
    r = FreeSpaceRunner(p_tilt, args.control_freq, args.seed)
    tg, tilts = make_targets("tilt", dt)
    log, a_mj = r.run("tilt", tg, tilts)
    a = log.arrays()
    # static samples: last 25% of each 2 s hold
    hold = int(2.0 / dt)
    idx = np.concatenate(
        [np.arange(k * hold + int(0.75 * hold), (k + 1) * hold) for k in range(len(a["t"]) // hold)]
    )
    R_all = np.stack([_rot_from_quat_xyzw(q) for q in a["ee_quat"]])
    load = identify_mass_com(a["ft_raw"][idx], R_all[idx])
    true_com = np.array([0.0, 0.0, -p_tilt.peg_len / 2])
    report["identification"] = {
        "n_static_samples": int(len(idx)),
        "mass_est_kg": load.mass,
        "mass_true_kg": args.peg_mass,
        "com_est_m": load.com.tolist(),
        "com_true_m": true_com.tolist(),
        "static_force_residual_rms_N": load.residual_force_rms,
        "static_torque_residual_rms_Nm": load.residual_torque_rms,
    }
    print("identified load:", json.dumps(report["identification"], indent=2))
    # sanity check the cacc convention: at rest, ee_acc_world must be ~0 (not +9.81)
    report["cacc_check_rest_accel_mps2"] = a_mj[idx][-5:].mean(axis=0).tolist()
    a["ft_comp"] = compensate_series(a["ft_raw"], R_all, load, a_mj)
    log.save(out / "tilt.npz")
    plot_ft_timeseries(
        a["t"],
        a["ft_raw"],
        out / "tilt_raw.png",
        title="Tilt 0..90 deg about y: raw sensor (gravity moves into Fx)",
    )
    plot_ft_timeseries(
        a["t"],
        a["ft_comp"],
        out / "tilt_comp.png",
        title="Tilt: after gravity+inertial compensation (identified m, COM)",
    )
    report["tilt"] = {
        "raw_rms_N": rms(a["ft_raw"][:, :3]),
        "comp_rms_N": rms(a["ft_comp"][:, :3]),
        "raw_torque_rms_Nm": rms(a["ft_raw"][:, 3:]),
        "comp_torque_rms_Nm": rms(a["ft_comp"][:, 3:]),
        "Fx_raw_at_90deg_N": float(np.median(a["ft_raw"][6 * hold + hold // 2 : 7 * hold, 0])),
    }

    # ---- 2. free-space motions on the plain scene ----------------------------------------
    results = {}
    for kind in ["hold", "descent_50", "descent_150", "sine_0.5", "sine_2", "sine_5"]:
        r = FreeSpaceRunner(p_plain, args.control_freq, args.seed, load=load)
        tg, _ = make_targets(kind, dt)
        log, a_mj = r.run(kind, tg)
        a = log.arrays()
        R_all = np.stack([_rot_from_quat_xyzw(q) for q in a["ee_quat"]])
        a_fd = estimate_accel(a["t"], a["ee_vel"][:, :3])
        comp_g = compensate_series(a["ft_raw"], R_all, load, None)
        comp_fd = compensate_series(a["ft_raw"], R_all, load, a_fd)
        comp_mj = compensate_series(a["ft_raw"], R_all, load, a_mj)
        comp_hf = a["ft_comp"]  # physics-rate FD compensation averaged per control step
        log.save(out / f"{kind}.npz")
        # also physics-rate raw stats (what the hf log sees)
        hf = a["ft_raw_hf"]
        res = {
            "raw_rms_N": rms(a["ft_raw"][:, :3]),
            "raw_minus_mg_rms_N": rms(a["ft_raw"][:, :3] - np.array([0, 0, mg])),
            "raw_hf_peak_dev_from_mg_N": float(np.max(np.abs(hf[:, 2] - mg))),
            "comp_gravity_only_rms_N": rms(comp_g[:, :3]),
            "comp_gravity_fd_accel_rms_N": rms(comp_fd[:, :3]),
            "comp_hf_fd_averaged_rms_N": rms(comp_hf[:, :3]),
            "comp_gravity_mj_accel_rms_N": rms(comp_mj[:, :3]),
            "peak_accel_mj_mps2": float(np.max(np.abs(a_mj[:, 2]))),
            "peak_accel_fd_mps2": float(np.max(np.abs(a_fd[:, 2]))),
            "peak_vel_mps": float(np.max(np.abs(a["ee_vel"][:, 2]))),
        }
        res["raw_pct_of_mg"] = 100 * res["raw_minus_mg_rms_N"] / mg
        res["comp_fd_pct_of_mg"] = 100 * res["comp_gravity_fd_accel_rms_N"] / mg
        res["comp_hf_pct_of_mg"] = 100 * res["comp_hf_fd_averaged_rms_N"] / mg
        res["comp_mj_pct_of_mg"] = 100 * res["comp_gravity_mj_accel_rms_N"] / mg
        results[kind] = res
        print(kind, json.dumps({k: round(v, 4) for k, v in res.items()}))
        plot_ft_timeseries(
            a["t"],
            a["ft_raw"],
            out / f"{kind}_raw.png",
            title=f"{kind}: raw sensor (free space)",
            t_hf=a["t_hf"],
            ft_hf=a["ft_raw_hf"],
        )
        # overlay of the three compensation variants for Fz
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(a["t"], a["ft_raw"][:, 2] - mg, label="raw − m·g", lw=1)
        ax.plot(a["t"], comp_g[:, 2], label="gravity comp (identified m)", lw=1)
        ax.plot(a["t"], comp_fd[:, 2], label="+ inertial, finite-diff accel @ control rate", lw=1)
        ax.plot(a["t"], comp_hf[:, 2], label="+ inertial, FD accel @ physics rate, averaged", lw=1)
        ax.plot(a["t"], comp_mj[:, 2], label="+ inertial, exact site accel", lw=1, ls="--")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("Fz residual [N] (sensor frame)")
        ax.set_title(f"{kind}: free-space residual after compensation (should be 0)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{kind}_comp.png", dpi=130)
        plt.close(fig)
    report["free_space"] = results

    # ---- 3. contact case: press on the rim, compare ft_comp with contact ground truth -------
    r = FreeSpaceRunner(p_plain, args.control_freq, args.seed, load=load)
    r.g.set_target([2e-3, 0, 0])
    r.g.settle(1.0)
    z_rim = -(p_plain.start_height - 0.01 - p_plain.peg_len - p_plain.hole_depth)
    for z in np.linspace(0, z_rim - 0.01, 60):
        r.g.step([2e-3, 0, z])
    for _ in range(int(1.0 / dt)):
        r.g.step([2e-3, 0, z_rim - 0.01])
    raw = r.g.ft.ft_raw()
    R = r.g.ee_rot()
    cw, n = r.g.contact_wrench()
    comp = r.g.ft_comp_last()
    # ft_comp is the wrench the LOAD applies to the world (sensor frame) = -(contact on load)
    cw_site = site_to_world(cw, R.T)
    report["contact_check"] = {
        "n_contacts": int(n),
        "ft_raw_N": raw[:3].tolist(),
        "ft_comp_N": comp[:3].tolist(),
        "contact_on_peg_world_N": cw[:3].tolist(),
        "minus_contact_in_site_N": (-cw_site[:3]).tolist(),
        "force_error_pct": float(
            100 * np.linalg.norm(comp[:3] + cw_site[:3]) / np.linalg.norm(cw_site[:3])
        ),
        "torque_comp_Nm": comp[3:].tolist(),
        "minus_contact_torque_site_Nm": (-cw_site[3:]).tolist(),
    }
    print("contact check:", json.dumps(report["contact_check"], indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "load_params.json").write_text(
        json.dumps({"mass": load.mass, "com": load.com.tolist()}, indent=2)
    )
    print(f"wrote {out / 'report.json'}")
    _ = mat_to_quat_xyzw


if __name__ == "__main__":
    main()
