#!/usr/bin/env python
"""M3: sensitivity of the F/T signal to solver and contact parameters.

Same scripted insertion (0.3 mm lateral offset, bottoms out on the hole floor) over a grid of
timestep, integrator, cone, solref time constant, noslip iterations, stiffness kp and approach
speed. Two designs are run:
  * OAT  — one-at-a-time around the baseline (cheap, readable)
  * FULL — the full factorial (864 runs, multiprocessing), used for the ranking and heatmaps
Outputs a tidy CSV, heatmaps and a ranked list of the parameters that move the force signal most.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from fvb.contacts import max_penetration
from fvb.control.gantry import Gantry, linear_descent_targets
from fvb.ft.filters import contact_onset_index, spike_metrics
from fvb.scenes.builder import SceneParams, build_model, insertion_depth
from fvb.viz.plots import plot_heatmap

GRID = {
    "timestep": [0.0005, 0.001, 0.002, 0.005],
    "integrator": ["Euler", "implicitfast"],
    "cone": ["pyramidal", "elliptic"],
    "solref_tc": [0.002, 0.005, 0.02],
    "noslip_iterations": [0, 5],
    "kp": [200.0, 800.0, 3000.0],
    "speed_mmps": [10.0, 50.0, 150.0],
}
BASELINE = {
    "timestep": 0.002,
    "integrator": "implicitfast",
    "cone": "elliptic",
    "solref_tc": 0.005,
    "noslip_iterations": 0,
    "kp": 800.0,
    "speed_mmps": 50.0,
}
X_OFFSET = 0.3e-3
CONTROL_FREQ = 20.0
TARGET_DEPTH = 0.050  # > 40 mm hole depth -> bottoms out


def run_one(cfg: dict) -> dict:
    p = SceneParams(
        timestep=cfg["timestep"],
        integrator=cfg["integrator"],
        cone=cfg["cone"],
        solref=(cfg["solref_tc"], 1.0),
        noslip_iterations=cfg["noslip_iterations"],
        kp=cfg["kp"],
    )
    speed = cfg["speed_mmps"] * 1e-3
    out = dict(cfg)
    try:
        m, d = build_model(p)
        g = Gantry(m, d, p, CONTROL_FREQ)
        g.set_target([X_OFFSET, 0, 0])
        g.settle(1.0)
        dt = 1.0 / CONTROL_FREQ
        z_end = -(p.start_height - 0.01 - p.peg_len - p.hole_depth) - TARGET_DEPTH
        targets = linear_descent_targets(0.0, z_end, speed, dt, xy=[X_OFFSET, 0], hold=1.0)
        t_hf, f_hf, pen, ncon, t_c, f_c, pos = [], [], [], [], [], [], []
        t0 = time.perf_counter()
        for tgt in targets:
            g.set_target(tgt)
            for _ in range(g.n_sub):
                import mujoco

                mujoco.mj_step(m, d)
                t_hf.append(d.time)
                f_hf.append(g.ft.ft_raw())
                pen.append(max_penetration(m, d, ["peg"]))
                ncon.append(d.ncon)
            t_c.append(d.time)
            f_c.append(g.ft.ft_raw())
            pos.append(g.ee_pos())
        wall = time.perf_counter() - t0
        t_hf, f_hf, pen = np.array(t_hf), np.array(f_hf), np.array(pen)
        t_c, f_c, pos = np.array(t_c), np.array(f_c), np.array(pos)
        if not np.all(np.isfinite(f_hf)):
            raise FloatingPointError("non-finite force")
        fm_hf = np.linalg.norm(f_hf[:, :3], axis=1)
        fm_c = np.linalg.norm(f_c[:, :3], axis=1)
        ncon = np.array(ncon)
        first = np.nonzero(ncon > 0)[0]
        on_hf = int(first[0]) if len(first) else None  # ground-truth onset
        sm_hf = spike_metrics(fm_hf, t_hf, on_hf)
        on_c = int(np.searchsorted(t_c, t_hf[on_hf])) if on_hf is not None else None
        on_c = min(on_c, len(t_c) - 1) if on_c is not None else None
        sm_c = spike_metrics(fm_c, t_c, on_c, window_s=0.15)
        # where an F/T threshold detector would have fired (may be before true contact)
        on_ft = contact_onset_index(fm_hf, 0.3, baseline_n=50)
        jz = pos[-1, 2] - (p.start_height - 0.01)
        depth = insertion_depth(p, jz)
        sim_s = t_hf[-1] - t_hf[0]
        # motion metrics: the peg's z trajectory at control rate, to ask whether force changes
        # without motion changing
        out.update(
            {
                "ok": True,
                "peak_F_hf_N": float(fm_hf.max()),
                "peak_F_c_N": float(fm_c.max()),
                "spike_height_hf_N": sm_hf["spike_height"],
                "spike_width_hf_ms": 1e3 * sm_hf["spike_width_s"],
                "spike_height_c_N": sm_c["spike_height"],
                "steady_F_N": sm_hf["steady"],
                "steady_Fz_N": float(np.median(f_c[-10:, 2])),
                "max_penetration_mm": 1e3 * float(pen.max()),
                "max_ncon": int(ncon.max()),
                "final_depth_mm": 1e3 * float(depth),
                "success": bool(depth > 0.039),
                "onset_t_s": float(t_hf[on_hf]) if on_hf is not None else float("nan"),
                "ft_detector_lead_s": (
                    float(t_hf[on_hf] - t_hf[on_ft])
                    if (on_hf is not None and on_ft is not None)
                    else float("nan")
                ),
                "wall_per_sim_s": float(wall / sim_s),
                "z_traj_rms_mm": 1e3 * float(np.sqrt(np.mean(pos[:, 2] ** 2))),
                "z_final_m": float(pos[-1, 2]),
                "fz_hf_std_last_0p5s_N": float(np.std(f_hf[t_hf > t_hf[-1] - 0.5, 2])),
            }
        )
    except Exception as e:  # noqa: BLE001
        out.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return out


def to_csv(rows: list[dict], path: Path) -> None:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in GRID, k))
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/m3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", type=int, default=1, help="run the full factorial (1) or OAT only")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "grid": GRID,
                "baseline": BASELINE,
                "x_offset_m": X_OFFSET,
                "control_freq": CONTROL_FREQ,
                "target_depth_m": TARGET_DEPTH,
            },
            indent=2,
        )
    )

    # ---- OAT ---------------------------------------------------------------------------
    oat_cfgs = [dict(BASELINE, _sweep="baseline")]
    for k, vals in GRID.items():
        for v in vals:
            if v != BASELINE[k]:
                oat_cfgs.append(dict(BASELINE, **{k: v}, _sweep=k))
    with mp.Pool(args.workers) as pool:
        oat = pool.map(run_one, oat_cfgs)
    to_csv(oat, out / "oat.csv")
    base = oat[0]
    print(
        "baseline:",
        json.dumps(
            {
                k: base[k]
                for k in (
                    "peak_F_hf_N",
                    "peak_F_c_N",
                    "spike_height_hf_N",
                    "spike_width_hf_ms",
                    "steady_F_N",
                    "max_penetration_mm",
                    "success",
                    "wall_per_sim_s",
                )
            }
        ),
    )
    print(
        f"{'param':<20}{'value':>12}{'peakF_hf':>10}{'spikeH':>9}{'spikeW_ms':>10}"
        f"{'steadyF':>9}{'pen_mm':>8}{'ok':>4}{'wall/s':>8}"
    )
    for r in oat[1:]:
        k = r["_sweep"]
        print(
            f"{k:<20}{str(r[k]):>12}{r.get('peak_F_hf_N', float('nan')):>10.2f}"
            f"{r.get('spike_height_hf_N', float('nan')):>9.2f}"
            f"{r.get('spike_width_hf_ms', float('nan')):>10.1f}"
            f"{r.get('steady_F_N', float('nan')):>9.2f}"
            f"{r.get('max_penetration_mm', float('nan')):>8.3f}"
            f"{str(r.get('success', '-')):>4}{r.get('wall_per_sim_s', float('nan')):>8.3f}"
        )

    # ---- full factorial ----------------------------------------------------------------
    if not args.full:
        return
    keys = list(GRID)
    cfgs = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*GRID.values())]
    print(f"full factorial: {len(cfgs)} runs on {args.workers} workers")
    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        rows = pool.map(run_one, cfgs, chunksize=4)
    print(f"done in {time.perf_counter() - t0:.0f} s; failures: {sum(not r['ok'] for r in rows)}")
    to_csv(rows, out / "full.csv")
    ok = [r for r in rows if r["ok"]]

    # ranking: for each parameter, the mean over all other settings of the range of each metric
    # when only that parameter changes (main-effect size), for force AND for motion.
    metrics = [
        "peak_F_hf_N",
        "peak_F_c_N",
        "spike_height_hf_N",
        "spike_width_hf_ms",
        "steady_F_N",
        "max_penetration_mm",
        "z_traj_rms_mm",
        "wall_per_sim_s",
    ]
    ranking = {}
    for k in keys:
        others = [o for o in keys if o != k]
        groups: dict[tuple, list[dict]] = {}
        for r in ok:
            groups.setdefault(tuple(r[o] for o in others), []).append(r)
        eff = {}
        for mname in metrics:
            rngs = [
                max(r[mname] for r in grp) - min(r[mname] for r in grp)
                for grp in groups.values()
                if len(grp) == len(GRID[k])
            ]
            eff[mname] = float(np.mean(rngs)) if rngs else float("nan")
        ranking[k] = eff
    force_key = "peak_F_hf_N"
    order = sorted(keys, key=lambda k: -ranking[k][force_key])
    lines = [
        "# Main effect = mean range of the metric when only this parameter varies\n",
        f"{'param':<20}" + "".join(f"{m:>18}" for m in metrics),
    ]
    for k in order:
        lines.append(f"{k:<20}" + "".join(f"{ranking[k][m]:>18.3f}" for m in metrics))
    (out / "ranking.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    (out / "ranking.json").write_text(
        json.dumps({"order_by_peak_F_hf": order, "effects": ranking}, indent=2)
    )

    # heatmaps: baseline slice, timestep x kp and timestep x solref for peak force + penetration
    def slice_mat(rowkey, colkey, metric):
        mat = np.full((len(GRID[rowkey]), len(GRID[colkey])), np.nan)
        for r in ok:
            if all(r[o] == BASELINE[o] for o in keys if o not in (rowkey, colkey)):
                mat[GRID[rowkey].index(r[rowkey]), GRID[colkey].index(r[colkey])] = r[metric]
        return mat

    for rk, ck in [
        ("timestep", "kp"),
        ("timestep", "solref_tc"),
        ("timestep", "speed_mmps"),
        ("kp", "speed_mmps"),
        ("solref_tc", "speed_mmps"),
    ]:
        for metric, cb in [
            ("peak_F_hf_N", "peak |F| at physics rate [N]"),
            ("spike_height_hf_N", "onset spike height [N]"),
            ("max_penetration_mm", "max penetration [mm]"),
        ]:
            plot_heatmap(
                slice_mat(rk, ck, metric),
                GRID[ck],
                GRID[rk],
                out / f"heat_{metric}__{rk}_x_{ck}.png",
                title=f"{metric} — {rk} × {ck} (others at baseline)",
                xlabel=ck,
                ylabel=rk,
                cbar=cb,
                fmt="{:.2f}" if metric == "max_penetration_mm" else "{:.1f}",
            )
    # integrator x cone at each timestep
    for metric in ["peak_F_hf_N", "spike_height_hf_N", "wall_per_sim_s"]:
        mat = np.full((len(GRID["timestep"]), 4), np.nan)
        cols = [f"{i}/{c}" for i in GRID["integrator"] for c in GRID["cone"]]
        for r in ok:
            if all(
                r[o] == BASELINE[o] for o in ("solref_tc", "noslip_iterations", "kp", "speed_mmps")
            ):
                mat[
                    GRID["timestep"].index(r["timestep"]),
                    cols.index(f"{r['integrator']}/{r['cone']}"),
                ] = r[metric]
        plot_heatmap(
            mat,
            cols,
            GRID["timestep"],
            out / f"heat_{metric}__timestep_x_intcone.png",
            title=f"{metric} — timestep × integrator/cone",
            xlabel="integrator/cone",
            ylabel="timestep [s]",
            cbar=metric,
            fmt="{:.2f}",
        )
    # unstable / tunnelling table
    bad = [r for r in rows if (not r["ok"]) or r["max_penetration_mm"] > 1.0 or (not r["success"])]
    (out / "unstable_or_tunnelling.json").write_text(json.dumps(bad, indent=1, default=str))
    print(f"runs with failure / >1 mm penetration / no insertion: {len(bad)}")
    _ = asdict, replace


if __name__ == "__main__":
    main()
