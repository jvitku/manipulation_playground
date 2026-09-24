#!/usr/bin/env python
"""Collect the time series and training/eval results the HTML report plots into one JSON.

Episodes (PLAN §5 .npz) are reduced to control-rate columns plus the per-interval max of the
physics-rate |F| (so the spikes survive the downsampling). Units in every key name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def episode(path: str | Path, label: str) -> dict | None:
    path = Path(path)
    if not path.exists():
        print(f"[skip] {path}")
        return None
    z = np.load(path)
    t = z["t"] - z["t"][0]
    T = len(t)
    hf = z["ft_raw_hf"]
    k = len(hf) // T
    hf_fmag_max = np.linalg.norm(hf[: k * T, :3], axis=1).reshape(T, k).max(1)
    d = {
        "label": label,
        "file": str(path),
        "t_s": t.round(4).tolist(),
        "speed_mmps": (1e3 * np.linalg.norm(z["ee_vel"][:, :3], axis=1)).round(2).tolist(),
        "vz_mmps": (1e3 * z["ee_vel"][:, 2]).round(2).tolist(),
        "z_mm": (1e3 * (z["ee_pos"][:, 2] - z["ee_pos"][0, 2])).round(3).tolist(),
        "fmag_comp_N": np.linalg.norm(z["ft_comp"][:, :3], axis=1).round(4).tolist(),
        "fmag_hfmax_N": hf_fmag_max.round(4).tolist(),
        "comp_N_Nm": z["ft_comp"].round(5).tolist(),
        "contact_N": np.linalg.norm(z["contact_wrench"][:, :3], axis=1).round(4).tolist(),
        "n_contacts": z["n_contacts"].astype(int).tolist(),
    }
    if "joint_torque" in z.files and z["joint_torque"].size:
        d["joint_torque_Nm"] = z["joint_torque"].round(3).tolist()
    return d


def load_json(p: str | Path):
    p = Path(p)
    if not p.exists():
        print(f"[skip] {p}")
        return None
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/report/report_data.json")
    args = ap.parse_args()
    o = Path("outputs")
    eps = [
        episode(o / "m1/offset_0mm.npz", "Track A gantry, aligned insert (0 mm)"),
        episode(o / "m1/offset_4mm.npz", "Track A gantry, 4 mm offset -> rim jam"),
        episode(o / "m4/press_slide_comp.npz", "Panda on Wipe: press and slide"),
        episode(o / "m6_pih/kp150_offset_1mm.npz", "Panda PegInHole, kp 150, 1 mm: wedge"),
        episode(o / "m6_pih/kp400_offset_0mm.npz", "Panda PegInHole, kp 400, 0 mm: insert"),
    ]
    nut = sorted(o.glob("m6/nut_seed*.npz"))
    if nut:
        eps.append(episode(nut[0], f"Panda NutAssemblyRound ({nut[0].stem})"))
    train = {}
    for name in ("force", "noforce", "mlp_force", "mlp_noforce"):
        h = load_json(o / f"s1/{name}/history.json")
        if h:
            train[name] = h
    data = {
        "episodes": [e for e in eps if e],
        "training": train,
        "eval": load_json(o / "s1/eval/eval.json"),
        "eval_sigma3": load_json(o / "s1/eval_sigma3/eval.json"),
        "noise_sweep": load_json(o / "s1_noise/sweep.json"),
        "m3_ranking": load_json(o / "m3/ranking.json"),
        "m5_fixed": load_json(o / "m5/fixed_grid.json"),
        "m2_report": load_json(o / "m2/report.json"),
        "videos": load_json(o / "s1/videos/summary.json"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
