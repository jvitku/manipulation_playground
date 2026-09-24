"""Build docs/stage1/index.html (Peg-in-Hole Stage 1 Report) from outputs/ and data/.

Needs the Stage 1 completion runs (PLAN §11): outputs/s1_arm{,_flangeref,_xyabs}, outputs/s1_phys,
outputs/diag/act_arm_*, outputs/s1_arm/videos, data/hdf5. Run inside the dev image:
`python docs/stage1/build.py`. Style and chart helpers are shared with docs/report/template.html.
"""

import datetime
import glob
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
O, DATA = REPO / "outputs", REPO / "data"

# Measured in the debugging passes (docs/FINDINGS.md "Stage 1 completion"): share of jams where
# the torque sign at the jam predicts the expert's next correction (hole > 0.6 mm away), and the
# trained model's predicted correction size relative to the expert's at correction steps.
CUE = {"flange_x": 0.54, "tip_x": 0.78, "flange_y": 0.96, "tip_y": 0.97}
COLLAPSE = {"nominal_delta": 0.99, "kprand_delta": 0.01, "kprand_xyabs": 0.99}


def load(p):
    return json.loads(Path(p).read_text())


def _finite(o):
    """NaN/inf -> None: evals with zero successes have a NaN mean step count, and browsers'
    JSON.parse rejects the NaN token Python's json writes."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def hist(p):
    h = load(p)
    return {k: h[k] for k in ("history", "best_val_l1", "params", "wall_s")}


def main() -> None:
    labels_seed = 3
    d = np.load(DATA / f"expert_trackA/ep{labels_seed:04d}_policy.npz")["action"][:, 0]
    a = np.load(DATA / f"expert_trackA_xyabs/ep{labels_seed:04d}_policy.npz")["action"][:, 0]
    vids = load(O / "s1_arm/videos/summary.json")
    seeds = sorted({r["seed"] for r in vids["runs"]})
    expert = load(DATA / "expert_arm/summary.json")
    export = []
    for f in sorted(glob.glob(str(DATA / "hdf5/*.hdf5"))):
        import h5py

        with h5py.File(f, "r") as h:
            export.append(
                {
                    "path": str(Path(f).relative_to(REPO)),
                    "task_spec": str(h["data"].attrs["task_spec"]),
                    "demos": len(h["data"]),
                    "samples": int(h["data"].attrs["total"]),
                    "train": len(h["mask/train"]),
                    "valid": len(h["mask/valid"]),
                    "size_mb": Path(f).stat().st_size / 1e6,
                }
            )
    n_tests = subprocess.run(
        ["pytest", "--collect-only", "-q"], cwd=REPO, capture_output=True, text=True
    ).stdout.count("::")
    act_variants = {"H 10 (default)": O / "s1_arm/force/history.json"}
    for k, v in (
        ("H 16", "H16"),
        ("H 20", "H20"),
        ("lr 1e-4, 80 ep", "lr1e-4"),
        ("250 ep", "e250"),
    ):
        act_variants[k] = O / f"diag/act_arm_{v}/history.json"
    data = {
        "arm_tip": load(O / "s1_arm/eval/eval.json"),
        "arm_flange": load(O / "s1_arm_flangeref/eval/eval.json"),
        "arm_xyabs": load(O / "s1_arm_xyabs/eval/eval.json"),
        "arm_train": {
            k: hist(O / f"s1_arm/{k}/history.json")
            for k in ("force", "noforce", "mlp_force", "mlp_noforce")
        },
        "act_variants": {k: hist(p) for k, p in act_variants.items()},
        "arm_expert": {k: expert[k] for k in ("n", "n_success", "mean_steps", "mean_jams")},
        "sweep": load(O / "s1_phys/sweep.json"),
        "kp_sweep": load(O / "s1_phys/kp_sweep.json"),
        "labels": {
            "seed": 1000 + labels_seed,
            "delta_x": (1e3 * d).round(3).tolist(),
            "abs_x": (1e3 * a).round(3).tolist(),
        },
        "cue": CUE,
        "collapse": COLLAPSE,
        "videos": vids,
        "traces": {str(s): load(O / f"s1_arm/videos/seed{s}_traces.json") for s in seeds},
        "export": export,
        "n_tests": n_tests,
    }
    for ev in ("arm_tip", "arm_flange", "arm_xyabs"):  # episodes are not plotted
        for r in data[ev]["policies"].values():
            r.pop("episodes", None)
    (HERE / "media").mkdir(exist_ok=True)
    for s in seeds:
        shutil.copyfile(
            O / f"s1_arm/videos/seed{s}_compare.mp4", HERE / f"media/seed{s}_compare.mp4"
        )

    base = (REPO / "docs/report/template.html").read_text()
    style = base[base.index("<style>") : base.index("</style>") + len("</style>")]
    lib = base[base.index("const D = JSON.parse") : base.index("// ---------- facts")]
    sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    html = (HERE / "template.html").read_text()
    html = html.replace("__STYLE__", style).replace("__CHARTLIB__", lib)
    blob = json.dumps(_finite(data), separators=(",", ":"), allow_nan=False)
    html = html.replace("__DATA__", blob.replace("</", "<\\/"))
    html = html.replace("__DATE__", datetime.date.today().isoformat()).replace("__SHA__", sha)
    (HERE / "index.html").write_text(html)
    print(f"index.html {len(html) / 1e3:.0f} kB, {len(seeds)} videos, {n_tests} tests")


if __name__ == "__main__":
    main()
