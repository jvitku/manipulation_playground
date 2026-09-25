"""Build docs/td3/index.html (Peg-in-Hole TD3 Report) from outputs/s1_td3/.

Needs the TD3 runs (scripts/18_train_td3.py; outputs/s1_td3/{force,noforce}_s<k>/) and, for the
videos, `13_policy_videos.py --task arm` on the TD3 checkpoints (outputs/s1_td3/videos). Run
inside the dev image: `python docs/td3/build.py`. Shares style and charts with docs/report.
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
O = REPO / "outputs"


def _finite(o):
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def rolling(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def thin(*cols, n: int = 500):
    idx = np.unique(np.linspace(0, len(cols[0]) - 1, min(n, len(cols[0]))).astype(int))
    return [np.asarray(c)[idx].round(4).tolist() for c in cols]


def run_data(d: Path) -> dict:
    prog = json.loads((d / "progress.json").read_text())
    eps = prog["episodes"]
    t = np.array([e["t"] for e in eps])
    ret = np.array([e["return"] for e in eps])
    reason = [e["reason"] for e in eps]
    w = 100
    frac = {k: rolling([r == k for r in reason], w) for k in ("success", "force_abort", "timeout")}
    tt, rr, ss, aa, oo, ll, pf = thin(
        t,
        rolling(ret, w),
        frac["success"],
        frac["force_abort"],
        frac["timeout"],
        rolling([e["len"] for e in eps], w),
        rolling([e["peak_F_N"] for e in eps], w),
    )
    losses = prog["losses"]
    lt = [x["t"] for x in losses]
    fp = d / "final.json"
    final = json.loads(fp.read_text()) if fp.exists() else None  # None: still training
    return {
        "name": d.name,
        "use_force": prog["meta"]["use_force"],
        "seed": prog["meta"]["args"]["seed"],
        "episodes": len(eps),
        "train": {
            "t": tt,
            "return": rr,
            "success": ss,
            "abort": aa,
            "timeout": oo,
            "len": ll,
            "peak_F": pf,
        },
        "loss": {
            "t": lt,
            "critic": [x.get("critic_loss") for x in losses],
            "actor": [x.get("actor_loss") for x in losses],
            "q": [x.get("q1_mean") for x in losses],
        },
        "evals": prog["evals"],
        "final": final,
    }


def main() -> None:
    runs = [
        run_data(Path(d))
        for d in sorted(glob.glob(str(O / "s1_td3/*_s*")))
        if (Path(d) / "progress.json").exists()
    ]
    cfg = json.loads((O / "s1_td3" / runs[0]["name"] / "config.json").read_text())
    bc = json.loads((O / "s1_arm/eval/eval.json").read_text())["policies"]
    bc = {k: {kk: v[kk] for kk in ("n", "success_rate", "mean_peak_F_N")} for k, v in bc.items()}
    data = {"runs": runs, "config": cfg, "bc": bc}
    vids = O / "s1_td3/videos/summary.json"
    if vids.exists():
        v = json.loads(vids.read_text())
        seeds = sorted({r["seed"] for r in v["runs"]})
        data["videos"] = v
        data["traces"] = {
            str(s): json.loads((O / f"s1_td3/videos/seed{s}_traces.json").read_text())
            for s in seeds
        }
        (HERE / "media").mkdir(exist_ok=True)
        for s in seeds:
            shutil.copyfile(
                O / f"s1_td3/videos/seed{s}_compare.mp4", HERE / f"media/seed{s}_compare.mp4"
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
    print(f"index.html {len(html) / 1e3:.0f} kB, {len(runs)} runs")


if __name__ == "__main__":
    main()
