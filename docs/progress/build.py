"""Build docs/progress/index.html: the hourly progress summary (PLAN §12).

Sources: docs/progress/journal.jsonl (one entry per experiment / fix, with a verdict), the git
log since the previous summary, and live training progress of every journal entry that is still
running (its ``runs`` directory of 18_train_td3.py outputs). docs/progress/state.json remembers
when the previous summary was made. Run in the dev image: `python docs/progress/build.py`.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MILESTONES = [
    ("V0", "VLA image, policy server, SmolVLA memory probe"),
    ("V1", "M16 bolt / nut parts, fixture, bin, bucket"),
    ("V2", "Franka Hand force, tactile pads, joint-torque sensor"),
    ("V3", "SortBoltsNuts env"),
    ("V4", "Scripted expert + 200 synthetic demos"),
    ("V5", "ACT behaviour-cloning baseline"),
    ("V6", "TD3 skills (pick, insert)"),
    ("V7", "SmolVLA fine-tune"),
    ("V8", "Force / tactile features and detectors"),
    ("V9", "Force-aware ACT, TD3, TA-SmolVLA"),
    ("V10", "Evaluation matrix and report"),
]


def _finite(o):
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout


def run_status(runs_dir: str) -> list[dict]:
    """Latest state of every run in an 18_train_td3.py output directory."""
    out = []
    for d in sorted(glob.glob(str(REPO / runs_dir / "*_s*"))):
        d = Path(d)
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        prog = (
            json.loads((d / "progress.json").read_text()) if (d / "progress.json").exists() else {}
        )
        final = json.loads((d / "final.json").read_text()) if (d / "final.json").exists() else None
        evals = prog.get("evals", [])
        out.append(
            {
                "name": d.name,
                "use_force": cfg.get("use_force"),
                "seed": cfg.get("args", {}).get("seed"),
                "evals": [{"t": e["t"], "success": e["success_rate"]} for e in evals],
                "steps_done": evals[-1]["t"] if evals else 0,
                "steps_total": cfg.get("args", {}).get("steps"),
                "final": None
                if final is None
                else {
                    "success": final["success_rate"],
                    "n": len(final["episodes"]),
                    "reasons": final["reasons"],
                },
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--next", default="", help="what happens next (free text)")
    ap.add_argument("--dry", action="store_true", help="do not advance the summary state")
    ap.add_argument(
        "--now", default=None, help="local time YYYY-MM-DDTHH:MM (the container is UTC)"
    )
    args = ap.parse_args()
    now = (
        datetime.datetime.fromisoformat(args.now)
        if args.now
        else datetime.datetime.now().replace(second=0, microsecond=0)
    )
    state_p = HERE / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"n": 0, "last": None}
    last = state["last"] or "2026-09-29T09:00"
    journal = [
        json.loads(line) for line in (HERE / "journal.jsonl").read_text().splitlines() if line
    ]
    commits = [
        dict(zip(("sha", "ts", "subject"), line.split("\t", 2), strict=True))
        for line in git("log", f"--since={last}", "--format=%h%x09%cI%x09%s").splitlines()
        if line
    ]
    running = [
        {**e, "status": run_status(e["runs"])}
        for e in journal
        if e["verdict"] == "running" and e.get("runs")
    ]
    baseline = next((e for e in journal if e["id"] == "RL0c"), None)
    data = {
        "now": now.isoformat(timespec="minutes"),
        "last": last,
        "n": state["n"] + 1,
        "journal": journal,
        "commits": commits,
        "running": running,
        "baseline": run_status(baseline["runs"]) if baseline else [],
        "milestones": MILESTONES,
        "next": args.next,
    }
    base = (REPO / "docs/report/template.html").read_text()
    style = base[base.index("<style>") : base.index("</style>") + len("</style>")]
    lib = base[base.index("const D = JSON.parse") : base.index("// ---------- facts")]
    html = (HERE / "template.html").read_text()
    html = html.replace("__STYLE__", style).replace("__CHARTLIB__", lib)
    blob = json.dumps(_finite(data), separators=(",", ":"), allow_nan=False)
    html = html.replace("__DATA__", blob.replace("</", "<\\/"))
    (HERE / "index.html").write_text(html)
    if not args.dry:
        state_p.write_text(json.dumps({"n": data["n"], "last": data["now"]}, indent=2))
    print(
        f"summary #{data['n']}: {len(journal)} journal entries, {len(commits)} commits since {last}"
    )


if __name__ == "__main__":
    main()
