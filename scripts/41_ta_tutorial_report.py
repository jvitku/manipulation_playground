#!/usr/bin/env python
"""Aggregate the tutorial runs (outputs/ta_tutorial/**) into docs/ta_tutorial/report_data.json
and build docs/ta_tutorial/index.html from docs/ta_tutorial/template.html.

Curves: per arm, the mean over seeds plus the per-seed min / max at every evaluation step.
Finals: per arm, per-seed values, mean, and a pooled Wilson 95 % interval over all episodes.
Runs that have not finished are skipped (re-run the script as results arrive).
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("outputs/ta_tutorial")
OUT = Path("docs/ta_tutorial")


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def _load(f: Path) -> dict:
    return json.loads(f.read_text()) if f.exists() else {}


def runs(exp: str):
    """{arm: [(seed, history, final, extras, config)]} for finished runs of one experiment."""
    out = defaultdict(list)
    for d in sorted((ROOT / exp).glob("*")):
        if not (d / "final.json").exists():
            continue
        m = re.match(r"(.*)_s(\d+)$", d.name)
        if not m:
            continue
        out[m.group(1)].append(
            (
                int(m.group(2)),
                *(
                    _load(d / f)
                    for f in ("history.json", "final.json", "extras.json", "config.json")
                ),
            )
        )
    return out


def curve(items, key, src="eval"):
    """Mean / min / max over seeds of eval[key] (src='eval') or a history list (src='hist')."""
    per = []
    for _, h, *_ in items:
        if src == "eval":
            per.append({e["step"]: e.get(key) for e in h.get("eval", [])})
        else:
            per.append({s: v for s, v in zip(h.get("step", []), h.get(key, []), strict=False)})
    steps = sorted(set.intersection(*(set(p) for p in per))) if per else []
    rows = []
    for s in steps:
        vals = [p[s] for p in per if p[s] is not None]
        if vals:
            rows.append(
                [s, round(sum(vals) / len(vals), 5), round(min(vals), 5), round(max(vals), 5)]
            )
    return rows


def final_stats(items, key="success", n_ep=None):
    vals = [f.get(key) for _, _, f, _, _ in items if f.get(key) is not None]
    r = {"seeds": len(vals), "values": [round(v, 4) for v in vals]}
    if vals:
        r["mean"] = round(sum(vals) / len(vals), 4)
        if n_ep:
            k = round(sum(vals) * n_ep)
            r["wilson95"] = wilson(k, n_ep * len(vals))
    return r


def outcome_mix(items):
    tot = defaultdict(int)
    for _, _, f, _, _ in items:
        for k, v in f.items():
            if k.startswith("n_"):
                tot[k[2:]] += v
    return dict(tot)


def main() -> None:
    data = {}
    # ---- T1
    t1 = runs("t1")
    data["t1"] = {
        arm: {
            "collision": curve(it, "collision"),
            "mode_err": curve(it, "mode_err"),
            "loss": curve(it, "loss", "hist"),
            "val_loss": curve(it, "val_loss", "hist"),
            "final": {k: final_stats(it, k) for k in ("collision", "mode_err", "left_frac")},
            "sweep": {s: v for s, v in it[0][3].get("sweep", {}).items()},
        }
        for arm, it in t1.items()
    }
    for arm in ("flow", "ddpm", "mse"):
        if arm in t1:
            ex = sorted(t1[arm])[0][3]
            data["t1"][arm]["samples"] = {k: v for k, v in ex.items() if k.startswith("samples_")}
            data["t1"][arm]["trace"] = ex.get("trace_c0")
            data["t1"][arm]["data_c0"] = ex.get("data_c0")
            if "field_c0" in ex:
                data["t1"][arm]["field"] = ex["field_c0"]
    # ---- T2
    t2 = runs("t2")
    data["t2"] = {}
    for arm, it in t2.items():
        e = {
            "success": curve(it, "success"),
            "collision": curve(it, "collision"),
            "loss": curve(it, "loss", "hist"),
            "val_loss": curve(it, "val_loss", "hist"),
            "final": final_stats(it, "success", 200),
            "final_collision": final_stats(it, "collision"),
            "cfg": {k: it[0][4].get(k) for k in ("gen", "horizon", "n_exec", "demos")},
        }
        sw = [x[3].get("sweep") for x in it if x[3].get("sweep")]
        if sw:
            e["sweep"] = {s: round(sum(w[s]["success"] for w in sw) / len(sw), 4) for s in sw[0]}
        ex = sorted(it)[0][3]
        e["paths"] = ex.get("paths", [])[:16]
        e["obstacle"] = ex.get("obstacle", [])[:16]
        e["goal"] = ex.get("goal", [])[:16]
        data["t2"][arm] = e
    # ---- T3 / T4
    for exp, n_ep in (("t3", 200), ("t4", 400)):
        data[exp] = {}
        for arm, it in runs(exp).items():
            e = {
                "success": curve(it, "success"),
                "loss": curve(it, "loss", "hist"),
                "val_loss": curve(it, "val_loss", "hist"),
                "aux": curve(it, "aux", "hist"),
                "final": final_stats(it, "success", n_ep),
                "outcomes": outcome_mix(it),
                "demo_episodes": it[0][3].get("demo_episodes"),
                "params": it[0][3].get("params"),
            }
            if exp == "t4":
                e["tasks"] = {t: final_stats(it, f"task{t}", n_ep // 4) for t in range(4)}
                e["task_curves"] = {t: curve(it, f"task{t}") for t in range(4)}
            ex = sorted(it)[0][3]
            e["traj"] = ex.get("traj", [])[:8]
            e["scene"] = ex.get("scene", {})
            data[exp][arm] = e
    # ---- T5
    data["t5"] = {}
    for arm, it in runs("t5").items():
        data["t5"][arm] = {
            "final": final_stats(it, "success", 50),
            "success": curve(it, "success"),
            "loss": curve(it, "loss", "hist"),
            "outcomes": outcome_mix(it),
            "demo_episodes": it[0][2].get("demo_episodes"),
        }
    m = Path("docs/stage2/matrix.json")
    if m.exists():
        rows = json.loads(m.read_text())
        bc = [r for r in rows["rows"] if r["label"].startswith("BC")]
        if bc:
            data["t5_bc"] = {"force": bc[0]["force"]["mean"], "noforce": bc[0]["noforce"]["mean"]}
    # ---- the real SmolVLA / ACT arms (V9)
    vla = {}
    for d in sorted(Path("outputs/eval_ta_insert").glob("*/final.json")):
        f = json.loads(d.read_text())
        vla[d.parent.name] = {"success": f["success"], "episodes": f["episodes"],
                              "wilson95": wilson(f["success"], f["episodes"])}  # fmt: skip
    data["vla"] = vla
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report_data.json").write_text(json.dumps(data, separators=(",", ":")))
    tpl = OUT / "template.html"
    if tpl.exists():
        html = tpl.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
        (OUT / "index.html").write_text(html)
    print({k: (list(v) if isinstance(v, dict) else v) for k, v in data.items() if k != "vla"}, vla)


if __name__ == "__main__":
    main()
