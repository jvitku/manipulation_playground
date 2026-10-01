#!/usr/bin/env python
"""V10: Stage 2 evaluation matrix with 95 % intervals (docs/stage2/matrix.{json,md}).

Every cell: per-seed success rates -> mean with a seed-level bootstrap 95 % interval, plus the
pooled episode-level Wilson 95 % interval; each force / no-force pair gets Welch's t and
Mann-Whitney U. Sources are the ``final.json`` files the training / evaluation scripts write.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROWS = [
    # (row label, metric, output dir pattern, seeds, episodes per seed)
    ("TD3+BC insert, centred grasps", "insert success", "outputs/td3_sort/{c}_s{s}", 10, 50),
    (
        "TD3+BC insert, randomised in-hand offsets",
        "insert success",
        "outputs/td3_sort_grand/{c}_s{s}",
        6,
        50,
    ),
    ("TD3+BC pick (held + lifted)", "pick success", "outputs/td3_pick/{c}_s{s}", 10, 50),
    ("TD3+BC pick (centred grasp)", "pick success", "outputs/td3_pick_centred/{c}_s{s}", 6, 50),
    ("BC (MLP) insert, randomised offsets", "insert success", "outputs/bc_insert/{c}_s{s}", 5, 50),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def boot(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main() -> None:
    out: dict = {"rows": []}
    md = [
        "| skill / learner | metric | with force | without force | Welch p | MW p |",
        "|---|---|---|---|---|---|",
    ]
    for label, metric, pat, n_seeds, n_ep in ROWS:
        cells = {}
        for c in ("force", "noforce"):
            rates = []
            for s in range(n_seeds):
                f = Path(pat.format(c=c, s=s)) / "final.json"
                if f.exists():
                    rates.append(json.loads(f.read_text())["success_rate"])
            r = np.array(rates)
            k = int(round(float(r.sum()) * n_ep))
            cells[c] = {
                "seeds": len(r),
                "rates": r.round(3).tolist(),
                "mean": float(r.mean()) if len(r) else float("nan"),
                "seed_boot95": boot(r) if len(r) > 1 else None,
                "episodes": int(len(r) * n_ep),
                "successes": k,
                "wilson95": wilson(k, int(len(r) * n_ep)),
            }
        a, b = np.array(cells["force"]["rates"]), np.array(cells["noforce"]["rates"])
        welch = float(stats.ttest_ind(a, b, equal_var=False).pvalue) if len(a) > 1 else None
        mw = float(stats.mannwhitneyu(a, b).pvalue) if len(a) > 1 else None
        out["rows"].append(
            {"label": label, "metric": metric, **cells, "welch_p": welch, "mannwhitney_p": mw}
        )

        def fmt(x):
            lo, hi = x["seed_boot95"] or (float("nan"), float("nan"))
            return f"{100 * x['mean']:.1f} % [{100 * lo:.1f}, {100 * hi:.1f}] (n={x['seeds']})"

        md.append(
            f"| {label} | {metric} | {fmt(cells['force'])} | {fmt(cells['noforce'])} | "
            f"{welch:.2g} | {mw:.2g} |"
        )
    seq = []
    for name, d in (
        ("sequencer, unconstrained picks, force", "outputs/seq_force_noisy"),
        (
            "sequencer, unconstrained pick + offset-robust insert, no force",
            "outputs/seq_noforce_grand",
        ),
        ("sequencer, unconstrained pick + offset-robust insert, force", "outputs/seq_force_grand"),
        ("sequencer, centred pick + offset-robust insert, force", "outputs/seq_force_centred"),
        ("sequencer, centred pick + offset-robust insert, no force", "outputs/seq_noforce_centred"),
    ):
        f = Path(d) / "summary.json"
        if f.exists():
            s = json.loads(f.read_text())
            seq.append(
                {
                    "name": name,
                    **s,
                    "episode_wilson95": wilson(s["success"], s["episodes"]),
                    "insert_wilson95": wilson(s["insert"][1], s["insert"][0]),
                }
            )
    out["sequencer"] = seq
    md += [
        "",
        "| full sort (10 episodes each) | episodes | parts | pick | insert after pick |",
        "|---|---|---|---|---|",
    ]
    for s in seq:
        lo, hi = s["insert_wilson95"]
        md.append(
            f"| {s['name']} | {s['success']}/{s['episodes']} | {s['parts']}/60 | "
            f"{s['pick'][1]}/{s['pick'][0]} | {s['insert'][1]}/{s['insert'][0]} "
            f"[{100 * lo:.0f}-{100 * hi:.0f} %] |"
        )
    d = Path("docs/stage2")
    d.mkdir(parents=True, exist_ok=True)
    (d / "matrix.json").write_text(json.dumps(out, indent=2))
    (d / "matrix.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
