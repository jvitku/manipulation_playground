"""A stand-in for nfv.lora_worker (lesson 20) that needs no /opt/vla, GPU or SmolVLA.

``python -m tests.lab.fake_lora_worker <train|eval|render|probe> ...`` writes files with the
real worker's names and shapes (see src/nfv/lora_worker.py), so tests/lab/test_task_nfv5.py can
run lab/tasks/task_nfv5.py's train() and evaluate() end to end on the CPU in seconds. Numbers
are fixed per mode: success head 20 %, lora 50 %, expert_full 55 % (deterministic per seed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SUCCESS = {"head": 0.2, "lora": 0.5, "expert_full": 0.55}
TRAINABLE = {"head": 1_635_152, "lora": 4_563_792, "expert_full": 99_880_992}
VRAM = {"head": 3.21, "lora": 3.61, "expert_full": 4.74}
LAYERS = ("model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj",
          "model.vlm_with_expert.lm_expert.layers.7.self_attn.v_proj")  # fmt: skip


def _dump(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _spectrum(r: int, low_rank: bool) -> dict:
    sv = [1.0 / (i + 1) for i in range(r)] + ([0.0] * 8 if low_rank else
                                               [0.05] * 300)  # fmt: skip
    e = np.square(sv)
    return {"shape": [960, 720], "sv": sv[:64], "n_sv": len(sv), "fro2": float(e.sum()),
            "energy_top_r": float(e[:r].sum() / e.sum()),
            "energy_top_16": float(e[:16].sum() / e.sum())}  # fmt: skip


def train(a) -> None:
    out = Path(a.out)
    with _mk(out) as f:
        for s in range(a.log_every, a.steps + 1, a.log_every):
            f.write(json.dumps({"step": s, "loss": 1.0 / s, "lr": 1e-4, "steps_per_s": 3.0,
                                "vram_gb": VRAM[a.mode]}) + "\n")  # fmt: skip
    _dump(out / "progress.json", {"step": a.steps, "message": f"{a.mode}: done"})
    (out / "adapter.pt").write_bytes(b"x" * 1000)
    total = 452_974_816 if a.mode == "lora" else 450_046_176
    res = {
        "mode": a.mode, "task": a.task, "steps": a.steps, "batch": a.batch,
        "lr": a.lr or 1e-4, "warmup": 1, "seed": a.seed, "demos": a.demos, "load_s": 1.0,
        "train_s": 1.0, "steps_per_s": 3.0, "peak_alloc_gb": VRAM[a.mode],
        "peak_reserved_gb": VRAM[a.mode] + 0.1,
        "trainable": {"total": total, "trainable": TRAINABLE[a.mode],
                      "trainable_frac": TRAINABLE[a.mode] / total, "groups": {}},
        "lora": ({"targets": ["model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj"],
                  "r": a.r, "alpha": a.alpha, "scale": a.alpha / a.r, "dropout": 0.0,
                  "n_layers": 128, "lora_params": 2_928_640} if a.mode == "lora" else None),
        "head_params": ["model.state_proj.weight"], "adapter_bytes": 1000,
        "adapter_tensor_bytes": 1000, "full_ckpt_bytes": 906_639_456, "final_loss": 0.4,
        "first_loss": 1.2,
        "spectrum": ({n: _spectrum(a.r, True) for n in LAYERS} if a.mode == "lora" else
                     {n: _spectrum(a.r, False) for n in LAYERS} if a.mode == "expert_full"
                     else {}),
        "bf16_lost_update_frac": 0.7 if a.mode == "expert_full" else None, "stopped": False,
    }  # fmt: skip
    if a.mode == "lora":
        res["merge_check"] = {"n_layers_merged": 128, "batch": 4, "fp32_max_abs": 2e-6,
                              "fp32_out_abs_mean": 0.74, "bf16_max_abs": 0.04,
                              "bf16_out_abs_mean": 0.74}  # fmt: skip
    _dump(out / "result.json", res)


def _mk(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    return (out / "metrics.jsonl").open("a")


def _mode(adapter: str) -> str:
    return Path(adapter).parent.name.split("_", 1)[-1]


def evaluate(a) -> None:
    from nfv.seeds import eval_seeds

    mode = _mode(a.adapter)
    seeds = eval_seeds(a.task, a.n, a.offset)
    k = int(round(SUCCESS[mode] * len(seeds)))
    eps = []
    for i, s in enumerate(seeds):
        ok = i < k
        e = {"seed": s, "outcome": "success" if ok else "timeout", "success": ok,
             "peak_force": 2.0 + i % 3, "instruction": "insert the peg", "steps": 40}  # fmt: skip
        if i < a.traces:
            e["trace"] = {"t": [0.1, 0.2], "fz": [0.0, -1.0], "mode": ["position"] * 2,
                          "subtask": ["approach", "descend"], "events": []}  # fmt: skip
        eps.append(e)
    _dump(Path(a.out), {
        "mode": mode, "task": a.task, "n": len(eps), "success": k / len(eps),
        "wilson95": [max(0.0, k / len(eps) - 0.1), min(1.0, k / len(eps) + 0.1)],
        "outcomes": {"success": k, "timeout": len(eps) - k}, "peak_force_mean": 3.0,
        "peak_force_max": 4.0, "chunk_ms": 120.0, "chunk_calls": 10, "wall_s": 1.0,
        "infer_vram_gb": 0.9, "episodes": eps,
    })  # fmt: skip


def render(a) -> None:
    from lab.core.video import frames_to_mp4

    frames_to_mp4((np.full((32, 32, 3), 40 * i, np.uint8) for i in range(3)), a.out, fps=10)
    _dump(Path(a.out).with_suffix(".json"), {"seed": a.seed})


def probe(a) -> None:
    _dump(Path(a.out), {"mode": "full", "batch": a.batch, "steps": a.steps,
                        "precision": a.precision, "fits": False, "error": "CUDA out of memory",
                        "peak_alloc_gb": 8.96, "peak_reserved_gb": 9.0})  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval", "render", "probe"):
        p = sub.add_parser(name)
        for flag in ("--task", "--demo-root", "--mode", "--adapter", "--label", "--title",
                     "--out", "--precision"):  # fmt: skip
            p.add_argument(flag, default="")
        for flag in ("--demos", "--seed", "--steps", "--batch", "--r", "--log-every", "--n",
                     "--offset", "--traces", "--threads"):  # fmt: skip
            p.add_argument(flag, type=int, default=0)
        for flag in ("--lr", "--alpha"):
            p.add_argument(flag, type=float, default=0.0)
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "render": render, "probe": probe}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
