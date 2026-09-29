#!/usr/bin/env python
"""PLAN §13 V0 (VLA image): does fine-tuning SmolVLA fit on this GPU, and how fast is inference?

Loads ``lerobot/smolvla_base``, builds random batches that match its pretrained input features
(through LeRobot's own pre-processor: tokenisation + normalisation), then for increasing batch
sizes runs real optimiser steps (frozen VLM, AdamW on the trainable parameters, bf16 autocast)
and records peak GPU memory and step time. Also times one action-chunk prediction.
Writes --out/probe.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="lerobot/smolvla_base")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=256, help="rendered camera size")
    ap.add_argument("--out", default="outputs/v0_smolvla_probe")
    args = ap.parse_args()

    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(args.base).to(dev)
    load_s = time.perf_counter() - t0
    cfg = policy.config
    pre, _post = make_pre_post_processors(cfg, pretrained_path=args.base)
    feats = {k: (str(v.type), list(v.shape)) for k, v in cfg.input_features.items()}
    n_params = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    res = {
        "base": args.base,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_GB": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "load_s": load_s,
        "params_M": n_params / 1e6,
        "trainable_M": n_train / 1e6,
        "input_features": feats,
        "chunk_size": cfg.chunk_size,
        "action_dim": list(cfg.output_features["action"].shape),
        "train": [],
    }
    print(json.dumps({k: v for k, v in res.items() if k != "train"}, indent=1), flush=True)

    def raw_batch(b: int) -> dict:
        batch = {}
        for k, v in cfg.input_features.items():
            shape = list(v.shape)
            if "image" in k:  # float images in [0, 1], rendered size (the policy pads to 512)
                batch[k] = torch.rand(b, 3, args.image_size, args.image_size)
            else:
                batch[k] = torch.randn(b, *shape)
        a = cfg.output_features["action"].shape
        batch["action"] = torch.randn(b, cfg.chunk_size, *a)
        batch["action_is_pad"] = torch.zeros(b, cfg.chunk_size, dtype=torch.bool)
        batch["task"] = ["sort the parts: bolts into the hole, nuts into bucket A"] * b
        return batch

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    policy.train()
    for b in args.batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            times = []
            for _ in range(args.steps):
                batch = pre(raw_batch(b))
                torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = policy.forward(batch)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
            row = {
                "batch": b,
                "ok": True,
                "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
                "step_s": min(times),
                "loss": float(loss),
            }
        except torch.OutOfMemoryError:
            row = {"batch": b, "ok": False, "error": "OOM"}
            opt.zero_grad(set_to_none=True)
        print(json.dumps(row), flush=True)
        res["train"].append(row)
        if not row["ok"]:
            break

    policy.eval()
    policy.reset()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        batch = pre(raw_batch(1))
        for _ in range(2):  # warm-up
            policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        t = time.perf_counter()
        n = 5
        for _ in range(n):
            chunk = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
    res["infer_chunk_s"] = (time.perf_counter() - t) / n
    res["chunk_shape"] = list(chunk.shape)
    print(json.dumps({"infer_chunk_s": res["infer_chunk_s"], "chunk_shape": res["chunk_shape"]}))
    (out / "probe.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out / 'probe.json'}")


if __name__ == "__main__":
    main()
