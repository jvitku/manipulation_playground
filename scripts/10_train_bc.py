#!/usr/bin/env python
"""Stage 1: train ACT-lite / MLP behaviour cloning on expert episodes, with or without force.

Writes best.pt (by validation L1), last.pt, norm.json, history.json and a loss plot to --out.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fvb.policy.data import compute_norm, load_episodes, make_windows, split_episodes
from fvb.policy.spec import spec_of_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/expert_trackA")
    ap.add_argument("--out", default="outputs/s1/force")
    ap.add_argument("--force", type=int, default=1, help="1: use force channels, 0: zero them")
    ap.add_argument("--model", choices=["act", "mlp"], default="act")
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch

    from fvb.policy.models import build_model, n_params

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    spec = spec_of_dataset(args.data)
    data_cfg_path = Path(args.data) / "config.json"
    data_cfg = json.loads(data_cfg_path.read_text()) if data_cfg_path.exists() else {}
    action_mode = data_cfg.get("task", {}).get("action_mode", "delta")
    fi = spec.force_idx
    eps = load_episodes(args.data, spec)
    train_eps, val_eps = split_episodes(eps, args.val_frac, args.seed)
    use_force = bool(args.force)
    norm = compute_norm(train_eps, use_force, fi)
    Xtr, Ytr, Mtr = make_windows(train_eps, args.H, args.K, use_force, norm, fi)
    Xva, Yva, Mva = make_windows(val_eps, args.H, args.K, use_force, norm, fi)
    print(
        f"device={device} episodes train/val={len(train_eps)}/{len(val_eps)} "
        f"windows train/val={len(Xtr)}/{len(Xva)} use_force={use_force} task={spec.name}"
    )

    model = build_model(args.model, spec.obs_dim, spec.act_dim, args.H, args.K).to(device)
    print(f"model={args.model} params={n_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = int(np.ceil(len(Xtr) / args.batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * steps_per_epoch)
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    Xtr_t, Ytr_t, Mtr_t = to(Xtr), to(Ytr), to(Mtr)
    Xva_t, Yva_t, Mva_t = to(Xva), to(Yva), to(Mva)

    def masked_l1(pred, y, m):
        return ((pred - y).abs().mean(-1) * m).sum() / m.sum()

    meta = {
        "model": args.model,
        "H": args.H,
        "K": args.K,
        "use_force": use_force,
        "task": spec.name,
        "action_mode": action_mode,
        "obs_dim": spec.obs_dim,
        "act_dim": spec.act_dim,
        "data": args.data,
        "seed": args.seed,
    }
    hist, best = [], float("inf")
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g).to(device)
        tr_loss = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch : (i + 1) * args.batch]
            loss = masked_l1(model(Xtr_t[idx]), Ytr_t[idx], Mtr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tr_loss += loss.item()
        model.eval()
        with torch.no_grad():
            va_loss = masked_l1(model(Xva_t), Yva_t, Mva_t).item()
            # per-channel val L1 on de-normalised actions (mm) for the first predicted step
            pred0 = model(Xva_t)[:, 0].cpu().numpy() * norm.act_std + norm.act_mean
            true0 = Yva[:, 0] * norm.act_std + norm.act_mean
            mae_mm = (1e3 * np.abs(pred0 - true0)).mean(0).tolist()
        hist.append(
            {
                "epoch": ep,
                "train_l1": tr_loss / steps_per_epoch,
                "val_l1": va_loss,
                "val_mae_mm_xyz": mae_mm,
                "lr": sched.get_last_lr()[0],
            }
        )
        print(json.dumps(hist[-1]))
        ck = {
            "state_dict": model.state_dict(),
            "norm": norm.to_json(),
            "meta": meta,
            "epoch": ep,
            "val_l1": va_loss,
        }
        torch.save(ck, out / "last.pt")
        if va_loss < best:
            best = va_loss
            torch.save(ck, out / "best.pt")
    hist_out = {
        "history": hist,
        "best_val_l1": best,
        "wall_s": time.perf_counter() - t0,
        "params": n_params(model),
        "n_windows_train": int(len(Xtr)),
        "device": device,
    }
    (out / "history.json").write_text(json.dumps(hist_out, indent=2))
    (out / "norm.json").write_text(json.dumps(norm.to_json(), indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot([h["train_l1"] for h in hist], label="train L1 (normalised)")
    ax.plot([h["val_l1"] for h in hist], label="val L1 (normalised)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked L1 on normalised action chunks")
    ax.set_title(f"{args.model}, force={use_force}: {n_params(model):,} params, {len(Xtr)} windows")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=130)
    print(f"best val L1 {best:.4f}; wrote {out}")


if __name__ == "__main__":
    main()
