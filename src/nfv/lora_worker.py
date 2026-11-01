"""Lesson 20's SmolVLA worker: one fine-tuning arm, one evaluation or one probe per process.

Runs in the lab image's venv (``/opt/vla/bin/python -m nfv.lora_worker ...``, cwd = repo root,
PYTHONPATH=src, HF_HOME=/workspace/outputs/hf_cache, HF_HUB_OFFLINE=1, MUJOCO_GL=osmesa); the
lab job (lab/tasks/task_nfv5.py, the lab's own python) spawns it - the house pattern of
nfv.rl.worker: file-based context, a ``STOP`` file is the stop button, metrics.jsonl rows are
re-logged by the parent with the arm's prefix.

    train   --mode head|lora|expert_full --task t3 --steps N --batch B --out DIR
            DIR/metrics.jsonl  {"step", "loss", "lr", "steps_per_s", "vram_gb"} rows
            DIR/progress.json  latest status
            DIR/adapter.pt     {"tensors": ONLY the trainable tensors, "config": {...}}
            DIR/result.json    trainable counts, peak VRAM, steps/s, adapter vs full checkpoint
                               bytes, the update spectrum of SPECTRUM_LAYERS, the merge check
                               (lora) and the bf16 lost-update fraction (expert_full)
    eval    --adapter DIR/adapter.pt --task t3 --n 100 --offset K --out FILE.json
            closed loop on nfv.seeds.eval_seeds(task, n, offset): per-episode outcome, peak
            force, Wilson 95 % CI, chunk-call latency, <= --traces traces (fvrollout schema)
    render  --adapter ... --seed S --out FILE.mp4 --label TEXT     one episode on video
    probe   --mode full --batch B --steps 3 --precision fp32|bf16 --out FILE.json
            does training EVERY parameter fit? peak VRAM, or the OOM

GPU memory is reported as torch.cuda.max_memory_allocated (tensors) and max_memory_reserved
(the caching allocator's pool) - the CUDA context (~0.3-0.5 GB) comes on top of both.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nfv import lora as L  # noqa: E402
from nfv import vla_smolvla as V  # noqa: E402

GB = 1024**3
FPS, VW, VH, SH = 10, 320, 240, 70  # the course's video layout (task_nfv0)


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def wilson95(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


class FileCtx:
    def __init__(self, out: Path):
        self.dir = Path(out)
        self.dir.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, **series) -> None:
        with (self.dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": int(step), **series}) + "\n")

    def status(self, **kw) -> None:
        _dump(self.dir / "progress.json", {**kw, "time": time.time()})

    def stopped(self) -> bool:
        return (self.dir / "STOP").exists()


def _demo_view(root: str, task: str, n: int):
    from nfv.data.dataset import DemoDataset

    ds = DemoDataset(root, task)
    n = max(1, min(int(n), len(ds.episodes)))
    ds.episodes = ds.episodes[:n]
    ds.index = {**ds.index, "n": n, "episodes": ds.index["episodes"][:n]}
    return ds


def _vram() -> dict:
    return {"peak_alloc_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 3)}  # fmt: skip


# ------------------------------------------------------------------------------------- train
def cmd_train(a) -> int:
    ctx = FileCtx(Path(a.out))
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True  # training speed; the merge check turns it off
    ctx.status(step=0, message=f"{a.mode}: loading the {a.task} demos")
    ds = _demo_view(a.demo_root, a.task, a.demos)
    stats = V.norm_stats(ds)
    sampler = V.Sampler(ds, chunk=V.CHUNK)
    ctx.status(step=0, message=f"{a.mode}: loading smolvla_base (offline)")
    t0 = time.perf_counter()
    policy = V.load_policy(dev)
    load_s = time.perf_counter() - t0
    w_before = {n: policy.get_submodule(n).weight.detach().float().cpu().clone()
                for n in V.SPECTRUM_LAYERS}  # fmt: skip
    rep = V.setup_mode(policy, a.mode, r=a.r, alpha=a.alpha, dropout=a.dropout)
    codec = V.Codec(policy, stats, dev)
    lr = a.lr if a.lr > 0 else V.MODE_LR[a.mode]
    warmup = a.warmup if a.warmup > 0 else max(1, min(1000, a.steps // 10))
    opt, params = V.make_optimizer(policy, lr)
    policy.train()
    torch.cuda.reset_peak_memory_stats()
    ctx.log(0, trainable_params=rep["trainable"]["trainable"],
            trainable_frac=rep["trainable"]["trainable_frac"])  # fmt: skip
    losses: list[float] = []
    win: list[float] = []
    t_run = time.perf_counter()
    t_win, k_win = time.perf_counter(), 0
    sps_hist: list[float] = []
    bf16_lost = None
    step = 0
    for step in range(a.steps):
        if a.mode == "expert_full" and step == warmup and bf16_lost is None:
            snap = {n: p.detach().clone() for n, p in policy.named_parameters()
                    if p.requires_grad and ".lm_expert." in n}  # fmt: skip
        else:
            snap = None
        batch = V.train_batch(codec, sampler.sample(a.batch, rng))
        loss = V.train_step(policy, opt, params, batch, V.lr_at(step, a.steps, lr, warmup))
        if not math.isfinite(loss):
            raise FloatingPointError(f"loss is {loss} at step {step}")
        if snap is not None:  # one peak-LR step: how much of it would bf16 storage lose?
            bf16_lost = V.bf16_lost_fraction(snap, policy)
            del snap
        losses.append(loss)
        win.append(loss)
        k_win += 1
        last = step == a.steps - 1
        if (step + 1) % a.log_every == 0 or last:
            torch.cuda.synchronize()
            now = time.perf_counter()
            sps = k_win / max(1e-9, now - t_win)
            if step + 1 > a.log_every:  # the first window holds the CUDA warm-up
                sps_hist.append(sps)
            ctx.log(step + 1, loss=float(np.mean(win)), lr=V.lr_at(step, a.steps, lr, warmup),
                    steps_per_s=round(sps, 3),
                    vram_gb=round(torch.cuda.max_memory_allocated() / GB, 3))  # fmt: skip
            ctx.status(step=step + 1, total=a.steps, loss=float(np.mean(win)),
                       message=f"{a.mode}: step {step + 1}/{a.steps}")  # fmt: skip
            win, k_win, t_win = [], 0, now
            if ctx.stopped():
                break
    train_s = time.perf_counter() - t_run
    steps_done = step + 1
    vram = _vram()
    opt.zero_grad(set_to_none=True)
    del opt
    torch.cuda.empty_cache()

    # ---- the adapter file: ONLY the trainable tensors + what is needed to rebuild ----
    tensors = L.trainable_state_dict(policy)
    config = {
        "mode": a.mode, "task": a.task, "r": a.r, "alpha": a.alpha, "dropout": a.dropout,
        "lora_targets": list(V.LORA_TARGETS) if a.mode == "lora" else [],
        "head_patterns": list(V.HEAD_PATTERNS), "stats": stats, "chunk": V.CHUNK,
        "n_exec": V.N_EXEC, "steps": steps_done, "batch": a.batch, "lr": lr,
        "warmup": warmup, "seed": a.seed, "demos": len(ds.episodes),
    }  # fmt: skip
    adapter = Path(a.out) / "adapter.pt"
    torch.save({"tensors": tensors, "config": config}, adapter.with_suffix(".pt.tmp"))
    os.replace(adapter.with_suffix(".pt.tmp"), adapter)

    # ---- the update spectrum of the representative expert layers ----
    spec = {}
    if a.mode == "lora":
        spec = V.lora_spectra(policy)
    elif a.mode == "expert_full":
        for n in V.SPECTRUM_LAYERS:
            dw = policy.get_submodule(n).weight.detach().float().cpu() - w_before[n]
            spec[n] = V.spectrum(dw, a.r)
    result = {
        "mode": a.mode, "task": a.task, "steps": steps_done, "batch": a.batch, "lr": lr,
        "warmup": warmup, "seed": a.seed, "demos": len(ds.episodes),
        "load_s": round(load_s, 2), "train_s": round(train_s, 2),
        "steps_per_s": round(float(np.median(sps_hist)) if sps_hist
                             else steps_done / max(1e-9, train_s), 3),
        **vram, "trainable": rep["trainable"],
        "lora": ({k: v for k, v in rep["lora"].items() if k != "per_layer"}
                 if rep["lora"] else None),
        "head_params": rep["head"],
        "adapter_bytes": adapter.stat().st_size,
        "adapter_tensor_bytes": L.state_bytes(tensors.values()),
        "full_ckpt_bytes": V.full_ckpt_bytes(policy),
        "final_loss": float(np.mean(losses[-max(1, a.log_every):])),
        "first_loss": float(np.mean(losses[: max(1, a.log_every)])),
        "spectrum": spec, "bf16_lost_update_frac": bf16_lost,
        "stopped": steps_done < a.steps,
    }  # fmt: skip
    if a.mode == "lora" and not a.no_merge_check:
        ctx.status(step=steps_done, message="lora: merge check")
        result["merge_check"] = merge_check(policy, codec, sampler, a.seed)
    _dump(Path(a.out) / "result.json", result)
    return 0


@torch.no_grad()
def merge_check(policy, codec, sampler, seed: int, b: int = 4) -> dict:
    """Adapter forward vs merged weights on a REAL batch with fixed noise and flow time.

    bf16: the model as trained (frozen weights bf16, adapters fp32, autocast) vs the adapters
    merged INTO the bf16 weights - the rounding a bf16 deployment would pay.
    fp32: the whole model upcast to fp32, TF32 off - merged vs unmerged: the claim (m5)."""
    policy.eval()
    g = torch.Generator(device=codec.device).manual_seed(seed + 12345)
    batch = V.train_batch(codec, sampler.sample(b, np.random.default_rng(seed + 12345)))
    noise = torch.randn((b, V.CHUNK, policy.config.max_action_dim), generator=g,
                        device=codec.device)  # fmt: skip
    t = torch.rand((b,), generator=g, device=codec.device) * 0.998 + 0.001
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v_a16 = V.velocity(policy, batch, noise, t)
    twin = copy.deepcopy(policy)
    n_merged = L.merge_lora(twin)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v_m16 = V.velocity(twin, batch, noise, t)
    del twin
    torch.cuda.empty_cache()
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        policy.float()
        v_a32 = V.velocity(policy, batch, noise, t)
        L.merge_lora(policy)
        v_m32 = V.velocity(policy, batch, noise, t)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    return {
        "n_layers_merged": n_merged, "batch": b,
        "fp32_max_abs": float((v_a32 - v_m32).abs().max()),
        "fp32_out_abs_mean": float(v_a32.abs().mean()),
        "bf16_max_abs": float((v_a16 - v_m16).abs().max()),
        "bf16_out_abs_mean": float(v_a16.abs().mean()),
    }  # fmt: skip


# ----------------------------------------------------------------------------- eval / render
def _load_arm(adapter: str, dev: str = "cuda"):
    ck = torch.load(adapter, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    policy = V.load_policy(dev, chunk=int(cfg["chunk"]), n_exec=int(cfg["n_exec"]))
    V.setup_mode(policy, cfg["mode"], r=int(cfg["r"]), alpha=float(cfg["alpha"]),
                 dropout=float(cfg.get("dropout", 0.0)))  # fmt: skip
    have = dict(policy.named_parameters())
    missing = [k for k in ck["tensors"] if k not in have]
    if missing:
        raise KeyError(f"adapter tensors not in the rebuilt model: {missing[:5]}")
    with torch.no_grad():
        for k, v in ck["tensors"].items():
            have[k].copy_(v.to(have[k].device, have[k].dtype))
    return policy.eval(), cfg


def _episode(env, actor, seed: int, trace: bool = False, frames=None, label: str = ""):
    """One closed-loop episode; ``frames`` (a list) collects video frames when given."""
    from nfv.sim.env import Action

    obs = env.reset(seed)
    instr = env.instruction
    gen = torch.Generator(device=actor.codec.device).manual_seed(int(seed))
    tmax = int(round(env.task.TIME_LIMIT_S * FPS))
    tr = {"t": [], "fz": [], "mode": [], "subtask": [], "events": []}
    fz_v: list[float] = []
    rend = None
    if frames is not None:
        import mujoco

        rend = mujoco.Renderer(env.model, height=VH, width=VW)

    def frame(status=""):
        rend.update_scene(env.data, camera="static")
        return _frame(rend.render().copy(), fz_v, tmax, f"{label}  t {env.t:4.1f} s  "
                      f"Fz {float(env.wrench[2]):+6.2f} N", status)  # fmt: skip

    info = {"done": False, "outcome": "timeout", "success": False}
    k, aborted = 0, False
    try:
        if rend is not None:
            frames.append(frame())
        while not info["done"] and k < tmax:
            chunk = actor.act(obs, instr, gen)  # [16, 4]
            for j in range(min(actor.n_exec, tmax - k)):
                obs, info = env.step(Action(dp=chunk[j, :3], gripper=float(chunk[j, 3])))
                k += 1
                if trace:
                    tr["t"].append(round(float(info["t"]), 2))
                    tr["fz"].append(round(float(obs.wrench[2]), 3))
                    tr["mode"].append("position")
                    tr["subtask"].append(env.task.SUBTASKS[obs.subtask_id])
                    if info["force_abort"] and not aborted:
                        tr["events"].append({"t": tr["t"][-1], "label": "abort"})
                    if info.get("outcome") == "success":
                        tr["events"].append({"t": tr["t"][-1], "label": "success"})
                aborted = aborted or bool(info["force_abort"])
                fz_v.append(float(obs.wrench[2]))
                if rend is not None:
                    frames.append(frame())
                if info["done"]:
                    break
        rec = {"seed": int(seed), "outcome": info["outcome"] or "timeout",
               "success": bool(info["success"]), "peak_force": round(float(env.peak_force), 3),
               "instruction": instr, "steps": k}  # fmt: skip
        if rend is not None:
            last = frame(rec["outcome"])
            frames.extend([last] * FPS)
        if trace:
            rec["trace"] = tr
        return rec
    finally:
        if rend is not None:
            rend.close()


def _frame(img: np.ndarray, fz: list, tmax: int, text: str, status: str = "") -> np.ndarray:
    """Static camera + a force strip (Fz over the episode, N), the course's video idiom."""
    import cv2

    strip = np.full((SH, VW, 3), 255, np.uint8)
    lo, hi = -12.0, 4.0

    def pt(i, v):
        return int(i / max(1, tmax) * (VW - 1)), int(SH - 2 - (np.clip(v, lo, hi) - lo)
                                                     / (hi - lo) * (SH - 4))  # fmt: skip

    cv2.line(strip, (0, pt(0, 0.0)[1]), (VW - 1, pt(0, 0.0)[1]), (200, 200, 200), 1)
    if len(fz) > 1:
        pts = np.array([pt(i, v) for i, v in enumerate(fz)], np.int32)
        cv2.polylines(strip, [pts], False, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(strip, "Fz (N)", (4, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1,
                cv2.LINE_AA)  # fmt: skip
    out = np.vstack([img, strip])
    cv2.rectangle(out, (0, 0), (VW, 18), (255, 255, 255), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1,
                cv2.LINE_AA)  # fmt: skip
    if status:
        col = (30, 140, 60) if status == "success" else (200, 50, 40)
        cv2.putText(out, status.upper(), (10, VH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2,
                    cv2.LINE_AA)  # fmt: skip
    return out


def _card(lines: list[str]) -> np.ndarray:
    import cv2

    img = np.full((VH + SH, VW, 3), 250, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (14, 70 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1,
                    cv2.LINE_AA)  # fmt: skip
    return img


def _env(task: str, video: bool):
    from nfv.sim.env import Env
    from nfv.sim.gantry_xml import GantryParams

    g = GantryParams(offwidth=VW, offheight=VH) if video else None
    return Env(task=task, render=True, gantry=g)


def cmd_eval(a) -> int:
    from nfv.seeds import eval_seeds

    out = Path(a.out)
    ctx = FileCtx(out.parent)
    torch.set_num_threads(max(1, a.threads))
    t0 = time.perf_counter()
    policy, cfg = _load_arm(a.adapter)
    codec = V.Codec(policy, cfg["stats"], "cuda")
    actor = V.SmolVLAActor(policy, codec)
    task = a.task or cfg["task"]
    env = _env(task, video=False)
    seeds = eval_seeds(task, a.n, a.offset)
    rows = []
    try:
        for i, s in enumerate(seeds):
            rows.append(_episode(env, actor, s, trace=i < a.traces))
            k = sum(r["success"] for r in rows)
            ctx.status(done=i + 1, total=len(seeds), success=k / (i + 1),
                       message=f"{cfg['mode']}: episode {i + 1}/{len(seeds)}, "
                               f"{k}/{i + 1} success")  # fmt: skip
            if (out.parent / "STOP").exists():
                break
    finally:
        env.close()
    n, k = len(rows), sum(r["success"] for r in rows)
    outcomes: dict[str, int] = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    pf = [r["peak_force"] for r in rows]
    _dump(out, {
        "mode": cfg["mode"], "task": task, "n": n, "success": round(k / max(1, n), 4),
        "wilson95": wilson95(k, n), "outcomes": outcomes,
        "peak_force_mean": round(float(np.mean(pf)), 3) if pf else 0.0,
        "peak_force_max": round(float(np.max(pf)), 3) if pf else 0.0,
        "chunk_ms": round(1000 * actor.time_s / max(1, actor.calls), 2),
        "chunk_calls": actor.calls, "wall_s": round(time.perf_counter() - t0, 1),
        "infer_vram_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
        "episodes": rows,
    })  # fmt: skip
    return 0


def cmd_render(a) -> int:
    from lab.core.video import frames_to_mp4

    policy, cfg = _load_arm(a.adapter)
    actor = V.SmolVLAActor(policy, V.Codec(policy, cfg["stats"], "cuda"))
    task = a.task or cfg["task"]
    env = _env(task, video=True)
    card = _card([a.title or "LoRA on SmolVLA (lesson 20)", f"arm {a.label} on {task}",
                  f"seed {a.seed}"])  # fmt: skip
    frames: list[np.ndarray] = [card] * 8
    try:
        rec = _episode(env, actor, a.seed, frames=frames, label=a.label)
    finally:
        env.close()
    frames_to_mp4(iter(frames), a.out, fps=FPS)
    _dump(Path(a.out).with_suffix(".json"), rec)
    return 0


# ------------------------------------------------------------------------------------- probe
def cmd_probe(a) -> int:
    """Training steps with EVERY parameter trainable: peak VRAM, or the OOM (claim m5)."""
    res = {"mode": a.mode, "batch": a.batch, "steps": a.steps, "precision": a.precision}
    rng = np.random.default_rng(a.seed)
    try:
        ds = _demo_view(a.demo_root, a.task, a.demos)
        sampler = V.Sampler(ds)
        policy = V.load_policy("cuda")
        if a.precision == "fp32":  # the lesson's recipe: fp32 master copies of what trains
            rep = V.setup_mode(policy, a.mode)
        else:  # bf16: lerobot's as-loaded dtypes, AdamW moments in bf16 too
            for p in policy.parameters():
                p.requires_grad_(True)
            rep = {"trainable": L.trainable_report(policy, V.GROUPS)}
        res["trainable"] = rep["trainable"]["trainable"]
        res["total"] = rep["trainable"]["total"]
        codec = V.Codec(policy, V.norm_stats(ds), "cuda")
        opt, params = V.make_optimizer(policy, V.MODE_LR.get(a.mode, 1e-4))
        policy.train()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(a.steps):
            V.train_step(policy, opt, params, V.train_batch(codec, sampler.sample(a.batch, rng)),
                         1e-5)  # fmt: skip
        torch.cuda.synchronize()
        res.update(fits=True, step_s=round((time.perf_counter() - t) / a.steps, 3), **_vram())
    except torch.OutOfMemoryError as e:
        res.update(fits=False, error="CUDA out of memory", detail=str(e).split("\n")[0][:300],
                   **_vram())  # fmt: skip
    _dump(Path(a.out), res)
    print(json.dumps(res))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nfv.lora_worker")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--task", default="t3")
        p.add_argument("--demo-root", default="lab/data/nfv/demos")
        p.add_argument("--demos", type=int, default=200)
        p.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--mode", choices=V.TRAIN_MODES, required=True)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=0.0, help="0 = the mode's default (MODE_LR)")
    t.add_argument("--warmup", type=int, default=0, help="0 = min(1000, steps // 10)")
    t.add_argument("--r", type=int, default=16)
    t.add_argument("--alpha", type=float, default=32.0)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--log-every", type=int, default=25)
    t.add_argument("--no-merge-check", action="store_true")
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--adapter", required=True)
    e.add_argument("--task", default="")
    e.add_argument("--n", type=int, default=100)
    e.add_argument("--offset", type=int, default=0)
    e.add_argument("--traces", type=int, default=3)
    e.add_argument("--threads", type=int, default=2)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_eval)

    r = sub.add_parser("render")
    r.add_argument("--adapter", required=True)
    r.add_argument("--task", default="")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--label", default="")
    r.add_argument("--title", default="")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_render)

    p = sub.add_parser("probe")
    common(p)
    p.add_argument("--mode", default="full", choices=V.MODES)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_probe)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
