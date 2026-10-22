"""Task 4 - "The force token" (id `force-token`): the repo's Stage-1 gantry pipeline in the lab.

Wraps, by import (nothing is shelled out and `src/fvb` is untouched):
  - collection  = `scripts/09_collect_expert.py::run_episode`'s loop on `fvb.policy.task.GantryTask`
                  (privileged scripted expert, the policy-facing `obs` / `action` rows), in memory;
  - training    = the epoch loop of `scripts/10_train_bc.py` (masked L1 on z-scored (H, K) windows,
                  AdamW + cosine, batch 256), with `fvb.policy.models.build_model("act"|"mlp")`;
  - evaluation  = `fvb.policy.rollout.TorchPolicy` closed loop on unseen holes (script 11), with
                  offscreen MuJoCo rendering of the first `render_n` episodes (script 13's camera).

`noise_mm` is DART-style action noise: while collecting, the executed action is the expert action
plus N(0, noise_mm) per axis, but the label stored for training stays the clean expert action
(the observation does see the noisy motion through `prev_action`). Stage 1's own datasets use 0.

Metrics (`metrics.jsonl`, one monotonic step stream): collection logs `collected=i` at step i
(1..demos); each epoch logs at step demos + epoch + 1 {epoch, train_l1, val_l1, val_mae_mm,
val_mae_mm_x/y/z, lr}. Checkpoints (`ckpt/`): one on each val-L1 improvement (label "best") and the
final / stop-time one (label "last"; `last.pt` is always the final epoch). Payload is script 10's:
    {"state_dict", "norm": Norm.to_json(), "meta": {model, H, K, use_force, task, action_mode,
     "model_kw", obs_dim, act_dim, data, seed}, "epoch", "val_l1", "params"}
`evaluate` rebuilds the model from `meta` only (never from form values).

result.json written by `evaluate`:
    {"n_episodes", "success", "counts": {success, force_abort, timeout}, "wilson95",
     "force_zeroed", "trained_with_force", "model", "H", "K", "clearance_mm", "offset_sigma_mm",
     "fps", "video": "rollout.mp4" | null,
     "buckets": [{"lo", "hi", "n", "success"}],             # by |hole offset| in mm
     "episodes": [{"seed", "offset_mm": [x, y], "outcome", "steps", "peak_F_N", "recoveries",
                   "final_depth_mm", "force": [[Fx,Fy,Fz,Tx,Ty,Tz]...] (compensated, after step i),
                   "obs_force": [[9 force obs channels]...] (what the net saw choosing step i),
                   "act": [[dx,dy,dz mm]...], "depth_mm": [...]}]}   # first render_n episodes
Video: for each rendered episode a 0.75 s title card (episode number, seed, hole offset), then the
3-D close-up with a |F| / Fz strip underneath (HUD: time, depth, |F|), then the result held 1 s.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from fvb.policy.data import compute_norm, make_windows, split_episodes
from fvb.policy.rollout import TorchPolicy
from fvb.policy.spec import GANTRY
from fvb.policy.task import GantryTask, TaskParams
from lab.core.runctx import atomic_write_json
from lab.core.video import frames_to_mp4
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.task1_bc_mse import OVERHEAD_S, wilson95

FPS = 20  # control rate = real time
OUTCOMES = ("success", "force_abort", "timeout")
BUCKET_EDGES = [0.0, 0.5, 1.0, 2.0, 3.0, 6.0, None]  # |offset| mm; last bucket open-ended (6+)
DEMO_SEED0, EVAL_SEED0 = 1000, 50000  # eval seeds are disjoint from data (train seed <= 40)
BATCH, WD, VAL_FRAC = 256, 1e-4, 0.1
ACT_KW = {"dropout": 0.1, "input_norm": True}  # script 10 defaults
FALLBACK_COLLECT_S_PER_EP, FALLBACK_S_PER_EPOCH, REF_DEMOS = 0.2, 1.5, 300
VW, VH, SH = 320, 240, 70  # video: close-up size and strip height


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------- collection
def collect_episode(seed: int, p: TaskParams, noise_mm: float) -> dict:
    """One expert episode (port of scripts/09 `run_episode`, in memory). Labels stay clean."""
    task = GantryTask(p, seed)
    rng = np.random.default_rng(seed + 555)
    obs_rows, act_rows, done, reason = [], [], False, None
    while not done:
        o = task.observe()
        a = task.encode(task.expert_action())
        obs_rows.append(o)
        act_rows.append(a.astype(np.float32))
        run = a + rng.normal(0.0, noise_mm * 1e-3, 3) if noise_mm > 0 else a
        done, reason = task.step(run)
    return {"obs": np.stack(obs_rows), "action": np.stack(act_rows), "success": reason == "success"}


# ------------------------------------------------------------------------------------- training
def train(ctx) -> None:
    import torch

    from fvb.policy.models import build_model, n_params

    P = ctx.params
    demos, epochs = int(P["demos"]), int(P["epochs"])
    kind, use_force = str(P["model"]), bool(P["use_force"])
    H, K, lr = int(P["H"]), int(P["K"]), float(P["lr"])
    noise_mm, seed = float(P["noise_mm"]), int(P.get("train_seed", 0))
    total = demos + epochs
    ctx.set_status(step=0, total_steps=total, message="collecting expert demonstrations")

    p = TaskParams()
    eps = []
    for i in range(demos):
        if not ctx.should_continue():
            ctx.set_status(state="stopped")
            return
        eps.append(collect_episode(DEMO_SEED0 + seed * 1000 + i, p, noise_mm))
        ctx.log(i + 1, collected=i + 1)
        ctx.set_status(step=i + 1, total_steps=total)
    n_ok = sum(e["success"] for e in eps)
    ctx.set_status(message=f"{demos} demonstrations ({n_ok} reached the goal), training")

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device()
    fi = GANTRY.force_idx
    train_eps, val_eps = split_episodes(eps, VAL_FRAC, seed)
    norm = compute_norm(train_eps, use_force, fi)
    Xtr, Ytr, Mtr = make_windows(train_eps, H, K, use_force, norm, fi)
    Xva, Yva, Mva = make_windows(val_eps, H, K, use_force, norm, fi)
    model_kw = dict(ACT_KW) if kind == "act" else {}
    model = build_model(kind, GANTRY.obs_dim, GANTRY.act_dim, H, K, **model_kw).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    spe = int(np.ceil(len(Xtr) / BATCH))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * spe)
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    Xtr_t, Ytr_t, Mtr_t, Xva_t, Yva_t, Mva_t = map(to, (Xtr, Ytr, Mtr, Xva, Yva, Mva))

    def masked_l1(pred, y, m):
        return ((pred - y).abs().mean(-1) * m).sum() / m.sum()

    meta = {"model": kind, "H": H, "K": K, "use_force": use_force, "task": GANTRY.name,
            "action_mode": p.action_mode, "model_kw": model_kw, "obs_dim": GANTRY.obs_dim,
            "act_dim": GANTRY.act_dim, "data": f"lab-inmemory:{demos}", "seed": seed}  # fmt: skip
    ctx.set_status(message=f"{kind} ({n_params(model):,} params) on {dev}, {len(Xtr)} windows")
    best, g = float("inf"), torch.Generator(device="cpu").manual_seed(seed)
    saved_ep = -1

    def save(ep, va, label):
        nonlocal saved_ep
        ck = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
              "norm": norm.to_json(), "meta": meta, "epoch": ep, "val_l1": va,
              "params": dict(P)}  # fmt: skip
        ctx.save_ckpt(demos + ep + 1, ck, label=label, metrics={"val_l1": va})
        saved_ep = ep

    last_ep, last_va = -1, float("nan")
    for ep in range(epochs):
        if not ctx.should_continue():
            if last_ep >= 0 and saved_ep != last_ep:
                save(last_ep, last_va, "last")
            ctx.set_status(state="stopped")
            return
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g).to(dev)
        tr = 0.0
        for i in range(spe):
            idx = perm[i * BATCH : (i + 1) * BATCH]
            loss = masked_l1(model(Xtr_t[idx]), Ytr_t[idx], Mtr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tr += loss.item()
        model.eval()
        with torch.no_grad():
            pred = model(Xva_t)
            va = masked_l1(pred, Yva_t, Mva_t).item()
            p0 = pred[:, 0].cpu().numpy() * norm.act_std + norm.act_mean
            t0 = Yva[:, 0] * norm.act_std + norm.act_mean
            mae = (1e3 * np.abs(p0 - t0)).mean(0)
        step = demos + ep + 1
        ctx.log(step, epoch=ep + 1, train_l1=tr / spe, val_l1=va, val_mae_mm=float(mae.mean()),
                val_mae_mm_x=float(mae[0]), val_mae_mm_y=float(mae[1]),
                val_mae_mm_z=float(mae[2]), lr=sched.get_last_lr()[0])  # fmt: skip
        ctx.set_status(step=step, total_steps=total)
        last_ep, last_va = ep, va
        if va < best:
            best = va
            if ep < epochs - 1:
                save(ep, va, "best")
    save(last_ep, last_va, "last")
    ctx.done(f"trained {kind} on {demos} demos ({'with' if use_force else 'no'} force), "
             f"best val L1 {best:.3f}")  # fmt: skip


# ------------------------------------------------------------------------------------- video
def _card(lines: list[str]) -> np.ndarray:
    import cv2

    img = np.full((VH + SH, VW, 3), 250, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (14, 80 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1,
                    cv2.LINE_AA)  # fmt: skip
    return img


def _strip(fmag: list[float], fz: list[float]) -> np.ndarray:
    """|F| (dark) and Fz (blue) against time; fixed axes -10..30 N over the 8 s horizon."""
    import cv2

    img = np.full((SH, VW, 3), 255, np.uint8)
    lo, hi, tmax = -10.0, 30.0, 160

    def pt(i, v):
        return int(i / tmax * (VW - 1)), int(
            SH - 2 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (SH - 4)
        )

    y0 = pt(0, 0.0)[1]
    cv2.line(img, (0, y0), (VW - 1, y0), (200, 200, 200), 1)
    for series, col in ((fz, (220, 110, 30)), (fmag, (40, 40, 40))):
        if len(series) > 1:
            pts = np.array([pt(i, v) for i, v in enumerate(series)], np.int32)
            cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
    cv2.putText(img, "|F| (dark), Fz (blue), N", (4, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                (90, 90, 90), 1, cv2.LINE_AA)  # fmt: skip
    return img


def _episode(policy, p: TaskParams, seed: int, render: bool):
    """Closed-loop episode (script 11 / rollout.rollout). A generator: yields video frames when
    `render`, returns the record (final series only for rendered episodes) via StopIteration."""
    import cv2

    task = GantryTask(replace(p, **policy.task_overrides), seed)
    policy.reset()
    rend = cam = None
    if render:
        import mujoco

        for wname in ("w_px", "w_nx", "w_py", "w_ny"):  # see-through walls, as in script 13
            gid = mujoco.mj_name2id(task.m, mujoco.mjtObj.mjOBJ_GEOM, wname)
            task.m.geom_rgba[gid] = [0.55, 0.5, 0.45, 0.35]
        rend = mujoco.Renderer(task.m, height=VH, width=VW)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [*task.hole_xy, 0.035]
        cam.distance, cam.azimuth, cam.elevation = 0.26, 125.0, -28.0
    fmag, fz = [], []
    ser = {"force": [], "obs_force": [], "act": [], "depth_mm": []}

    def frame(status=""):
        rend.update_scene(task.d, camera=cam)
        img = np.vstack([rend.render().copy(), _strip(fmag, fz)])
        txt = (
            f"t {task.k / FPS:4.2f} s  depth {1e3 * task.depth():5.1f} mm  "
            f"|F| {task.force_norm():5.2f} N"
        )
        cv2.rectangle(img, (0, 0), (VW, 18), (255, 255, 255), -1)
        cv2.putText(img, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1, cv2.LINE_AA)
        if status:
            col = (30, 140, 60) if status == "success" else (200, 50, 40)
            cv2.putText(img, status.upper(), (10, VH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2,
                        cv2.LINE_AA)  # fmt: skip
        return img

    try:
        return (yield from _run(policy, task, render, frame, fmag, fz, ser, seed))
    finally:
        if rend is not None:
            rend.close()


def _run(policy, task, render, frame, fmag, fz, ser, seed):
    if render:
        yield frame()
    done, reason, peak, n_jams, retracting = False, None, 0.0, 0, False
    while not done:
        obs = task.observe()
        a = policy.act(obs)
        f = task.force_norm()
        if a[2] > 1e-4 and f > 1.0 and not retracting:
            n_jams += 1
            retracting = True
        if a[2] < 0:
            retracting = False
        done, reason = task.step(a)
        peak = max(peak, task.force_norm())
        if render:
            comp = task.g.ft_comp_last()
            ser["force"].append(np.round(comp, 3).tolist())
            ser["obs_force"].append(np.round(obs[GANTRY.force_idx], 3).tolist())
            ser["act"].append(np.round(1e3 * np.asarray(a, float), 3).tolist())
            ser["depth_mm"].append(round(1e3 * task.depth(), 3))
            fmag.append(float(np.linalg.norm(comp[:3])))
            fz.append(float(comp[2]))
            yield frame()
    if render:
        last = frame(reason)
        for _ in range(FPS):  # hold the result for 1 s
            yield last
    rec = {"seed": seed, "offset_mm": np.round(1e3 * task.hole_xy, 3).tolist(), "outcome": reason,
           "steps": task.k, "peak_F_N": round(peak, 3), "recoveries": n_jams,
           "final_depth_mm": round(1e3 * task.depth(), 3)}  # fmt: skip
    return {**rec, **ser} if render else rec


def _drain(gen):
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def evaluate(ctx) -> None:
    P = ctx.params
    n = int(P.get("n_episodes", 50))
    render_n = max(0, min(int(P.get("render_n", 4)), n))
    seed = int(P.get("seed", 0))
    clearance, sigma = float(P.get("clearance_mm", 0.5)), float(P.get("offset_sigma_mm", 1.5))
    import torch

    torch.set_num_threads(1)  # batch-1 inference: more threads only thrash on a loaded host
    policy = TorchPolicy(P["ckpt"], "cpu")  # architecture comes from ckpt["meta"] only
    trained_with_force = bool(policy.use_force)
    zeroed = bool(P.get("force_zeroed", False)) or not trained_with_force
    policy.use_force = not zeroed  # TorchPolicy.act zeroes the force channels when False
    p = TaskParams(clearance=clearance * 1e-3, offset_sigma=sigma * 1e-3)
    seeds = [EVAL_SEED0 + seed * 1000 + i for i in range(n)]
    ctx.set_status(step=0, total_steps=n)

    rows, rendered = [], []

    stopped = False

    def frames():
        nonlocal stopped
        for i in range(render_n):
            if not ctx.should_continue():
                stopped = True
                return
            gen = _episode(policy, p, seeds[i], True)
            ox, oy = (1e3 * GantryTask(p, seeds[i]).hole_xy).tolist()
            title = [f"episode {i + 1} of {render_n}", f"seed {seeds[i]}",
                     f"hole offset ({ox:+.2f}, {oy:+.2f}) mm", f"clearance {clearance:g} mm",
                     "force input: " + ("ZEROED" if zeroed else "on")]  # fmt: skip
            for _ in range(int(0.75 * FPS)):
                yield _card(title)
            rec = yield from gen
            rendered.append(rec)

    video = None
    if render_n:
        try:
            frames_to_mp4(frames(), ctx.dir / "rollout.mp4", fps=FPS)
            video = "rollout.mp4"
            rows.extend(rendered)
        except Exception as e:  # noqa: BLE001 - a render failure must not lose the statistics
            print(f"video rendering failed ({type(e).__name__}: {e}); re-running unrendered")
            ctx.set_status(message=f"video failed: {type(e).__name__}: {e}")
            (ctx.dir / "rollout.mp4").unlink(missing_ok=True)
            # the rendered episodes are re-run without rendering: same seeds, so same outcomes,
            # but no per-step series (the viewer then shows tables only)
            for i in range(render_n):
                rows.append(_drain(_episode(policy, p, seeds[i], False)))
        if stopped:
            ctx.set_status(state="stopped")
            return
        ctx.set_status(step=render_n, total_steps=n)
    for i in range(render_n, n):
        if not ctx.should_continue():
            ctx.set_status(state="stopped")
            return
        rows.append(_drain(_episode(policy, p, seeds[i], False)))
        ctx.set_status(step=i + 1, total_steps=n)

    k = sum(r["outcome"] == "success" for r in rows)
    cnt = Counter(r["outcome"] for r in rows)
    mags = np.array([np.hypot(*r["offset_mm"]) for r in rows])
    ok = np.array([r["outcome"] == "success" for r in rows])
    edges = BUCKET_EDGES
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (mags >= lo) & (mags < hi) if hi is not None else (mags >= lo)
        buckets.append({"lo": lo, "hi": hi, "n": int(sel.sum()),
                        "success": float(ok[sel].mean()) if sel.any() else None})  # fmt: skip
    atomic_write_json(
        ctx.dir / "result.json",
        {"n_episodes": n, "success": k / max(1, n),
         "counts": {o: int(cnt.get(o, 0)) for o in OUTCOMES}, "wilson95": wilson95(k, n),
         "force_zeroed": zeroed, "trained_with_force": trained_with_force,
         "model": policy.meta["model"], "H": policy.H, "K": policy.K, "clearance_mm": clearance,
         "offset_sigma_mm": sigma, "fps": FPS, "video": video,
         "buckets": buckets, "episodes": rows[:render_n]},
    )  # fmt: skip
    ctx.done("evaluated")


# ------------------------------------------------------------------------------------ estimate
def estimate(params: dict) -> float:
    cal = registry.calib("force-token")
    collect = cal.get("collect_s_per_ep") or FALLBACK_COLLECT_S_PER_EP
    per_epoch = (cal.get("s_per_epoch") or FALLBACK_S_PER_EPOCH) * int(params["demos"]) / REF_DEMOS
    if params.get("model") == "mlp":
        per_epoch *= cal.get("mlp_factor", 0.3)
    overhead = cal.get("overhead_s", OVERHEAD_S)
    return int(params["demos"]) * collect + int(params["epochs"]) * per_epoch + overhead


SPEC = TaskSpec(
    id="force-token",
    order=5,
    title="The force token",
    subtitle="Stage 1 on the gantry: a history transformer feels the jam and finds the hole.",
    params=[
        Param("demos", "Expert demonstrations", "int", 300, min=50, max=500, step=50,
              help="collected at run start on the gantry (cheap: about a minute for 300)"),
        Param("model", "Policy", "choice", "act", choices=["act", "mlp"]),
        Param("use_force", "Feed the force channels", "bool", True),
        Param("epochs", "Epochs", "int", 40, min=5, max=80, step=5),
        Param("lr", "Learning rate", "float", 3e-4, min=1e-4, max=3e-3),
        Param("H", "History length H", "int", 10, min=4, max=20, step=1),
        Param("K", "Action chunk K", "int", 5, min=1, max=10, step=1),
        Param("noise_mm", "Demo action noise (mm)", "float", 0.0, min=0.0, max=2.0, step=0.1,
              help="expert execution noise (mm), labels stay clean; 0 = Stage-1 replica, "
              "0.5 = robustness experiment"),
        Param("train_seed", "Training seed", "int", 0, min=0, max=40),
    ],
    eval_params=[
        Param("n_episodes", "Episodes", "int", 50, min=2, max=200, step=10),
        Param("offset_sigma_mm", "Hole offset sigma (mm)", "float", 1.5, min=0.5, max=3.0,
              step=0.5, help="trained at 1.5; the hole position is hidden from the policy"),
        Param("clearance_mm", "Clearance (mm)", "float", 0.5, min=0.3, max=1.0, step=0.1),
        Param("force_zeroed", "Zero the force input (twin)", "bool", False),
        Param("render_n", "Episodes to render", "int", 4, min=0, max=8, step=1),
        Param("seed", "Eval seed", "int", 0, min=0, max=999),
    ],
    charts=[
        {"id": "collect", "title": "Demonstrations collected",
         "series": [{"key": "collected", "label": "episodes"}], "log_y": False},
        {"id": "l1", "title": "Masked L1 (z-scored action chunks)",
         "series": [{"key": "train_l1", "label": "train"}, {"key": "val_l1", "label": "val"}],
         "log_y": False},
        {"id": "mae", "title": "Validation action error (mm, first chunk step)",
         "series": [{"key": "val_mae_mm", "label": "mean"}, {"key": "val_mae_mm_x", "label": "x"},
                    {"key": "val_mae_mm_y", "label": "y"}, {"key": "val_mae_mm_z", "label": "z"}],
         "log_y": False},
    ],
    viewer="video",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="force-token.html",
)  # fmt: skip
registry.register(SPEC)
