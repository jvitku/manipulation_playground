"""Lesson 20 - "LoRA: fine-tuning a large VLA on a small budget" (id `lora-vla`).

SmolVLA (``lerobot/smolvla_base``, 450 M parameters, arXiv 2506.01844) fine-tuned on the course's
canonical demos (lesson 15, ``data_root()/nfv/demos/<task>/``) three ways, and LoRA (nfv.lora, from
scratch) put next to the two classic alternatives:

    A head         only the new-embodiment projections train (state/action in/out + the
                   time-action MLP, 1.6 M) - the linear-probe baseline
    B lora         head + LoRA r/alpha on the attention q/k/v/o of the 16 VLM text layers AND the
                   16 action-expert layers (nfv.vla_smolvla.LORA_TARGETS, 128 layers)
    C expert_full  lerobot's standard SmolVLA recipe: VLM frozen, the whole action expert trains
    D full         every parameter trainable - only MEASURED: peak VRAM at the same batch, or OOM

The lab's own python (3.11, torch 2.5.1) cannot run lerobot 0.6.1, so every GPU step happens in a
WORKER process on the lab image's separate venv (``/opt/vla/bin/python -m nfv.lora_worker``, see
docker/Dockerfile.lab) - lesson 19's house pattern (nfv.rl.worker): file-based context, a STOP
file is the stop button, the parent re-logs each worker's metrics.jsonl rows with the arm prefix
(``a_loss``, ``b_steps_per_s``, ...; arm-local steps, so the three loss curves overlay). The
arms run SEQUENTIALLY on the GPU (one 450 M model at a time); the D probe runs first, on a clean
GPU, in fp32-master precision (the recipe every arm trains with) and in lerobot's as-loaded bf16.

TRAIN presets: ``single`` (one arm: ``mode``, rank ``r`` 4/8/16/32, ``alpha``) and
``claims_suite`` (A, B, C in one job, + the D probe). ``lr`` 0 = each mode's pre-registered
default (nfv.vla_smolvla.MODE_LR). Run dir layout:
    <run>/arms/<key>_<mode>/   the worker's metrics.jsonl, progress.json, adapter.pt (ONLY the
                               trainable tensors + config), result.json, worker.log
    <run>/probe_full_<precision>.json
Checkpoint (small, saved after every arm): {"kind": "nfv-lora-suite", "task", "preset",
"arms": [{"key", "label", "mode", "adapter": "arms/<..>/adapter.pt" (relative to the run dir),
"train": the worker's result.json}], "probe_full": {"fp32": .., "bf16": ..}, "params"} - the
adapters themselves stay in the run dir (C's is ~400 MB; the lab writes every checkpoint twice,
step file + last.pt, and must not copy it).

EVAL rolls every arm closed-loop on the SAME ``episodes`` held-out seeds
(``nfv.seeds.eval_seeds(task, episodes, eval_seed * episodes)``), one eval worker per arm (up to
``parallel`` at once - the sim is CPU-bound, each worker holds one bf16 model, ~1 GB), then
renders the before/after video on one seed: the first seed where B succeeds and A fails, else
the first any arm solves, else the first seed; every arm on that seed, concatenated.

result.json (viewer ``fvrollout``; the "LoRA anatomy" panel reads lora_arms / lora_spectrum):
    {"task", "episodes", "n_episodes", "fps", "video", "video_seed", "claims_task_note",
     "arms": {label: {"mode", "n", "success", "wilson95", "outcomes", "peak_force_mean",
                      "peak_force_max", "chunk_ms"}},
     "lora_arms": [{"label", "mode", "trainable", "total", "trainable_pct", "peak_vram_gb",
                    "peak_reserved_gb", "steps_per_s", "adapter_mb", "full_ckpt_mb",
                    "final_loss", "success", "wilson95", "r", "alpha", "lr", "steps", "batch"}],
     "probe_full": {"fp32": {fits, peak_alloc_gb, peak_reserved_gb, step_s|error}, "bf16": ..},
     "lora_spectrum": {"r", "layers": [{"name", "shape", "lora": {"sv", "energy_top_r"},
                       "full": {"sv", "energy_top_r", "energy_top_16"}}],
                       "full_mean_energy_top16"},
     "merge_check": B's {fp32_max_abs, bf16_max_abs, ...}, "bf16_lost_update_frac": C's,
     "lora_targets": [128 module names], "head_params": [...],
     "outcomes": {"success", "counts", "wilson95", "per_task": {label: row}},
     "peak_force": {"mean", "max"}, "claims": [nfv.claims.M5_CLAIMS rows],
     "episodes_traces": [<= 10, the fvrollout trace schema]}
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from lab.core.runctx import atomic_write_json
from lab.core.video import frames_to_mp4
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.task1_bc_mse import wilson95
from lab.tasks.task_nfv0 import FPS, MAX_TRACES, _demo_root
from nfv.claims import M5_CLAIMS, NOT_MEASURED, verdict
from nfv.seeds import TASK_NAMES

PRESETS = ("single", "claims_suite")
TRAIN_MODES = ("head", "lora", "expert_full")  # nfv.vla_smolvla.TRAIN_MODES (no torch import here)
# (metric prefix, label, mode) - the suite's arms in training order
SUITE = (("a", "A head", "head"), ("b", "B lora", "lora"), ("c", "C expert_full", "expert_full"))
RANKS = [4, 8, 16, 32]
VLA_PY = "/opt/vla/bin/python"  # the lab image's lerobot venv (docker/Dockerfile.lab)
WORKER = "nfv.lora_worker"
POLL_S = 2.0
N_TRACE = 3  # per arm (3 arms -> 9 traces <= MAX_TRACES)
D_LIMIT_GB = 9.5  # m5.full_ft_does_not_fit: "OOM or > 9.5 GB"
GPU_GB = 10.0  # the RTX 3080's 10 GB; an OOM counts as "needed at least this"
# calib.json fallbacks (measured 2026-11-01, RTX 3080, batch 16, the two 5000-step suites):
# samples/s per mode (steps/s x batch), worker load, one D probe; the suites took 81 min
FALLBACK_SAMPLES_PER_S = {"head": 52.1, "lora": 48.4, "expert_full": 48.9}
FALLBACK_LOAD_S, FALLBACK_PROBE_S, OVERHEAD_S = 10.0, 8.0, 10.0


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _worker_env(threads: int = 4) -> dict:
    env = dict(os.environ)
    src = str(_repo() / "src")
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), HF_HUB_OFFLINE="1",
               MUJOCO_GL=env.get("MUJOCO_GL", "osmesa"),
               HF_HOME=env.get("HF_HOME") or str(_repo() / "outputs" / "hf_cache"))  # fmt: skip
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _spawn(args: list[str], log: Path, threads: int = 4) -> subprocess.Popen:
    """``/opt/vla/bin/python -m nfv.lora_worker <args>`` (NFV_VLA_PY / NFV_LORA_WORKER override
    the interpreter / module - the tests' fake worker)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    py = os.environ.get("NFV_VLA_PY", VLA_PY)
    mod = os.environ.get("NFV_LORA_WORKER", WORKER)
    with log.open("ab") as f:
        return subprocess.Popen([py, "-m", mod, *args], stdout=f, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=_worker_env(threads),
                                cwd=_repo())  # fmt: skip


def _tail(path: Path, n: int = 1500) -> str:
    try:
        return path.read_text()[-n:]
    except OSError:
        return ""


def _read(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _arms(P: dict) -> list[tuple[str, str, str]]:
    if P["preset"] == "single":
        return [("", f"single {P['mode']}", str(P["mode"]))]
    return list(SUITE)


def _arm_dir(key: str, mode: str) -> str:
    return f"arms/{key + '_' if key else ''}{mode}"


class _Drain:
    """Re-log a worker's metrics.jsonl rows into the run with the arm's prefix."""

    def __init__(self, ctx, path: Path, prefix: str):
        self.ctx, self.path, self.prefix, self.off = ctx, path, prefix, 0

    def __call__(self) -> None:
        if not self.path.exists():
            return
        with self.path.open() as f:
            f.seek(self.off)
            chunk = f.read()
        end = chunk.rfind("\n")
        if end < 0:
            return
        self.off += len(chunk[: end + 1].encode())
        for line in chunk[: end + 1].splitlines():
            row = json.loads(line)
            step = int(row.pop("step"))
            p = f"{self.prefix}_" if self.prefix else ""
            self.ctx.log(step, **{f"{p}{k}": v for k, v in row.items()})


def _wait(ctx, procs: dict, on_poll, stop_dirs: list[Path]) -> bool:
    """Poll until every process exits; False when the user stopped (STOP files written)."""
    stopped = False
    try:
        while any(p.poll() is None for p in procs.values()):
            time.sleep(POLL_S)
            on_poll()
            if not stopped and not ctx.should_continue():
                stopped = True
                for d in stop_dirs:
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "STOP").touch()
    except BaseException:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        raise
    on_poll()
    return not stopped


# ------------------------------------------------------------------------------------- train
def train(ctx) -> None:
    P = ctx.params
    arms = _arms(P)
    steps, batch, task = int(P["steps"]), int(P["batch"]), str(P["task"])
    root = _demo_root()
    if not (root / task / "index.json").exists():
        raise FileNotFoundError(f"no {task} demos under {root} - record them in lesson 15 first")
    common = ["--task", task, "--demo-root", str(root), "--demos", str(int(P["demos"]))]
    probe: dict = {}
    if P["preset"] == "claims_suite" or bool(P.get("probe_full")):
        for prec in ("fp32", "bf16"):  # D: does training EVERY parameter fit? (first, clean GPU)
            msg = f"D probe: every parameter trainable ({prec}), batch {batch}"
            ctx.set_status(step=0, total_steps=len(arms) * steps, message=msg)
            out, log = ctx.dir / f"probe_full_{prec}.json", ctx.dir / f"probe_{prec}.log"
            p = _spawn(["probe", *common, "--mode", "full", "--batch", str(batch), "--steps", "3",
                        "--precision", prec, "--out", str(out)], log)  # fmt: skip
            if not _wait(ctx, {prec: p}, lambda: None, []):
                ctx.set_status(state="stopped")
                return
            probe[prec] = _read(out) or {"fits": None, "error": _tail(log, 400)}
    done: list[dict] = []
    for i, (key, label, mode) in enumerate(arms):
        rel = _arm_dir(key, mode)
        d = ctx.dir / rel
        args = ["train", *common, "--mode", mode, "--steps", str(steps), "--batch", str(batch),
                "--lr", str(float(P["lr"])), "--seed", str(int(P["seed"]) + 101 * i),
                "--r", str(int(P["r"])), "--alpha", str(float(P["alpha"])),
                "--log-every", str(max(10, steps // 200)), "--out", str(d)]  # fmt: skip
        proc = _spawn(args, d / "worker.log")
        drain = _Drain(ctx, d / "metrics.jsonl", key)

        def poll(i=i, label=label, d=d, drain=drain):
            drain()
            pr = _read(d / "progress.json", {})
            k = int(pr.get("step", 0))
            ctx.set_status(step=i * steps + k, total_steps=len(arms) * steps,
                           message=f"arm {i + 1}/{len(arms)} ({label}): "
                                   f"{pr.get('message', 'starting the worker')}")  # fmt: skip

        ok = _wait(ctx, {label: proc}, poll, [d])
        if proc.returncode != 0:
            if not ok:
                ctx.set_status(state="stopped")
                return
            raise RuntimeError(f"arm {label} failed:\n{_tail(d / 'worker.log')}")
        res = _read(d / "result.json")
        if res is None:
            raise RuntimeError(f"arm {label} wrote no result.json:\n{_tail(d / 'worker.log')}")
        done.append({"key": key, "label": label, "mode": mode, "adapter": f"{rel}/adapter.pt",
                     "train": res})  # fmt: skip
        payload = {"kind": "nfv-lora-suite", "task": task, "preset": P["preset"],
                   "arms": done, "probe_full": probe, "params": dict(P)}  # fmt: skip
        last = i == len(arms) - 1 and ok
        ctx.save_ckpt((i + 1) * steps, payload, label="final" if last else f"{label} trained",
                      metrics={f"{a['key'] or 'single'}_loss": round(a["train"]["final_loss"], 4)
                               for a in done})  # fmt: skip
        if not ok:
            ctx.set_status(state="stopped")
            return
    parts = ", ".join(f"{a['label']} loss {a['train']['final_loss']:.3f} "
                      f"({a['train']['trainable']['trainable'] / 1e6:.2f} M trainable, "
                      f"{a['train']['peak_alloc_gb']:.2f} GB)" for a in done)  # fmt: skip
    dtxt = ""
    if probe.get("fp32"):
        q = probe["fp32"]
        dtxt = ("; D full: " + (f"fits at {q.get('peak_alloc_gb')} GB" if q.get("fits")
                                else "OOM"))  # fmt: skip
    ctx.done(f"{task}: {parts}{dtxt}")


# ------------------------------------------------------------------------------------- eval
def _concat_videos(parts: list[Path], out: Path) -> None:
    import imageio.v2 as imageio

    def frames():
        for p in parts:
            r = imageio.get_reader(p)
            try:
                yield from r
            finally:
                r.close()

    frames_to_mp4(frames(), out, fps=FPS)


def pick_video_seed(evs: dict[str, dict], order: list[str]) -> int:
    """First seed where B (lora) succeeds and A (head) fails, else the first any arm solves."""
    seeds = [e["seed"] for e in evs[order[0]]["episodes"]]
    ok = {lbl: {e["seed"]: e["success"] for e in evs[lbl]["episodes"]} for lbl in order}
    by_mode = {evs[lbl]["mode"]: lbl for lbl in order}
    a, b = by_mode.get("head"), by_mode.get("lora")
    if a and b:
        for s in seeds:
            if ok[b].get(s) and not ok[a].get(s):
                return s
    for s in seeds:
        if any(ok[lbl].get(s) for lbl in order):
            return s
    return seeds[0]


def evaluate(ctx) -> None:
    import torch

    P = ctx.params
    episodes, eseed, video = int(P["episodes"]), int(P["eval_seed"]), bool(P["video"])
    par = max(1, int(P.get("parallel", 3)))
    ctx.set_status(message="reading the lora-vla checkpoint")
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    if ck.get("kind") != "nfv-lora-suite":
        raise ValueError("not a lora-vla checkpoint (kind nfv-lora-suite)")
    run_dir = Path(P["ckpt"]).resolve().parent.parent
    task, arms = str(ck["task"]), list(ck["arms"])
    offset = eseed * episodes
    wd = ctx.dir / "workers"
    pending = list(arms)
    running: dict[str, subprocess.Popen] = {}
    evs: dict[str, dict] = {}

    def poll():
        n_done = sum(e["n"] for e in evs.values())
        n_run = sum(int((_read(wd / a["key"] / "progress.json", {}) or {}).get("done", 0))
                    for a in arms if a["label"] in running)  # fmt: skip
        msg = (f"evaluating {len(arms)} arms on {episodes} shared {task} seeds "
               f"({len(evs)}/{len(arms)} done, {len(running)} running)")  # fmt: skip
        ctx.set_status(step=n_done + n_run, total_steps=len(arms) * episodes, message=msg)

    while pending or running:
        while pending and len(running) < par:
            a = pending.pop(0)
            out = wd / (a["key"] or "single") / "eval.json"
            running[a["label"]] = _spawn(
                ["eval", "--adapter", str(run_dir / a["adapter"]), "--task", task, "--n",
                 str(episodes), "--offset", str(offset), "--traces", str(N_TRACE),
                 "--out", str(out)], out.parent / "worker.log", threads=2)  # fmt: skip
        time.sleep(POLL_S)
        for lbl, p in list(running.items()):
            if p.poll() is None:
                continue
            a = next(x for x in arms if x["label"] == lbl)
            sub = wd / (a["key"] or "single")
            if p.returncode != 0:
                for q in running.values():
                    if q.poll() is None:
                        q.terminate()
                raise RuntimeError(f"eval of {lbl} failed:\n{_tail(sub / 'worker.log')}")
            evs[lbl] = _read(sub / "eval.json")
            del running[lbl]
        poll()
        if not ctx.should_continue():
            for q in running.values():
                q.terminate()
            ctx.set_status(state="stopped")
            return
    order = [a["label"] for a in arms]
    video_name, vseed = None, None
    if video:
        vseed = pick_video_seed(evs, order)
        parts = []
        try:
            for a in arms:
                ctx.set_status(message=f"rendering {a['label']} on seed {vseed} for the video")
                seg = wd / f"video_{a['key'] or 'single'}.mp4"
                p = _spawn(["render", "--adapter", str(run_dir / a["adapter"]), "--task", task,
                            "--seed", str(vseed), "--label", a["label"],
                            "--title", "LoRA on SmolVLA (lesson 20)",
                            "--out", str(seg)], seg.with_suffix(".log"))  # fmt: skip
                if p.wait() != 0:
                    raise RuntimeError(_tail(seg.with_suffix(".log")))
                parts.append(seg)
            _concat_videos(parts, ctx.dir / "rollout.mp4")
            video_name = "rollout.mp4"
        except Exception as e:  # noqa: BLE001 - a render failure must not lose the stats
            ctx.set_status(message=f"video failed: {type(e).__name__}: {e}"[:300])
            (ctx.dir / "rollout.mp4").unlink(missing_ok=True)
        for p2 in parts:
            p2.unlink(missing_ok=True)
    result = assemble(ck, evs, episodes, video_name, vseed)
    atomic_write_json(ctx.dir / "result.json", result)
    arm_txt = ", ".join(f"{lbl} {s['success']:.0%}" for lbl, s in result["arms"].items())
    verdicts = "; ".join(f"{r['id'].split('.')[-1]}: {r['verdict']}" for r in result["claims"]
                         if r["verdict"] != NOT_MEASURED)  # fmt: skip
    ctx.done(f"{task}: {arm_txt} on {episodes} shared seeds - {verdicts}")


# ------------------------------------------------------------------------- result + claims
def _mb(b) -> float | None:
    return None if b is None else round(float(b) / 1e6, 2)


def assemble(ck: dict, evs: dict[str, dict], episodes: int, video: str | None = None,
             video_seed: int | None = None) -> dict:  # fmt: skip
    """result.json from the checkpoint payload and the eval workers' outputs (pure: tested)."""
    task = str(ck["task"])
    arms = list(ck["arms"])
    rows, per_task, lora_arms, traces = {}, {}, [], []
    for a in arms:
        ev = evs.get(a["label"])
        tr = a["train"]
        tot = tr["trainable"]["total"]
        row = None
        if ev:
            row = {"mode": a["mode"], "n": ev["n"], "success": ev["success"],
                   "wilson95": ev["wilson95"], "outcomes": ev["outcomes"],
                   "peak_force_mean": ev["peak_force_mean"],
                   "peak_force_max": ev.get("peak_force_max"),
                   "chunk_ms": ev.get("chunk_ms")}  # fmt: skip
            rows[a["label"]] = row
            per_task[a["label"]] = {"n": ev["n"], "success": ev["success"],
                                    "wilson95": ev["wilson95"],
                                    "aborts": ev["outcomes"].get("force_abort", 0),
                                    "peak_force_mean": ev["peak_force_mean"]}  # fmt: skip
            for e in ev["episodes"]:
                if "trace" in e and len(traces) < MAX_TRACES:
                    traces.append({"task": task, "group": a["label"], **{k: v for k, v in
                                   e.items() if k != "trace"}, **e["trace"]})  # fmt: skip
        lora = tr.get("lora") or {}
        lora_arms.append({
            "label": a["label"], "mode": a["mode"], "trainable": tr["trainable"]["trainable"],
            "total": tot, "trainable_pct": round(100 * tr["trainable"]["trainable"] / tot, 4),
            "groups": tr["trainable"].get("groups"),
            "peak_vram_gb": tr["peak_alloc_gb"], "peak_reserved_gb": tr.get("peak_reserved_gb"),
            "steps_per_s": tr["steps_per_s"], "adapter_mb": _mb(tr["adapter_bytes"]),
            "full_ckpt_mb": _mb(tr["full_ckpt_bytes"]), "final_loss": round(tr["final_loss"], 4),
            "first_loss": round(tr.get("first_loss", float("nan")), 4),
            "success": row["success"] if row else None,
            "wilson95": row["wilson95"] if row else None,
            "r": lora.get("r"), "alpha": lora.get("alpha"), "lora_layers": lora.get("n_layers"),
            "lr": tr["lr"], "steps": tr["steps"], "batch": tr["batch"],
        })  # fmt: skip
    b = next((a for a in arms if a["mode"] == "lora"), None)
    c = next((a for a in arms if a["mode"] == "expert_full"), None)
    spec = lora_spectrum(b["train"] if b else None, c["train"] if c else None)
    all_eps = [e for ev in evs.values() for e in ev["episodes"]]
    k = sum(e["success"] for e in all_eps)
    result = {
        "task": task, "episodes": episodes, "n_episodes": len(all_eps), "fps": FPS,
        "video": video, "video_seed": video_seed,
        "claims_task_note": "the M5 claims are judged on the task pre-registered in "
                            "outputs/lora_vla/PREREG.txt (T3; T0 if every arm is <= 5 % on "
                            "T3) - this result fills them for its own task",
        "arms": rows, "lora_arms": lora_arms, "probe_full": ck.get("probe_full") or {},
        "lora_spectrum": spec,
        "merge_check": (b["train"].get("merge_check") if b else None),
        "bf16_lost_update_frac": (c["train"].get("bf16_lost_update_frac") if c else None),
        "lora_targets": ((b["train"].get("lora") or {}).get("targets") if b else None),
        "head_params": (arms[0]["train"].get("head_params") if arms else None),
        "outcomes": {"success": round(k / max(1, len(all_eps)), 4),
                     "counts": dict(Counter(e["outcome"] for e in all_eps)),
                     "wilson95": wilson95(k, len(all_eps)), "per_task": per_task},
        "peak_force": {"mean": round(float(np.mean([e["peak_force"] for e in all_eps])), 3)
                       if all_eps else 0.0,
                       "max": round(float(np.max([e["peak_force"] for e in all_eps])), 3)
                       if all_eps else 0.0},
        "episodes_traces": traces, "params": ck.get("params", {}),
    }  # fmt: skip
    result["claims"] = claims_rows(lora_arms, rows, result)
    return result


def lora_spectrum(b: dict | None, c: dict | None) -> dict | None:
    """Singular values of B's ΔW = (alpha/r) B@A and of C's ΔW = W_after - W_before, per layer."""
    if not b and not c:
        return None
    sb, sc = (b or {}).get("spectrum") or {}, (c or {}).get("spectrum") or {}
    r = int(((b or {}).get("lora") or {}).get("r") or 16)
    layers = []
    for name in list(dict.fromkeys([*sb, *sc])):
        row = {"name": name, "shape": (sb.get(name) or sc.get(name))["shape"]}
        if name in sb:
            row["lora"] = {"sv": sb[name]["sv"][: r + 8], "energy_top_r": sb[name]["energy_top_r"]}
        if name in sc:
            row["full"] = {"sv": sc[name]["sv"], "energy_top_r": sc[name]["energy_top_r"],
                           "energy_top_16": sc[name]["energy_top_16"]}  # fmt: skip
        layers.append(row)
    full = [L["full"]["energy_top_16"] for L in layers if "full" in L]
    return {"r": r, "layers": layers,
            "full_mean_energy_top16": float(np.mean(full)) if full else None}  # fmt: skip


def claims_rows(lora_arms: list[dict], rows: dict, result: dict) -> list[dict]:
    by = {a["mode"]: a for a in lora_arms}
    A, B, C = by.get("head"), by.get("lora"), by.get("expert_full")
    filled: dict[str, dict] = {}

    def ok(x):
        return x is not None and x.get("success") is not None

    if B:
        frac = B["trainable"] / B["total"]
        filled["m5.lora_param_fraction"] = {
            "measured": f"B trains {B['trainable']:,} of {B['total']:,} parameters "
            f"({100 * frac:.2f} %): {B.get('lora_layers')} LoRA layers r={B['r']} + the "
            "projections",
            "verdict": verdict("a <= 0.02", {"a": frac}),
        }
    if ok(B) and ok(C):
        ci = (B["wilson95"][0] - C["wilson95"][1], B["wilson95"][1] - C["wilson95"][0])
        filled["m5.lora_matches_expert_ft"] = {
            "measured": f"B lora {B['success']:.0%} vs C expert_full {C['success']:.0%} "
            f"(gap {B['success'] - C['success']:+.0%}, CI {ci[0]:+.1%}..{ci[1]:+.1%})",
            "verdict": verdict("a-b >= -0.10", {"a": B["success"], "b": C["success"]}, ci=ci),
        }
    if ok(B) and ok(A):
        ci = (B["wilson95"][0] - A["wilson95"][1], B["wilson95"][1] - A["wilson95"][0])
        filled["m5.lora_beats_head_only"] = {
            "measured": f"B lora {B['success']:.0%} vs A head {A['success']:.0%} "
            f"(gap {B['success'] - A['success']:+.0%}, CI {ci[0]:+.1%}..{ci[1]:+.1%})",
            "verdict": verdict("a-b >= 0.10", {"a": B["success"], "b": A["success"]}, ci=ci),
        }
    if B and C and C.get("peak_vram_gb"):
        ratio = B["peak_vram_gb"] / C["peak_vram_gb"]
        filled["m5.lora_memory"] = {
            "measured": f"peak allocated {B['peak_vram_gb']:.2f} GB (B) vs "
            f"{C['peak_vram_gb']:.2f} GB (C) at batch {B['batch']} (ratio {ratio:.2f})",
            "verdict": verdict("a <= 0.6", {"a": ratio}),
        }
    d = (result.get("probe_full") or {}).get("fp32")
    if d and d.get("fits") is not None:
        gb = float(d.get("peak_alloc_gb") or 0.0)
        val = GPU_GB if not d["fits"] else gb
        bf = (result.get("probe_full") or {}).get("bf16") or {}
        btxt = ""
        if bf.get("fits") is not None:
            btxt = ("; in lerobot's as-loaded bf16 (no fp32 master): "
                    + (f"fits, {bf.get('peak_alloc_gb')} GB" if bf["fits"] else "OOM"))  # fmt: skip
        filled["m5.full_ft_does_not_fit"] = {
            "measured": (f"D (every parameter, fp32 master + AdamW) at batch {d.get('batch')}: "
                         + (f"fits, peak {gb:.2f} GB allocated / {d.get('peak_reserved_gb')} GB "
                            "reserved" if d["fits"]
                            else f"CUDA OOM after {gb:.2f} GB allocated")) + btxt,
            "verdict": verdict(f"a >= {D_LIMIT_GB}", {"a": val}),
        }  # fmt: skip
    spec = result.get("lora_spectrum") or {}
    if spec.get("full_mean_energy_top16") is not None:
        per = ", ".join(f"{L['name'].split('lm_expert.')[-1]} {L['full']['energy_top_16']:.2f}"
                        for L in spec["layers"] if "full" in L)  # fmt: skip
        filled["m5.update_is_low_rank"] = {
            "measured": f"C's ΔW = W_after - W_before: top-16 singular values carry "
            f"{spec['full_mean_energy_top16']:.2f} of ||ΔW||_F^2 on average ({per})",
            "verdict": verdict("a >= 0.5", {"a": spec["full_mean_energy_top16"]}),
        }
    mc = result.get("merge_check")
    if mc and mc.get("fp32_max_abs") is not None:
        filled["m5.merge_is_exact"] = {
            "measured": f"max |v_adapter - v_merged| = {mc['fp32_max_abs']:.2e} in fp32 on a real "
            f"batch (|v| mean {mc.get('fp32_out_abs_mean', float('nan')):.2f}); merged into bf16 "
            f"weights instead: {mc.get('bf16_max_abs', float('nan')):.2e} (rounding)",
            "verdict": verdict("a <= 1e-4", {"a": mc["fp32_max_abs"]}),
        }
    out = []
    for c in M5_CLAIMS:
        f = filled.get(c.id)
        out.append({
            "id": c.id, "statement": c.statement, "metric": c.metric, "expected": c.expect,
            "measured": f["measured"] if f else "not measured in this run",
            "verdict": f["verdict"] if f else NOT_MEASURED, "source": c.source, "url": c.url,
        })  # fmt: skip
    return out


# ---------------------------------------------------------------------------------- estimate
def estimate(params: dict) -> float:
    cal = registry.calib("lora-vla")
    sps = {**FALLBACK_SAMPLES_PER_S, **(cal.get("samples_per_s") or {})}
    load = cal.get("load_s", FALLBACK_LOAD_S)
    modes = [params["mode"]] if params["preset"] == "single" else [m for _, _, m in SUITE]
    t = sum(int(params["steps"]) * int(params["batch"]) / sps[m] + load for m in modes)
    if params["preset"] == "claims_suite" or params.get("probe_full"):
        t += 2 * cal.get("probe_s", FALLBACK_PROBE_S)
    return t + cal.get("overhead_s", OVERHEAD_S)


def check(params: dict) -> None:
    if params["preset"] == "single" and params["mode"] not in TRAIN_MODES:
        raise ValueError(f"mode must be one of {TRAIN_MODES}")
    if int(params["r"]) not in RANKS:
        raise ValueError(f"rank r must be one of {RANKS}")


SPEC = TaskSpec(
    id="lora-vla",
    order=20,
    title="LoRA: fine-tuning a large VLA on a small budget",
    subtitle="SmolVLA (450 M) on the course's demos: train only the heads, LoRA adapters "
             "(~1 % of the weights) or the whole action expert - success, memory, adapter size, "
             "and the spectrum of what full fine-tuning actually changed.",  # fmt: skip
    params=[
        Param("preset", "Preset", "choice", "single", choices=list(PRESETS),
              help="single = one arm; claims_suite = A head, B lora, C expert_full in one job "
                   "(+ the D 'train everything' memory probe)"),
        Param("mode", "Mode (single preset)", "choice", "lora", choices=list(TRAIN_MODES),
              help="head = new-embodiment projections only; lora = head + LoRA on every "
                   "attention q/k/v/o of the VLM and the action expert; expert_full = lerobot's "
                   "recipe (VLM frozen, whole action expert)"),
        Param("r", "LoRA rank r", "choice", 16, choices=RANKS,
              help="adapter rank: parameters per layer = r x (in + out)"),
        Param("alpha", "LoRA alpha", "float", 32.0, min=1.0, max=128.0,
              help="ΔW = (alpha / r) B A; alpha = 2r is the common choice"),
        Param("task", "Task", "choice", "t3", choices=list(TASK_NAMES),
              help="t3 = peg insertion (hole hidden up to +-4 mm off the marker); t0 = reach "
                   "& grasp (visible) - the pre-registered fallback when every arm is <= 5 % "
                   "on t3"),
        Param("steps", "Training steps per arm", "int", 5000, min=10, max=50000, step=10,
              long=True),
        Param("batch", "Batch size", "int", 16, min=1, max=64),
        Param("lr", "Peak learning rate (0 = mode default)", "float", 0.0, min=0.0, max=1e-2,
              help="0 = head 1e-3, lora 2e-4, expert_full 1e-4 (SmolVLA's preset); warmup "
                   "min(1000, steps/10), cosine to 2.5 %"),
        Param("demos", "Demos (first n episodes)", "int", 200, min=8, max=400, step=4),
        Param("seed", "Seed", "int", 0, min=0, max=9999,
              help="init + batch sampling; suite arms use seed + 101*i"),
        Param("probe_full", "Measure D (train everything) first", "bool", False,
              help="the suite always measures it; single runs only when ticked"),
    ],  # fmt: skip
    eval_params=[
        Param("episodes", "Shared held-out seeds", "int", 100, min=2, max=500,
              help="every arm rolls on the same seeds"),
        Param("eval_seed", "Eval seed", "int", 0, min=0, max=99,
              help="shifts the window inside the task's eval-seed block"),
        Param("video", "Render the same-seed video", "bool", True),
        Param("parallel", "Eval workers in parallel", "int", 3, min=1, max=3,
              help="one SmolVLA per worker (~1 GB VRAM each); the sim is CPU-bound"),
    ],  # fmt: skip
    charts=[
        {"id": "loss", "title": "Flow-matching loss per arm (arm-local steps)",
         "series": [{"key": "loss", "label": "single"}, {"key": "a_loss", "label": "A head"},
                    {"key": "b_loss", "label": "B lora"},
                    {"key": "c_loss", "label": "C expert_full"}], "log_y": True},
        {"id": "vram", "title": "Peak GPU memory allocated (GB)",
         "series": [{"key": "vram_gb", "label": "single"}, {"key": "a_vram_gb", "label": "A head"},
                    {"key": "b_vram_gb", "label": "B lora"},
                    {"key": "c_vram_gb", "label": "C expert_full"}], "log_y": False},
        {"id": "speed", "title": "Training speed (steps/s)",
         "series": [{"key": "steps_per_s", "label": "single"},
                    {"key": "a_steps_per_s", "label": "A head"},
                    {"key": "b_steps_per_s", "label": "B lora"},
                    {"key": "c_steps_per_s", "label": "C expert_full"}], "log_y": False},
        {"id": "lr", "title": "Learning rate",
         "series": [{"key": "lr", "label": "single"}, {"key": "a_lr", "label": "A head"},
                    {"key": "b_lr", "label": "B lora"}, {"key": "c_lr", "label": "C expert_full"}],
         "log_y": True},
    ],  # fmt: skip
    viewer="fvrollout",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="lora-vla.html",
    check=check,
)  # fmt: skip
registry.register(SPEC)
