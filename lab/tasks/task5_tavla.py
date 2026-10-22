"""Task "TA-VLA mini" (id `ta-vla`): a two-stream transformer policy with the TA-VLA torque token,
trained on the 2-D contact world with four commands (T4 of the tutorial, `docs/ta_tutorial.md`).

Env: `contact2d.t4_scenes` / `collect("t4", ...)`: two holes (red / blue), four commands
(insert red, insert blue, press gently 2 N, press firmly 6 N).
The network is `fvb.ta_tutorial.nets.TAVLAMini`:
  prefix  [16 image-patch tokens | 5 language tokens | state token (peg x, z)]     "VLM" weights
  suffix  [torque token (10 x 2 force history) | 8 noisy action tokens (+ time)]  "action expert"
The hole positions are NOT given as numbers: the policy must read them from a 32x32 side-view
image. Following `scripts/40_ta_tutorial.py::RenderingTAVLA` the dataset stores the 5-number
`scene` per frame and the image is rendered on the fly inside `encode_prefix` (train and eval),
so memory stays small and the policy's `make_cond` passes `scene`, not pixels.

Params -> model: `torque_pos` none|prefix|suffix, `zero_torque` (token input zeroed: the
no-force twin, same parameters), `aux_weight` (TA-VLA's auxiliary head: future 8 force readings,
`aux_dim` 16, `L = L_action + w * L_torque`; 0 = no head), `d` 64|128 (layers 4, heads 4).

Checkpoint (`last.pt`) = the `train_loop` dict {"model", "ema", "norm", "arch", "params", "gen"};
`arch` is every TAVLAMini constructor argument (act_dim, horizon, state_dim, torque_dim,
torque_pos, zero_torque, aux_dim, image_hw, vocab, lang_len, d, layers, heads) + "n_exec";
`evaluate` rebuilds the model from it alone.

result.json written by `evaluate`:
    {"n_episodes", "success", "counts", "wilson95", "tasks", "sample_steps", "sensor_noise",
     "torque_pos", "trained_with_force", "force_zeroed", "commands": [{"task", "name", "n",
     "success", "wilson95", "counts"}] (4 rows, n=0 when filtered out),
     "episodes": [{"z0", "holes" (2), "target" (-1 = press), "pos", "force", "act", "outcome",
                   "task" (name), "task_id"}]}  # <= 16
Field mapping for the viewer's force chip: "force" = the torque token's input. `trained_with_force`
is false when trained with `zero_torque` or `torque_pos=none`; `force_zeroed` is true when the
token input was zeroed at evaluation (the eval param `blind`, or a twin trained blind).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from fvb.ta_tutorial import contact2d as c2d
from fvb.ta_tutorial.data import ChunkDataset, split_episodes
from fvb.ta_tutorial.nets import TAVLAMini
from fvb.ta_tutorial.train import ChunkPolicy
from lab.core.loop import train_loop
from lab.core.runctx import atomic_write_json
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.task1_bc_mse import OVERHEAD_S, _device, _Norm, wilson95
from lab.tasks.task3_flow_bc import make_gen

# chunk 8, re-plan every 4 (as the tutorial's T4; a 16/8 chunk plateaued ~20 points lower)
HORIZON, N_EXEC, LAYERS, HEADS, AUX_DIM = 8, 4, 4, 4, 16
OUTCOMES = ("success", "force_abort", "wrong_hole", "timeout")
EXPERT_NOISE, TRAIN_K, TRAIN_EVAL_N, MAX_VIEW_EPISODES = 0.05, 10, 200, 16
FALLBACK_STEPS_PER_S, FALLBACK_COLLECT_S_PER_EP, FALLBACK_EVAL_S = 30.0, 0.05, 15.0
TASK_FILTER = {"all": (0, 1, 2, 3), "insert": (0, 1), "press": (2, 3)}


class RenderingTAVLA(TAVLAMini):
    """Same as `RenderingTAVLA` in scripts/40_ta_tutorial.py (that file is not importable): renders
    the camera image from `cond["scene"]` on the model's device inside the prefix encoder."""

    def encode_prefix(self, cond):
        if "image" not in cond:
            cond = dict(cond, image=c2d.render(cond["scene"]))
        return super().encode_prefix(cond)


def make_arch(P: dict) -> dict:
    return {
        "act_dim": 2, "horizon": HORIZON, "state_dim": 2, "torque_dim": 2 * c2d.HIST,
        "torque_pos": P["torque_pos"], "zero_torque": bool(P["zero_torque"]),
        "aux_dim": AUX_DIM if float(P["aux_weight"]) > 0 else 0, "image_hw": 32,
        "vocab": len(c2d.VOCAB), "lang_len": c2d.LANG_LEN, "d": int(P["d"]), "layers": LAYERS,
        "heads": HEADS, "n_exec": N_EXEC,
    }  # fmt: skip


def build(arch: dict) -> RenderingTAVLA:
    kw = {k: v for k, v in arch.items() if k != "n_exec"}
    return RenderingTAVLA(**kw)


def _trained_with_force(arch: dict) -> bool:
    return arch["torque_pos"] != "none" and not arch["zero_torque"]


def _load(ck: dict, dev: str):
    arch = ck["arch"]
    m = build(arch)
    m.load_state_dict(ck["ema"])
    norm = {k: (a.to(dev), s.to(dev)) for k, (a, s) in ck["norm"].items()}
    return m.to(dev).eval(), _Norm(arch["horizon"], norm, dev), arch


def _rollout(model, ds_like, n, seed, n_exec, K, sensor_noise, tasks=(0, 1, 2, 3), record=0):
    env = c2d.t4_scenes(n, seed, tasks)
    env.noise = sensor_noise
    dev = ds_like.device

    def make_cond(o):
        return {k: torch.as_tensor(o[k], device=dev) for k in ("state", "torque", "scene", "lang")}

    pol = ChunkPolicy(model, make_gen(), ds_like, make_cond, n_exec, K, seed=seed)
    record = min(record, n)
    pos = [[env.pos[i].tolist()] for i in range(record)]
    force = [[[0.0, 0.0]] for _ in range(record)]
    act = [[] for _ in range(record)]
    while not env.done.all():
        live = ~env.done.copy()
        a = pol.act(c2d.t4_obs(env))
        env.step(a)
        for i in range(record):
            if live[i]:
                act[i].append(np.asarray(a[i], float).tolist())
                pos[i].append(env.pos[i].tolist())
                force[i].append(env.hist[i, -1].tolist())
    return env, pos, force, act


def _per_task(env) -> dict:
    return {f"c{k}": float(env.success[env.task == k].mean()) if (env.task == k).any() else 0.0
            for k in range(4)}  # fmt: skip


def train(ctx) -> None:
    P = ctx.params
    steps, demos, seed = int(P["steps"]), int(P["demos"]), int(P.get("train_seed", 0))
    batch, lr, aux_w = int(P["batch"]), float(P["lr"]), float(P["aux_weight"])
    eval_every = int(P.get("eval_every", 1000))
    dev = _device()
    arch = make_arch(P)
    eps = c2d.collect("t4", demos, seed * 1000 + 1, EXPERT_NOISE)
    ok = [e for e in eps if e["success"]]
    ctx.set_status(message=f"demonstrations: {len(ok)} of {demos} episodes succeeded")
    if len(ok) < 2:
        ctx.fail(f"only {len(ok)} successful demonstrations out of {demos}")
        return
    keep = ("action", "force", "state", "torque", "scene", "lang")
    tr, va = split_episodes([{k: e[k] for k in keep} for e in ok], 0.1, seed)
    ds = ChunkDataset(tr, HORIZON, device=dev)
    vds = ChunkDataset(va, HORIZON, device=dev, norm=ds.norm)
    model = build(arch)

    def eval_fn(m):
        env, *_ = _rollout(m, ds, TRAIN_EVAL_N, 10_000 + seed, N_EXEC, TRAIN_K, 0.05)
        return {"success": float(env.success.mean()), **_per_task(env)}

    done = train_loop(
        ctx, model, make_gen(), ds, vds, steps=steps, batch=batch, lr=lr,
        warmup=max(1, min(500, steps // 10)), aux_weight=aux_w, eval_every=eval_every,
        ckpt_every=max(1, steps // 4), eval_fn=eval_fn, log_every=max(1, min(50, steps // 4)),
        seed=seed, arch=arch,
    )  # fmt: skip
    if not done:
        ctx.set_status(state="stopped")
        return
    ctx.done(f"trained on {len(ok)} successful demos")


def evaluate(ctx) -> None:
    P = ctx.params
    n, seed = int(P.get("n_episodes", 400)), int(P.get("seed", 0))
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    dev = _device()
    model, ds_like, arch = _load(ck, dev)
    twf = _trained_with_force(arch)
    blind = bool(P.get("blind", False))
    if blind:
        model.zero_torque = True  # same parameters, token input zeroed
    flt = P.get("tasks", "all")
    K = int(P.get("eval_steps") or 8)
    noise = float(P.get("sensor_noise", 0.05))
    env, pos, force, act = _rollout(model, ds_like, n, 20_000 + seed, int(arch["n_exec"]), K,
                                    noise, TASK_FILTER[flt], MAX_VIEW_EPISODES)  # fmt: skip
    k = int(env.success.sum())
    cnt = Counter(env.outcome.tolist())
    commands = []
    for t in range(4):
        m = env.task == t
        nt, kt = int(m.sum()), int(env.success[m].sum())
        ct = Counter(env.outcome[m].tolist())
        commands.append({
            "task": t, "name": c2d.TASKS[t], "n": nt, "success": kt / nt if nt else 0.0,
            "wilson95": wilson95(kt, nt), "counts": {o: int(ct.get(o, 0)) for o in OUTCOMES},
        })  # fmt: skip
    r4 = lambda v: np.round(np.asarray(v, float), 4).tolist()  # noqa: E731
    episodes = [
        {
            "z0": r4(env.z0[i]), "holes": r4(env.holes[i]), "target": int(env.target[i]),
            "pos": r4(pos[i]), "force": r4(force[i]),
            "act": np.round(np.asarray(act[i], float), 3).tolist(),
            "outcome": str(env.outcome[i]), "task": c2d.TASKS[int(env.task[i])],
            "task_id": int(env.task[i]),
        }
        for i in range(min(n, MAX_VIEW_EPISODES))
    ]  # fmt: skip
    atomic_write_json(
        ctx.dir / "result.json",
        {
            "n_episodes": n, "success": k / n, "counts": {o: int(cnt.get(o, 0)) for o in OUTCOMES},
            "wilson95": wilson95(k, n), "tasks": flt, "sample_steps": K, "sensor_noise": noise,
            "torque_pos": arch["torque_pos"], "trained_with_force": twf,
            "force_zeroed": blind or not twf, "commands": commands, "episodes": episodes,
        },
    )  # fmt: skip
    ctx.done("evaluated")


def estimate(params: dict) -> float:
    cal = registry.calib("ta-vla")
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    if int(params.get("d", 128)) == 64:
        sps *= cal.get("d64_speedup", 1.0)
    collect = cal.get("collect_s_per_ep") or FALLBACK_COLLECT_S_PER_EP
    overhead = cal.get("overhead_s", OVERHEAD_S)
    n_evals = max(1, -(-int(params["steps"]) // max(1, int(params.get("eval_every", 1000)))))
    return (params["steps"] / sps + n_evals * cal.get("eval_s", FALLBACK_EVAL_S)
            + params["demos"] * collect + overhead)  # fmt: skip


SPEC = TaskSpec(
    id="ta-vla",
    order=6,
    title="TA-VLA mini",
    subtitle="A two-stream transformer with a torque token: image + words + state in, actions out.",
    params=[
        Param("demos", "Demonstrations", "int", 800, min=200, max=3000, step=100),
        Param("steps", "Training steps", "int", 15000, min=2000, max=60000, step=1000, long=True),
        Param("torque_pos", "Torque token position", "choice", "suffix",
              choices=["none", "prefix", "suffix"]),
        Param("zero_torque", "Zero the torque input (twin)", "bool", False),
        Param("aux_weight", "Aux future-torque loss weight w", "float", 0.1, min=0.0, max=0.5,
              step=0.05),
        Param("lr", "Learning rate", "float", 3e-4, min=1e-4, max=3e-3),
        Param("batch", "Batch size", "int", 256, min=32, max=512, step=32),
        Param("d", "Model width d", "choice", 128, choices=[64, 128]),
        Param("eval_every", "Closed-loop eval every N steps", "int", 1000, min=100, max=10000,
              step=100),
        Param("train_seed", "Training seed", "int", 0, min=0, max=40),
    ],
    eval_params=[
        Param("n_episodes", "Episodes", "int", 400, min=50, max=2000, step=50),
        Param("tasks", "Commands", "choice", "all", choices=["all", "insert", "press"]),
        Param("blind", "Blind the torque token at eval (zero its input)", "bool", False),
        Param("sensor_noise", "Force sensor noise", "float", 0.05, min=0.0, max=0.2, step=0.01),
        Param("eval_steps", "Sampling steps K", "choice", 8, choices=[1, 2, 4, 8, 16, 32]),
        Param("seed", "Seed", "int", 0, min=0, max=9999),
    ],
    charts=[
        {"id": "loss", "title": "Flow loss (train / held-out)",
         "series": [{"key": "loss", "label": "train"}, {"key": "val_loss", "label": "val"}],
         "log_y": True},
        {"id": "aux", "title": "Aux future-torque MSE (when w > 0)",
         "series": [{"key": "aux", "label": "aux"}], "log_y": True},
        {"id": "success", "title": "Closed-loop success per command (200 fixed scenes)",
         "series": [{"key": "success", "label": "all"}, {"key": "c0", "label": "insert red"},
                    {"key": "c1", "label": "insert blue"}, {"key": "c2", "label": "press gently"},
                    {"key": "c3", "label": "press firmly"}], "log_y": False},
        {"id": "lr", "title": "Learning rate", "series": [{"key": "lr", "label": "lr"}],
         "log_y": False},
    ],
    viewer="tavla",
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="ta-vla.html",
)  # fmt: skip
registry.register(SPEC)
