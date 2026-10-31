"""Task "Embeddings & conditioning" (id `embeddings`): one command-conditioned flow policy, four
ways to tell it which command to follow.

Env: `contact2d.t4_scenes` / `collect("t4", ...)`: two holes (red = hole 0, blue = hole 1) and four
commands (`contact2d.TASKS`: insert red, insert blue, press gently 2 N, press firmly 6 N). There is
no camera. The network input is [state | torque | encoding]:
  state   = peg (x, z) and, with `hole_obs` (default on), the *perceived* hole xs (red, blue):
            the true hole + N(0, EST_SIGMA) drawn once per episode - a stand-in for a perception
            module, so every arm can find its hole and only the COMMAND encoding differs.
            (`hole_obs` off: only `cont` knows where the holes are - a confound, kept as a knob.)
  torque  = the last 10 force readings (20)
  enc     = onehot (4)   one-hot of the command id
            binary (2)   the id's two bits
            text   (d)   `contact2d.tokens(task)` (LANG_LEN ids) -> nn.Embedding(len(VOCAB), d),
                         masked mean over the real tokens; trained jointly with the policy - the
                         miniature of TAVLAMini's `nn.Embedding` language path (nets.py)
            cont   (3)   [perceived target hole x (0 for press), target force N (0 for insert),
                         is_insert]: the physically meaningful code
The MLPDenoiser is the same in all arms; the text arm wraps it in `CondDenoiser` (embedding table +
MLP) so the embedding is part of the checkpointed model. The four arms train sequentially in one
run on the SAME demonstrations, the same train/val split and the same fixed closed-loop scenes,
with metric prefixes `onehot_ / binary_ / text_ / cont_`.

Normalisation (itself a lesson point): `state` and `torque` are z-scored by ChunkDataset as usual.
onehot / binary / text pass through UNNORMALISED (they are categorical: a z-score would turn
{0, 1} into values that depend on how often a command occurs, and ids/embeddings have no scale).
`cont` carries physical scales (metres, newtons, a flag) and IS z-scored, via ChunkDataset's `obs`
statistics. Dataset keys: onehot/binary -> `scene` (a key ChunkDataset never normalises), text ->
`lang`, cont -> `obs`.

Checkpoint payload (`last.pt`), like task 1's multi-part layout:
    {<arm>: P, ..., "arms": [trained arms in order], "arch": {arm: arch}, "gen": "flow",
     "params": ...}
with P = the train_loop dict {"model", "ema", "norm", "arch", "params", "gen"} plus "encoding": arm
and "score" (last closed-loop success). arch = {"act_dim", "horizon", "state_dim", "hidden",
"depth", "enc", "d_embed", "hole_obs", "n_exec", "est_sigma"}.

result.json written by `evaluate`:
    {"n_episodes", "success", "counts", "wilson95", "encoding", "arms", "tasks", "sample_steps",
     "est_sigma", "sensor_noise", "hole_obs",
     "commands": [{"task", "name", "n", "success", "wilson95", "counts"}] (4 rows, n=0 if filtered),
     "encodings": {"commands": [names], "arms": [...], "labels": {arm: [dim names]},
                   "vectors": {arm: [4 x dim]}, "tokens": [4 x LANG_LEN], "note"},
     "episodes": [{"z0", "holes" (2), "target" (-1 = press), "est" (perceived holes, 2), "pos",
                   "force", "act", "outcome", "task" (name), "task_id", "enc"}]}  # <= 16
`enc` is the vector the network consumed (raw; for text the pooled learned embedding).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch import nn

from fvb.ta_tutorial import contact2d as c2d
from fvb.ta_tutorial.data import ChunkDataset, split_episodes
from fvb.ta_tutorial.nets import MLPDenoiser
from fvb.ta_tutorial.train import ChunkPolicy
from lab.core.loop import train_loop
from lab.core.runctx import atomic_write_json
from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec
from lab.tasks.task1_bc_mse import DEPTH, HIDDEN, OVERHEAD_S, _device, _Norm, wilson95
from lab.tasks.task3_flow_bc import make_gen

ARMS = ("onehot", "binary", "text", "cont")
ENC_KEY = {"onehot": "scene", "binary": "scene", "text": "lang", "cont": "obs"}
OUTCOMES = ("success", "force_abort", "wrong_hole", "timeout")
MAX_VIEW_EPISODES, TRAIN_EVAL_N = 16, 200
EXPERT_NOISE, EST_SIGMA = 0.05, 0.01
NOMINAL_X = 0.05  # hole x shown in the cont vector of the encodings block
FALLBACK_STEPS_PER_S, FALLBACK_COLLECT_S_PER_EP, FALLBACK_EVAL_S = 100.0, 0.05, 8.0
SHORT = ("insert red", "insert blue", "press gently", "press firmly")
TASK_FILTER = {"all": (0, 1, 2, 3), "insert": (0, 1), "press": (2, 3)}


def enc_dim(kind: str, d_embed: int) -> int:
    return {"onehot": 4, "binary": 2, "cont": 3, "text": d_embed}[kind]


def enc_labels(kind: str, d_embed: int) -> list[str]:
    return {
        "onehot": [f"is {s}" for s in SHORT],
        "binary": ["bit 0", "bit 1"],
        "cont": ["hole x (m)", "force (N)", "is insert"],
        "text": [f"e{i}" for i in range(d_embed)],
    }[kind]


def enc_raw(kind: str, task: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Raw conditioning array for command ids `task` (n,) and perceived hole xs `est` (n, 2):
    float32 (n, dim) for onehot/binary/cont, int64 token ids (n, LANG_LEN) for text."""
    # n = commands in the batch, L = LANG_LEN text tokens
    task = np.asarray(task, int)  # command ids [n]
    if kind == "onehot":
        return np.eye(4, dtype=np.float32)[task]  # row lookup: [n] -> [n, 4]
    if kind == "binary":
        return np.stack([task & 1, (task >> 1) & 1], 1).astype(np.float32)  # 2 x [n] -> [n, 2]
    if kind == "text":
        return np.array([c2d.tokens(int(t)) for t in task], np.int64)  # token ids [n, L]
    ins = task <= 1  # [n] bool: insert commands
    x = np.where(ins, est[np.arange(len(task)), np.minimum(task, 1)], 0.0)  # target hole x [n]
    f = np.array([c2d.PRESS_N.get(int(t), 0.0) for t in task])  # target force [n]
    return np.stack([x, f, ins.astype(float)], 1).astype(np.float32)  # 3 x [n] -> [n, 3]


class CondDenoiser(nn.Module):
    """MLPDenoiser over [state | torque | enc]; the text arm owns the embedding table (`emb`)."""

    def __init__(self, arch: dict):
        super().__init__()
        self.kind, self.key = arch["enc"], ENC_KEY[arch["enc"]]
        self.d = enc_dim(self.kind, arch["d_embed"])  # E = encoding dims fed to the net
        self.emb = None
        if self.kind == "text":
            # the text arm's word-embedding table: [V, E] with V = vocab size, E = d_embed
            self.emb = nn.Embedding(len(c2d.VOCAB), arch["d_embed"], padding_idx=0)
        self.net = MLPDenoiser(arch["act_dim"], arch["horizon"], arch["state_dim"] + 20 + self.d,
                               arch["hidden"], arch["depth"])  # fmt: skip

    def encode(self, e: torch.Tensor) -> torch.Tensor:
        # B = batch, L = text tokens, E = encoding dims; e [B, E] floats or [B, L] token ids
        if self.emb is None:
            return e.float()  # onehot / binary / cont pass through: [B, E]
        m = (e != 0).float().unsqueeze(-1)  # mean over the real tokens, not the padding: [B, L, 1]
        # lookup self.emb(e): ids [B, L] -> vectors [B, L, E]; masked mean pools over L
        return (self.emb(e) * m).sum(1) / m.sum(1).clamp_min(1.0)  # [B, L, E] -> [B, E]

    def forward(self, x_t, t, cond):
        # B = batch, S = state dims, E = encoding dims, O = obs dims, H = horizon, A = action dims
        # concat [B, S] + [B, 20] + [B, E] -> obs [B, O]
        obs = torch.cat([cond["state"], cond["torque"], self.encode(cond[self.key])], 1)
        return self.net(x_t, t, {"obs": obs})  # x_t [B, H, A] -> velocity [B, H, A]


def _build(arch: dict) -> CondDenoiser:
    return CondDenoiser(arch)


def _arch(horizon, n_exec, kind, d_embed, hole_obs) -> dict:
    return {"act_dim": 2, "horizon": horizon, "state_dim": 4 if hole_obs else 2, "hidden": HIDDEN,
            "depth": DEPTH, "enc": kind, "d_embed": d_embed, "hole_obs": hole_obs,
            "n_exec": n_exec, "est_sigma": EST_SIGMA}  # fmt: skip


def _state(pos: np.ndarray, est: np.ndarray, hole_obs: bool) -> np.ndarray:
    return (np.concatenate([pos, est], 1) if hole_obs else pos).astype(np.float32)


def _episodes(ok: list[dict], kind: str, hole_obs: bool, seed: int) -> list[dict]:
    """Demonstrations -> dataset episodes for one arm (the perception noise is seeded per episode
    index, so every arm sees the same perceived holes)."""
    out = []
    for j, e in enumerate(ok):
        T = len(e["action"])
        holes = e["scene"][0, 1:3]
        est = holes + np.random.default_rng(seed * 100003 + j).normal(0, EST_SIGMA, 2)
        est_t = np.tile(est, (T, 1))
        raw = enc_raw(kind, np.full(T, e["task"]), est_t)
        out.append({"action": e["action"], "torque": e["torque"],
                    "state": _state(e["state"], est_t, hole_obs), ENC_KEY[kind]: raw})  # fmt: skip
    return out


def _rollout(model, ds_like, kind, hole_obs, n, seed, n_exec, K, est_sigma, sensor_noise,
             tasks=(0, 1, 2, 3), record=0):  # fmt: skip
    env = c2d.t4_scenes(n, seed, tasks)
    env.noise = sensor_noise
    est = env.holes + np.random.default_rng(seed + 5).normal(0, est_sigma, env.holes.shape)
    dev = ds_like.device
    key = ENC_KEY[kind]

    def make_cond(o):
        # n = episodes, S = state dims (4 with hole_obs), E = encoding dims, L = text tokens
        raw = enc_raw(kind, env.task, est) if kind != "text" else o["lang"]  # [n, E] or ids [n, L]
        return {
            "state": torch.as_tensor(_state(o["state"], est, hole_obs), device=dev),  # [n, S]
            "torque": torch.as_tensor(o["torque"], device=dev),  # [n, 20]
            key: torch.as_tensor(raw, device=dev),  # [n, E]; text: token ids [n, L]
        }

    # the policy samples action chunks [n, H, 2] every n_exec steps (H = chunk horizon)
    pol = ChunkPolicy(model, make_gen(), ds_like, make_cond, n_exec, K, seed=seed)
    record = min(record, n)
    pos = [[env.pos[i].tolist()] for i in range(record)]
    force = [[[0.0, 0.0]] for _ in range(record)]
    act = [[] for _ in range(record)]
    while not env.done.all():
        live = ~env.done.copy()
        a = pol.act(c2d.t4_obs(env))  # obs -> the next action [n, 2]
        env.step(a)  # apply [n, 2]
        for i in range(record):
            if live[i]:
                act[i].append(np.asarray(a[i], float).tolist())
                pos[i].append(env.pos[i].tolist())
                force[i].append(env.hist[i, -1].tolist())
    return env, est, pos, force, act


def _per_command(env, key="success"):
    return {k: float(env.success[env.task == k].mean()) if (env.task == k).any() else 0.0
            for k in range(4)}  # fmt: skip


def train(ctx) -> None:
    P = ctx.params
    steps, demos = int(P["steps"]), int(P["demos"])
    horizon, seed = int(P["horizon"]), int(P.get("seed", 0))
    n_exec = max(1, min(int(P.get("n_exec", 8)), horizon))
    batch, lr, K = int(P["batch"]), float(P["lr"]), int(P.get("sample_steps", 8))
    d_embed, hole_obs = int(P.get("d_embed", 16)), bool(P.get("hole_obs", True))
    eval_every = int(P.get("eval_every", 1000))
    ckpt_every = int(P.get("ckpt_every", max(1, steps // 4)))
    enc = P.get("encodings", "all")
    arms = ARMS if enc == "all" else (enc,)
    dev = _device()
    gen = make_gen()

    eps = c2d.collect("t4", demos, seed * 1000 + 1, EXPERT_NOISE)
    ok = [e for e in eps if e["success"]]
    ctx.set_status(message=f"demonstrations: {len(ok)} of {demos} episodes succeeded")
    if len(ok) < 2:
        ctx.fail(f"only {len(ok)} successful demonstrations out of {demos}")
        return
    total = len(arms) * steps
    holder: dict = {}
    for i, arm in enumerate(arms):
        arch = _arch(horizon, n_exec, arm, d_embed, hole_obs)
        tr, va = split_episodes(_episodes(ok, arm, hole_obs, seed), 0.1, seed)
        ds = ChunkDataset(tr, horizon, device=dev)
        vds = ChunkDataset(va, horizon, device=dev, norm=ds.norm)
        model = _build(arch)

        def eval_fn(m, arm=arm, ds=ds):
            env, *_ = _rollout(m, ds, arm, hole_obs, TRAIN_EVAL_N, 10_000 + seed, n_exec, K,
                               EST_SIGMA, 0.05)  # fmt: skip
            per = _per_command(env)
            holder[arm + "_score"] = float(env.success.mean())
            return {f"{arm}_success": float(env.success.mean()),
                    **{f"{arm}_c{k}": v for k, v in per.items()}}  # fmt: skip

        def wrap(p, arm=arm):
            holder[arm] = {**p, "encoding": arm, "score": holder.get(arm + "_score", -1.0)}
            done = {a: holder[a] for a in arms if a in holder}
            return {**done, "arms": list(done), "arch": {a: done[a]["arch"] for a in done},
                    "gen": p["gen"], "params": ctx.params}  # fmt: skip

        ok_run = train_loop(
            ctx, model, gen, ds, vds, steps=steps, batch=batch, lr=lr, prefix=f"{arm}_",
            step_offset=i * steps, total_steps=total, warmup=max(1, min(500, steps // 10)),
            eval_every=eval_every, ckpt_every=ckpt_every, eval_fn=eval_fn,
            log_every=max(1, min(50, steps // 4)), seed=seed, arch=arch, payload_wrap=wrap,
            local_step_key="arm_step",
        )  # fmt: skip
        if not ok_run:
            ctx.set_status(state="stopped")
            return
    ctx.done(f"trained {len(arms)} encoding(s) on {len(ok)} successful demos")


def _load(ck: dict, arm: str, dev: str):
    arch = ck[arm]["arch"]
    m = _build(arch)
    m.load_state_dict(ck[arm]["ema"])
    norm = {k: (a.to(dev), s.to(dev)) for k, (a, s) in ck[arm]["norm"].items()}
    return m.to(dev).eval(), _Norm(arch["horizon"], norm, dev), arch


def _encodings_block(ck: dict, dev: str) -> dict:
    labels, vectors = {}, {}
    est = np.array([[NOMINAL_X, -NOMINAL_X]] * 4)
    tasks = np.arange(4)
    for arm in ck["arms"]:
        arch = ck[arm]["arch"]
        labels[arm] = enc_labels(arm, arch["d_embed"])
        raw = enc_raw(arm, tasks, est)
        if arm == "text":
            m, *_ = _load(ck, arm, dev)
            with torch.no_grad():
                raw = m.encode(torch.as_tensor(raw, device=dev)).cpu().numpy()
        vectors[arm] = np.round(raw.astype(float), 4).tolist()
    return {
        "commands": list(c2d.TASKS), "arms": list(ck["arms"]), "labels": labels,
        "vectors": vectors, "tokens": [c2d.tokens(t) for t in range(4)],
        "note": f"cont: x shown for a hole at {NOMINAL_X:+.2f} m; text: pooled learned embedding",
    }  # fmt: skip


def evaluate(ctx) -> None:
    P = ctx.params
    n, seed = int(P.get("n_episodes", 400)), int(P.get("seed", 0))
    ck = torch.load(P["ckpt"], map_location="cpu", weights_only=False)
    arms = list(ck.get("arms") or [])
    want = P.get("encoding", "best")
    if want == "best":
        want = max(arms, key=lambda a: ck[a].get("score", -1.0)) if arms else ""
    if want not in arms:
        ctx.fail(f"encoding {want!r} is not in this checkpoint (trained: {arms or 'none'})")
        return
    flt = P.get("tasks", "all")
    tasks = TASK_FILTER[flt]
    est_sigma = float(P.get("est_sigma", EST_SIGMA))
    sensor_noise = float(P.get("sensor_noise", 0.05))
    dev = _device()
    model, ds_like, arch = _load(ck, want, dev)
    K = int(P.get("eval_steps") or (ck.get("params") or {}).get("sample_steps") or 8)
    n_exec = max(1, min(int(arch.get("n_exec", 8)), arch["horizon"]))
    env, est, pos, force, act = _rollout(
        model, ds_like, want, arch["hole_obs"], n, 20_000 + seed, n_exec, K, est_sigma,
        sensor_noise, tasks, MAX_VIEW_EPISODES,
    )  # fmt: skip
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
    m_ep = min(n, MAX_VIEW_EPISODES)
    raw = enc_raw(want, env.task[:m_ep], est[:m_ep])
    with torch.no_grad():
        enc_used = model.encode(torch.as_tensor(raw, device=dev)).cpu().numpy()
    episodes = [
        {
            "z0": r4(env.z0[i]), "holes": r4(env.holes[i]), "target": int(env.target[i]),
            "est": r4(est[i]), "pos": r4(pos[i]), "force": r4(force[i]),
            "act": np.round(np.asarray(act[i], float), 3).tolist(),
            "outcome": str(env.outcome[i]), "task": c2d.TASKS[int(env.task[i])],
            "task_id": int(env.task[i]), "enc": r4(enc_used[i]),
        }
        for i in range(m_ep)
    ]  # fmt: skip
    atomic_write_json(
        ctx.dir / "result.json",
        {
            "n_episodes": n, "success": k / n, "counts": {o: int(cnt.get(o, 0)) for o in OUTCOMES},
            "wilson95": wilson95(k, n), "encoding": want, "arms": arms, "tasks": flt,
            "sample_steps": K, "est_sigma": est_sigma, "sensor_noise": sensor_noise,
            "hole_obs": bool(arch["hole_obs"]), "commands": commands,
            "encodings": _encodings_block(ck, dev), "episodes": episodes,
        },
    )  # fmt: skip
    ctx.done("evaluated")


def estimate(params: dict) -> float:
    cal = registry.calib("embeddings")
    sps = cal.get("steps_per_s") or FALLBACK_STEPS_PER_S
    collect = cal.get("collect_s_per_ep") or FALLBACK_COLLECT_S_PER_EP
    overhead = cal.get("overhead_s", OVERHEAD_S)
    eval_s = cal.get("eval_s", FALLBACK_EVAL_S) * int(params.get("sample_steps", 8)) / 8
    n_arms = 4 if params.get("encodings", "all") == "all" else 1
    n_evals = max(1, -(-int(params["steps"]) // max(1, int(params.get("eval_every", 1000)))))
    train_s = n_arms * (params["steps"] / sps + n_evals * eval_s)
    return train_s + params["demos"] * collect + overhead


SPEC = TaskSpec(
    id="embeddings",
    order=3,
    title="Embeddings & conditioning",
    subtitle="Tell one policy four commands with a one-hot, two bits, learned word embeddings, or "
             "physical numbers.",
    params=[
        Param("demos", "Demonstrations", "int", 600, min=200, max=3000, step=100),
        Param("steps", "Training steps per encoding", "int", 6000, min=1000, max=30000, step=1000),
        Param("lr", "Learning rate", "float", 1e-3, min=1e-4, max=3e-3),
        Param("batch", "Batch size", "int", 256, min=64, max=512, step=64),
        Param("horizon", "Action chunk", "choice", 16, choices=[8, 16]),
        Param("n_exec", "Actions executed per chunk", "choice", 8, choices=[4, 8, 16]),
        Param("d_embed", "Text embedding size", "choice", 16, choices=[8, 16, 32]),
        Param("encodings", "Encodings to train", "choice", "all",
              choices=["all", "onehot", "binary", "text", "cont"]),
        Param("hole_obs", "Tell every arm where the holes are", "bool", True),
        Param("sample_steps", "Sampling steps K", "choice", 8, choices=[1, 2, 4, 8, 16, 32]),
        Param("eval_every", "Closed-loop eval every N steps", "int", 1000, min=100, max=10000,
              step=100),
    ],
    eval_params=[
        Param("encoding", "Encoding to evaluate", "choice", "best",
              choices=["best", "onehot", "binary", "text", "cont"]),
        Param("n_episodes", "Episodes", "int", 400, min=50, max=2000, step=50),
        Param("tasks", "Commands", "choice", "all", choices=["all", "insert", "press"]),
        Param("est_sigma", "Hole-perception error (sigma)", "float", 0.01, min=0.0, max=0.04,
              step=0.005),
        Param("sensor_noise", "Force sensor noise", "float", 0.05, min=0.0, max=0.2, step=0.01),
        Param("eval_steps", "Sampling steps K", "choice", 8, choices=[1, 2, 4, 8, 16, 32]),
        Param("seed", "Seed", "int", 0, min=0, max=9999),
    ],
    charts=[
        {"id": "loss", "title": "Flow loss (train, per encoding)",
         "series": [{"key": f"{a}_loss", "label": a} for a in ARMS], "log_y": True,
         "x_key": "arm_step"},
        {"id": "success", "title": "Closed-loop success (200 fixed scenes, all commands)",
         "series": [{"key": f"{a}_success", "label": a} for a in ARMS], "log_y": False,
         "x_key": "arm_step"},
        *[{"id": f"cmd{k}", "title": f"Success: {c2d.TASKS[k]}",
           "series": [{"key": f"{a}_c{k}", "label": a} for a in ARMS], "log_y": False,
           "x_key": "arm_step"}
          for k in range(4)],
    ],
    viewer="embeddings",
    vecgrids=True,
    estimate=estimate,
    train=train,
    evaluate=evaluate,
    lesson="embeddings.html",
)  # fmt: skip
registry.register(SPEC)
