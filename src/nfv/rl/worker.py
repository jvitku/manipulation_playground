"""One RL seed (or one evaluation) per process — the unit lesson 19's lab job runs in parallel.

    python -m nfv.rl.worker train --base CKPT --task t3 --seed 0 --env-steps 20000 --out DIR
    python -m nfv.rl.worker eval  --suite CKPT --which base|<seed> --n 100 --out FILE.json

Why processes and not one loop: a CPU step is render/physics/python-bound (~66 env steps/s
for one process at 1 thread, ~50 each with several in parallel on a busy 24-core host), and
torch's intra-op threads buy almost nothing (base.act 48 ms at 1 thread vs 41 ms at 4). So 5
seeds as 5 one-thread processes take the wall time of one seed — about 8-11 min for 20 k env
steps with the in-training evaluations — where a sequential loop would take 5x that.

``train`` writes into DIR: ``metrics.jsonl`` (every ctx.log row, the lab's own format),
``progress.json`` (latest set_status), ``residual.pt`` (the residual payload: TD3.state(meta)
+ "arch"; the frozen base is NOT embedded per seed — the lab job stores it once in the suite
checkpoint), ``result.json`` (run_residual_rl's dict without the agent). A file ``STOP`` in
DIR makes should_continue() return False (the parent's stop button).

``eval`` re-evaluates the frozen base (``--which base``) or one saved residual (``--which``
= its seed) of a suite checkpoint on ``nfv.seeds.eval_seeds(task, n, offset)`` and writes the
nfv.rl.loop.evaluate dict to FILE.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # before torch is imported
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import torch  # noqa: E402

from nfv.rl.loop import ResidualActor, evaluate, policy_from_payload, run_residual_rl  # noqa: E402
from nfv.rl.residual import DP_RES, make_agent  # noqa: E402
from nfv.seeds import eval_seeds  # noqa: E402
from nfv.sim.env import Env  # noqa: E402


def _dump(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


class FileCtx:
    """RunContext duck-type over a plain directory (see the module docstring)."""

    def __init__(self, out: Path, params: dict):
        self.dir = Path(out)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.params = params
        self._last = 0.0

    def log(self, step: int, **series) -> None:
        with (self.dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": int(step), **series}) + "\n")

    def set_status(self, **kw) -> None:
        _dump(self.dir / "progress.json", {**kw, "time": time.time()})

    def should_continue(self) -> bool:
        return not (self.dir / "STOP").exists()

    def save_ckpt(self, step: int, payload: dict, label=None, metrics=None):
        payload = {k: v for k, v in payload.items() if k != "base"}  # base stored once, upstream
        tmp = self.dir / "residual.pt.tmp"
        torch.save({**payload, "step": int(step), "metrics": metrics}, tmp)
        os.replace(tmp, self.dir / "residual.pt")


def actor_from_payload(res: dict, base) -> ResidualActor:
    """Rebuild a saved residual (TD3.state + "arch") on top of a frozen base Policy."""
    arch = res.get("arch", {})
    rf = bool(arch.get("residual_force", False))
    agent = make_agent(rf, hidden=int(res["cfg"]["hidden"]))
    agent.actor.load_state_dict(res["actor"])
    agent.actor.eval()
    return ResidualActor(base, agent, rf, dp_res=float(arch.get("dp_res_m", DP_RES)))


def _jsonable(out: dict) -> dict:
    return {k: v for k, v in out.items() if k != "agent"}


def cmd_train(a) -> int:
    params = {k: v for k, v in vars(a).items() if k != "fn"}
    ctx = FileCtx(Path(a.out), params)
    out = run_residual_rl(
        a.base, task=a.task, seed=a.seed, env_steps=a.env_steps, ctx=ctx, arm=a.arm or None,
        n_envs=a.n_envs, warmup=a.warmup, eval_every=a.eval_every, n_eval=a.n_eval,
        bc_floor=a.bc_floor, peak_force_penalty=a.peak_force_penalty,
        hole_shaping=a.hole_shaping,
        residual_force=a.residual_force, dp_res=a.dp_res,
    )  # fmt: skip
    _dump(ctx.dir / "result.json", _jsonable(out))
    return 0


def cmd_eval(a) -> int:
    torch.set_num_threads(1)
    suite = torch.load(a.suite, map_location="cpu", weights_only=False)
    base = policy_from_payload(suite["base"])
    task = str(suite["task"])
    if a.which == "base":
        actor, off = ResidualActor(base, make_agent(hidden=32)), True
    else:
        actor, off = actor_from_payload(suite["residuals"][str(a.which)], base), False
    env = Env(task=task, render=True)
    ev = evaluate(actor, env, task, eval_seeds(task, a.n, a.offset), off=off, trace_first=True)
    env.close()
    _dump(Path(a.out), {"which": a.which, "task": task, **ev})
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nfv.rl.worker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--base", required=True)
    t.add_argument("--arm", default="")
    t.add_argument("--task", default="t3")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--env-steps", type=int, default=20_000)
    t.add_argument("--out", required=True)
    t.add_argument("--n-envs", type=int, default=4)
    t.add_argument("--warmup", type=int, default=1_000)
    t.add_argument("--eval-every", type=int, default=4_000)
    t.add_argument("--n-eval", type=int, default=20)
    t.add_argument("--bc-floor", type=float, default=1.0)
    t.add_argument("--peak-force-penalty", type=float, default=0.0)
    t.add_argument("--hole-shaping", type=float, default=0.0)
    t.add_argument("--residual-force", action="store_true")
    t.add_argument("--dp-res", type=float, default=DP_RES)
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("eval")
    e.add_argument("--suite", required=True)
    e.add_argument("--which", required=True)
    e.add_argument("--n", type=int, default=100)
    e.add_argument("--offset", type=int, default=0)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_eval)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
