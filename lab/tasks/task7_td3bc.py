"""Task 12 - "TD3+BC" (id `td3-bc`): the recipe that makes RL reliable (PLAN §12, R1b/R4).

Extends task 11 through its seams (`task6_td3._run` / `_train_loop` hooks); nothing of the loop is
copied. Evaluation, the video viewer, checkpoint contract (`TD3Policy`) and the metrics of task 11
are reused unchanged. What is added is the repo's R4 recipe, as `scripts/18_train_td3.py
--demos 100 --bc-weight 2.5 --bc-final 1.0 --bc-decay-steps 100000` ran it (journal R1b/R3b/R4,
docs/FINDINGS.md "R1-R4 (TD3)"):

  - demos: `n_demos` scripted-expert episodes (script 18's `collect_demos`: the privileged arm
    expert through the RL env, real observations and rewards, actions = expert delta / scale,
    seeds 900000+) go into the replay buffer AND a separate demo buffer;
  - actor loss (inside `fvb.policy.td3.TD3.update`, which is untouched):
        q_weight * (-Q / max(mean|Q|, 1)) + lambda(t) * mean |pi(o_d) - a_d|^2,
    `o_d, a_d` a fresh batch from the demo buffer every actor update;
  - lambda(t) = floor + (start - floor) * max(0, 1 - t / bc_decay_steps), t = environment steps
    (script 18's formula): 2.5 -> 1.0, and never 0 (R1, which decayed to 0, collapsed).
    The script decays over 100k of a 150k run; the lab default is 66k of 100k (same ratio).

Metrics: task 11's, plus `demos_collected` (step 0, per episode) and in the loss rows `bc_loss`
(imitation error on a demo batch) and `bc_weight` (lambda, window mean). Checkpoint meta gains
`bc` = {n_demos, bc_weight_start, bc_weight_floor, bc_decay_steps, demo_stats}.
"""

from __future__ import annotations

import numpy as np

from lab.tasks import registry
from lab.tasks import task6_td3 as t6
from lab.tasks.base import Param, TaskSpec
from lab.tasks.task1_bc_mse import OVERHEAD_S

DEMO_SEED0 = 900_000  # script 18's demo seeds, disjoint from training (1e6+) and eval seeds
FALLBACK_COLLECT_S_PER_EP = 6.0


def bc_weight_at(t: int, start: float, floor: float, decay_steps: int) -> float:
    """Script 18: floor + (start - floor) * max(0, 1 - t / decay_steps)."""
    return floor + (start - floor) * max(0.0, 1.0 - t / max(decay_steps, 1))


def check_params(P: dict) -> None:
    """Cross-param check (-> 422): lambda(t) only decays, so start must be >= floor."""
    if float(P["bc_weight_start"]) < float(P["bc_weight_floor"]):
        raise ValueError("bc_weight_start must be >= bc_weight_floor "
                         "(for plain TD3 with demos in the buffer set both to 0)")  # fmt: skip


def collect_demos(ctx, env, n: int, bufs) -> dict | None:
    """Script 18's `collect_demos` loop (arm branch), with progress and stop polling.
    None if stopped meanwhile."""
    ok = 0
    for i in range(n):
        if not ctx.should_continue():
            return None
        o, done, info = env.reset(DEMO_SEED0 + i), False, {}
        while not done:
            a = np.clip(env.task.encode(env.task.expert_action()) / env.scale, -1, 1)
            a = a.astype(np.float32)
            o2, r, term, trunc, info = env.step(a)
            for b in bufs:
                b.add(o, a, r, o2, term)
            o, done = o2, term or trunc
        ok += info["reason"] == "success"
        ctx.log(0, demos_collected=i + 1)
        ctx.set_status(message=f"collecting expert demonstrations {i + 1}/{n} ({ok} succeeded)")
    return {"episodes": n, "success": ok, "transitions": bufs[0].n}


class BCHooks:
    def __init__(self, P: dict):
        self.n_demos = int(P["n_demos"])
        self.start = float(P["bc_weight_start"])
        self.floor = float(P["bc_weight_floor"])
        self.decay = int(P["bc_decay_steps"])
        self.demo_buf = None
        self.meta_extra: dict = {}

    def meta(self) -> dict:
        return {"bc": {"n_demos": self.n_demos, "bc_weight_start": self.start,
                       "bc_weight_floor": self.floor, "bc_decay_steps": self.decay}}  # fmt: skip

    def final_meta(self) -> dict:
        """Meta known only after `prepare` (demo_stats); merged into every checkpoint."""
        return {"bc": {**self.meta()["bc"], **self.meta_extra}}

    def buffer_size(self, steps: int) -> int:
        """Replay capacity: steps + the demo transitions (about 250 per episode at most), so the
        demos are not overwritten near the end of a run. Script 18 used 1M."""
        return min(1_000_000, steps + self.n_demos * 250)

    def prepare(self, ctx, env, buf, rng) -> bool:
        from fvb.policy.td3 import ReplayBuffer

        # legend: O = obs dims, A = action dims; buffer rows hold o [O], a [A] transitions
        self.demo_buf = ReplayBuffer(env.obs_dim, env.act_dim, max(1, self.n_demos) * 250)
        stats = collect_demos(ctx, env, self.n_demos, [buf, self.demo_buf])
        if stats is None:
            return False
        self.stats = stats
        self.meta_extra = {"demo_stats": stats}
        ctx.set_status(message=f"demos: {stats['success']}/{stats['episodes']} succeeded, "
                       f"{stats['transitions']} transitions; training")  # fmt: skip
        return True

    def update(self, agent, buf, rng, t: int) -> dict:
        w = bc_weight_at(t, self.start, self.floor, self.decay)
        out = agent.update(buf, rng, self.demo_buf, w)
        out["bc_weight"] = w
        return out


def train(ctx) -> None:
    t6._run(ctx, BCHooks(ctx.params))


def estimate(params: dict) -> float:
    cal = registry.calib("td3-bc")
    steps = int(params["env_steps"])
    sps = cal.get("steps_per_s") or t6.FALLBACK_STEPS_PER_S
    eval_s = cal.get("eval_s") or t6.FALLBACK_EVAL_S
    collect = cal.get("collect_s_per_ep") or FALLBACK_COLLECT_S_PER_EP
    n_evals = steps // max(1, int(params["eval_every"])) + 1
    return (steps / sps + n_evals * eval_s + int(params["n_demos"]) * collect
            + cal.get("overhead_s", OVERHEAD_S))  # fmt: skip


def _params() -> list[Param]:
    ps = list(t6.SPEC.params)
    i = next(k for k, p in enumerate(ps) if p.name == "use_force")
    new = [
        Param(
            "n_demos",
            "Expert demonstrations",
            "int",
            100,
            min=10,
            max=300,
            step=10,
            help="scripted-expert episodes put in the replay buffer at the start (R4: 100)",
        ),
        Param(
            "bc_weight_start",
            "BC weight at the start",
            "float",
            2.5,
            min=0.0,
            max=10.0,
            step=0.5,
            help="weight of the imitation term in the actor loss",
        ),
        Param(
            "bc_weight_floor",
            "BC weight floor",
            "float",
            1.0,
            min=0.0,
            max=5.0,
            step=0.5,
            help="the weight decays to this and stays: R1 decayed to 0 and collapsed",
        ),
        Param(
            "bc_decay_steps",
            "BC decay (steps)",
            "int",
            66_000,
            min=5_000,
            max=300_000,
            step=1_000,
            help="linear decay length; R4 used 100k of a 150k run",
        ),
    ]
    return ps[:i] + new + ps[i:]


SPEC = TaskSpec(
    id="td3-bc",
    order=12,
    title="TD3+BC: the recipe that makes RL reliable",
    subtitle="Put 100 expert demonstrations in the replay buffer and keep an imitation term in the "
    "actor loss: the repo's R1-R4 search, 5 of 10 seeds instead of 0.",
    params=_params(),
    eval_params=t6.SPEC.eval_params,
    charts=[
        *t6.SPEC.charts,
        {"id": "bc", "title": "Imitation: BC loss and its weight lambda(t)",
         "series": [{"key": "bc_loss", "label": "bc loss (demo batch)"},
                    {"key": "bc_weight", "label": "lambda(t)"}], "log_y": False},
        {"id": "collect", "title": "Demonstrations collected",
         "series": [{"key": "demos_collected", "label": "episodes"}], "log_y": False},
    ],
    viewer="video",
    estimate=estimate,
    train=train,
    evaluate=t6.evaluate,
    lesson="td3-bc.html",
    check=check_params,
)  # fmt: skip
registry.register(SPEC)
