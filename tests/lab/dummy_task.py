"""Tiny fast task for API tests; registered by import (server process and job subprocess)."""

import json
import time

from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec


def _train(ctx):
    steps = int(ctx.params["steps"])
    delay = float(ctx.params["delay"])
    for i in range(1, steps + 1):
        if not ctx.should_continue():
            ctx.set_status(state="stopped", step=i)
            return
        ctx.log(i, loss=1.0 / i)
        ctx.set_status(step=i, total_steps=steps)
        time.sleep(delay)
    ctx.save_ckpt(steps, {"w": 1}, label="final", metrics={"loss": 1.0 / steps})
    ctx.done()


def _evaluate(ctx):
    (ctx.dir / "result.json").write_text(
        json.dumps({"ckpt": ctx.params["ckpt"], "n": ctx.params["n"]})
    )
    ctx.done()


SPEC = TaskSpec(
    id="dummy",
    order=99,
    title="Dummy",
    subtitle="test task",
    params=[
        Param("steps", "Steps", "int", 5, 1, 1000),
        Param("delay", "Delay", "float", 0.02, 0, 10),
    ],
    eval_params=[Param("n", "N", "int", 3, 1, 10)],
    charts=[
        {"id": "l", "title": "loss", "series": [{"key": "loss", "label": "loss"}], "log_y": False}
    ],
    viewer="t1",
    estimate=lambda p: p["steps"] * 14.0,
    train=_train,
    evaluate=_evaluate,
    lesson="dummy.html",
)
registry.register(SPEC)
