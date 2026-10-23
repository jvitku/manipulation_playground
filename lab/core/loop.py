"""Shared training loop for the lab tasks: a RunContext-aware port of `fvb.ta_tutorial.train.train`.

Same mechanics (AdamW, linear warm-up + cosine LR, grad-clip 1.0, EMA copy that is the evaluated
and checkpointed model). Differences: `ctx.should_continue()` is polled every step, metrics go to
`ctx.log`, checkpoints to `ctx.save_ckpt`, progress to `ctx.set_status`.

Checkpoint payload written by the loop (`payload_wrap` may nest/extend it):
    {"model": state_dict, "ema": state_dict, "norm": {k: (mean, std)}, "arch": dict,
     "params": ctx.params, "gen": gen.name}
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

import torch

from fvb.ta_tutorial.train import val_loss


def _cpu_state(m) -> dict:
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def train_loop(
    ctx,
    model,
    gen,
    ds,
    val_ds,
    *,
    steps: int,
    batch: int,
    lr: float,
    warmup: int,
    wd: float = 1e-4,
    ema: float = 0.999,
    aux_weight: float = 0.0,
    log_every: int = 50,
    eval_every: int,
    ckpt_every: int,
    eval_fn: Callable[[object], dict] | None,
    seed: int = 0,
    arch: dict | None = None,
    prefix: str = "",
    step_offset: int = 0,
    total_steps: int | None = None,
    payload_wrap: Callable[[dict], dict] | None = None,
    local_step_key: str | None = None,
):
    """Train `model`; returns True if all steps ran, False if stopped early
    (a final ckpt is saved). `prefix` is prepended to loss/val_loss/lr metric names;
    `step_offset`/`total_steps` place this phase inside a longer multi-phase run (status bar,
    metric and checkpoint step numbering) and keep checkpoint file names unique across phases.
    `local_step_key`, when set, adds this phase's own step (1..steps) to every metric row under
    that key, so charts can overlay phases on a shared x-axis (global `step` stays monotonic)."""

    def local(step):
        return {local_step_key: step} if local_step_key else {}

    torch.manual_seed(seed)
    dev = ds.device
    model.to(dev)
    ema_m = copy.deepcopy(model).eval()
    for p in ema_m.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = max(1, warmup)
    total = total_steps or steps

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        return lr * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))

    def save(step):
        payload = {
            "model": _cpu_state(model),
            "ema": _cpu_state(ema_m),
            "norm": {k: (m.cpu(), s.cpu()) for k, (m, s) in ds.norm.items()},
            "arch": arch or {},
            "params": ctx.params,
            "gen": gen.name,
        }
        if payload_wrap:
            payload = payload_wrap(payload)
        ctx.save_ckpt(step_offset + step, payload, metrics={"loss": last_loss})

    g = torch.Generator(device="cpu").manual_seed(seed)
    acc, acc_aux, nacc, last_loss, step = 0.0, 0.0, 0, None, 0
    completed = True
    for step in range(1, steps + 1):
        if not ctx.should_continue():
            step -= 1
            completed = False
            break
        model.train()
        idx = torch.randint(0, ds.n, (batch,), generator=g).to(dev)
        cond, x0, mask, aux_t = ds.batch(idx)
        per, extra = gen.loss(model, x0, cond, mask)
        loss = per.mean()
        if aux_weight > 0 and "aux" in extra and aux_t is not None:
            la = (extra["aux"] - aux_t).pow(2).mean()
            loss = loss + aux_weight * la
            acc_aux += float(la)
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema_m.parameters(), model.parameters(), strict=True):
                pe.lerp_(pm, 1 - ema)
        acc += float(per.mean())
        nacc += 1
        if step % log_every == 0:
            last_loss = acc / nacc
            series = {f"{prefix}loss": last_loss, f"{prefix}lr": lr_at(step)}
            if val_ds is not None:
                series[f"{prefix}val_loss"] = val_loss(ema_m, gen, val_ds)
            if aux_weight > 0:
                series[f"{prefix}aux"] = acc_aux / nacc
            ctx.log(step_offset + step, **series, **local(step))
            ctx.set_status(step=step_offset + step, total_steps=total)
            acc, acc_aux, nacc = 0.0, 0.0, 0
        if eval_fn is not None and (step % eval_every == 0 or step == steps):
            ctx.log(
                step_offset + step,
                **{k: float(v) for k, v in eval_fn(ema_m).items()},
                **local(step),
            )
        if step % ckpt_every == 0 and step < steps:
            save(step)
    if nacc and last_loss is None:
        last_loss = acc / nacc
    ctx.set_status(step=step_offset + step, total_steps=total)
    save(step)
    return completed
