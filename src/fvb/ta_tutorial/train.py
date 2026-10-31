"""Training loop, closed-loop chunked policy, evaluation helpers.

The loop is the whole recipe of pi0 / SmolVLA fine-tuning in miniature:

1. sample a batch of (observation, expert chunk) pairs;
2. ``gen.loss``: draw a noise level, noise the chunk, ask the network for the velocity (flow) or
   the noise (DDPM), MSE against the true one (padding masked);
3. TA-VLA: add ``aux_weight * MSE(aux_head, future force)`` when the model has the head;
4. AdamW, cosine learning-rate decay after a warm-up, gradient clipping;
5. keep an exponential moving average (EMA) of the weights and evaluate the EMA copy -
   standard for diffusion policies, it smooths the noisy late-training updates.

The training loss of a generative head is a poor progress signal: it is an average over all
noise levels, dominated by the noisy (easy-to-be-wrong, impossible-to-be-right) ones, so it
flattens early while closed-loop success keeps improving. That is why ``history`` records the
closed-loop evaluation next to the loss.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Cfg:
    steps: int = 20_000
    batch: int = 256
    lr: float = 3e-4
    warmup: int = 500
    wd: float = 1e-4
    ema: float = 0.999
    aux_weight: float = 0.0
    log_every: int = 100
    eval_every: int = 2000
    seed: int = 0


def train(model, gen, ds, cfg: Cfg, val_ds=None, eval_fn=None, log=print) -> tuple[dict, object]:
    torch.manual_seed(cfg.seed)
    dev = ds.device
    model.to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    def lr_at(s):
        if s < cfg.warmup:
            return cfg.lr * (s + 1) / cfg.warmup
        return (
            cfg.lr
            * 0.5
            * (1 + math.cos(math.pi * (s - cfg.warmup) / max(1, cfg.steps - cfg.warmup)))
        )

    hist = {"step": [], "loss": [], "aux": [], "val_loss": [], "lr": [], "eval": [], "time_s": []}
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    acc, acc_aux, nacc, t0 = 0.0, 0.0, 0, time.time()
    for step in range(1, cfg.steps + 1):
        model.train()
        idx = torch.randint(0, ds.n, (cfg.batch,), generator=g).to(dev)
        cond, x0, mask, aux_t = ds.batch(idx)
        per, extra = gen.loss(model, x0, cond, mask)
        loss = per.mean()
        if cfg.aux_weight > 0 and "aux" in extra and aux_t is not None:
            la = (extra["aux"] - aux_t).pow(2).mean()
            loss = loss + cfg.aux_weight * la
            acc_aux += float(la)
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
                pe.lerp_(pm, 1 - cfg.ema)
        acc += float(per.mean())
        nacc += 1
        if step % cfg.log_every == 0:
            hist["step"].append(step)
            hist["loss"].append(acc / nacc)
            hist["aux"].append(acc_aux / nacc)
            hist["lr"].append(lr_at(step))
            hist["time_s"].append(time.time() - t0)
            hist["val_loss"].append(val_loss(ema, gen, val_ds) if val_ds is not None else None)
            acc, acc_aux, nacc = 0.0, 0.0, 0
        if eval_fn is not None and (step % cfg.eval_every == 0 or step == cfg.steps):
            r = eval_fn(ema)
            r["step"] = step
            hist["eval"].append(r)
            log(f"step {step} loss {hist['loss'][-1]:.4f} val {hist['val_loss'][-1]} eval {r}")
    return hist, ema


@torch.no_grad()
def val_loss(model, gen, ds, n: int = 2048, seed: int = 123) -> float:
    """Held-out loss with FIXED noise draws (same t and noise every call) so the curve is smooth
    enough to compare across steps."""
    model.eval()
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    idx = torch.arange(min(n, ds.n), device=ds.device)
    cond, x0, mask, _ = ds.batch(idx)
    per, _ = gen.loss(model, x0, cond, mask)
    torch.random.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return float(per.mean())


class ChunkPolicy:
    """Closed loop: every ``n_exec`` steps sample a chunk of H actions for all episodes at once
    (from noise, ``steps`` integration steps, prefix encoded once), then play its first
    ``n_exec`` actions open-loop. ``make_cond(obs) -> raw tensors``; ``ds`` normalises them."""

    def __init__(self, model, gen, ds, make_cond, n_exec: int, steps: int = 10, seed: int = 0):
        self.model, self.gen, self.ds, self.make_cond = model.eval(), gen, ds, make_cond
        self.H, self.n_exec, self.steps = ds.H, n_exec, steps
        self.g = torch.Generator(device=ds.device).manual_seed(seed)
        self.queue, self.k = None, 0

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        # N = parallel episodes, H = chunk horizon, A = action dims
        if self.queue is None or self.k >= self.n_exec:
            cond = self.ds.cond(self.make_cond(obs))  # dict of normalised [N, ...] tensors
            n = next(v for v in cond.values() if torch.is_tensor(v)).shape[0]  # N
            shape = (n, self.H, self.ds.t["x0"].shape[-1])  # [N, H, A]
            if hasattr(self.model, "cache"):
                with self.model.cache(cond):
                    self.queue = self.gen.sample(self.model, shape, cond, self.steps, self.g)
            else:
                self.queue = self.gen.sample(self.model, shape, cond, self.steps, self.g)
            self.queue = self.queue.clamp(-1, 1).cpu().numpy()  # chunks [N, H, A]
            self.k = 0
        a = self.queue[:, self.k]  # [N, H, A] -> [N, A], action k of every chunk
        self.k += 1
        return a


def run_closed_loop(env, policy, obs_fn) -> None:
    while not env.done.all():
        env.step(policy.act(obs_fn(env)))
