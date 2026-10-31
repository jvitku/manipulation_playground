"""Train a nfv Policy on recorded demos: AdamW, warmup + cosine LR, grad-clip 1.0, EMA.

``fit`` accepts the lab's RunContext duck-type — the methods lab/core/loop.py uses:
``should_continue()``, ``log(step, **series)``, ``set_status(**kw)``,
``save_ckpt(step, payload, metrics=...)`` — so the lesson-16 task wrapper can pass a real
RunContext straight in. ``ctx=None`` trains without a run directory (tests, scripts); any of
the four methods may equally be missing on a custom ctx and is then skipped.

The batcher mirrors ``nfv.data.dataset.DemoDataset.sample_batch`` (uniform chunk starts over
all steps, zero-padded chunk tails flagged by pad_mask) and adds the one thing the model needs
that the dataset does not serve: the episode's instruction as token ids ("text"). It reads
only public DemoDataset attributes (episodes, lengths, starts, index), so a test can feed a
tiny fake with the same fields.

Checkpoint payload (house shape, lab/core/loop.py): {"model", "ema", "norm", "arch",
"params", "gen"} — "norm" is empty because the action scaling is fixed (see nfv.model.build),
"arch" is the ModelConfig as a dict.

Shape legend: B = batch, H = chunk, T = max_text; the batch is the dataset contract + "text".
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict

import numpy as np
import torch

from nfv.model.config import ModelConfig, build
from nfv.model.text import tokenize_batch

_OBS_KEYS = ("rgb_static", "rgb_wrist", "state", "wrench", "wrench_window")
_CHUNK_KEYS = ("dp", "gripper", "f_target", "s_fv2", "s_lin")


def episode_text_ids(dataset, max_text: int) -> torch.Tensor:
    """Tokenize every episode's instruction (from the dataset index) -> int64 [n_episodes, T]."""
    return tokenize_batch([ep["instruction"] for ep in dataset.index["episodes"]], max_text)


def sample_batch_with_text(
    dataset, text_ids: torch.Tensor, batch_size: int, rng: np.random.Generator, chunk: int
) -> dict[str, torch.Tensor]:
    """One training batch: the DemoDataset batch contract plus "text" int64 [B, T].

    Mirrors DemoDataset.sample_batch (uniform starts over all steps; a start near the episode
    end gets a zero-padded tail and a False pad_mask) so the instruction can be joined per
    sampled episode — the dataset's own batch does not say which episode a row came from.
    """
    picks = dataset.starts[rng.integers(0, len(dataset.starts), size=batch_size)]  # [B, 2]
    out: dict[str, list] = {k: [] for k in (*_OBS_KEYS, *_CHUNK_KEYS, "pad_mask", "subtask")}
    for i, t in picks:
        ep, n = dataset.episodes[i], int(dataset.lengths[i])
        for k in _OBS_KEYS:
            out[k].append(ep[k][t])
        valid = min(chunk, n - t)
        for k in _CHUNK_KEYS:
            a = ep[k][t : t + valid].astype(np.float32)  # [valid, ...]
            if valid < chunk:  # zero-pad past the episode end; pad_mask marks it
                a = np.concatenate(
                    [a, np.zeros((chunk - valid, *a.shape[1:]), dtype=np.float32)], axis=0
                )
            out[k].append(a)
        mask = np.zeros(chunk, dtype=bool)
        mask[:valid] = True
        out["pad_mask"].append(mask)
        out["subtask"].append(np.int64(ep["subtask"][t]))
    batch: dict[str, torch.Tensor] = {}
    for k in ("rgb_static", "rgb_wrist"):  # uint8 HWC -> float CHW in [0, 1] (batch time)
        u8 = torch.from_numpy(np.stack(out[k]))  # [B, 96, 96, 3] uint8
        batch[k] = u8.permute(0, 3, 1, 2).float() / 255.0  # [B, 3, 96, 96]
    for k in ("state", "wrench", "wrench_window", *_CHUNK_KEYS):
        batch[k] = torch.from_numpy(np.stack(out[k]).astype(np.float32))
    batch["pad_mask"] = torch.from_numpy(np.stack(out["pad_mask"]))  # [B, H] bool
    batch["subtask"] = torch.from_numpy(np.stack(out["subtask"]))  # [B] int64
    batch["text"] = text_ids[picks[:, 0]]  # [B, T] int64, the sampled episodes' instructions
    return batch


def _call(ctx, method: str, *args, **kw):
    """Call ``ctx.method`` when the ctx provides it (duck-type, every method optional)."""
    fn = getattr(ctx, method, None)
    return fn(*args, **kw) if callable(fn) else None


def fit(
    cfg: ModelConfig,
    dataset,
    ctx=None,
    *,
    steps: int,
    lr: float = 3e-4,
    batch: int = 64,
    eval_every: int = 0,
    callbacks: Sequence[Callable] = (),
    seed: int = 0,
    warmup: int | None = None,
    weight_decay: float = 1e-4,
    log_every: int = 100,
    ckpt_every: int = 2000,
    device: str | None = None,
):
    """Build the Policy from ``cfg`` and train it on ``dataset``; returns (policy, history).

    * optimizer: AdamW on policy.net (the EMA shadow is never optimized), grad-clip 1.0;
    * schedule: linear warmup (default steps/50, at least 10) then cosine to 0;
    * every ``log_every`` steps: {"loss", "lr", "steps_per_s", "loss_dp", "loss_gripper"} to
      ctx.log and into the returned history (loss averaged since the last log);
    * every ``eval_every`` steps (0 = off) and at the end: each callback is called
      ``cb(policy, step)``; a returned dict is logged and appended to the history;
    * checkpoints through ctx.save_ckpt every ``ckpt_every`` steps and at the end;
    * determinism: one ``seed`` fixes the init, the batch sampling and the flow noise, so the
      same call yields the same losses (the unit-tested contract).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    policy = build(cfg).to(device)
    opt = torch.optim.AdamW(policy.net.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)  # batch sampling stream
    text_ids = episode_text_ids(dataset, cfg.max_text)
    warmup = max(10, steps // 50) if warmup is None else max(1, warmup)

    def lr_at(s: int) -> float:
        if s < warmup:
            return lr * (s + 1) / warmup
        return lr * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))

    def save(step: int) -> None:
        cpu = {k: v.detach().cpu().clone() for k, v in policy.net.state_dict().items()}
        ema = {k: v.detach().cpu().clone() for k, v in policy.ema.state_dict().items()}
        payload = {
            "model": cpu,
            "ema": ema,
            "norm": {},  # action scaling is fixed in nfv.model.build, nothing fitted
            "arch": asdict(cfg),
            "params": getattr(ctx, "params", {}),
            "gen": "flow",
        }
        _call(ctx, "save_ckpt", step, payload, metrics={"loss": last_loss})

    history: list[dict] = []
    acc = {"loss": 0.0, "loss_dp": 0.0, "loss_gripper": 0.0}
    nacc, last_loss, step = 0, None, 0
    t_last = time.monotonic()
    for step in range(1, steps + 1):
        if ctx is not None and _call(ctx, "should_continue") is False:
            step -= 1
            break
        policy.net.train()
        b = sample_batch_with_text(dataset, text_ids, batch, rng, cfg.chunk)
        b = {k: v.to(device) for k, v in b.items()}
        loss, aux = policy.loss(b)
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 1.0)
        opt.step()
        policy.ema_update()
        acc["loss"] += float(loss)
        acc["loss_dp"] += aux["loss_dp"]
        acc["loss_gripper"] += aux["loss_gripper"]
        nacc += 1
        if step % log_every == 0:
            now = time.monotonic()
            metrics = {k: v / nacc for k, v in acc.items()}
            metrics["lr"] = lr_at(step)
            metrics["steps_per_s"] = round(nacc / max(now - t_last, 1e-9), 3)
            last_loss, t_last = metrics["loss"], now
            acc, nacc = dict.fromkeys(acc, 0.0), 0
            history.append({"step": step, **metrics})
            _call(ctx, "log", step, **metrics)
            _call(ctx, "set_status", step=step, total_steps=steps)
        if eval_every and callbacks and (step % eval_every == 0 or step == steps):
            for cb in callbacks:
                out = cb(policy, step)
                if out:
                    out = {k: float(v) for k, v in out.items()}
                    history.append({"step": step, **out})
                    _call(ctx, "log", step, **out)
        if ckpt_every and step % ckpt_every == 0 and step < steps:
            save(step)
    if nacc and last_loss is None:
        last_loss = acc["loss"] / nacc
    _call(ctx, "set_status", step=step, total_steps=steps)
    save(step)
    return policy, history
