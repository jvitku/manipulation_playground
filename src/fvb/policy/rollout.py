"""Closed-loop policy rollout on the hidden-hole gantry task (numpy + torch inference)."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from fvb.policy.data import Norm, apply_ablation
from fvb.policy.spec import get_spec
from fvb.policy.task import GantryTask, TaskParams


class TorchPolicy:
    """Loads a checkpoint written by scripts/10_train_bc.py and runs it step by step."""

    def __init__(self, ckpt_path: str | Path, device: str = "cpu"):
        import torch

        from fvb.policy.models import build_model

        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.meta = ck["meta"]
        self.H, self.K = self.meta["H"], self.meta["K"]
        self.use_force = bool(self.meta["use_force"])
        # task settings the policy was trained with; rollout applies them to the task params
        self.task_overrides = {"action_mode": self.meta.get("action_mode", "delta")}
        if self.meta.get("task") == "gantry_kp":
            self.task_overrides["obs_kp"] = True
        self.norm = Norm.from_json(ck["norm"])
        self.spec = get_spec(self.meta.get("task"))
        self.model = build_model(
            self.meta["model"], self.spec.obs_dim, self.spec.act_dim, self.H, self.K
        ).to(device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.device = device
        self.torch = torch
        self.buf: deque = deque(maxlen=self.H)

    def reset(self) -> None:
        self.buf.clear()

    def act(self, obs: np.ndarray, exec_mode: str = "first") -> np.ndarray:
        o = apply_ablation(obs, self.use_force, self.spec.force_idx)
        o = (o - self.norm.obs_mean) / self.norm.obs_std
        if not self.buf:
            for _ in range(self.H - 1):
                self.buf.append(o)
        self.buf.append(o)
        x = self.torch.as_tensor(
            np.stack(self.buf)[None], dtype=self.torch.float32, device=self.device
        )
        with self.torch.no_grad():
            chunk = self.model(x)[0].cpu().numpy()
        chunk = chunk * self.norm.act_std + self.norm.act_mean
        return chunk[0] if exec_mode == "first" else chunk


class UntrainedPolicy(TorchPolicy):
    """Same architecture + normalisation as a trained checkpoint, freshly initialised weights."""

    def __init__(self, ckpt_path: str | Path, init_seed: int = 0, device: str = "cpu"):
        super().__init__(ckpt_path, device=device)
        from fvb.policy.models import build_model

        self.torch.manual_seed(init_seed)
        sp = self.spec
        self.model = build_model(self.meta["model"], sp.obs_dim, sp.act_dim, self.H, self.K)
        self.model.to(device).eval()


def load_policy(path: str | Path, device: str = "cpu", untrained_seed: int | None = None):
    """BC (TorchPolicy) or TD3 (TD3Policy) checkpoint; ``untrained_seed`` -> random weights."""
    import torch

    meta = torch.load(path, map_location="cpu", weights_only=False).get("meta", {})
    if meta.get("algo") == "td3":
        from fvb.policy.td3 import TD3Policy

        return TD3Policy(str(path), device, untrained_seed)
    if untrained_seed is not None:
        return UntrainedPolicy(path, untrained_seed, device)
    return TorchPolicy(path, device)


def rollout(policy, p: TaskParams, seed: int, log=None, make_task=None) -> dict:
    """One closed-loop episode. ``make_task(p, seed)`` builds the task (default: gantry); a task
    exposes observe / expert_action / step / depth / force_norm / hole_xy / k."""
    if policy is not None and getattr(policy, "task_overrides", None):
        p = replace(p, **policy.task_overrides)
    task = (make_task or GantryTask)(p, seed)
    if hasattr(policy, "reset"):
        policy.reset()
    done, reason, peak, n_jams, retracting = False, None, 0.0, 0, False
    while not done:
        obs = task.observe()
        a = policy.act(obs) if policy is not None else task.encode(task.expert_action())
        f = task.force_norm()
        if a[2] > 1e-4 and f > 1.0 and not retracting:
            n_jams += 1
            retracting = True
        if a[2] < 0:
            retracting = False
        done, reason = task.step(a, log)
        peak = max(peak, task.force_norm())
    return {
        "seed": seed,
        "success": reason == "success",
        "reason": reason,
        "steps": task.k,
        "peak_F_N": peak,
        "n_recoveries": n_jams,
        "hole_xy_mm": (task.hole_xy * 1e3).tolist(),
        "final_depth_mm": 1e3 * task.depth(),
    }


def evaluate(policy, p, seeds: list[int], make_task=None) -> dict:
    rows = [rollout(policy, p, s, make_task=make_task) for s in seeds]
    return summarize(rows)


def summarize(rows: list[dict]) -> dict:
    ok = np.array([r["success"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    return {
        "n": len(rows),
        "success_rate": float(ok.mean()),
        "mean_steps_success": float(steps[ok].mean()) if ok.any() else float("nan"),
        "mean_peak_F_N": float(np.mean([r["peak_F_N"] for r in rows])),
        "mean_recoveries": float(np.mean([r["n_recoveries"] for r in rows])),
        "reasons": {
            k: int(sum(r["reason"] == k for r in rows)) for k in {r["reason"] for r in rows}
        },
        "episodes": rows,
    }


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))
