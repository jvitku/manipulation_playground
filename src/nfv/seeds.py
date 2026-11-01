"""Disjoint seed ranges for demos, training and evaluation.

Every episode-level random choice (object placement, hidden thresholds, instruction pick,
expert action noise) is derived from one integer seed, so a seed fully identifies an episode.
The three uses draw from disjoint integer ranges (same scheme as lab/tasks/task4_force_token.py,
which uses demo seeds from 1000 and eval seeds from 50000):

    train      [0, 1_000)              network init / shuffling, never an episode
    demo       [1_000, 41_000)         10_000-seed block per task, demos recorded from here
    eval       [50_000, 90_000)        10_000-seed block per task, held-out closed-loop rollouts

A policy therefore never sees an eval seed's placement during training, and the demo set of one
task never collides with another task's.
"""

from __future__ import annotations

TASK_NAMES = ("t0", "t1", "t2", "t3")

TRAIN_SEED_RANGE = (0, 1_000)
DEMO_SEED0 = 1_000
EVAL_SEED0 = 50_000
SEEDS_PER_TASK = 10_000  # per-task block size inside the demo and eval ranges

assert TRAIN_SEED_RANGE[1] <= DEMO_SEED0
assert DEMO_SEED0 + len(TASK_NAMES) * SEEDS_PER_TASK <= EVAL_SEED0


def _block(seed0: int, task: str, n: int, offset: int) -> list[int]:
    if task not in TASK_NAMES:
        raise ValueError(f"unknown task {task!r}, expected one of {TASK_NAMES}")
    if not 0 <= offset and n + offset <= SEEDS_PER_TASK:
        raise ValueError(f"n={n} offset={offset} exceeds the {SEEDS_PER_TASK}-seed task block")
    base = seed0 + TASK_NAMES.index(task) * SEEDS_PER_TASK + offset
    return list(range(base, base + n))


def demo_seeds(task: str, n: int, offset: int = 0) -> list[int]:
    """The first ``n`` demo seeds of ``task`` (skip ``offset`` seeds first)."""
    return _block(DEMO_SEED0, task, n, offset)


def eval_seeds(task: str, n: int, offset: int = 0) -> list[int]:
    """The first ``n`` held-out evaluation seeds of ``task``."""
    return _block(EVAL_SEED0, task, n, offset)


def train_seeds(n: int) -> list[int]:
    """Seeds for network init / data shuffling. Never used to generate an episode."""
    if n > TRAIN_SEED_RANGE[1] - TRAIN_SEED_RANGE[0]:
        raise ValueError(f"only {TRAIN_SEED_RANGE[1] - TRAIN_SEED_RANGE[0]} train seeds exist")
    return list(range(TRAIN_SEED_RANGE[0], TRAIN_SEED_RANGE[0] + n))
