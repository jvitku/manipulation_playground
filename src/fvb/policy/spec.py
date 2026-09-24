"""Per-task observation/action layout, so training, evaluation and export are task-agnostic.

Every checkpoint stores ``meta["task"]``; every expert dataset stores ``task_spec`` in its
``config.json``. Checkpoints and datasets from before the registry are gantry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    name: str
    obs_dim: int
    act_dim: int
    force_idx: slice  # channels zeroed in the no-force ablation
    obs_groups: dict[str, slice]  # named slices of the observation (HDF5 export, docs)


GANTRY = TaskSpec(
    name="gantry",
    obs_dim=18,
    act_dim=3,
    force_idx=slice(6, 15),
    obs_groups={
        "rel_pos": slice(0, 3),  # m, peg position relative to the episode start
        "vel": slice(3, 6),  # m/s
        "ft_comp": slice(6, 12),  # N, N·m, compensated, sensor frame (= world on the gantry)
        "ft_hf_summary": slice(12, 15),  # max|F| N, min Fz N, max|T| N·m over the interval
        "prev_action": slice(15, 18),  # m, previous target delta
    },
)

ARM = TaskSpec(
    name="arm",
    obs_dim=20,
    act_dim=3,
    force_idx=slice(8, 17),
    obs_groups={
        "rel_pos": slice(0, 3),  # m, peg-tip position relative to the episode start (world)
        "vel": slice(3, 6),  # m/s, F/T-site linear velocity (world)
        "tilt": slice(6, 8),  # peg axis x, y components (world; 0 = vertical)
        "ft_comp_world": slice(8, 14),  # N, N·m, compensated, world axes, torque about the tip
        "ft_hf_summary": slice(14, 17),  # max|F| N, max|T| N·m, std|F| N over the interval
        "prev_action": slice(17, 20),  # m, previous tip-target delta
    },
)

GANTRY_KP = TaskSpec(
    name="gantry_kp",
    obs_dim=19,
    act_dim=3,
    force_idx=GANTRY.force_idx,
    obs_groups={**GANTRY.obs_groups, "log_kp": slice(18, 19)},  # log(kp / 800 N/m)
)

SPECS = {s.name: s for s in (GANTRY, ARM, GANTRY_KP)}


def get_spec(name: str | None) -> TaskSpec:
    return SPECS[name or "gantry"]


def spec_of_dataset(data_dir: str | Path) -> TaskSpec:
    cfg = Path(data_dir) / "config.json"
    if cfg.exists():
        return get_spec(json.loads(cfg.read_text()).get("task_spec"))
    return GANTRY
