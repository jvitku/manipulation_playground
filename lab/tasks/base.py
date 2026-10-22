"""Task spec dataclasses: tunable parameters and the per-task contract."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Param:
    name: str
    label: str
    kind: str  # "int" | "float" | "choice" | "bool"
    default: object
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list | None = None
    help: str = ""
    long: bool = False  # True marks params that make runs hours-long


@dataclass
class TaskSpec:
    id: str
    order: int
    title: str
    subtitle: str
    params: list[Param]
    eval_params: list[Param]
    charts: list[dict]  # {"id","title","series":[{"key","label"}],"log_y":bool}
    viewer: str  # "t1" | "traj2d" | "contact2d" | "video"
    estimate: Callable[[dict], float]  # params -> seconds, using calib()
    train: Callable
    evaluate: Callable
    lesson: str  # filename under lab/web/lessons/
