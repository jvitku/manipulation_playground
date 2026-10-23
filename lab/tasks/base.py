"""Task spec dataclasses: tunable parameters and the per-task contract."""

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Param:
    """A tunable parameter. `kind` is "int" | "float" | "choice" | "bool" | "ckpt".

    kind "ckpt": a checkpoint of another task, picked from a dropdown the frontend fills from
    `ckpt_task`'s checkpoints (GET /api/tasks/<ckpt_task>). The form sends the checkpoint id
    "<run_id>/<file>"; on a training start the server checks that the id exists under that
    task's runs (422 otherwise) and replaces it with the absolute file path in the run's config.
    """

    name: str
    label: str
    kind: str  # "int" | "float" | "choice" | "bool" | "ckpt"
    default: object
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list | None = None
    help: str = ""
    long: bool = False  # True marks params that make runs hours-long
    ckpt_task: str = ""  # kind "ckpt": the task whose checkpoints the dropdown lists


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
    check: Callable | None = None  # cross-param check on typed train params; raises ValueError
    # True when the task's result viewer draws vector grids (shows the "vector grids" toggle)
    vecgrids: bool = False
    # checkpoints that live outside lab/data: name -> (repo-relative path, label). Listed in the
    # evaluate dropdown as id "reference/<name>"; only this allowlist is ever resolved.
    reference_ckpts: dict[str, tuple[str, str]] = field(default_factory=dict)
