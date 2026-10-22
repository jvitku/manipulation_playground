"""Task registry stub: specs keyed by `spec.id`, each with `.train(ctx)` and `.evaluate(ctx)`."""

_TASKS: dict[str, object] = {}


def register(spec) -> None:
    _TASKS[spec.id] = spec


def get(task_id: str):
    try:
        return _TASKS[task_id]
    except KeyError:
        raise KeyError(f"unknown task {task_id!r}; registered: {sorted(_TASKS)}") from None
