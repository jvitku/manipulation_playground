"""Task registry: specs keyed by `spec.id`, param validation, calibration lookup."""

import json
import math
from pathlib import Path

CALIB_PATH = Path(__file__).with_name("calib.json")

_TASKS: dict[str, object] = {}


def register(spec):
    names = [p.name for p in (getattr(spec, "params", None) or [])]
    names += [p.name for p in (getattr(spec, "eval_params", None) or [])]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"task {spec.id!r}: param names in both params and eval_params: {dups}")
    _TASKS[spec.id] = spec
    return spec


def get(task_id: str):
    try:
        return _TASKS[task_id]
    except KeyError:
        raise KeyError(f"unknown task {task_id!r}; registered: {sorted(_TASKS)}") from None


def all_specs() -> list:
    return sorted(_TASKS.values(), key=lambda s: getattr(s, "order", 0))


def _coerce(p, value):
    if p.kind == "bool":
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "on"):
                return True
            if low in ("false", "0", "no", "off", ""):
                return False
            raise ValueError(f"param {p.name!r}: not a bool: {value!r}")
        return bool(value)
    if p.kind == "choice":
        if value not in (p.choices or []):
            raise ValueError(f"param {p.name!r}: {value!r} not in {p.choices}")
        return value
    if p.kind in ("int", "float"):
        try:
            num = float(value)
            if not math.isfinite(num):
                raise ValueError("non-finite")
            out = int(num) if p.kind == "int" else num
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"param {p.name!r}: not a finite {p.kind}: {value!r}") from None
        if p.min is not None:
            out = max(out, int(p.min) if p.kind == "int" else p.min)
        if p.max is not None:
            out = min(out, int(p.max) if p.kind == "int" else p.max)
        return out
    raise ValueError(f"param {p.name!r}: unknown kind {p.kind!r}")


def validate_params(spec, form: dict, which: str = "all") -> dict:
    """Coerce to typed values, clamp to min/max, fill defaults; unknown keys raise ValueError.

    `which` selects the key space: "train" (spec.params), "eval" (spec.eval_params), "all" (both).
    """
    if which == "train":
        params = list(spec.params)
    elif which == "eval":
        params = list(spec.eval_params)
    elif which == "all":
        params = list(spec.params) + list(spec.eval_params)
    else:
        raise ValueError(f"which must be 'train', 'eval' or 'all', got {which!r}")
    known = {p.name: p for p in params}
    unknown = sorted(set(form) - set(known))
    if unknown:
        raise ValueError(f"unknown params: {unknown}")
    return {p.name: _coerce(p, form.get(p.name, p.default)) for p in params}


def calib(task_id: str) -> dict:
    """Entry for `task_id` in calib.json, re-read on every call; {} if missing."""
    try:
        data = json.loads(CALIB_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return data.get(task_id, {})
