"""EpisodeLogger -> ``.npz`` + sibling ``.json`` (schema: PLAN.md §5).

Arrays are time-major at **control** rate; ``*_hf`` keys are at physics rate. Every
episode writes its full config, seed, git SHA and package versions next to the data.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# key -> (trailing shape, dtype). None in shape = free dimension.
SCHEMA: dict[str, tuple[tuple[int | None, ...], type]] = {
    "t": ((), np.float64),
    "ee_pos": ((3,), np.float64),
    "ee_quat": ((4,), np.float64),
    "ee_vel": ((6,), np.float64),
    "ft_raw": ((6,), np.float64),
    "ft_world": ((6,), np.float64),
    "ft_comp": ((6,), np.float64),
    "contact_wrench": ((6,), np.float64),
    "n_contacts": ((), np.int64),
    "action": ((None,), np.float64),
}
HF_SCHEMA = {"ft_raw_hf": ((6,), np.float64), "t_hf": ((), np.float64)}
OPTIONAL = {"joint_torque": ((None,), np.float64), "image": ((None, None, 3), np.uint8)}
SCALARS = {"success": np.bool_}


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def package_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("mujoco", "robosuite", "scipy"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            pass
    return out


@dataclass
class EpisodeLogger:
    config: dict[str, Any]
    seed: int
    _rows: dict[str, list] = field(default_factory=dict)
    _hf: dict[str, list] = field(default_factory=dict)
    success: bool = False

    def step(self, **kw: Any) -> None:
        """Append one control-rate sample. Keys must be in the schema."""
        for k, v in kw.items():
            if k not in SCHEMA and k not in OPTIONAL:
                raise KeyError(f"{k!r} is not a schema key (PLAN.md §5)")
            self._rows.setdefault(k, []).append(np.asarray(v))

    def step_hf(self, t: float, ft_raw: np.ndarray) -> None:
        """Append one physics-rate F/T sample."""
        self._hf.setdefault("t_hf", []).append(float(t))
        self._hf.setdefault("ft_raw_hf", []).append(np.asarray(ft_raw, dtype=np.float64))

    def arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for k, rows in self._rows.items():
            spec = SCHEMA.get(k) or OPTIONAL[k]
            out[k] = np.stack(rows).astype(spec[1])
        for k, rows in self._hf.items():
            out[k] = np.stack(rows).astype(HF_SCHEMA[k][1])
        out["success"] = np.bool_(self.success)
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        arrs = self.arrays()
        validate(arrs)
        np.savez_compressed(path, **arrs)
        meta = {
            "config": self.config,
            "seed": self.seed,
            "git_sha": git_sha(),
            "versions": package_versions(),
            "T": int(len(arrs["t"])),
            "keys": sorted(arrs.keys()),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=_json_default))
        return path


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def validate(arrs: dict[str, np.ndarray]) -> None:
    """Raise if ``arrs`` violates PLAN.md §5 (keys, shapes, dtypes, finite, monotonic t)."""
    missing = [k for k in SCHEMA if k not in arrs]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    T = len(arrs["t"])
    for k, (shape, dtype) in {**SCHEMA, **OPTIONAL}.items():
        if k not in arrs:
            continue
        a = arrs[k]
        if a.shape[0] != T:
            raise ValueError(f"{k}: leading dim {a.shape[0]} != T={T}")
        if len(a.shape) - 1 != len(shape):
            raise ValueError(f"{k}: rank {a.shape} vs schema {shape}")
        for got, want in zip(a.shape[1:], shape, strict=True):
            if want is not None and got != want:
                raise ValueError(f"{k}: shape {a.shape} vs schema {shape}")
        if a.dtype != dtype:
            raise ValueError(f"{k}: dtype {a.dtype} != {dtype}")
        if np.issubdtype(a.dtype, np.floating) and not np.all(np.isfinite(a)):
            raise ValueError(f"{k}: non-finite values")
    for k, (shape, dtype) in HF_SCHEMA.items():
        if k in arrs:
            a = arrs[k]
            if a.shape[1:] != shape or a.dtype != dtype or not np.all(np.isfinite(a)):
                raise ValueError(f"{k}: bad hf array {a.shape} {a.dtype}")
    if "ft_raw_hf" in arrs and "t_hf" in arrs and len(arrs["ft_raw_hf"]) != len(arrs["t_hf"]):
        raise ValueError("ft_raw_hf / t_hf length mismatch")
    if T > 1 and not np.all(np.diff(arrs["t"]) > 0):
        raise ValueError("t not strictly increasing")
    if "t_hf" in arrs and len(arrs["t_hf"]) > 1 and not np.all(np.diff(arrs["t_hf"]) > 0):
        raise ValueError("t_hf not strictly increasing")
    if "success" not in arrs:
        raise ValueError("missing success")


def load(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path)
    with np.load(path.with_suffix(".npz")) as z:
        arrs = {k: z[k] for k in z.files}
    meta = json.loads(path.with_suffix(".json").read_text())
    return arrs, meta
