"""RunContext: the trainer-side half of the run-directory contract.

Run dir files: config.json (in), status.json / metrics.jsonl / ckpt/ (out), control.json (in).
status.json and control.json are replaced atomically; metrics.jsonl is append-only.
"""

import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

POLL_S = 0.5


def atomic_write_json(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunContext:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        cfg = json.loads((self.dir / "config.json").read_text())
        self.task_id: str = cfg["task"]
        self.params: dict = cfg["params"]
        self._t0 = time.time()
        self._last_poll = 0.0
        self._stop = False

    def log(self, step: int, **series: float) -> None:
        with (self.dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **series}) + "\n")

    def set_status(self, **kw) -> None:
        path = self.dir / "status.json"
        try:
            status = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            status = {}
        status.update(kw)
        status.update(pid=os.getpid(), wall_s=round(time.time() - self._t0, 1), updated=_now())
        atomic_write_json(path, status)

    def _read_cmd(self) -> str:
        try:
            return json.loads((self.dir / "control.json").read_text()).get("cmd", "run")
        except (FileNotFoundError, json.JSONDecodeError):
            return "run"

    def should_continue(self) -> bool:
        """Poll control.json (at most every POLL_S); block while paused; False once stopped."""
        if self._stop:
            return False
        if time.time() - self._last_poll < POLL_S:
            return True
        self._last_poll = time.time()
        cmd = self._read_cmd()
        if cmd == "pause":
            self.set_status(state="paused")
            while cmd == "pause":
                time.sleep(0.1)
                cmd = self._read_cmd()
            self.set_status(state="running")
        if cmd == "stop":
            self._stop = True
            return False
        return True

    def save_ckpt(
        self, step: int, payload: dict, label: str | None = None, metrics: dict | None = None
    ) -> Path:
        import torch

        d = self.dir / "ckpt"
        d.mkdir(exist_ok=True)
        path = d / f"step_{step:08d}.pt"
        torch.save(payload, path)
        torch.save(payload, d / "last.pt")
        idx_path = d / "index.json"
        index = json.loads(idx_path.read_text()) if idx_path.exists() else []
        index.append(
            {"file": path.name, "step": step, "label": label, "metrics": metrics, "created": _now()}
        )
        atomic_write_json(idx_path, index)
        return path

    def done(self, msg: str = "") -> None:
        self.set_status(state="done", msg=msg)

    def fail(self, msg: str) -> None:
        self.set_status(state="failed", msg=msg)
