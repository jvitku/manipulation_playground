"""Job helpers (server side): spawn a detached trainer, read its run dir, send control."""

import json
import os
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from lab.core.runctx import atomic_write_json


def new_job_dir(root: Path, task_id: str) -> Path:
    stamp = datetime.now(UTC).strftime("%y%m%d-%H%M%S")
    return Path(root) / task_id / f"{stamp}-{secrets.token_hex(2)}"


def start_job(kind: str, job_dir: Path) -> int:
    job_dir = Path(job_dir)
    atomic_write_json(job_dir / "status.json", {"state": "running"})
    with (job_dir / "job.log").open("ab") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "lab.run_job", kind, str(job_dir)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=Path(__file__).resolve().parents[2],  # repo root, so `-m lab.run_job` resolves
        )
    # The child also records its own pid; only fill it in if it has not got that far yet,
    # so a fast-finishing child's terminal state is never clobbered.
    status = read_status_raw(job_dir)
    if "pid" not in status:
        atomic_write_json(job_dir / "status.json", {**status, "pid": proc.pid})
    return proc.pid


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie (our own un-reaped child) is dead for our purposes
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def read_status_raw(job_dir: Path) -> dict:
    try:
        return json.loads((Path(job_dir) / "status.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_status(job_dir: Path) -> dict:
    st = read_status_raw(job_dir)
    if not st:
        return {"state": "pending"}
    pid = st.get("pid")
    if st.get("state") in ("running", "paused") and pid and not _alive(pid):
        return {**st, "state": "failed", "msg": f"process died (pid {pid}) without finishing"}
    return st


def read_metrics(job_dir: Path, since: int = 0) -> list[dict]:
    try:
        text = (Path(job_dir) / "metrics.jsonl").read_text()
    except FileNotFoundError:
        return []
    lines = text.split("\n")
    lines.pop()  # after split, the last item is "" (complete) or a partial line (no newline)
    return [json.loads(ln) for ln in lines[since:] if ln.strip()]


def send_control(job_dir: Path, cmd: str) -> None:
    if cmd not in ("run", "pause", "stop"):
        raise ValueError(f"bad control cmd {cmd!r}")
    atomic_write_json(Path(job_dir) / "control.json", {"cmd": cmd})
