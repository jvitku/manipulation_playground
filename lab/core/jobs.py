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


def proc_start(pid: int) -> int | None:
    """Process start time in clock ticks since boot (/proc/<pid>/stat field 22), or None."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def start_job(kind: str, job_dir: Path) -> int:
    job_dir = Path(job_dir)
    atomic_write_json(job_dir / "status.json", {"state": "running"})
    try:
        with (job_dir / "job.log").open("ab") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "lab.run_job", kind, str(job_dir)],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                cwd=Path(__file__).resolve().parents[2],  # repo root, so `-m lab.run_job` resolves
            )
    except OSError as e:  # never leave an eternal "running" dir behind
        atomic_write_json(
            job_dir / "status.json", {"state": "failed", "msg": f"could not start job: {e}"}
        )
        raise
    # The pid goes to its own file: the parent never read-modify-writes status.json after the
    # spawn, so it cannot clobber the child's state. read_status falls back to it for liveness.
    # The start time guards against pid reuse (a restarted container hands out the same pids).
    atomic_write_json(job_dir / "job.pid", {"pid": proc.pid, "start": proc_start(proc.pid)})
    return proc.pid


def _alive(pid: int, start: int | None = None) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie (our own un-reaped child) is dead for our purposes
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
    except OSError:
        return True
    if fields[0] == "Z":
        return False
    # a different start time means the pid now belongs to an unrelated process
    return start is None or int(fields[19]) == start


def read_status_raw(job_dir: Path) -> dict:
    try:
        return json.loads((Path(job_dir) / "status.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_pid_info(job_dir: Path) -> tuple[int | None, int | None]:
    """(pid, start time) from job.pid; old files hold a bare int (no start time)."""
    try:
        v = json.loads((Path(job_dir) / "job.pid").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None
    if isinstance(v, dict):
        return v.get("pid"), v.get("start")
    return (v if isinstance(v, int) else None), None


def read_pid_file(job_dir: Path) -> int | None:
    return read_pid_info(job_dir)[0]


def read_status(job_dir: Path) -> dict:
    st = read_status_raw(job_dir)
    if not st:
        return {"state": "pending"}
    file_pid, start = read_pid_info(job_dir)
    pid = st.get("pid") or file_pid
    if pid != file_pid:
        start = None  # only the spawned pid has a recorded start time
    if st.get("state") in ("running", "paused") and pid and not _alive(pid, start):
        return {**st, "state": "failed", "msg": f"process died (pid {pid}) without finishing"}
    return st


TERMINAL = ("done", "failed", "stopped")


def sweep_stale(*roots: Path) -> int:
    """Mark every non-terminal job dir under roots/<task>/<job> failed. Called once at server
    startup: jobs are children of the container, so none outlives a restart."""
    n = 0
    for root in roots:
        for d in sorted(Path(root).glob("*/*")):
            if not d.is_dir():
                continue
            st = read_status_raw(d)
            if st and st.get("state") in TERMINAL:
                continue
            atomic_write_json(
                d / "status.json", {**st, "state": "failed", "msg": "server restarted"}
            )
            n += 1
    return n


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
