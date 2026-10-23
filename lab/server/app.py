"""FastAPI server of the lab web app: tasks, runs, evals, GPU info, static files."""

import dataclasses
import json
import os
import re
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

import lab.tasks  # noqa: F401  (registers the tasks)
from lab.core import jobs
from lab.core.runctx import atomic_write_json
from lab.tasks import registry


class RevalidatingStaticFiles(StaticFiles):
    """Static files that browsers must revalidate (ETag -> cheap 304), so a restart never leaves
    a stale cached frontend."""

    def file_response(self, *args, **kwargs):
        r = super().file_response(*args, **kwargs)
        r.headers["Cache-Control"] = "no-cache"
        return r


WEB = Path(__file__).resolve().parents[1] / "web"
REPO = Path(__file__).resolve().parents[2]
REF_RUN = "reference"  # pseudo run id of allowlisted reference checkpoints ("reference/<name>")
FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" '
    'rx="7" fill="#2f6fed"/><path d="M8 23V9l8 9 8-9v14" fill="none" stroke="#fff" '
    'stroke-width="3" stroke-linejoin="round"/></svg>'
)
SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
ACTIVE = ("pending", "queued", "running", "paused")
# Estimate label thresholds: >= 45 min is a long run, >= 1.5 h an overnight job.
LONG_S, OVERNIGHT_S = 45 * 60, 90 * 60


def data_root() -> Path:
    return Path(os.environ.get("LAB_DATA") or Path(__file__).resolve().parents[1] / "data")


def estimate_label(seconds: float) -> str:
    if seconds < 60:
        label = f"~{max(1, round(seconds))} s"
    elif seconds < 90 * 60:
        label = f"~{max(1, round(seconds / 60))} min"
    else:
        label = f"~{seconds / 3600:.1f} h"
    if seconds >= OVERNIGHT_S:
        label += " — overnight job"
    elif seconds >= LONG_S:
        label += " — long run"
    return label


def _safe(name: str) -> str:
    if not SAFE.match(name) or name in (".", ".."):
        raise HTTPException(404, "not found")
    return name


def _spec(task_id: str):
    try:
        return registry.get(_safe(task_id))
    except KeyError:
        raise HTTPException(404, f"unknown task {task_id!r}") from None


def _resolve_ckpt(task_id: str, ckpt_id: str) -> Path:
    """Checkpoint id "<run_id>/<file>" of `task_id` -> its file; 404 if unsafe/missing/foreign."""
    run_id, _, fname = str(ckpt_id or "").partition("/")
    if run_id == REF_RUN:  # allowlist of the task spec, never a client-supplied path
        ref = registry.get(_safe(task_id)).reference_ckpts.get(fname)
        f = (REPO / ref[0]) if ref else None
        if f is None or not f.is_file():
            raise HTTPException(404, "checkpoint not found")
        return f.resolve()
    try:
        _safe(task_id)
        _safe(run_id)
        _safe(fname)
    except HTTPException:
        raise HTTPException(404, "checkpoint not found") from None
    runs_dir = (data_root() / "runs" / task_id).resolve()
    ckpt = (runs_dir / run_id / "ckpt" / fname).resolve()
    if not ckpt.is_relative_to(runs_dir) or not ckpt.is_file():
        raise HTTPException(404, "checkpoint not found")
    return ckpt


def _newest_state(task_id: str) -> str | None:
    for d in _job_dirs("runs", task_id)[:1]:
        return jobs.read_status(d).get("state")
    return None


def _params_dicts(params) -> list[dict]:
    return [dataclasses.asdict(p) for p in params]


def _dir(kind: str, task_id: str, job_id: str) -> Path:
    d = data_root() / kind / _safe(task_id) / _safe(job_id)
    if not d.is_dir():
        raise HTTPException(404, "not found")
    return d


def _job_dirs(kind: str, task_id: str) -> list[Path]:
    root = data_root() / kind / task_id
    return sorted((d for d in root.iterdir() if d.is_dir()), reverse=True) if root.is_dir() else []


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _start(kind: str, root_kind: str, task_id: str, config: dict) -> Path:
    d = jobs.new_job_dir(data_root() / root_kind, task_id)
    d.mkdir(parents=True)
    atomic_write_json(d / "config.json", config)
    atomic_write_json(d / "control.json", {"cmd": "run"})
    atomic_write_json(d / "status.json", {"state": "queued"})
    jobs.start_job(kind, d)
    return d


def _gpu() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.splitlines()[0]
        used, total = (int(float(x)) for x in out.split(","))
        return {"used_mb": used, "total_mb": total}
    except Exception:  # noqa: BLE001 - no GPU / no nvidia-smi / odd output: report zeros
        return {"used_mb": 0, "total_mb": 0}


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # every job is a child of this container, so none outlives a restart: whatever is still
    # non-terminal on disk was interrupted (and its pid may since have been reused)
    jobs.sweep_stale(data_root() / "runs", data_root() / "evals")
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="lab", lifespan=_lifespan)

    @app.get("/api/tasks")
    def list_tasks():
        def est_label(s):
            try:
                return estimate_label(
                    float(s.estimate(registry.validate_params(s, {}, which="train")))
                )
            except Exception:  # noqa: BLE001 - a broken estimate must not take the ladder down
                return None

        return [
            {
                "est_label": est_label(s),
                "last_state": _newest_state(s.id),
                "id": s.id,
                "order": s.order,
                "title": s.title,
                "subtitle": s.subtitle,
                "running": any(
                    jobs.read_status(d).get("state") in ACTIVE for d in _job_dirs("runs", s.id)
                ),
            }
            for s in registry.all_specs()
        ]

    @app.get("/api/tasks/{task_id}")
    def task_detail(task_id: str):
        s = _spec(task_id)
        runs, ckpts = [], []
        for d in _job_dirs("runs", s.id):
            st = jobs.read_status(d)
            cfg = _read_json(d / "config.json", {})
            runs.append(
                {
                    "run_id": d.name,
                    "created": d.name,
                    "state": st.get("state"),
                    "step": st.get("step"),
                    "total_steps": st.get("total_steps"),
                    "params": cfg.get("params", {}),
                }
            )
            for c in reversed(_read_json(d / "ckpt" / "index.json", [])):
                ckpts.append(
                    {
                        "id": f"{d.name}/{c['file']}",
                        "run_id": d.name,
                        "step": c.get("step"),
                        "label": c.get("label"),
                        "metrics": c.get("metrics"),
                        "created": c.get("created"),
                    }
                )
        for name, (path, label) in s.reference_ckpts.items():
            if (REPO / path).is_file():
                ref = {"id": f"{REF_RUN}/{name}", "run_id": REF_RUN, "label": label}
                ckpts.append(
                    {**ref, "step": None, "metrics": None, "created": None, "reference": True}
                )
        return {
            "spec": {
                "params": _params_dicts(s.params),
                "eval_params": _params_dicts(s.eval_params),
                "charts": s.charts,
                "viewer": s.viewer,
                "vecgrids": s.vecgrids,
                "lesson_url": f"/lessons/{s.lesson}",
            },
            "runs": runs,
            "checkpoints": ckpts,
        }

    @app.post("/api/tasks/{task_id}/runs")
    def start_run(task_id: str, body: dict):
        s = _spec(task_id)
        try:
            params = registry.validate_params(s, body.get("params") or {}, which="train")
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        for p in s.params:  # kind "ckpt": a checkpoint of another task, id -> absolute path
            if p.kind != "ckpt":
                continue
            if not params.get(p.name):
                raise HTTPException(422, f"param {p.name!r}: pick a {p.ckpt_task} checkpoint")
            try:
                params[p.name] = str(_resolve_ckpt(p.ckpt_task, params[p.name]))
            except HTTPException:
                raise HTTPException(
                    422, f"param {p.name!r}: no such {p.ckpt_task} checkpoint"
                ) from None
        d = _start("train", "runs", s.id, {"task": s.id, "params": params})
        return {"run_id": d.name}

    @app.get("/api/runs/{task_id}/{run_id}")
    def run_state(task_id: str, run_id: str, since: int = 0):
        d = _dir("runs", task_id, run_id)
        metrics = jobs.read_metrics(d, since)
        return {"status": jobs.read_status(d), "metrics": metrics, "next": since + len(metrics)}

    @app.post("/api/runs/{task_id}/{run_id}/control", status_code=204)
    def control(task_id: str, run_id: str, body: dict):
        d = _dir("runs", task_id, run_id)
        try:
            jobs.send_control(d, body.get("cmd"))
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return Response(status_code=204)

    @app.post("/api/tasks/{task_id}/estimate")
    def estimate(task_id: str, body: dict):
        s = _spec(task_id)
        try:
            params = registry.validate_params(s, body.get("params") or {}, which="train")
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        seconds = float(s.estimate(params))
        return {"seconds": seconds, "label": estimate_label(seconds)}

    @app.post("/api/tasks/{task_id}/evals")
    def start_eval(task_id: str, body: dict):
        s = _spec(task_id)
        ckpt = _resolve_ckpt(s.id, str(body.get("ckpt", "")))
        try:
            params = registry.validate_params(s, body.get("params") or {}, which="eval")
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        params["ckpt"] = str(ckpt.resolve())
        d = _start("eval", "evals", s.id, {"task": s.id, "params": params})
        return {"eval_id": d.name}

    @app.get("/api/evals/{task_id}/{eval_id}")
    def eval_state(task_id: str, eval_id: str):
        d = _dir("evals", task_id, eval_id)
        return {"status": jobs.read_status(d), "result": _read_json(d / "result.json")}

    @app.get("/api/evals/{task_id}/{eval_id}/file/{name}")
    def eval_file(task_id: str, eval_id: str, name: str):
        d = _dir("evals", task_id, eval_id)
        f = d / _safe(name)
        if not f.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(f)

    @app.get("/favicon.ico", include_in_schema=False)
    @app.get("/favicon.svg", include_in_schema=False)
    def favicon():
        return Response(FAVICON, media_type="image/svg+xml", headers={"Cache-Control": "no-cache"})

    @app.get("/api/gpu")
    def gpu():
        return _gpu()

    @app.get("/")
    def index():
        f = WEB / "index.html"
        if not f.is_file():
            raise HTTPException(404, "index.html not built yet")
        return FileResponse(f, headers={"Cache-Control": "no-cache"})

    if WEB.is_dir():
        (WEB / "lessons").mkdir(exist_ok=True)
        app.mount("/lessons", RevalidatingStaticFiles(directory=WEB / "lessons"), name="lessons")
        app.mount("/static", RevalidatingStaticFiles(directory=WEB), name="static")
    return app


app = create_app()
