# Experiment lab

Web app for running and inspecting the toy experiments (training, closed-loop evaluation, lessons).
FastAPI backend (`lab/server`), static front end (`lab/web`), tasks in `lab/tasks`.

## Build / run / stop

```bash
make lab-build   # build fvb-lab:latest (FROM fvb-train; needs fvb-train built: make build-train)
make lab         # docker compose up -d lab; prints the URLs
make lab-stop    # stop the service
make lab-test    # builds the image if needed, then pytest tests/lab in a one-off fvb-lab container (not the running service)
```

Tests run in the fvb-lab image (the dev image has no fastapi); `make lab-test` runs `lab-build` first.
`make lab` does not rebuild a stale image: run `make lab-build` first after Dockerfile changes.

URL: `http://<host>:8430` (port bound on 0.0.0.0). The service has `restart: unless-stopped`, so it
survives reboots until `make lab-stop`. It runs as uid 1001 (this machine's owner uid; adjust `user:` in compose if different) with the repo mounted at
`/workspace`. After editing server or task code, `docker compose -f docker/compose.yaml restart lab`
(task modules are imported at startup; `calib.json` is re-read per request).

## Data layout

Everything the app writes lives under `lab/data/` (gitignored):

- `lab/data/runs/<task>/<run_id>/` - `config.json`, `status.json`, `metrics.jsonl`, `job.log`, `ckpt/`
  (`last.pt`, `step_*.pt`, `index.json`)
- `lab/data/evals/<task>/<eval_id>/` - `config.json`, `status.json`, `result.json` (rollouts for the viewer)

## Time estimates and calibration

Each task's `estimate(params)` returns seconds from `lab/tasks/calib.json`. For `bc-mse` the keys are
`steps_per_s` (per-model steps per second; the run trains two models, so this is about half the raw
optimizer rate), `collect_s_per_ep` and `overhead_s`; estimate = `steps/steps_per_s + demos*collect_s_per_ep + overhead_s`.

Re-calibrate (different GPU, changed model size): start a default `bc-mse` run via the UI or
`POST /api/tasks/bc-mse/runs`, read `wall_s` from its `status.json`, then set
`steps_per_s = steps / (wall_s - overhead_s - demos*collect_s_per_ep)` and check
`POST /api/tasks/bc-mse/estimate` is within ~2x of the wall time. Measured on an RTX 3080 (shared):
default run 12.2-12.8 s, `steps_per_s` 354.

## Adding a task

Create `lab/tasks/taskN_<name>.py` with a `TaskSpec` (id, order, title, params as `Param(...)`, `estimate`),
a `train(ctx)` that writes checkpoints/metrics through `lab.core.loop.train_loop` / the run context, an
`evaluate(ctx)` that writes `result.json`, and a lesson page in `lab/web/lessons/`; end the module with
`registry.register(SPEC)` and import it where the other tasks are imported. Copy `task1_bc_mse.py` as the pattern, add an entry to `calib.json`, and a test in `tests/lab/`.
