# Experiment lab

Web app for running and inspecting the toy experiments (training, closed-loop evaluation, lessons).
FastAPI backend (`lab/server`), static front end (`lab/web`), tasks in `lab/tasks`.

## The ladder (nominal time = default parameters, RTX 3080, shared)

| # | id | what it shows | default run |
|---|---|---|---|
| 1 | `bc-mse` | behaviour cloning with MSE averages two expert modes into a crash | ~13 s |
| 2 | `flow` | same data and net, flow matching samples both modes | ~13 s |
| 3 | `embeddings` | one policy, four commands: one-hot / bits / learned word embeddings / continuous goal vector | ~2 min |
| 4 | `flow-bc` | flow policy reading the force history to find a hole by feel | ~40 s |
| 5 | `force-token` | the repo's Stage-1 gantry pipeline (ACT/MLP, force token on/off, DART noise) | ~2 min |
| 6 | `attention` | fetch the value stored under a key with one attention layer - watch Q, K and V form | ~2 s |
| 7 | `vit-mnist` | read digits with patches, a CLS token and attention - see where CLS looks | ~8 s |
| 8 | `cross-attn` | turn "23 Oct 2026" into "2026-10-23" - see which source character each output reads | ~12 s |
| 9 | `find-digit` | a command token reads an image: is the queried digit left, right or absent? | ~1 min |
| 10 | `ta-vla` | two-stream transformer with a torque token: image + words + state in, actions out | ~4 min |
| 11 | `td3` | hidden-hole insertion learned from reward alone | ~25 min |
| 12 | `td3-bc` | TD3 with demonstrations in the buffer and an imitation term | ~30 min |
| 13 | `residual-rl` | frozen flow policy plus a TD3-learned residual | ~2 min |
| 14 | `tokenize` | one task, three action interfaces: continuous regression, per-dim bins, autoregressive tokens — with a hands-on tokenization playground | ~30 s |

The ladder page shows each card's estimate at the default parameters and the state of its newest
run (`GET /api/tasks`: `est_label`, `last_state`). Runs of `td3`/`td3-bc` are long; they and any
other run survive closing the browser (the job is a detached process; reattach from the run list).

Task 5 lists two **reference checkpoints** (the repo's `outputs/s1/force/best.pt` and
`outputs/s1/noforce/best.pt`, labelled "reference (repo Stage 1)") in its evaluate dropdown. They
are an allowlist on the task spec (`TaskSpec.reference_ckpts`: name -> repo-relative path, label),
exposed as ids `reference/<name>`; no other path outside `lab/data` is ever resolved, and the usual
traversal rules apply. They are listed only when the file exists.

## Build / run / stop

```bash
make lab-build   # build fvb-lab:latest (FROM fvb-train; needs fvb-train built: make build-train)
make lab         # docker compose up -d lab; prints the URLs
make lab-stop    # stop the service
make lab-test    # builds the image if needed, then pytest tests/lab in a one-off fvb-lab container (not the running service)
```

Tests run in the fvb-lab image (the dev image has no fastapi); `make lab-test` runs `lab-build` first.
`make lab` does not rebuild a stale image: run `make lab-build` first after Dockerfile changes.

URL: `http://desktop153-jv.marekrosa.online:8430` (any host: `http://<host>:8430`) (port bound on 0.0.0.0). The service has `restart: unless-stopped`, so it
survives reboots until `make lab-stop`. It runs as uid 1001 (this machine's owner uid; adjust `user:` in compose if different) with the repo mounted at
`/workspace`. After editing server or task code, `docker compose -f docker/compose.yaml restart lab`
(task modules are imported at startup; `calib.json` is re-read per request).

## Data layout

Everything the app writes lives under `lab/data/` (gitignored):

- `lab/data/runs/<task>/<run_id>/` - `config.json`, `status.json`, `metrics.jsonl`, `job.log`, `ckpt/`
  (`last.pt`, `step_*.pt`, `index.json`)
- `lab/data/evals/<task>/<eval_id>/` - `config.json`, `status.json`, `result.json` (rollouts for the viewer)

## Time estimates and calibration

Each task's `estimate(params)` returns seconds from its entry in `lab/tasks/calib.json` (re-read per
request, no restart needed); `estimate_label` formats it (`~15 s`, `~3 min`, `~1.2 h`; `long run`
from 45 min, `overnight job` from 1.5 h). The entry holds measured rates, e.g. for `bc-mse`
`steps_per_s` (per-model steps per second; the run trains two models, so about half the raw
optimizer rate), `collect_s_per_ep`, `overhead_s`; other tasks add `eval_s`, `s_per_epoch`, ...
(see each task's `estimate`).

Re-calibrate (different GPU, changed model size): start a default run of the task via the UI or
`POST /api/tasks/<id>/runs`, read `wall_s` from its `status.json`, solve for the rate in the
entry (e.g. `steps_per_s = steps / (wall_s - overhead_s - collection)`), then check
`POST /api/tasks/<id>/estimate` is within ~2x of the wall time. Measured on an RTX 3080 (shared):
`bc-mse` default run 12.2-12.8 s, `steps_per_s` 354.

## Adding a task

1. `lab/tasks/taskN_<name>.py` with a `TaskSpec` (see `lab/tasks/base.py`): `id`, `order` (ladder
   position), `title`, `subtitle`, `params` / `eval_params` as `Param(...)`, `charts`, `viewer`,
   `estimate`, `train`, `evaluate`, `lesson`; end with `registry.register(SPEC)` and import the
   module in `lab/tasks/__init__.py`. Param names must not repeat between `params` and
   `eval_params`. Optional: `check` (cross-param validation of the typed train params, raise
   `ValueError` -> HTTP 422), `vecgrids=True` (the viewer draws vector grids; shows the toggle),
   `reference_ckpts` (allowlisted checkpoints outside `lab/data`). `live` names a `LabViewers` renderer that is fed the newest metrics row carrying the key
   `attn` (the live attention panel under the charts).
2. `Param.kind`: `int|float|choice|bool|ckpt`. `ckpt` takes a checkpoint of another task
   (`ckpt_task`); the server validates the id and swaps in the absolute path. `long=True` flags
   hours-long settings.
3. `train(ctx)` receives a `RunContext` (`lab/core/runctx.py`): `ctx.params`, `ctx.log(step, **metrics)`
   (one monotonic step stream -> `metrics.jsonl` -> the charts; a chart spec may set `x_key` to plot against another row field, e.g. embeddings' per-arm `arm_step`, so phases overlay), `ctx.set_status` (progress text goes under the key `message`; the terminal `ctx.done(msg)`/`ctx.fail(msg)` write `msg`, which wins in the UI), `ctx.save_ckpt(step, payload, label=, metrics=)`
   (-> `ckpt/index.json`), `ctx.should_continue()` (poll it so pause/stop work; on stop set
   `state="stopped"`), `ctx.done(msg)`. Gradient loops can use `lab.core.loop.train_loop`. Checkpoint
   payloads must be self-describing (a `meta` dict) so `evaluate` rebuilds the model from the file
   alone, never from form values.
4. `evaluate(ctx)` loads `ctx.params["ckpt"]` and writes `result.json` (+ optional files such as
   an mp4, served at `/api/evals/<task>/<eval>/file/<name>`). The payload is the contract with the
   viewer: document its schema in the module docstring (see `task4_force_token.py`).
5. Viewer: add or reuse one in `lab/web/viewers.js` (`LabViewers.register(name, fn(result,
   container, ctx))`); it must degrade to a JSON fallback on unknown payloads.
6. Lesson: `lab/web/lessons/<id>.html`. Math in KaTeX, delimiters `$$...$$` (display) and `\(...\)` (inline); a bare `$...$` is not typeset. Grounding rule: every
   number on the page is either measured by this app (say which run) or cited; anything else is
   written as an explanation, not a result. Refer to other tasks by ladder order ("task 4
   (flow-bc)").
7. Code snippets: a lesson shows real code in collapsed `<details class="code">` blocks:
   `<pre><code class="language-python" data-src="repo/relative/path.py">` with HTML-escaped text and
   elisions written exactly `# ...`. Every other line must exist in the `data-src` file (compared
   whitespace-stripped); `tests/lab/test_snippets.py` fails on drift.
8. `calib.json` entry for the task, and a test `tests/lab/test_taskN.py` (spec shape, a tiny train
   -> checkpoint -> evaluate round trip through the real code).

## Operations

- `make lab-build` / `make lab` / `make lab-stop` / `make lab-test` (above). After editing server
  or task code: `docker compose -f docker/compose.yaml restart lab` (modules are imported at
  startup; static files and `calib.json` are served fresh). **Restarting the container kills every
  running job**; at startup the server marks all interrupted (non-terminal) runs and evals `failed`
  ("server restarted"), and `job.pid` records the process start time so a reused pid never looks alive.
- Training and evaluation are detached jobs (`lab/run_job.py`); closing the browser never stops
  them, and an overnight run keeps going and shows up in the run list on return. Stop/pause via the
  UI buttons.
- Lint: `ruff check` and `ruff format --check` on the dev image; JS: `node --check lab/web/*.js`.
