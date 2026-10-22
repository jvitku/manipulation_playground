# The Lab — interactive learning web app for force-aware manipulation

Date: 2026-10-22. Approved in chat by the owner (jvitku); "start implementing" given.

## Intent

A local web app — a guided lab for learning robotic manipulation policies professionally:
imitation learning, generative action heads (flow matching / diffusion), force/torque-aware
policies, TA-VLA, and RL fine-tuning. The owner mostly **reads and watches**, but turns the
knobs a practitioner really turns (learning rate, demo count, reward weights, force on/off,
chunk size) and sees the consequences live. Everything is built on the working code in this
repo (`src/fvb`), so every lesson's claims are backed by experiments that already ran here.

## Goals

- A ladder of 8 tasks, easy → complex, each a self-contained lesson page:
  task context, training data (what, how collected), training method, exact architecture
  with an **SVG schematic showing where the loss is computed, from what, and its formula**,
  and the curated parameter set.
- Live-controllable training per task: **start / pause / resume / stop / reset**, live charts
  of the signals that method actually produces (train loss, fixed-noise val loss, closed-loop
  eval success, reward, critic loss, Q values).
- **Runtime estimate per parameter configuration**, updating as knobs change, calibrated from
  measured throughput on this PC. Every run labeled before start (~2 min / ~40 min /
  ~6 h — overnight). Long runs keep training with the browser closed; the page re-attaches.
- **Checkpoints**: saved during/after training; reference checkpoints from the repo's past
  results where available. A dropdown picks one; evaluation runs server-side with
  **changeable evaluation parameters** (hole offset range, sensor noise, force-zeroed twin,
  episode count, Euler steps…), and shows the rollout: 2-D tasks as an interactive in-browser
  animation (trajectory JSON, scrubber, synced force trace), MuJoCo tasks as rendered video +
  synced plots, plus an outcome table (success/abort/timeout, Wilson 95% CI).

## Non-goals / anti-goals

- Not a code sandbox: the user does not edit model code through the UI.
- No full SmolVLA training in the lab (hours per epoch, VRAM contention); its results are
  linked as reference material in the TA-VLA lesson.
- Anti-goal: a config dump of every hyperparameter. Each task exposes a small curated set.
- Anti-goal: fake/simplified lesson content that contradicts the repo's measured results.

## The task ladder

1. **bc-mse** — Behaviour cloning, the floor. Toy T1 (two expert modes) + T2 closed loop with
   the Regression head (`fvb.ta_tutorial.toy`, `generative.Regression`). Watch MSE converge
   and the policy still collide: averaging multimodal demos. ~1–2 min.
2. **flow** — Flow matching / DDPM fixes it. Same data, generative head; knobs: head type
   (flow/ddpm), Euler/DDIM steps, chunk horizon H, n_exec. T1 scatter + T2 closed loop. ~2–5 min.
3. **flow-bc** — Flow-matching from demos on the 2-D contact peg-search (contact2d T3,
   MLPDenoiser). Convergence is the lesson: train loss flattens early while closed-loop
   success keeps rising (loss is health, eval is the metric). Knobs: demos, lr, steps, batch,
   force on/off. ~2–6 min.
4. **force-token** — The force token on the real MuJoCo gantry insertion (Stage 1 pipeline:
   scripts 09/10/11, ACT-lite). Force vs zeroed twin; the repo's 50/50 vs 5/50 result.
   Eval renders MuJoCo video. ~5–15 min.
5. **ta-vla** — TAVLAMini on contact2d T4 (image + language + state + torque token).
   Knobs: torque position (none/prefix/suffix), zero-torque twin, aux loss weight, demos.
   The block-attention schematic is the centerpiece. ~10–30 min.
6. **td3** — RL from scratch: TD3 on the arm insertion (scripts/18, `fvb.policy.td3`).
   The seed lottery. Knobs: seed, reward weights (force penalty, time cost, shaping),
   exploration noise. ~30 min–2 h, flagged.
7. **td3-bc** — TD3+BC, the recipe that works. Knobs: demo count, bc_weight start/floor/decay.
   ~1–3 h, overnight-friendly.
8. **residual-rl** — RL fine-tuning a generative policy: residual TD3 over a frozen flow
   policy from task 3 (T5 pattern, scripts/42). Depends on a task-3 checkpoint. Long, flagged.

### Additions (owner, 2026-10-22, during execution)

- **Embeddings & conditioning task** (ladder position 3, after flow matching; later tasks shift
  down): on the 2-D contact world with two holes and four commands (insert red / insert blue /
  press gently / press firmly), train the same policy once per conditioning representation —
  one-hot(4), binary (2 bits), text tokens through a learned embedding, and a continuous
  goal vector (target hole x-offset + target press force). Overlaid success curves per
  encoding; eval picks the encoding. Lesson: how the conditioning representation changes
  sample efficiency and generalization.
- **Vector-grid visualization (`vecgrid`)**: reusable renderer drawing any vector/matrix as a
  grid of colored cells (diverging palette: −1 red, 0 black, +1 green; hover shows the value;
  tokens-based so themes/colorblind variants work). Wired into the embeddings task (the four
  encodings side by side) and into the rollout viewers of the early tasks (normalized
  observation, torque history, sampled action chunk, synced to the scrubber). One
  **show/hide toggle per task page**, persisted in the browser, in case rendering drags.

## How it gets made (owner delegated these to the builder)

- **Location**: new top-level `lab/` in this repo; branch `lab`.
- **Server**: FastAPI + uvicorn in a new `fvb-lab` image (`docker/Dockerfile.lab`,
  FROM `fvb-train` + pip fastapi/uvicorn), compose service `lab`, **bound to 0.0.0.0**
  (the owner connects remotely), fixed port 8430.
- **Jobs**: each training/eval run is a subprocess inside the container
  (`python -m lab.run_job <run_dir>`). The run dir is the contract: `config.json`,
  `control.json` (pause/resume/stop commands), `metrics.jsonl` (appended),
  `status.json`, `ckpt/`. Pause = polled control file; stop = graceful save+exit;
  reset = new run. Server tails files; frontend polls (~1 s). Jobs survive browser close.
- **Frontend**: single-page vanilla JS + vendored uPlot, dark/light aware, inline SVG
  schematics, served statically by FastAPI.
- **Estimates**: per-task calibration constants (steps/s, collection s/episode) measured on
  this PC during development, stored with the task spec; estimate = f(params), shown live.
- **GPU**: shares the RTX 3080 (10 GB) with other containers; jobs must tolerate that; the
  UI shows current GPU memory so the owner can judge before starting a long run.
- Original `src/fvb` code is reused, not forked; small non-breaking extensions (e.g. an
  optional hook in a training loop) are allowed where wrapping would mean duplication.

## Left to the builder

Exact lesson texts (grounded in `docs/FINDINGS.md`, `docs/progress/journal.jsonl`, the
Stage 1/2 reports), schematic layouts, chart composition per task, eval parameter sets per
task, calibration methodology, checkpoint retention policy.
