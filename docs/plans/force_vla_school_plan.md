# The force-VLA school: lessons 15–19 (branch `force_vla2`)

31 Oct 2026 · v2 — the uploaded `nano_force_vla` plan (source: the user's PLAN.md export of
claude.ai/artifact/RkwDwB6hjRxP7z2DidYJuW) validated against this repository and rewritten to
live **inside the Manipulation Lab** instead of a new repo. Changes from v1 are collected in
§1; everything not listed there is kept from the source plan.

## 1. Validation round 1 — findings and corrections

| v1 assumption | Found | Consequence |
|---|---|---|
| Target GPU is a GTX 1080 Ti (11 GB, sm_61, fp32) | The dev box is an **RTX 3080 10 GB (sm_86)**; the 1080 Ti is the portability target | Keep the fp32-only, <10 M-param, <3 GB discipline (it is what makes the course portable AND leaves room next to the lab's other jobs on 10 GB). Budgets below re-stated for the 3080; 1080 Ti numbers are ~2.5× longer |
| Pin `torch==2.14.0+cu126` (last Pascal wheels) | The `fvb-lab` image ships **torch 2.5.1+cu124**, which still includes sm_61 | No pin change needed here; the pin note stays in the lesson text for learners on Pascal. fp32 only, no autocast (also: bf16 is pointless on Pascal and the 3080 gains little at these sizes) |
| MuJoCo 3.x + EGL offscreen must be set up | **mujoco 3.3.0 in the image; EGL rendering proven** — tasks 5/11/12 already render rollout.mp4 on this GPU | M0 starts from a working renderer |
| "Frontend: open question — confirm the pc_robot app" | Confirmed: it is this repo's lab (FastAPI + static web, port 8430, 14 lessons) | §3: full integration, no new app, no SQLite, no WebSockets |
| New repo `nano_force_vla/` | The repo already has `src/fvb` with `ft/` (force compensation, filters, frames), `scenes/builder.py` (plain-MuJoCo XML building), `policy/td3.py` (the exact TD3+BC used by lessons 11–13), `vla/lerobot_convert.py` | New code goes to **`src/nfv/`** in this repo and *reuses* those; the robosuite-based `fvb/envs` are NOT used (robosuite is not in the fvb-lab image — the plan's "no robosuite" rule is confirmed and kept) |
| Prior-work numbers (gantry 50/50 vs 5/50; TA-VLA force vs zeroed) | Match `docs/FINDINGS.md` and the force-token / ta-vla lessons | Motivation section can cite the lessons directly |
| JAX parity test as an optional extra at test time | jax/flax are not in the fvb-lab image | Parity by **fixture**: run the reference JAX `LIMoEBlock` once on host CPU (`uv run --with jax,flax,flaxformer`), dump input/weights/output to `tests/nfv/fixtures/limoe_parity.npz`; the repo test compares the PyTorch port against the fixture, no JAX at test time |
| 15 Hz / 480×640, 8×A100 recipes | n/a here | Already replaced by the nano recipe in v1; unchanged |

New requirement from the user (31 Oct): **ablations are first-class and the papers' claims are
validated automatically by the course** — §6.

## 2. Goal (unchanged) + the teaching contract

Show, with numbers measured on this PC, two gaps: a force **input** beats vision alone where
contact carries information the camera cannot see (ForceVLA, arXiv 2505.22159), and a force
**output** through a hybrid controller beats a force input alone where the task demands a
target force (ForceVLA2, arXiv 2603.15169). Module 4 adds runtime RL on the frozen model
(ForceRFT 2609.22840, DSRL 2506.15799 as a flag). Every design choice exists to make the two
gaps large, reproducible and cheap to ablate. The expected ordering nanoVLA < nanoForceVLA <
nanoForceVLA2 is a design target and the evaluation must be able to show it failing.

The course continues the existing ladder: the learner arrives knowing flow matching (2, 4),
action chunking (4, 5), tokens/attention/two-stream VLAs (6–10, force token in 5, TA-VLA in
10), TD3/residual RL (11–13) and tokenization (14). Lessons 15–19 assume exactly that and
cross-reference it.

## 3. Lab integration (replaces v1's "Frontend demo" and "Repository layout")

**No new app.** Five new lessons, orders 15–19, each a normal lab task (TaskSpec, train/eval
jobs, lesson fragment, viewer). Everything the v1 frontend section asked for maps onto an
existing lab mechanism:

| v1 panel | Lab mechanism |
|---|---|
| Module ladder, estimates, run status | the existing ladder (`/api/tasks`, calib estimates) |
| Train panel, config form from a dataclass, live curves, ETA | TaskSpec params → generated form; charts from metrics.jsonl; existing pause/stop; the verbose job phases |
| Evaluate, checkpoint picker, Wilson intervals, twin button | eval form; `ckpt` picker (ISO-stamped names); result.json carries Wilson CIs; the twin is an eval param `force_noise = off/zero/shuffle/gaussian` so one extra eval = the twin row |
| Live rollout view, force traces, gate bars, subtask, controller mode | post-hoc per-episode **`fvrollout` viewer**: rollout.mp4 (the proven video path) + synchronized force/target traces, expert-gate bars, subtask string, per-axis controller mode, with a scrub slider + prev/next (the house `stepButtons`) |
| Compare, 3 checkpoints lock-step on one seed, shared force axis | eval params: optional `ckpt_b`, `ckpt_c` (kind "ckpt", same task) — one eval job rolls all checkpoints on identical seeds/physics and the viewer renders side-by-side columns with one force axis |
| RL panel: per-seed curves, residual magnitude | charts (one series per seed) + the rollout viewer's residual trace |
| SQLite registry, WebSockets, JPEG streams | not needed: run dirs + status.json + metrics.jsonl + 400 ms polling already survive reloads |

**Repository layout** (inside manipulation_playground):

```
src/nfv/                  # plain python + mujoco, importable in the fvb-lab image
  sim/      gantry_xml.py (build the MJCF string), env.py, sensors.py, controllers.py,
            tasks.py (T0..T4 definitions), experts.py
  data/     record.py, dataset.py (npz shards -> RAM uint8), lerobot_export.py (follows fvb/vla)
  model/    config.py (ModelConfig dataclass = every ablation flag), vision.py, text.py,
            trunk.py, expert.py, flow.py, force_in.py, limoe_port.py (ForceVLA reference port),
            forcevla2.py (cross_attn, moe3, hybrid_head), build.py (registry: config -> module)
  rl/       residual.py (actor), loop.py (runtime TD3+BC on frozen base; reuses fvb.policy.td3)
  claims.py # the claims registry (§6)
  seeds.py  # demo / train / eval seed ranges, disjoint
third_party/forcevla/     # read-only reference clone at 79de88f (Apache-2.0), parity source
lab/tasks/task_nfv0_testbed.py ... task_nfv4_rl.py     # thin TaskSpec wrappers over src/nfv
lab/web/lessons/{force-sim,nano-vla,nano-force-vla,nano-force-vla2,force-rl}.html
lab/web/viewers.js        # + fvrollout, fvcompare, claims viewers
tests/nfv/                # unit tests (sim, controllers, model shapes, parity fixture, claims)
tests/lab/test_task_nfv*.py   # task-level tests in the house shape
```

**House conventions checklist** (every new lesson, enforced by the existing guards + review):
"The question" opener · Motivation box · Theory box (both collapsed, end of method prose) ·
schematic SVG with `data-i` + blkinfo popup for EVERY block (purpose, shapes, update frequency
— the user asked for extra depth here) · ≥3 verbatim snippets with `data-src` (drift guard) ·
tensor-shape legends + `[B, ...]` inline comments in the quoted sources · citations as links ·
"What to watch" with honest measured numbers · Knobs matching the TaskSpec · text ≤6.62 px/char
inside SVG boxes · KaTeX only `$$...$$`/`\(...\)`.

## 4. The models (kept from v1, d = 256 trunk; deltas only)

The v1 architecture table stands (nanoVLA ≈ 7.0 M, +ForceVLA ≈ 8.9 M, +ForceVLA2 ≈ 8.7 M);
implementation deltas:

- The nano trunk is written fresh in `src/nfv/model` in the readable style of `lab/tasks/minitf.py`
  (the lessons quote it, so it must read like lesson material), but is its own code — minitf
  stays the attention school's teaching artifact.
- `limoe_port.py` ports `LIMoEBlock` + the `Pi0_Guidance` force path (encoder block, top-1
  tokens-choose router with jitter 0.01 and the group-of-16 capacity cap, ReLU experts,
  residual, out-proj, **last-H sum into the expert output**) with the reference's names kept
  and the Apache-2.0 notice. Parity: the npz fixture test (§1). The port also logs the
  fraction of tokens dropped by the capacity cap.
- ForceVLA2 modules from the paper (no code exists); where the paper is silent the choice is
  written in the docstring and the lesson: the hybrid controller law (per-axis selection
  `S_i = 1[|f_target,i| > 0.5 N]`, admittance `ẋ_i = K_a (f_target,i − f_meas,i)` clamped to
  5 cm/s, 0.1 N hysteresis, 25 N abort), progress threshold with hysteresis, H = 16 with 8
  executed steps.
- All switches in one `ModelConfig`: `force_in = none | prefix_linear | prefix_moe |
  concat_after | fvlmoe | forcevla2 | prompt_bins`; `moe = top1_capacity | top1 | soft`;
  `n_experts`; `balance_loss`; `expert_inputs = all | vision_only | none`; `force_prompt`;
  `head = position | hybrid`; `force_noise = off | zero | shuffle | gaussian`. Every variant
  is a config, never a subclass.

## 5. Simulator, tasks, demonstrations (kept; reuse notes)

Gantry (x, y, z, yaw + parallel gripper) in plain MuJoCo 3 XML built by `gantry_xml.py`
(style of `fvb/scenes/builder.py`); wrist F/T site, tool-weight compensation and filtering
reuse `fvb/ft/compensate.py` and `filters.py`; sign convention documented once (contact
pushing up on the tool reads negative F_z). Two cameras (static + wrist) at 96×96 EGL. 10 Hz
policy, 200 Hz controller, 2 ms physics. Controllers as in v1 (position PD; hybrid admittance
as above; 25 N abort = failure).

Tasks T0 reach/grasp, T1 press-until-click (hidden 3–9 N threshold), T2 wipe at a commanded
force (3 N "gently" / 8 N "firmly"; hidden surface height ±8 mm, tilt ±3°), T3 peg insert
(hole offset ±4 mm off the visible marker, 0.5 mm clearance), T4 knob-to-stop (optional).
Scripted privileged experts with the earlier project's action noise; 200 demos/task default
(50/100/400 for the data curve); npz shards + RAM uint8 (~1.6 GB/task); LeRobot v3 exporter
after `fvb/vla/lerobot_convert.py`. Demo, train and eval seeds disjoint (`nfv/seeds.py`),
100 eval seeds, every rate with a 95 % Wilson interval.

## 6. The claims board — ablations first-class, validated automatically (NEW)

Every quantitative claim the course makes — the papers' ablation rows scaled to mini, and the
course's own design targets — lives in **`src/nfv/claims.py`** as data:

```python
Claim(id="fv.table3.fvlmoe_vs_none",
      statement="FVLMoE beats no-force on T1 (paper: 80% vs 45% on plug insertion)",
      module="nano-force-vla", task="t1",
      arms={"a": {"force_in": "fvlmoe"}, "b": {"force_in": "none"}},
      metric="success", expect="a-b >= 0.15",   # expected direction + margin, mini-scale
      source="ForceVLA Table 3, arXiv 2505.22159")
```

Mechanics (pure house patterns):

- Each module task's train form has `preset = paper_default | <named ablation rows> |
  claims_suite`. `claims_suite` trains every arm its module's claims need **in one job**,
  sequentially with fresh seeds (exactly how task 3 trains four encoding arms and task 14
  trains three heads), checkpointing each arm; the form's estimate shows the real cost
  (a full module suite ≈ 2–3 h on the 3080 — an explicit long run like task 11).
- The suite's evaluate rolls every arm on the shared held-out seeds and writes
  `result.claims`: per claim — both arms' metrics with Wilson CIs, the expected relation, and
  a verdict `CONFIRMED / NOT CONFIRMED / INCONCLUSIVE` (inconclusive when the CIs overlap the
  margin). A `claims` viewer renders the board: statement, paper source (linked), expected,
  measured, verdict chip (green/red/grey).
- The lesson's "What to watch" embeds the claims board reading; the module's **gate** is a
  named subset of claim ids. Twin and force-to-noise checks are claims like any other, so
  "is force actually used" is validated by default, not as an extra.
- Single runs stay available: `preset = paper_default` trains one model fast; every named
  ablation row is also a preset so the learner can reproduce any single bar.

Claims shipped by default: ForceVLA Table 3 (5 force-entry routes on T1), the twin gap, the
FACT force-to-noise drop, ForceVLA2's build-up chain (+prompt, +encoder, +MoE on T2/T3),
`expert_inputs` ablation, force-into-trunk collapse, the missing-from-the-paper
"ForceVLA2 inputs + position head" row, hybrid-head force RMSE < 1 N on T2, M4's residual
gains (median over 5 RL seeds, per-seed curves), and the T0 no-regression checks.

## 7. The five lessons (orders 15–19)

| # | id | title | what it adds | default run on the 3080 |
|---|---|---|---|---|
| 15 | `force-sim` | The contact testbed | gantry, F/T sensing (sign, gravity, filtering, what is invisible at control rate), position vs hybrid control, scripted experts, demos; viewer: `fvrollout` on expert episodes; "a position controller cannot hold a force on an unknown surface" measured (K_p 150 vs 800 wipe trace) | demos ~5 min; controller demos ~1 min |
| 16 | `nano-vla` | nanoVLA: a tiny VLA from scratch | vision/text/trunk/expert/flow on T0+T1; chunking ablations; why generative (T1 two modes); claims: language needed on two-button T1, Euler steps, chunk/exec trade | ~15 min train |
| 17 | `nano-force-vla` | ForceVLA: force as an input | φ_F + the LIMoE port; where-force-enters claims suite (Table 3 mini); twin + noise claims; expert-usage instrumentation (which expert fires in contact vs free space) | ~20 min; suite ~2.5 h |
| 18 | `nano-force-vla2` | ForceVLA2: force as an output | force prompt + subtask machine, cross-attn encoder with bypass, 3-expert dense MoE, hybrid head + controller; claims: build-up chain, position-head control row, force RMSE, gate-tilt in contact | ~25 min; suite ~3 h |
| 19 | `force-rl` | Runtime RL on the frozen VLA | residual TD3+BC (reuses `fvb.policy.td3`, the 11–13 recipe: BC floor never 0), DSRL flag; claims: T3 median +10 pts over 5 seeds, peak force down, T0/T1 no regression, residual ≈ 0 in free space | ~30 min per RL seed |
|  |  |  | cross-module: data-efficiency curve claim (50/100/200/400 on T3), the three-model T2 compare rollout (the centrepiece) |  |

Each lesson: full house conventions (§3 checklist), schematics with every block's purpose /
shapes / update frequency in blkinfo popups, Motivation box covering *why each technique is in
this experiment*, Theory box with the papers, and the claims board wired into "What to watch".

## 8. Milestones and gates (adapted; each lands WITH its lesson)

1. **M0 testbed + lesson 15.** Gate: expert ≥ 95 % on T0–T3 (100 seeds); hybrid controller
   holds 5 N ± 1 N on T2 at three surface heights; sensor sign + gravity compensation unit
   test; `fvrollout` renders an expert episode end-to-end in the browser.
2. **M1 nanoVLA + lesson 16.** Gate: T0 ≥ 95 %; T1 in 30–70 % (else re-randomise T1);
   20 k steps < 20 min on the 3080; flow unit test (unimodal toy mean); claims board renders.
3. **M2 nanoForceVLA + lesson 17.** Gate: parity fixture < 1e-5; T1 ≥ 80 % with `fvlmoe` and
   twin ≥ 20 pts lower; noise drops T1; `prefix_moe` < `concat_after` (reported either way);
   capacity-drop fraction logged.
4. **M3 nanoForceVLA2 + lesson 18.** Gate: T2 ≥ 75 % with force RMSE < 1 N while
   `head=position` < 20 %; T3 peak force ≥ 30 % below M2; gate weights tilt to the force
   expert in contact (plotted); force-into-trunk collapse reproduced.
5. **M4 runtime RL + lesson 19.** Gate: T3 median success +10 pts over frozen base (5 seeds),
   peak force down, no T0/T1 regression, residual ≈ 0 in free space.

Conventions: fp32 only; every run journaled by the lab itself (run dirs); no success rate
without interval and twin; smoke tests before long runs; `MUJOCO_GL=egl`; tests CPU-only in
the image unless the GPU is verified idle.

## 9. Risks (updated)

| Risk | Mitigation |
|---|---|
| 10 GB VRAM shared with the user's lab jobs | <3 GB budget; eval subprocess separate; check active jobs before suites |
| Force token ignored at image scale (the V9 result) | tasks hide the decisive quantity; twin/noise are **claims**, so the failure is visible by default; T1 re-randomised if vision solves it |
| Position head "solves" T2 by digging in | 25 N abort = failure; force RMSE inside the success criterion |
| MoE collapse to one expert | usage logged per eval; balance-loss flag; capacity-drop fraction reported |
| Claims suite too slow to be run | per-claim presets allow single rows; suite steps knob; estimates honest in the form |
| ForceVLA reference inference bug (None prefix concat) | port follows `compute_loss` + cached prefix; documented in lesson 17 ("reading a codebase against its paper") |
| ForceVLA2 ambiguities copied wrongly | every unspecified choice written in docstring + lesson; claims test behaviour, not prose |
| Robosuite absent from the image | not used; plain MuJoCo only (validated) |

## 10. Sources

ForceVLA arXiv 2505.22159 (code: github.com/ft-robotic/ForceVLA, Apache-2.0, commit 79de88f) ·
ForceVLA2 arXiv 2603.15169 · FACT arXiv 2608.01402 · ForceBench (π0.5 numbers) · ForceRFT
arXiv 2609.22840 · DSRL arXiv 2506.15799 · RoFormer/RT/π0/SmolVLA as already cited in lessons
1–14 · this repo: docs/FINDINGS.md, the force-token and ta-vla lessons, fvb/ft, fvb/policy/td3.
