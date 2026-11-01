# Force-VLA school — technical development log

Every change that was needed to make the experiments (and their claims) work, in the order
they were found. Each entry: **symptom → how it was found → root cause → change → what to
take away**. The lessons carry the same entries in their "Tech development" collapsibles;
this file is the master copy. Numbers are measured on the dev box (RTX 3080, fvb-lab image).

## Before any code: validating the plan

**P1 — "robosuite is not in the image" was false.**
Symptom: the plan (v2) justified the plain-MuJoCo gantry by robosuite's absence. Found by: the
independent plan reviewer running `python -c "import robosuite"` in the image — it imports
(1.5.2, installed by the base `fvb` image). Change: the gantry stays, but on its real grounds
(isolates contact physics, exact Δp, MJCF is itself lesson material). Take-away: verify an
environment claim by executing it, never by reading a Dockerfile you think you remember.

**P2 — "EGL rendering proven" was the wrong mechanism.**
Found by: the same review — the lab service runs with `MUJOCO_GL=osmesa` and without the
`graphics` driver capability; only an unused dev service has EGL. The existing rollout videos
were software-rendered all along. Change: measured osmesa at 96×96 (2.75 ms/frame, 363 fps) and
adopted it as the course renderer — rendering never competes with training for the GPU.
Take-away: check the *deployed* service config, not the one you would have written.

**P3 — the JAX parity recipe could not run.** `flaxformer` is not on PyPI. Change: the fixture
generator imports the vendored copy inside the pinned ForceVLA clone (PYTHONPATH), runs once on
host CPU, and ships an `.npz` fixture so tests never need JAX.

**P4 — eval forms could not take checkpoint parameters.** The server resolved kind-"ckpt"
params only for training jobs; the plan's compare/twin/cross-module claims needed them at eval.
Change: one shared resolver for both; "optional" = `default=None`, because task 13's *required*
checkpoint already declares `""` (a falsy default) — the naive rule would have broken it.

**P5 — a whole package silently missing from a commit.** `src/nfv/data` was created, tests
passed, the commit "succeeded" — and the package was not in it. Found by: reading
`git show --stat` after the commit. Cause: an unanchored `data/` line in `.gitignore` matched
every `data/` directory in the repo. Change: anchored to `/data/`. Take-away: after committing
new directories, read the commit's file list.

## Lesson 15 — the contact testbed (M0)

**15.1 — the timing contract was arithmetically impossible.** The brief asked for "50 physics
steps per controller tick"; 2 ms physics against a 200 Hz controller is 2.5 steps per tick.
Change: alternating 2/3 substeps (20 ticks = exactly 50 steps = 100 ms), asserted in a test.

**15.2 — force-controlled wiping chattered.** Symptom: in-band time 30 % at the worst surface
tilt. Found by: the gate script's in-band metric. Cause: the repo-default stiff contact at
0.1 m/s sliding. Change: a compliant wipe tip (`solref 0.02 1`) and a 0.05 m/s sweep → ~90 % in
band, T2 mean peak 13.7 → 7.7 N. A real wipe pad is compliant too.

**15.3 — the position baseline aborted at every stiffness.** A naive position-mode press
integrated its command down to the joint floor and hit the 25 N abort for any K_p (command
windup). Change: the baseline commands a fixed depth relative to its own command; documented in
its docstring as an insight in itself.

**15.4 — a claim that could never be decided.** c3 was briefed as "position-mode in-band time ≤
20 %". Measured: in-band time has a geometric floor (band width / contact stiffness ≈ 5 mm over a
16 mm hidden height range) → 31.6 %; the per-episode "held the band" rate is 15/100 with a CI
that straddles 0.20 for any affordable n. Change: a two-arm claim on T2's own criterion — hybrid
minus position ≥ 0.5 → 100/100 vs 15/100, gap 85 % (CI 73–91 %) → CONFIRMED. Take-away: a claim
needs a metric whose noise floor is below its margin.

**15.5 — "INCONCLUSIVE" at the default episode count.** The expert scored 20/20, yet c1 read
INCONCLUSIVE: the Wilson CI (83.9–100 %) straddles the 0.95 bar. At 100 episodes: CONFIRMED.
Kept as the lesson's teaching point about what a perfect small sample does not prove.

**15.6 — the "deliberately wrong" exercise was not wrong.** K_a × 10 was meant to oscillate;
measured, it holds 5.00 N (peak 5.09) — the 5 cm/s clamp and the compliant pad absorb it.
Change: the wrong exercise became a sign flip (tool climbs at the clamp, F_z stuck at −0.02 N).
Take-away: verify the failure you teach.

**15.7 — the T1 expert taught nothing a position policy can learn** (found in lesson 16, fixed
here). The press expert pressed through the *hybrid* controller: every recorded press step had
`dp = 0` and the descent lived in `f_target`. Change: a pure position press (1 mm per 10 Hz step
from a 6 mm hover, privileged stop on the click latch) — the stop moment stays visible only in
the wrench. Re-verified: worst-case threshold 3 N → 100/100, peak/threshold ≤ 1.13 (bound 1.5);
M0 gate re-passed 100/100 on every task; a unit test now guards that press-phase `dp_z < 0`.

## Lesson 16 — nanoVLA (M1)

**16.1 — an all-padding instruction produced NaN.** Blanking the text (the no-language arm) made
the text self-attention softmax over an all-masked row. Change: "blank" = one constant `<unk>`
token.

**16.2 — round 1: T1 at 3/100 instead of the designed 30–70 %.** Found by: the claims board (c5
red, *below* the window). Investigation: the policy cloned the demos perfectly (loss 0.0037) and
hovered 15 mm above the cap until timeout (92/100 timeouts; it touched a button in 8/100).
Cause: entry 15.7. After the fix: T1 45/100 (CI 35.6–54.8) → c5 CONFIRMED, and the failure mix
finally looks like T1 (45 wrong button, 10 force aborts, 0 timeouts). Take-away: the board's
job is exactly this — a red verdict is a measurement, and this one found a data bug.

**16.3 — the instruction is ignored (c6).** With pressing fixed, the language arm and the
blanked arm tie (45 vs 46/100); button choice is at chance; their training losses are equal
(0.0042 vs 0.0040). Mechanism: in flow matching the noisy interpolant x_τ already leaks which
mode the demo took, so the words reduce no loss; and a wrong button costs nothing at training
time. Round-2 lever (instruction dropout + classifier-free guidance) is measured in lesson 17.

**16.4 — the claim c4 sits inside its own confidence band.** T0: 88/100 on CPU and 92/100 on GPU
for the same seeds (per-device noise streams in the flow sampler); 20 k steps → 97/100; then a
"500-episode" eval reported 200 episodes — the form's `episodes` cap (200) silently clipped the
request. Change: cap raised to 1000. With 400 demos: 484/500 = 96.8 %, CI 94.9–98.0 — the bar
(0.95) lies inside the band by 0.1 pt, so the verdict stays INCONCLUSIVE, accepted as honest.
Take-away: read back what actually ran, not what you asked for.

**16.5 — a test failure that only existed in suite order.** Importing the `nfv.model.build`
*submodule* rebound the package attribute `build` (the function) → `TypeError: 'module' object is
not callable`, only when another test imported the submodule first. Change: the package
`__init__` loads the submodule, then rebinds the name.

**16.6 — wall time is not GPU time.** The suite took 86 min instead of ~33 because another job
had the host at load 60–74: the batcher is CPU-bound. Change: none needed — but budgets now quote
"idle box" and the calibration records the measured rate.

## Lesson 17 — ForceVLA (M2)

**17.1 — parity needed a dtype pin.** The reference rounds router logits and expert inputs to
bfloat16 (a TPU artefact); the port is fp32. Change: the fixture generator pins those two dtypes
to fp32 to compare the math → max |port − reference| = 9.5e-07.

**17.2 — the capacity cap works at training and vanishes at inference.** Instrumentation showed
62–67 % of tokens dropped by the capacity cap during training, 0 % at act time. Cause: the
reference's own grouping rule — at batch 1 the 85 tokens factor into groups of 1, where the
capacity floor of 4 can never bind. (A "17-token group" is likewise impossible: 17 is prime.)
Ported as-is, documented; the routing stats report both regimes.

**17.3 — the reference's inference path cannot run.** `sample_actions` concatenates a `None`
prefix into the LIMoE call while the cached prefix is unused. The port uses one definition for
loss and act.

**17.4 — round 1: force did nothing.** Every arm landed at 43–56/100; fvlmoe 45/100 = exactly the
force-blind bar; zeroing the force changed nothing (twin +1). Diagnosis, measured: the language
ceiling from 16.3 caps every arm near the 50 % button lottery; one 10 Hz wrench sample per
0.8 s open-loop chunk arrives too late to gate a press; behaviour cloning gives no incentive to
use force at all. Notably fvlmoe's flow loss was 29 % lower and bought zero behaviour.
Take-away: loss is not the metric; hiding the quantity from the camera is necessary, not
sufficient. Round 2 (below) attacks the first two causes.

**17.5 — "shuffle the force" was a no-op.** At batch 1 a random permutation is the identity —
and on CUDA `randperm(1)` still consumes a generator draw (on CPU it does not), so the measured
±2 pt jitter was RNG-stream displacement, not corrupted force. The lesson states the two-layer
mechanism.

**17.6 — a restart stamped a healthy run "failed".** The lab's startup sweep marks every
non-terminal run dir failed; an agent's out-of-app training was running at the time (the jobs
check printed it, but the shell chain did not gate on it). The training was unaffected and its
own `done()` repaired the status. Change: restarts only behind a probe that exits non-zero on
any active job.

**17.7 — round 2 (instruction dropout + CFG, reactive re-planning):** results pending — filled
in when the re-run lands.
