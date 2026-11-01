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

**17.7 — round 2's grid stalled on its own job.** The 8-cell evaluation grid stopped after two
cells: "refusing: 1 compute process on the GPU". The driver's idle check counted the agent's own
5-arm suite, launched in parallel to save wall time. Change: overlapping one's own jobs became an
explicit opt-in (`ALLOW_SHARED=1`). Take-away: an idle check cannot tell your jobs from someone
else's — overlap must be explicit, never the default.

**17.8 — new knobs could not run through the live lab.** The running server imports task modules
at startup, so its forms rejected the new parameters (HTTP 422), and restarts are gated. Change:
a driver writes a standard job directory and runs the same job entry point in a fresh container,
so the runs still appear in the app's lists; after the next gated restart the forms carry the
knobs.

**17.9 — faster re-planning fixed the presses, not the score.** Re-planning every 2 steps instead
of 8 took force failures from 45 to 7 to 0 out of 700 episodes, yet fvlmoe moved only 45 → 47 →
48 and the twin gap stayed at −1. Cause: the aborts were real chunk staleness, but stopping a
press only needs fresher *camera* frames, not the wrench — and the binding constraint was always
the button choice (match 45–56 %, chance). Take-away: separate failure modes before crediting a
lever; removing failures that never bound the score leaves the score flat.

**17.10 — classifier-free guidance had nothing to amplify.** Across guidance weight {1, 3} ×
re-plan {8, 2}, button match stayed at 45–48.5 %. Found by measuring the gap guidance works on:
‖v_cond − v_blank‖ / ‖v_cond‖ was 3.4 % before instruction dropout and **1.6 %** after it (cosine
≥ 0.996). Cause: the flow interpolant already reveals which button the demo pressed, so the target
is predictable without the words, and training with blanked instructions converges toward
ignoring them even harder. Take-away: guidance amplifies a difference the training objective
paid for; it cannot create one — measure the conditional-minus-blank gap before tuning w.

**17.11 — the training curve overstated success.** The periodic 20-seed closed-loop points ended
at 70 % and 65 %; the 100-episode held-out evals measured 46 % and 44 %. A 20-seed estimate
carries a ±20-point interval and used a different seed window. The board reads only the
100-episode evals.

**17.12 — an unexplained collapse, honestly left open.** In round 3 the MoE-before-the-trunk arm
fell to 1/100 (88 timeouts, 11 aborts) — confirming the paper's "MoE before the VLM collapses"
row — after scoring 52 in round 1. Two things changed at once for that arm (the dropout retrain
and guidance weight 3), and the one evaluation that would separate them was blocked by a busy
GPU. Take-away: when a pre-registered recipe changes two things, budget the separating
diagnostic up front.

**17.13 — a mechanism quantity was never stored.** Round 1's instruction-following rate could not
be recomputed afterwards: results stored success per instruction and press counts, never the
target per episode. Change: evals now store `button_match`. Take-away: store the quantity a
claim's mechanism depends on, not just the headline rate.

## Lesson 18 — ForceVLA2 (M3)

**18.1 — the paper's progress label cannot drive its own subtask machine.** Symptom: fed the
experts' eq.-10 progress traces, the machine advanced on step 1 of every episode. Found by:
driving the machine with the label over 4 experts × 4 seeds and comparing to the expert's
subtask per step — agreement 0.15–0.61. Cause: eq. 10 restarts at every expert *phase*, not every
subtask, and its free-space factor exp(−2l) already reads 0.82 at l = 0.1 m, so the label sits
near the 0.8 threshold from the start. Change: a hindsight label, k/(n−1) within each run of one
subtask label (served by the dataset as `s_sub`). Take-away: a progress label must restart at
the same granularity the prompt switches on — test it by driving the machine with it *before*
training on it.

**18.2 — distance-based progress fails too.** The linear phase progress (1 − l/l₀) never left
"approach" on the wipe and the peg tasks: force-settling and spiral-search phases do not shrink a
distance. The hindsight label reaches 0.83–0.93 agreement on all four tasks; the residual gap is
a deliberate switch at ~80 % of each subtask — early, never late.

**18.3 — the prompt vocabulary listed words no demo carries.** The wipe and insert phases run to
the episode's end, so "done" is never labelled; the machine would have prompted a word absent
from training. Change: prompts use only the subtasks the demos actually label. Take-away: the
run-time prompt vocabulary must equal the training-label vocabulary.

**18.4 — "exceeds 1" with a bounded s.** The paper advances when s exceeds 1 while bounding s to
[0, 1]. Change: advance at s ≥ 0.8, re-arm only after s < 0.3 — the hysteresis also stops a noisy
s near the threshold from skipping subtasks inside one chunk. Known limit: a one-step subtask run
cannot re-arm the machine.

**18.5 — a residual would make the force bypass redundant.** With a standard residual around
the cross-attention, the force token reaches E'_state anyway. Change: no residual; tested — with
the cross-attention zeroed, the state gradient is exactly 0 while all six wrench axes still reach
the gate and experts. Consequence worth teaching: "experts see only state" still contains
vision, because E'_state is a weighted sum of the prefix's values.

**18.6 — the parameter budget differs from the plan.** Measured +1.13 M (cross-attention 263 k,
experts 789 k) against the plan's ≈ +1.7 M, which matches no d → 2d → d expert configuration.
The lesson quotes the measured number.

**18.7 — risk: sampling noise can switch an axis into force control in free space.** The
controller selects an axis at |f| > 0.5 N — 0.02 in model units — so residue on the f_target
dimensions could flip an axis into admittance mid-air. The lesson-18 eval measures the
spurious-selection rate before any deadband is added.

## Lesson 19 — runtime RL on the frozen VLA (M4)

**19.1 — on the button task the residual had nothing to fix.** Symptom: two 5k-step RL runs left
T1 success flat (0.25 → 0.25, 0.25 → 0.20). Found by: evaluating the frozen bases alone *before*
any RL, with outcome counts — fvlmoe 17/40 successes, 22 wrong button, 1 force abort; the
force-blind base 14/40, 23 wrong button, 3 force failures. Cause: a 1 mm/step press almost never
overshoots, so 55–70 % of failures are the wrong button, and a residual that sees no image and
no instruction cannot tell which button is right. The residual's ceiling on T1 is the
force-failure share — 2.5–7.5 points. Change: lesson 19 moves to the peg task (T3), the plan's
original choice, where the hidden hole offset is exactly what a wrench-reading residual can
search for; the eval also reports success conditional on no non-force failure. Take-away:
measure the base's failure anatomy before you choose what RL should fix.

**19.2 — rendering, not learning, was the bottleneck.** 33.7 env steps/s; timing each component
showed 26 ms of a ~33 ms step went to rendering both cameras, though only the base reads images,
and only every 8th step. Change: render only on the steps the base plans from → 66.5 env
steps/s, with outcomes and peak forces verified identical.

**19.3 — the plan's ±5 mm residual bound was too large.** One exploratory step at 5 mm against
a ~1000 N/m button is 5 N — enough to cross the over-force band of a 3 N threshold in a single
step, so the residual would take over the press instead of correcting it. Change: ±2 mm. Take-
away: scale a residual's bound to the base's step size times the contact stiffness.

**19.4 — the thread count changes the outputs.** The same base eval gave mean peak force 5.385013
at 4 threads and 5.385015 at 1 — CPU reductions summed in a different order. Change: RL runs pin
`torch_threads=1` and record it; a seed is reproducible only with the thread count fixed too.

**19.5 — the behaviour-cloning target moved to visited states.** For a residual the BC target is
always "residual = 0", so demonstrations would only contribute their states — and building the
base's next-action input for them would mean re-running the frozen base on every demo frame.
Change: BC toward zero on the replay buffer's own states, through the demo slot TD3 already has.

**19.6 — the BC floor is enforced in code.** Lesson 12's failure (annealing the imitation weight
to zero) is now impossible to configure by accident: floors below 1.0 are refused.

**19.7 — a scratch file named `profile.py` broke torch.** `cProfile` imports `profile`, which
picked up the script; torch's optimizer import then failed with a misleading circular-import
error. Take-away: never name scratch files after standard-library modules.
