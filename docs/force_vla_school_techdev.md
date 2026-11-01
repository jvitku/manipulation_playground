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

**18.8 — a position head cannot even reach the plate.** Both position-head arms scored 0/100
with zero contact steps (peak force ~0.5 N): they hover. Found by the eval's contact-step count,
then reading the wipe expert: its engage step is `dp = 0` with a force target, and its wipe
steps keep `dp_z = 0` — the descent to the surface exists only in `f_target`, executed by the
admittance controller. A position-head clone learns `dp_z ≈ 0`. Consequence for reading the
board: on the wipe task the force output also carries the motion to the surface, so the 99-point
gap measures "can the policy reach the surface at all", not only force tracking. (Same family as
entry 15.7 — what the demonstrator puts in which action channel decides what can be learned.)

**18.9 — the force-tracking bar missed on "firmly".** The full model wiped 99/100 but its contact
RMSE was 1.155 N against the 1.0 N bar: 0.715 N for the 3 N instruction, 1.41 N for 8 N, with
peaks up to 11 N — likely touchdown impacts at the higher target (not yet verified). The task's
own success band (±1.5 N for 80 % of contact) passes; the plan's stricter gate does not. Bar kept;
both numbers reported.

**18.10 — the subtask prompt is load-bearing, measured.** Freezing the prompt at "approach" for the
whole episode dropped the full model from 99 % to 15/30 (firmly 2/14, gently 13/16) — the subtask
word matters most for the 8 N instruction.

**18.11 — spurious force-axis selection, measured.** The risk from 18.7: x/y were selected for
force control on 0.23 % of 2,144 free-space steps — no deadband needed. The z-selection rate in
free space (87.7 %) mixes the legitimate force-controlled descent with any spurious selections
and cannot be split with the current counters.

**18.12 — the gate tilt is real, but not where expected.** The force expert's weight rises from
0.19 (free space) to 0.29 (contact) averaged over token groups — claim confirmed — but the tilt
comes from the vision and text tokens (0.06 → 0.27, 0.13 → 0.34); the force token's own
force-expert weight goes slightly the other way (0.44 → 0.39). Report the breakdown, not just
the average.

**18.13 — budgeting on a saturated host.** Training ran at 9 steps/s instead of the calibrated 25
(load ~115 on 24 cores), so the suite was cut to 8k steps per arm; the 100-episode eval ran as 8
parallel windows over one shared seed block and was merged (counts add exactly; RMSE and gate
means re-weighted by contact steps and decisions). Windowed evals over a fixed seed block
parallelise without changing the seeds.

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

**19.8 — the T3 base starts almost at zero.** The base reached 0/20, 0/20, 1/20 at its periodic
checks and 2/100 on the shared seeds (67 timeouts, 31 force aborts): the hole sits up to ±4 mm
off the visible marker with 0.5 mm clearance, so a behaviour-cloned policy aims at the marker,
lands on the rim and pushes. Consequence: with a success-only reward, 20k training steps saw at
most 2 % successes, the critic went flat (Q ≈ −0.35) and the BC floor held the residual at
2–3 % of its bound. All four M4 bars stayed red (median gain +0 over 5 seeds). Take-away:
runtime RL refines a base that already succeeds sometimes — ForceRFT starts from a working
policy too; from a 2 % base, sparse reward has nothing to propagate.

**19.9 — a reward the policy cannot observe is noise with a sign.** A hole-direction shaping term
(computed from the simulator's hidden hole position, reward-only) made everything worse: force
aborts 37–60/100, peak force +15 %, the residual largest in free space, critic loss ~300× higher.
Lesson 13's shaping worked because the actor observed a hole estimate; here it sees only
wrench, state and the base's intent, so the shaped reward is unpredictable from its inputs. Kept
as a documented knob, default off.

**19.10 — the effect that did appear: push less.** Every plain-reward seed cut force aborts from
31 to 10–18/100 (converted to timeouts) while success stayed flat; in 4 of 5 seeds the mean
vertical residual in contact points upward (+0.013 to +0.031 mm/step, 1–3 mm less depth over a
contact phase). The abort penalty was the only signal training actually collected, and the
residual learned exactly that. It is what the 7 % peak-force drop measures.

**19.11 — shared hosts cost 3× throughput.** 17–20 env steps/s per worker against 66 on an idle
box (load 70–76 from other jobs) — 5 seeds took 24 min instead of 12. Budget 25–30 min for five
seeds on a shared machine.

**19.12 — liveness shown for the wrong reason.** Jobs started in fresh containers recorded
`pid 1`, which in the lab container is its own always-alive init: the UI showed them as running
by luck, and a server restart would mark them failed. Real runs belong to the live lab once the
task is registered.

**19.13 — the oversized bound does its damage while exploring.** With a 5 mm residual bound, 20 %
of the first training episodes ended in a force failure (0 % at 2 mm) and the contact residual
reached 1.25 vs 0.40 mm/step — yet the deterministic eval was identical by 4k steps. On hardware
the exploratory episodes are the ones that break things: watch the training force-failure rate,
not the eval curve.

**19.14 — reward alone had nothing to propagate; corrections did.** Round 2 simulated a deployment:
a privileged operator takes over when the peg stalls on the rim or the force passes 12 N (≤ 20 %
of steps), its corrections supervise the actor, and the critics learn only from autonomous
segments without bootstrapping across a take-over — ForceRFT's recipe. Same base, same seeds:
median success 2 % → 42 % (+40; every seed +34 to +50) against +0 for reward alone, peak force
−24 %. Take-away: the residual's purpose at deployment is to absorb a dense correction signal;
from a weak base a sparse reward alone is not enough.

**19.15 — an intervention rate that hits its budget measures the budget.** The "autonomy rises"
claim failed at a ratio of 0.99: the step-weighted operator rate sat at exactly the 20 % budget
in every bin, because the triggers wanted more than the budget allowed. The robot-side signals
did move (episodes the operator touched 73 % → 60 %; success among untouched episodes 48 % → 71 %).
Take-away: report operator *demand* (triggers fired), not operator *supply*, or set the budget
above demand.

**19.16 — a trigger that fires on slowness can never step back.** With an unlimited operator, the
trained residuals were taken over in 86–98 % of episodes at the same step as the base, yet alone
they succeed 36–52 %: "1 s in contact without 1 mm of depth" is also what a careful learned
search looks like. A simulated operator should step in on failure (force rising toward abort, no
progress for a long time), as HIL-SERL's humans do.

**19.17 — imitated contact corrections leak into free space.** The residual was no longer quiet in
free space (0.40 vs 0.44 mm/step in contact): corrections exist only in contact states, and BC
toward zero at a 50 % batch share did not pin the free-space output. Free-space states need more
weight toward zero (and torque inputs scaled so the actor can read the hole direction).

**19.18 — what the residual actually fills: a channel gap again.** The base stalls ~15 mm deep in a
hole it has found, with dp_z ≈ 0, because its demonstrations pushed through the force target a
position-head base cannot emit. The operator pushes in position mode, and that vertical gap is
what the residual learned — the same "which action channel carries the motion" lesson as 15.7
and 18.8.

## Lesson 20 — LoRA on a large pretrained VLA (M5)

Runs: T3 suite `lab/data/runs/lora-vla/261101-210003-27ae` (81 min) + eval `261101-222133-94d9`;
pre-registered T0 fallback suite `261101-222627-da39` + eval `261101-234737-865a`
(`outputs/lora_vla/PREREG.txt`). SmolVLA 450 M, batch 16, 5000 steps per arm.

**20.1 — lerobot 0.6.1 cannot live in the lab's Python.** The brief's recipe was a venv made from
the lab interpreter. Found by: reading the installed lerobot's wheel METADATA in the `fvb-vla`
image before writing the Dockerfile — `Requires-Python: >=3.12`; the lab image is
`python:3.11` (robosuite's numpy<2 world). Change: the `/opt/vla` venv brings its own CPython
3.12 (uv-managed python-build-standalone under `/opt/uv-python`; uv is uninstalled from the lab
python afterwards), then torch 2.7.1+cu126, `lerobot[smolvla]==0.6.1`, mujoco 3.3.0, imageio.
The lab's python is untouched (3.11, torch 2.5.1, numpy 1.26). Image 7.79 → 13.93 GB (+6.1 GB).
Take-away: a dependency's Python floor is as binding as its torch pin — check both.

**20.2 — smolvla_base stores the action expert in bf16 too, so lerobot's recipe updates bf16
weights.** Nothing looks wrong in training. Found by: listing parameter dtypes after
`from_pretrained`: the 350 M VLM AND 96.6 M of the expert are bf16 (the expert is built from
the VLM's bf16 text config); only the re-created cross-attention k/v (1.6 M) and the
projections are fp32. AdamW on bf16 weights keeps bf16 moments, and an update below half a
bf16 ulp (~0.4 % of the weight) rounds away. Measured at one peak-LR step of the expert-only
fine-tune: 83 % (T3) / 85 % (T0) of the moved expert elements would have been lost in bf16
storage (72 % in the 100-step pilot). Change: every trainable tensor gets an fp32 master copy,
frozen weights stay bf16, compute runs under `torch.autocast(bfloat16)` (Ampere bf16, no loss
scaling). Take-away: at 450 M, "bf16 training" must mean bf16 compute, not bf16 weights.

**20.3 — an adapter is a drop-in only if it keeps the attributes the host reads.** SmolVLA's
attention does `x.to(layer.self_attn.q_proj.weight.dtype)` before every projection. Found by:
reading `smolvlm_with_expert.py` before writing `nfv.lora`. Change: `LoRALinear` exposes
`weight` / `bias` / `in_features` / `out_features` as read-through properties of the frozen
base (unit-tested with a bf16 base). Take-away: wrap by interface, not just by forward().

**20.4 — LoRA did not cut training memory to 60 %; activations dominate at 450 M / batch 16.**
B peaked at 3.61 GB vs C's 4.74 GB (ratio 0.76, claim NOT CONFIRMED on both suites). Found by:
the per-arm peaks — the head-only arm A (1.6 M trainable) already peaks at 3.21 GB = 0.68 x C,
a floor no adapter method can beat at this batch. Cause: (i) `state_proj` feeds the VLM prefix,
so every mode back-propagates through all 16 frozen VLM layers and stores their activations
(2 cameras x 64 image tokens at 512 px + 48 text + 1 state token); (ii) the 0.91 GB of weights
are resident in every arm; (iii) what LoRA removes is C's fp32 expert gradients + AdamW moments
(~1.2 GB). B's +0.4 GB over A is the 128 adapter paths (the adapter input is upcast to fp32).
D (every parameter) does not fit at all: CUDA OOM after 8.96 GB at batch 16, in fp32-master and
in as-loaded bf16. Take-away: LoRA saves memory in proportion to the frozen parameters'
optimizer state; the paper's 2/3 saving is a GPT-3-scale, optimizer-dominated number. For
activation memory, the levers are batch size and gradient checkpointing.

**20.5 — on T3 every arm force-aborts: the dp-only action space cannot express the expert's
contact phase.** T3: A 0 %, B 0 %, C 3 % on 100 shared seeds; 95-100 % of the episodes end in a
force abort (mean peak |F| 25-26 N), against the nano base's 31 % aborts / 67 % timeouts.
Found by: the outcome counts + splitting the T3 demos by controller mode: 62 % of the demo steps
are hybrid force control in search/insert (f_target_z −3.8 N, dp_z exactly 0), while the
free-space descent runs at −14.7 mm/step in position mode. Cause: the lesson's action is dp xyz
+ gripper (as specified), so the force-controlled phase has no channel; one descent step past
contact commands 14.7 mm x 800 N/m ≈ 12 N, two reach the 25 N abort, and a chunk executes 8
steps open-loop. Change: none to the arms (pre-registered); the pre-registered fallback (every
arm <= 5 % on T3) ran the identical suite on T0. Take-away: the action space decides what can
be imitated — the same channel gap as 15.7, 18.8 and 19.18.

**20.6 — at the floor, "matches within 10 points" is confirmed by vacuity.** On T3 the claim
B ≥ C − 0.10 read CONFIRMED (0 % vs 3 %, gap CI −8.5..+2.7) while B ≥ A + 0.10 failed — two
arms at 0 % "match". Found by: reading the T3 board before the T0 run. Take-away: a matching
claim needs a task where the reference succeeds; the floor fallback was pre-registered for
exactly this, and on T0 (A 13 %, B 99 %, C 100 %) both success claims are decisive.

**20.7 — what full fine-tuning changed IS low-rank (a result, kept here as a calibration).** C's
ΔW = W_after − W_before on three expert attention layers: the top-16 singular values carry
0.88 / 0.83 / 0.98 of ||ΔW||²_F on T0 (0.88 / 0.85 / 0.98 on T3); a random Gaussian matrix of
the same shapes puts 0.07 / 0.18 / 0.07 there. The late o_proj update is essentially rank 5
(singular values 1.21, 1.05, 0.88, 0.74, 0.68, then 0.30). The merge is exact in fp32 (max
|Δv| 3.1e-6 on a real batch) but not in bf16 storage (5.5e-2): keep adapters fp32 and merge
in fp32 before casting.
