# FINDINGS — Stage 0

Each section: date, commit, command line, findings with numbers + units, plots, and what it means
for a force-aware policy. Frames: "sensor" = MuJoCo F/T site frame, "world" = MuJoCo world.

## M0 — Repo + container + API probe (2026-09-19)

Commit: initial scaffold (this commit). Command: `make build && make test && make m0`
(host has no `make`; the `docker compose run` equivalent was used). Artifact: `outputs/m0/probe.json`.

- **Versions:** python 3.11.16, mujoco 3.3.0, robosuite 1.5.2, numpy 1.26.4, scipy 1.17.1. Build
  from a clean clone succeeds; `requirements.lock` is the container's `pip freeze`.
- **Offscreen GL works** on CPU with `MUJOCO_GL=osmesa` (64×64 test render OK), so M1/M4 MP4s are
  possible.
- **PLAN §2 confirmed:** `Wipe` obs has `robot0_contact` but no `robot0_eef_force`; the F/T sensors
  are `gripper0_right_force_ee` / `gripper0_right_torque_ee` (3 dims each) on site
  `gripper0_right_ft_frame`; `env.robots[0].ee_force["right"]` returns *exactly* the raw sensor
  values (bit-identical in the probe). Model timestep 0.002 s, control 20 Hz → 25 physics steps
  per action, 100 solver iterations, integrator Euler (0), cone elliptic (1) on the robosuite side.
  BASIC controller → `OSC_POSE`, kp = 150, damping_ratio = 1, impedance_mode = fixed,
  action dim 6 on `Wipe`.
- **PLAN §2 corrected:** `env.robots[0].torques` is `None` in 1.5.2. The applied arm torques are
  `env.sim.data.ctrl[0:7]` (= OSC controller `.torques` = `qfrc_actuator[0:7]`, gear 1). PLAN §2/§5
  and CLAUDE.md updated.
- **Raw Track-A scene:** loads and steps; at rest the free-hanging 0.1 kg peg reads
  `ft_force = [0, 0, +0.981] N`, `ft_torque = 0`, no contacts. So the sensor reports **+Z for a
  load hanging below the site** (the peg pulls down on the wrist; sensor is positive in the
  direction of the force the *child* exerts on the *parent*… or equivalently minus the force the
  parent exerts on the child). M1 derives this properly from motion.
- **Container gotcha:** `numba` and matplotlib caches were created by root at build time; running as
  the host uid failed with `RuntimeError: cannot cache function`. Fixed by `chmod 1777` on
  `/tmp/numba`, `/tmp/mpl` in the Dockerfile and `HOME=/tmp` in compose.

What this means for a force-aware policy: the robosuite "F/T" is just a MuJoCo site sensor — no
noise, no bias, no filtering — so any realism (and every artifact) has to be understood at the
MuJoCo level first, which is what Track A is for.

## M1 — Track A: first insertion and the F/T time series (2026-09-19)

Command: `make m1` (= `python scripts/01_gantry_insert.py --out outputs/m1`, defaults: offsets
0/0.3/1/4 mm, 50 mm/s descent, 20 Hz control, kp = 800 N/m, commanded tip depth 50 mm so the
aligned case bottoms out on the 40 mm-deep hole floor, then lateral pushes 0.3/1/4 mm).
Artifacts: `outputs/m1/offset_*mm.{npz,json,mp4}`, `*_ft.png`, `*_fmag.png`, `*_fz_vs_depth.png`,
`lateral_vs_offset.png`, `lateral_vs_push.png`, `sign_convention.json`, `summary.json`.

**Sign convention (derived, `sign_convention.json`).** Free-hanging 0.1 kg peg: `Fz = +0.981 N`.
Kick the target up: peak peg accel +16.9 m/s² → `Fz = +2.667 N`; kick down: −36.5 m/s² →
`Fz = −2.667 N`. Both equal `m·(g + a_z)` exactly. So the MuJoCo force sensor reports **the force
the parent body (wrist) applies to the child body (peg)**, in the site frame, +Z up here. Contact
pushing the peg *up* (bottoming out, rim) therefore makes Fz *negative*: steady bottomed-out
reading −12.90 N while the ground-truth contact force is +13.88 N up — the difference is exactly
the 0.981 N peg weight the sensor also carries. Sensor Fz = contact Fz − m·g (world = site here).

**Contact-onset transient, physics vs control rate** (`offset_0mm_fmag.png`, `offset_0mm_ft.png`).
At 2 ms physics rate the impact of the 50 mm/s descent onto the floor is a one-step spike of
−12.5 N (−3.4, −0.9, −0.2 N in the following steps: a 4-step, ≈8 ms impulse), after which Fz sits
near 0 N (peg supported by floor, spring not yet wound up). At 20 Hz control rate the sample after
onset reads −0.4 N: **the impact spike is invisible** to a policy observing at control rate. What
the 20 Hz signal does show is the spring winding up in 2 N steps (kp × 2.5 mm per control step),
i.e. a *staircase* rather than the smooth 40 N/s ramp (kp × 50 mm/s). The peak seen at control
rate is 13.27 N vs 13.27 N at physics rate — the peak is the steady wind-up, not the impact.

**Free-space "contamination" preview.** Even before contact the 20 Hz stepped targets excite the
spring: Fz ripples 0.72–1.14 N at physics rate (nominal 0.981 N), i.e. ±20 % of m·g. M2 quantifies
this properly.

**Jam vs insert.** With a rigid Cartesian gantry (no tilt DoF) and a chamferless square hole the
outcome is binary. 0 and 0.3 mm offsets (< 0.5 mm clearance) insert to 40.0 mm with *zero* wall
contact during the descent. 1 mm and 4 mm both land on the rim at depth 0.07 mm and behave
identically: the spring keeps winding up while the target descends, settling at
`Fz = −44.85 N`. Check: `kp × z_err = 800 × 49.9 mm = 39.9 N` (spring) + 4.9 N (wrist link weight
on the same spring) + 0.98 N (peg) = 45.8 N ground-truth contact, minus 0.98 N peg weight seen by
the sensor = 44.85 N ✓. The "jam force" is therefore not a property of the jam but of *how far
the target keeps going after the stall* — a scripted descent with no force feedback is what makes
it large. The sag at rest is 7.36 mm = (0.5 + 0.1 kg)·g / kp, so commanded and achieved depth
differ by that much in free space.

**Lateral force vs offset** (`lateral_vs_push.png`). Pushing the inserted peg sideways:
`Fx = kp × (x_target − x_actual)` to 4 significant figures in every case (0.229/0.752/3.006 N for
0.3/1/4 mm at 0 mm entry offset), and the torque sensor reads `Ty = −Fx × 0.060 m` (the contact
is at the peg tip, 60 mm below the site) — 0.180 N·m at 3 N. Surprise: at 0 mm entry offset the
peg moved only 0.24 mm under a 4 mm push, less than the 0.5 mm clearance, with 4 contacts (floor
only). It never reached the wall: floor friction μ·N = 0.6 × 13.9 N ≈ 8.3 N > 3 N push. The
0.3 mm-entry case did reach the wall (x = 0.494 mm, 9 contacts). A lateral force reading does not
by itself say *what* you are pushing against.

**Tests.** `test_ft_static.py`: hanging peg |Fz| = m·g within 1 % (actual error 1e-7), doubling
mass doubles Fz.

What this means for a force-aware policy: (1) at 20 Hz the contact *impact* is not observable —
only the quasi-static wind-up is, so a "spike detector" must run on physics-rate history
(`ft_raw_hf`) or the detector is really a stiffness×displacement estimator; (2) the magnitude of
the steady force is set by the controller (kp × overshoot), not by the task, so force targets
learned under one kp do not transfer; (3) sign and frame must be pinned before any learning:
here +Fz means the wrist is pulling *up* on the peg.

## M2 — Contamination: gravity and inertia (2026-09-19)

Command: `make m2` (= `python scripts/02_contamination.py --out outputs/m2`, 20 Hz control,
kp = 800, 0.1 kg peg). Artifacts: `outputs/m2/{tilt,hold,descent_50,descent_150,sine_0.5,sine_2,
sine_5}.{npz,json}`, `*_raw.png`, `*_comp.png`, `tilt_{raw,comp}.png`, `report.json`,
`load_params.json`.

**Identification from static poses (tilt 0→90° about y, 10 holds, 100 static samples).**
Least squares gives m = 0.10000 kg (true 0.1), COM = (0, 0, −0.0300) m (true −0.030), static
residual 8e-7 N. At 90° tilt the full 0.981 N moves from Fz into Fx (`tilt_raw.png`) — the
"contamination" is the load weight rotating through the sensor frame, and it is removed exactly
by `m·Rᵀ(−g)` with the identified parameters (`tilt_comp.png`).

**How big is free-space contamination vs M1 contact forces?** RMS of `ft_raw − m·g` as % of
m·g (0.981 N), peak deviation at physics rate in brackets:

| motion | raw RMS % of m·g | peak dev [N] | peak accel [m/s²] |
|---|---|---|---|
| hold | 0.0 | 0.00 | 0.0 |
| descent 50 mm/s | 6.5 | 0.33 | 1.5 |
| descent 150 mm/s | 12.9 | 1.00 | 4.4 |
| sine 0.5 Hz, 10 mm | 5.5 | 0.21 | 0.9 |
| sine 2 Hz, 10 mm | 22.6 | 0.78 | 3.1 |
| sine 5 Hz, 10 mm | 53.2 | 2.09 | 7.5 |

A 2 N inertial swing on a 0.1 kg load is small next to the 13–45 N contact forces in M1, but it
is the *same size* as the M1 lateral-push forces (0.2–3 N) and larger than any sensible contact
detection threshold. The ratio scales with load mass: robosuite's `Wipe` tool turned out to be
only 0.03 kg (M4), but a parallel gripper is ~0.9 kg, i.e. 9× these numbers (M4 emulates that).

**How well does compensation work? It depends entirely on where the acceleration comes from.**
Residual RMS as % of m·g:

| motion | gravity only | + FD accel @ 20 Hz (control rate) | + FD accel @ 500 Hz, averaged per control step | + exact MuJoCo site accel |
|---|---|---|---|---|
| descent 150 | 12.9 | 14.8 | **0.30** | 0.0 |
| sine 2 Hz | 22.6 | 25.1 | **0.22** | 0.0 |
| sine 5 Hz | 53.2 | 88.1 | **2.9** | 0.0 |

- With MuJoCo's exact site acceleration the residual is 0 to machine precision: the simulated
  sensor is *exactly* `m·Rᵀ(a − g)` plus contact. There is no noise, bias, or sensor dynamics.
- Differentiating the 20 Hz velocity is **worse than not compensating at all**: the one-sample
  lag of a causal difference at 50 ms turns the correction into a phase-shifted copy of the
  error (`sine_2_comp.png`, green vs orange). Central differences at 20 Hz merely break even.
- Differencing at physics rate (2 ms) and averaging the compensated wrench over the control
  interval brings 2 Hz to 0.2 % and 5 Hz to 2.9 %. This is what a real F/T driver does (1 kHz
  compensation → downsample), so `Gantry.step` now does it and `ft_comp` in every episode is
  produced this way. `test_compensation.py` pins < 5 % at 2 Hz (actual 0.22 %).
- **MuJoCo gotcha:** `mj_objectAcceleration` / `cacc` returns *proper* acceleration (+9.81 ẑ at
  rest) because `mj_rnePostConstraint` sets the world acceleration to −g. `Gantry.ee_acc_world`
  adds gravity back. Verified: rest reading 0.000 m/s² after the fix (`report.json:
  cacc_check_rest_accel_mps2`).

**Does `ft_comp` match ground truth in contact?** Static press on the rim (2 mm offset,
4 contacts): `ft_comp = (0, 0, −13.870) N`, summed `mj_contactForce` on the peg = (0, 0, +13.870)
N world; error 1e-10 %. Torque: `ft_comp Ty = 0.1283 N·m` = −(contact torque about the site)
within 1e-10. `test_contact_consistency.py` pins 2 %. **Sign convention for `ft_comp`:** it is
the wrench the load applies *to the environment* (minus the contact wrench on the load), so
pressing down reads −Fz. PLAN §5 updated. `mj_contactForce` sign: the returned force (contact
frame, normal from geom1 to geom2) is the force **on geom2**; `fvb.contacts.geom_contact_wrench`
flips the sign when the peg is geom1 — the 1e-10 agreement above is the check.

What this means for a force-aware policy: compensation is cheap and exact *if* it runs at the
physics/driver rate before downsampling; do it in the data pipeline, not in the policy. A policy
fed raw 20 Hz F/T plus 20 Hz proprioception cannot learn to compensate inertia itself, because the
information (acceleration) is aliased away at that rate. Contact detection thresholds must be set
above the *uncompensated* inertial swing unless compensation is in the loop (M4 tests this on
the arm).

## M3 — Solver and contact-parameter sensitivity (2026-09-19)

Command: `make m3` (= `python scripts/03_solver_sweep.py --out outputs/m3`): one-at-a-time sweep
(`oat.csv`) plus the full 864-run factorial (`full.csv`, 3 s wall on 30 workers), 0.3 mm offset,
20 Hz control, insertion bottoms out on the hole floor. Artifacts: `ranking.{txt,json}`,
`heat_<metric>__<row>_x_<col>.png` (15 heatmaps + 3 integrator/cone maps),
`unstable_or_tunnelling.json`.

**Ranking — main effect on the physics-rate peak |F| (mean range in N when only that parameter
varies, all other combinations averaged):** approach speed 30.2, kp 29.5, solref time constant
24.9, timestep 16.0, cone 0.36, noslip 0.00, integrator 0.00. For the *motion* (z-trajectory
RMS): speed 13.6 mm, kp 12.2 mm, then solref 0.05, timestep 0.05, cone 0.02 mm, noslip/integrator
0. So **solref and timestep change the force signal by 16–25 N without changing the motion by
more than 0.05 mm** — the answer to "which parameters change force without changing motion".
Integrator (Euler vs implicitfast) and `noslip_iterations` change nothing at all in this scene
(identical to the last digit); the cone type changes the peak by < 0.4 N.

**The onset spike is contact stiffness, and the timestep silently sets it.** At 150 mm/s, kp 800
(`heat_peak_F_hf_N__timestep_x_solref_tc.png`):

| timestep \ solref tc | 0.002 s | 0.005 s | 0.02 s |
|---|---|---|---|
| 0.0005 | **94 N** (spike 81, 1 ms) | 37 N | 12.8 N, pen 1.18 mm |
| 0.001 | **99 N** | 38 N | 12.8 N, pen 1.17 mm |
| 0.002 | 44 N | 36 N | 12.8 N, pen 1.00 mm |
| 0.005 | 21 N | 21 N | 12.8 N, pen 1.12 mm |

Steady force is 12.9 N in every cell. MuJoCo clamps the solref time constant to ≥ 2×timestep
(documented behaviour), so solref 0.002 at dt 0.002/0.005 is really 0.004/0.010 and the stiff
contact simply *does not exist* at the default timestep — the 94–99 N impact is physical for
the parameters as written and appears only when the timestep is small enough to represent it.
Conversely solref 0.02 (soft) removes the spike entirely but lets the peg sink 1.0–1.2 mm into
the floor (2.5 mm with kp 3000 at dt 0.005) — 2× the 0.5 mm hole clearance, i.e. tunnelling-scale
error for this task. All 100 runs flagged for > 1 mm penetration are at 150 mm/s and 84 of them
have solref 0.02. No run diverged (0 NaN failures in 864).

**What the policy sees vs what physics does.** The control-rate peak (`peak_F_c_N`) has main
effects speed 0.003 N, solref 0.2 N, timestep 0.04 N, kp 27.8 N. At 20 Hz *none* of the
solver-driven spikes are visible; only kp (through the quasi-static wind-up) is. Every spike in
the table above is 1–5 ms wide (`spike_width_hf_ms` = 1–2 physics steps) and lands between
control samples.

**Stiffness × speed** (`heat_peak_F_hf_N__kp_x_speed_mmps.png`): steady force = kp × overshoot
(6.9 / 12.9 / 34.8 N for kp 200 / 800 / 3000), peak at 150 mm/s = 38 / 36 / 70 N. Speed sets the
spike, kp sets the plateau; low kp does *not* protect against the impact spike (38 N at kp 200).

**F/T-threshold detector false triggers (preview of M4).** With a 0.3 N threshold on the raw
physics-rate |F| − baseline, the detector fired *before* true contact (ground truth: `ncon > 0`)
in 288 of 864 runs: 0 % at 10 mm/s, 33 % at 50 mm/s, 67 % at 150 mm/s, up to 1.44 s early —
the inertial transient at the start of the descent (M2: 0.33 N at 50 mm/s, 1.0 N at 150 mm/s)
exceeds the threshold.

**Cost:** wall-clock per simulated second 0.005 / 0.013 / 0.026 / 0.051 s for dt 0.005 / 0.002 /
0.001 / 0.0005 (this tiny scene; robosuite scenes are ~50× heavier).

**Defensible default for data collection:** `timestep 0.001, implicitfast, elliptic, solref
(0.005, 1), noslip 0` — the spike it produces (38 N at 150 mm/s) is within 3 % of the dt 0.0005
value (37 N), so the force signal is converged with respect to the timestep, penetration stays
≤ 0.29 mm (< clearance), and it costs 2× the robosuite default. Keep `kp` and approach speed as
*logged experiment variables*, never fixed defaults: they move the force more than any solver
knob. If robosuite's dt 0.002 must be kept (M4–M6), note that its onset spikes are ~half of the
converged value and carry a solref ≥ 0.004 s that the XML does not state.

What this means for a force-aware policy: the peak-force *distribution* in a dataset is a
property of (timestep, solref, approach speed) as much as of the task; a policy trained on
dt 0.002 data would see ~2× larger, ~2× shorter spikes on a dt 0.0005 sim or on hardware with a
stiff contact — and it will never see them at all through a 20 Hz observation. Peak-force
features must come from the physics-rate buffer (`ft_raw_hf`), and augmentation over
solref/timestep is the sim-side equivalent of domain randomization for force.

## M4 — Track B: robosuite Panda on `Wipe` (2026-09-19)

Command: `make m4` (= `python scripts/04_robosuite_wipe_ft.py --out outputs/m4`; BASIC/OSC_POSE,
kp 150, 20 Hz, seed 0). Artifacts: `report.json`, `load_params.json`, `free_space_{0.2,0.5,1.0}.*`,
`press_slide_{raw,comp}.{npz,json,mp4}` + `_raw/_comp/_fmag/_joint_torque.png`, `heavy_tool/`.

**API facts pinned.** `env.robots[0].ee_force["right"]` equals the raw `gripper0_right_force_ee`
sensor to the last bit (max diff 0.0) — it is a plain sensordata read. The sensor site
`gripper0_right_ft_frame` sits on body `gripper0_right_wiping_gripper` (parent
`robot0_right_hand`); everything distal weighs **0.03 kg** with COM at the site. Sign convention
is the same as Track A: rest reading `(0.0007, 0.041, −0.291) N` = `m·Rᵀ(−g)` exactly, i.e. the
force the hand applies to the tool, in the site frame, whose z axis points *down* (world
`(0.11, 0.00, −0.99)`). **Frame trap:** `obs["robot0_eef_quat"]` is the grip site, rotated
exactly 90° about z relative to `ft_frame`; using it to rotate the wrench swaps Fx/Fy. Episodes
now log the F/T-site pose as `ee_pos/ee_quat`. **Seeding trap:** robosuite 1.5.2 draws init
noise and placements from `env.rng` (a Generator shared by reference with the placement
sampler); `np.random.seed` has no effect. `ArmRig.reset(seed)` reseeds that Generator in place;
verified bit-identical episodes. **Joint torques** via `sim.data.ctrl[0:7]` (M0).

**Load identification on the arm** (OSC rotation deltas ±20° about x and y, 78 static samples):
m = 0.02999 kg, COM = (−1.5e-5, 1.1e-5, −1.2e-4) m, residual 2.8e-5 N. With the tool mass
patched to 0.9 kg (`ArmRig.set_tool_mass`): m = 0.89999 kg, residual 6.5e-5 N. Same LSQ as
Track A, works unchanged.

**Free-space F/T on the arm vs Track A.** Alternating ±z then ±x moves, peak EE speed 0.25 m/s,
peak accel 7.3 m/s² (larger than any Track A motion):

| tool | m·g | raw swing, control rate (peak) | raw swing, physics rate (peak) | after compensation (peak) |
|---|---|---|---|---|
| Wipe tool 0.03 kg | 0.29 N | 0.125 N | 0.41 N | 0.004 N |
| 0.9 kg gripper | 8.83 N | 3.95 N | 11.3 N | 0.13 N |

Physics-rate compensation (same code path as Track A, hooked into robosuite's per-substep
`_update_observables`) removes 97–99 % of the swing. The arm adds nothing Track A didn't have:
the sensor is still exactly `m·Rᵀ(a−g)` + contact. What the arm *does* add is a 3× larger
physics-rate-vs-control-rate ratio (0.41 vs 0.125 N) because the OSC excites the arm at
frequencies the 20 Hz observation cannot see. The joint-torque signal is dominated by gravity
(RMS 25–43 N·m); contact changes each joint by only 1–5 N·m, comparable to what a
posture change does — it is a poor stand-in for the wrist sensor.

**Press and slide** (descend at 0.1 m/s, 1 N threshold, hold target 10 mm below the contact
height, slide +y): contact at t = 2.15 s detected on the same control step as ground truth
(`n_contacts > 0`). Onset: physics-rate Fz jumps −0.3 → **54.6 N** in one step and decays over
~200 ms to 17 N; the control-rate sample reads 33 N (the 56 N spike is 60 % under-observed, as
in M1/M3). Steady press: `ft_comp` world = `(0.25, 0.40, −15.75) N` vs summed contact wrench
`(−0.24, −0.39, +15.24) N` — sign-opposite as they should be (tool-on-table vs table-on-tool),
3 % apart because `ft_comp` is a control-interval average while the contact sum is an end-of-
interval sample (Fz std 0.95 N at physics rate during the slide). **Pressing straight down gives
a world −Z wrench of the tool on the table ✓**, while the raw sensor reads
`(0.33, −1.92, +14.82) N` in its own (z-down) frame. Later in a longer slide the force climbed to
72 N as the arm approached its workspace limit — that rise is *posture*, not task, another reason
force targets do not transfer between configurations.

**OSC `kp` is not a stiffness.** A 10 mm press with kp = 150 gives 15–17 N, not 1.5 N: robosuite's
OSC computes `F = Λ (kp·Δx − kd·ẋ)` with Λ the task-space inertia matrix (~1 kg-scale here), so
the effective Cartesian stiffness is Λ·kp and configuration dependent. M5 quantifies it.

**Does a threshold detector false-trigger before/after compensation?**
Sweep threshold × descend speed (0.4 / 1.0 action ≈ 0.10 / 0.25 m/s), raw vs compensated:

| tool | threshold | raw, 0.1 m/s | raw, 0.25 m/s | compensated (any) |
|---|---|---|---|---|
| 0.03 kg | 0.1 / 0.3 / 1.0 N | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| 0.9 kg | 0.3 N | **1** false | **4** false | 0 |
| 0.9 kg | 1.0 N | 0 | **1** false | 0 |
| 0.9 kg | 3.0 N | 0 | 0 | 0 |

With the 30 g wiping tool the inertial swing (0.125 N) never reaches even a 0.1 N threshold, so
the plan's expected false triggers do not occur — the tool is too light. With a realistic 0.9 kg
gripper the raw signal false-triggers at any threshold below ~4 N when moving fast; the
compensated signal never does, and a 0.3 N threshold becomes usable. Every run still found the
true contact afterwards (no misses).

What this means for a force-aware policy: (1) on the arm, everything measured in Track A holds
and compensation is equally exact; (2) contact detection thresholds are a property of *tool mass
× acceleration*, so a policy trained with a light tool will learn thresholds that false-trigger
with a heavier one unless inputs are compensated; (3) the observation must carry the F/T-site
frame (or the wrench in world), not the grip-site frame; (4) OSC gains must be reported as
Λ·kp, and effective stiffness in N/m measured, before force magnitudes across datasets can be
compared.

## M5 — Controller stiffness vs contact force (2026-09-19)

Command: `make m5` (= `python scripts/05_robosuite_gain_sweep.py --out outputs/m5`; Wipe, seed 0,
descend at −0.4 action, 1 N compensated detector, 10 mm press, 40-step slide). Artifacts:
`fixed_grid.json`, `variable_kp.json`, `fixed_kp*_z*.npz`, `varkp_*.npz`,
`stiffness_tradeoff.png`, `variable_kp.png`.

**OSC kp is an acceleration gain; the tool's stiffness is Λ·kp.** Steady press force (ζ = 1)
divided by the commanded 10 mm:

| kp [1/s²] | 50 | 150 | 400 | 1000 |
|---|---|---|---|---|
| steady force [N] | 5.5 | 15.8 | 29.4 | 108.5 |
| effective stiffness [N/m] | 550 | 1580 | 2950 | 10850 |
| onset peak, physics rate [N] | 29.3 | 46.3 | 56.9 | 118.8 |
| onset peak, control rate [N] | 16.5 | 33.3 | 50.9 | 113.3 |
| free-space lag per 20 mm/step command [mm] | 16.9 | 14.7 | 12.0 | 8.6 |
| steps to settle at hover (3 mm) | 43 | 30 | 21 | 15 |

For kp ≤ 150 the stiffness is 10.5 × kp → task-space inertia Λ_z ≈ 10.5 kg in this posture;
above that the steady force grows faster than kp because the press also drags the tool
(friction) and drives the arm toward its limits. The **trade-off**: going from kp 50 to 1000 cuts
the free-space lag 2× and the settling time 3×, and raises the steady contact force 20× and the
onset peak 4×. Damping ratio: ζ = 2 halves the achievable descent speed (0.062 → 0.033 m/s at
kp 50) and with it the impact peak (29 → 15 N); ζ = 0.7 lets the press ring (kp 400: 101 N peak,
64 N "steady" vs 57 / 29 N at ζ = 1). Impact speed, not kp, sets the physics-rate spike: at ζ = 1
the tool arrives at 0.06–0.15 m/s and the spike is 29–119 N, all of it between two 20 Hz samples.

**Switching to low stiffness on contact (`impedance_mode="variable_kp"`, kp raw in action[:6],
`kp_limits` raised to [0, 1000]).** Approach at kp_free, switch to kp_contact when the
compensated detector fires, with 0/1/2 control steps (0/50/100 ms) of extra latency during which
the descent command continues:

| kp_free → kp_contact | latency | peak physics [N] | peak control [N] | steady [N] | peak after switch, control / physics [N] |
|---|---|---|---|---|---|
| 1000 → 1000 | 0 | 118.8 | 113.3 | 108.5 | 185.6 / 186.3 |
| 1000 → 150 | 0 | 117.2 | 48.3 | 17.2 | 34.9 / 50.4 |
| 1000 → 50 | 0 | 117.2 | 48.3 | 6.7 | 27.8 / 48.2 |
| 1000 → 1000 | 1 | 220.5 | 217.7 | 106.0 | 184.7 / 185.5 |
| 1000 → 50 | 1 | 220.5 | 217.7 | 7.0 | 18.4 / 26.3 |
| 400 → 400 | 0 | 56.9 | 50.9 | 29.4 | 50.9 / 56.9 |
| 400 → 50 | 0 | 54.0 | 31.3 | 5.4 | 31.3 / 54.0 |
| 400 → 50 | 2 | 104.9 | 101.7 | 4.9 | 5.1 / 7.2 |

- With zero latency, dropping 1000 → 50 reduces the control-rate peak 113 → 48 N (−58 %) and the
  steady force 108 → 6.7 N (−94 %), while the **physics-rate impact is untouched (119 → 117 N)**:
  it is over before the controller can see it. What the switch removes is the *wind-up* after
  the impact, which is the part a 20 Hz policy actually observes.
- One control step of latency at kp 1000 doubles the peak (119 → 220 N) whether or not the
  switch follows — the cost of latency is paid at the high stiffness. After the switch the force
  collapses within one step (e.g. 220 N → 18 N for 1000 → 50, delay 1).
- The whole benefit of variable stiffness is therefore in *predicting* contact (switching
  before impact) or in bounding the latency; a reactive switch buys the steady-state, not the
  spike.

What this means for a force-aware policy: (1) force magnitudes in demonstrations are ≈ Λ·kp ×
overshoot and must be logged with kp, ζ and posture to be interpretable; (2) the observable
(20 Hz) peak scales with stiffness ≈ linearly, so a policy can learn "stiff = high force"; (3) the
unobservable impact scales with approach speed, so speed near expected contact is the lever a
policy has *before* contact, stiffness is the lever *after* — this is the case for policies that
output stiffness (or an "expect contact" flag) as part of the action.

## M6 — Peg-like task on the arm + episode recorder (2026-09-19)

Commands: `make m6` = `python scripts/06_robosuite_nut_round.py --out outputs/m6` (8 seeds run:
`--n 8`), then `07_record_episodes.py --track A --n 50 --noise-mm 0.5`, `--track B --task Wipe
--n 20`, `--track B --task NutAssemblyRound --n 10`. Artifacts: `outputs/m6/results.json`,
`nut_seed*.{npz,json,mp4}`, `nut_seed*_{comp,fmag}.png`; datasets in `data/track{A,B_*}/`
(`ep0000.npz`… + `summary.json` + `peak_force_distribution.png`, git-ignored).

**NutAssemblyRound scripted grasp → lift → transport → lower → release (7-D action, Panda
gripper 0.52 kg identified to 0.5200 kg).** Success **7/8** by the env's own criterion (nut
centre within 3 cm of peg2 and below table + 5 cm). Phase forces (compensated, sensor frame):

| phase | typical mean / peak |F| | notes |
|---|---|---|
| grasp (fingers close on handle) | 0.2–13 N / 1–26 N | handle squeezed against the table when the fingers land 6 mm low; internal finger↔nut forces do not pass through the wrist sensor |
| lift + transport | 0.19 N / 0.35–0.54 N | free flight; residual = **0.19 N = the nut's weight** (0.019 kg·g): the load model identified before grasping no longer matches the payload |
| lower onto peg | 0.2 N until contact; spikes 30–33 N (79–94 N physics rate) in 3/8 | peg top hits the gripper/nut assembly |
| mate | 0.2 N (5/8, contact-free) or 12–20 N mean / 16–31 N peak (2/8) | ring rides down the peg with lateral force + 1 N·m torque |
| release | < 1 N | |

Failure modes seen: (1) *dropped in transport* (1/8 at carry gain 0.4; 1/3 at gain 1.0 before
the fix): the OSC overshoots the 5 cm/step saturated command, the nut slips out of the pads, and
a naive closed-loop "follow the nut centre" then drives the arm into a joint limit pressing the
table at 100 N — the recorder now aborts the episode with `failure_mode="dropped_in_transport"`;
(2) *grasp too high* (all episodes before the fix): the nut is spawned 6 cm above the table and
settles in the first ~0.3 s, so poses read at reset are wrong — read them after settling.

**A 20 Hz ground-truth contact flag misses real contact.** Seed 6: the assembly hits the peg top
at t = 9.162 s with 93.6 N at physics rate (32.5 N at control rate), as a bouncing contact of
2–26 ms bursts; the control samples at 9.20, 9.25 and 9.30 s all fall in gaps between bursts, so
`n_contacts` reads 0 for three consecutive steps while the F/T sees the impact. The wrist sensor
is a better contact detector than an end-of-step contact query — at control rate *both* are
undersampled, but the sensor value integrates the interval's impulse.

**Uncompensated rotational inertia.** Yaw-aligning the gripper at 1 rad/s² produces a 2.2 N·m
`Tz` spike and 0.5 N·m in `Tx/Ty` with no contact (`nut_seed0_comp.png`, t = 0.45 s):
`fvb.ft.compensate` deliberately drops the `Iω̇ + ω×Iω` terms. On a 0.52 kg gripper that is the
size of the mating torques (1 N·m) — Track B needs the rotational terms before torque can be used
for contact detection during re-orientation.

**Recorder + schema (`07_record_episodes.py`, `test_logger_schema.py`).** Every file written is
re-loaded and validated (keys, shapes, dtypes, finite, monotonic `t`/`t_hf`, `ft_raw_hf` length
= T × physics steps); optional `image` stream verified at 96×128. Datasets:

| set | n | success | peak |F| control rate, median [min, max] | physics rate | hf/control ratio |
|---|---|---|---|---|---|
| Track A, σ = 0.5 mm | 50 | 28 | 9.9 N [9.9, 41.9] | 13.5 N [12.2, 40.9] | 1.24 (max 1.44) |
| Track A, σ = 2 mm (PLAN default) | 50 | 1 | 41.9 N | 40.9 N | 0.98 |
| Track B Wipe | 20 | 20 | 37.0 N [28, 52] | 61.8 N [33, 88] | 1.68 (max 1.91) |
| Track B NutAssemblyRound | 10 | 9 | 27.1 N [1.6, 54] | 51.2 N [7.5, 94] | 2.03 (max 5.5) |

Track A is bimodal: inserted episodes peak at 9.9 N (bottoming out, kp 800 × 12 mm overshoot),
rim-jams at 41.9 N — the peak force *is* the success label, which is exactly the shortcut a
policy would learn. With the plan's 2 mm noise on a 0.5 mm clearance only 1/50 inserts, so the
main set uses 0.5 mm (Makefile updated). Track B peaks are 1.7–2× larger at physics rate than
at control rate; the ratio reaches 5.5 in the nut task (brief impacts), so the control-rate
peak is not a usable proxy for the physical peak there. Recording cost: 0.1 s (A) / 0.85 s
(Wipe) / 1.7 s (Nut) per episode, wall-clock.

**Stretch (custom single-arm `PegInHole` env): not attempted**, per the plan's ordering. Track A
already answers the peg-in-hole questions and NutAssemblyRound supplies the arm-side mating
signal; the missing piece is only the sub-mm-clearance jam on the arm.

What this means for a force-aware policy: (1) the payload changes the gravity model at grasp
time — either re-identify after grasp or give the policy the uncompensated residual as a
"holding something" cue (0.19 N here, but 9 N for a 0.9 kg object); (2) contact labels for
training must come from the physics-rate F/T impulse, not from a control-rate contact query;
(3) rotational inertia matters on the arm as soon as the gripper re-orients.

---

## Stage 0 summary

**Sensor convention.** A MuJoCo `force`/`torque` sensor on a site reports the wrench the
*parent* body applies to the *child* subtree, in the site frame (derived in M1: hanging peg reads
`+m·g`, `Fz = m(g + a_z)` under vertical acceleration). robosuite's `ee_force` is that raw
sensordata; its `ft_frame` site has z pointing *down* and is rotated 90° about z from the
grip-site frame used by `robot0_eef_quat`. `ft_comp` in the §5 logs is sign-flipped to the
wrench the tool applies to the environment (pressing down ⇒ −Z in world).

**Gravity/inertial contamination.** Exactly `m·Rᵀ(a − g)` (plus torque `c × F`); m and COM
recover by least squares from 3–4 static orientations to 1e-5 relative error on both tracks. RMS
contamination in free space: 5–53 % of m·g for 0.5–5 Hz motion of a 0.1 kg peg; on the arm 0.125 N
(30 g tool) to 4 N (0.9 kg gripper) at 0.25 m/s. Compensation reduces it to 0.2–3 % **only if
the acceleration is differenced at physics rate and the result averaged per control step**;
differencing at 20 Hz makes it worse than doing nothing. Rotational inertia (not modelled) shows
up as 2 N·m torque spikes when the gripper re-orients. A grasped payload shifts the residual by
its weight.

**Sim-parameter sensitivity (M3, 864 runs).** Main effects on physics-rate peak force: approach
speed 30 N, kp 30 N, solref time constant 25 N, timestep 16 N, cone 0.4 N, integrator 0,
noslip 0. solref and timestep move the force 16–25 N while moving the trajectory < 0.05 mm.
MuJoCo clamps solref ≥ 2·dt, so the stiff-contact impact (94 N) exists only at dt ≤ 1 ms; soft
contact (solref 0.02) hides the spike but penetrates 1–2.5 mm. **Chosen defaults:** `timestep
0.001, implicitfast, elliptic, solref (0.005, 1), noslip 0` (converged within 3 % of dt 0.0005,
penetration < clearance, 2× robosuite cost). robosuite's own dt 0.002 halves the onset spike.

**Stiffness vs peak force (M5).** robosuite OSC `kp` is an acceleration gain; effective tool
stiffness = Λ·kp ≈ 10.5·kp N/m at low kp (Λ_z ≈ 10.5 kg). kp 50→1000: tracking lag 17→9 mm per
step, steady contact 5.5→108 N, onset peak 29→119 N (physics) / 16→113 N (control). Switching
1000→50 at zero latency removes 58 % of the observable peak and 94 % of the steady force but
none of the 2 ms impact; one 50 ms step of latency at high kp doubles the peak. Approach speed
governs the spike, stiffness governs everything after it.

**Recommended preprocessing for a learning pipeline.**
- *Compensation:* identify m, COM per tool from static poses; compensate at driver/physics
  rate (FD acceleration from the same-rate velocity) and average per control step; re-identify
  or flag after grasp; add rotational terms if the wrist re-orients.
- *Rates & history:* keep the physics-rate buffer (`ft_raw_hf`, 25 samples per 20 Hz step here);
  a force token should summarise at least the last control interval's max-|F| and impulse, not
  just the sample. Contact labels from the hf impulse, never from a control-rate contact query.
- *Filtering:* none before the peak/impulse features; a causal 2nd-order Butterworth at 5–10 Hz
  (`fvb.ft.filters.butter_lowpass`) for the "steady" channel only — the 1–5 ms spikes are signal
  in this domain, and any low-pass removes them.
- *Normalisation:* per-track ranges observed — Track A |F| ∈ [0, 45] N (kp 800), torque
  ≤ 0.2 N·m; Track B |F| ∈ [0, 120] N at kp ≤ 400 (up to 220 N with latency at kp 1000),
  torque ≤ 2.5 N·m. Normalise by a fixed physical scale (e.g. 50 N, 1 N·m) rather than dataset
  statistics, and log kp/ζ/dt/solref with every episode since force scales with all four.
- *Frames:* store the wrench and the pose in the *same* site frame (done in §5), and the world
  rotation alongside; never the grip-site quaternion with the ft-site wrench.
- *Solver metadata as a domain-randomisation axis:* timestep and solref change peak forces 2×
  without changing motion; sample them when generating data, and log them.

## M6 stretch — custom single-arm `PegInHole` on the Panda (2026-09-21)

Command: `python scripts/08_robosuite_peg_in_hole.py --out outputs/m6_pih` (offsets 0 / 0.3 / 1 /
4 mm × OSC kp 150 / 400, 0.5 mm clearance, seed 0) and `07_record_episodes.py --track B --task
PegInHole --n 20 --noise-mm 0.5`. Artifacts: `outputs/m6_pih/results.json`, `kp*_offset_*mm.{npz,
json}`, `*_comp.png`, `*_fmag.png`, `kp*_offset_4mm.mp4`; `data/trackB_PegInHole/`.

**The env (`fvb.envs.PegInHole`, registered with robosuite).** A 19 mm × 100 mm cylindrical peg
(`CylinderObject`, 28 g) welded to the Panda flange in place of the gripper, a four-box square hole
with parametrised clearance fixed to the table (the plan's `PlateWithHoleObject` has a 100 mm
square opening — 2–4 cm of clearance around any peg, so it was not reused), a `force`/`torque`
sensor pair on a flange site, tip/axis/hole observables, dense reward and a depth-based success
check. Three things had to be discovered before a 0.5 mm-clearance insertion could succeed at
all, and each is a lesson in its own right:

1. **`CylinderObject` writes `margin="0.001"`** — a 1 mm collision skin. With 0.5 mm clearance the
   peg "touched" the rim at 1.0 mm above the plate with 0.01 mm of lateral error, every time. The
   env now zeroes the margin. Any robosuite object used for tight tolerances needs this.
2. **The OSC settles ~1.1 mm off its goal** (PD in task space, model-based gravity compensation,
   no integral action): the scripted approach integrates the residual tip error until it is
   < 0.2 mm. A learned policy has to close this loop itself.
3. **A zero rotation delta does not hold orientation.** With delta actions the orientation goal
   is re-anchored to the current pose every step, so contact torque pivots the peg freely:
   7–18° of tilt during the first descents, where 0.5 mm over 100 mm allows 0.3°. The script now
   servos the peg axis to vertical with world-frame rotation-vector deltas (the Panda home pose
   already has the flange 8° off vertical). Result: ≤ 0.08° tilt when free, ≤ 0.25° in contact.

**Offset sweep (tip servo ≤ 0.2 mm, descent 3 mm/step, 40 N stop, then a 2 mm push held):**

| kp | offset | result | depth | peak |F| control / physics | held force (site frame) | Ty |
|---|---|---|---|---|---|---|
| 150 | 0 / 0.3 mm | inserted | 39.8 mm | 0.0 / 0.3 N | 0 | 0 |
| 150 | 1 mm | **wedged** | 3.4 mm | 27.0 / 79.4 N | (−1.5, 0, 0.9) N | −0.14 N·m |
| 150 | 4 mm | rim | 0 | 26.9 / 79.3 N | (0.2, 0, 4.1) N | 0 |
| 400 | 0 / 0.3 mm | inserted, bottoms out | 40.0 mm | 35 / 79–91 N | (9.9, 0, 16.9) N | 0.8 N·m |
| 400 | 1 / 4 mm | rim, force limit | 0 | 49 / 117 N | (14, 0, 23) N | 1.2 N·m |

- Same binary outcome as Track A — pass below the clearance, jam above it — with one arm-only
  behaviour: at kp 150 and 1 mm offset the peg **wedges 3.4 mm in** over ~6 s
  (`kp150_offset_1mm_comp.png`): Fz 13.5 N + Fx 8 N on the rim, then the lateral compliance and
  the xy servo drag the tip across the edge; Fx falls through zero and reverses (−2.4 N) as the
  second wall is met and Ty follows Fx × lever. That Fx sign flip with a growing Ty is the
  two-point-contact jam signature the peg-in-hole literature describes, and it appears only when
  the arm is compliant enough to let the tip move.
- The rim impact at 3 mm/step is 79 N at physics rate vs 27 N at control rate (kp 150), 117 vs
  49 N (kp 400) — the arm's 20 Hz observation sees a third of it, consistent with M4/M5.
- Held rim force is Λ·kp × the 2 mm push: 4 N at kp 150, 23 N at kp 400 (the same 10 kg-scale Λ
  as M5), so the *steady* jam force again measures the controller, not the jam.

**Recorded set, σ = 0.5 mm lateral noise, 20 episodes:** 13 inserted, 7 jams (1 force limit, 6
timeouts). Successful episodes peak at 11.3 N control / 54 N physics (bottoming out), failures at
25 N / 78 N; physics/control peak ratio median 4.25 (max 83 on a clean insertion whose only
force is a 2 ms bottom-out tick). 17 s wall for 20 episodes of 166 steps.

What this means for a force-aware policy: the arm reproduces every Track A result and adds
three requirements a policy must meet on its own — close the ~1 mm steady-state OSC error,
actively hold orientation (a zero delta is not a hold), and read the Fx-reversal / Ty-growth
signature to tell a wedge from a rim stall. All three are visible in `ft_comp` at 20 Hz; the
impact itself is not.

---

# Stage 1 — Tier 1 behaviour cloning with a force token (2026-09-22)

Commands: `make build-train && make s1` (= `09_collect_expert.py --n 300`, `10_train_bc.py
--force 1/0`, `11_eval_bc.py --n 50`); MLP baselines and the σ = 3 mm eval were run by hand.
Artifacts: `data/expert_trackA/` (300 episodes, §5 files + `ep*_policy.npz`), `outputs/s1/{force,
noforce,mlp_force,mlp_noforce}/{best.pt,history.json,loss.png}`, `outputs/s1/eval/eval.{json,png}`,
`outputs/s1/eval_sigma3/`. Design and gotchas: the Tier 1 explainer (artifact) and
`fvb/policy/task.py`.

**Task.** Track A gantry, M3 defaults (dt 1 ms, kp 800), 0.5 mm clearance, hole centre drawn
from N(0, 1.5 mm) per axis (clipped ±4 mm) and **hidden**: the policy sees peg position relative
to the episode start, velocity, `ft_comp`, three physics-rate force summaries and its previous
action (18 dims × H = 10 history). Output: K = 5 future target deltas (clipped ±5 mm/step).
Success = 30 mm depth within 8 s. A privileged scripted expert (descend 50 mm/s; on |F| > 3 N
with stalled depth: retract 3 mm, step ≤ 0.5 mm toward the true hole, descend) succeeds 300/300
with 2.4 jam-recoveries per episode; the only information about the correction direction in the
observation is the torque at the jam (sign(dx) = −sign(Ty) in 78 %, sign(dy) = +sign(Tx) in 83 %
of corrections; the rest are corner contacts).

**Models.** ACT-lite: linear embed → 3-layer transformer encoder (d 128, 4 heads) → 5 learned
queries through a 2-layer decoder → linear, 800 k parameters; MLP baseline 116 k. Masked L1 on
z-scored chunks, AdamW 3e-4, cosine, 40 epochs, batch 256, 270/30 episode split (9 772 / 1 035
windows). **16 s per training run on the RTX 4060 Laptop** (1.4 GB free of 8 GB was enough);
data collection 25 s; evaluation of 50 episodes ~1 min per policy.

**Closed-loop results, 50 fresh seeds, σ = 1.5 mm:**

| policy | val L1 | success | steps (success) | peak |F| | retracts / episode |
|---|---|---|---|---|---|
| expert (privileged) | — | 50/50 | 37.4 | 3.7 N | 2.6 |
| ACT-lite + force | 0.073 | **50/50** | 36.9 | 3.4 N | 2.6 |
| ACT-lite, force zeroed | 0.152 | **5/50** | — | 7.6 N | 19.7 |
| MLP + force | 0.132 | 45/50 | 67.1 | 3.7 N | 8.1 |
| MLP, force zeroed | 0.158 | 5/50 | — | 7.9 N | 20.3 |

- The five no-force successes are the episodes whose offset is within the clearance or lands
  in on the first try; on |offset| ≥ 0.5 mm the no-force policies succeed in 4/49 vs 49/49 (ACT)
  and 44/49 (MLP) with force. Median final depth without force: 0.0 mm — on the rim at timeout.
- The no-force policy *does* learn the jam → retract reflex (from the velocity stall and the
  previous action), but with no direction signal it retracts ~20 times per episode and dithers,
  at 2× the peak force. This is the multi-modality failure predicted in the explainer §6: the
  same observation precedes +x and −x corrections, and L1 regression averages them.
- Validation loss separates the variants only mildly (0.07 vs 0.15) because 90 % of the steps
  are "descend 2.5 mm" and both fit those; the closed-loop gap is 10×. Loss ≠ success.
- The history transformer beats the MLP on the same data with force (50 vs 45/50; 37 vs 67
  steps; 2.6 vs 8.1 retracts): reading the torque sign and executing a multi-step manoeuvre is
  easier from a sequence than from a flattened window.

**Generalisation to larger offsets (σ = 3 mm, twice the training distribution):** expert 50/50
with 5.1 recoveries; ACT-lite + force **50/50** with 5.4 recoveries (57 steps); MLP + force 28/50.
The learned correction is a per-jam local rule (read Tx/Ty, step 0.5 mm), so it composes to
offsets it never saw.

What this means for a force-aware policy: on a task where position observations cannot
disambiguate the contact geometry, a 0.8 M-parameter policy trained in 16 s on 300 simulated
episodes goes from 10 % to 100 % success by being given the compensated wrench, and generalises
beyond the training offsets. The signal it uses is tiny — 0.018 N·m of torque at a 3 N jam — and
exists only because Stage 0 established a noiseless, compensated, correctly-framed sensor; on
hardware the same rule needs the compensation pipeline and a torque resolution well below
0.02 N·m. Next steps from here: DAgger if a harder variant (noise, timestep/solref
randomisation, the arm's 1 mm OSC error) drops success; injected sensor noise to find the
torque-resolution floor; the same experiment on `fvb.envs.PegInHole`.

## Stage 1b — sensor-noise floor (2026-09-22)

Command: `python scripts/12_noise_sweep.py` (train image). Gaussian noise added to the raw
sensor at every 1 kHz physics step, before compensation; torque std ∈ {0, 0.002, 0.005, 0.01,
0.02, 0.05} N·m with force std 20× that in N. For each level: 300 noisy expert episodes → train
ACT-lite + force → evaluate at that noise (50 episodes); the noise-free policy from Stage 1 is
also evaluated at each level. Artifacts: `outputs/s1_noise/sweep.json`, `noise_sweep.png`,
per-level `t*_f*/{train,eval}/`.

| torque σ [N·m] | force σ [N] | expert | trained at this noise | trained noise-free |
|---|---|---|---|---|
| 0 | 0 | 50/50 | 50/50 | 50/50 |
| 0.002 | 0.05 | 50/50 | 50/50 | 50/50 |
| 0.005 | 0.1 | 50/50 | 50/50 | 50/50 |
| 0.01 | 0.2 | 50/50 | 50/50 | 41/50 (4.4 retracts) |
| 0.02 | 0.5 | 50/50 | 50/50 (2.7 retracts) | 3/50 (0.7 retracts: never retracts) |
| 0.05 | 1.0 | 50/50 | 20/50 (15 retracts: dithers) | 2/50 |

- The signal is 0.018 N·m of torque at the jam. The policy trained at each level is unaffected
  up to σ = 0.02 N·m — *the same size as the signal* — because compensation averages 50
  physics-rate samples per control step (σ/√50 ≈ 0.003 N·m effective) and the network also
  integrates over its 10-step history. At σ = 0.05 (2.8× the signal) it drops to 40 % and dithers
  like the no-force policy did.
- The noise-free policy is far more brittle: 82 % at 0.01 N·m, 6 % at 0.02. Its failure mode is
  the opposite of dithering — it stops retracting at all (0.7 retracts/episode), because its
  jam detector was fitted to noiseless `ft_comp` and the physics-rate summary channels (max |F|,
  max |T|) that noise inflates: on noisy data those channels no longer mean "contact".
  Training with the noise present teaches it to rely on the averaged channels instead.
- For hardware: an ATI Nano/Mini-class sensor has torque noise well below 0.002 N·m, and the
  expert's 3 N threshold could be raised to make the torque signal larger (it scales with the
  jam force). The practical requirement is that the training data contain the sensor's real
  noise, not that the sensor be quiet.

What this means for a force-aware policy: noise robustness comes from two averaging stages
(driver-rate compensation + temporal context) and from training on noisy data; a policy fit to
a clean simulator will silently lose its contact detector on a real sensor.
