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
detection threshold. The ratio scales with load mass: robosuite's Panda gripper is ~0.7 kg
(M4 will measure), so the same accelerations give ~7× these numbers.

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
