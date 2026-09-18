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
