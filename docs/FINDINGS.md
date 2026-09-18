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
