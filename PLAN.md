# PLAN.md — Stage 0: F/T readings on peg-in-hole (MuJoCo + robosuite, Docker + Python)

## 1. Goal

> Install MuJoCo + robosuite first and get F/T readings on a peg-in-hole task before touching any VLA. This teaches the real problem (contact spikes, gravity/inertia contamination, solver sensitivity) fastest and cheapest.

Done means: a reproducible Docker repo where one command produces plots and a written `docs/FINDINGS.md` answering the questions in §7. No learning, no GPU required.

Non-goals for this stage: policies, datasets at scale, VLAs, tactile sim, real hardware. §9 lists the hooks left for them.

## 2. Facts verified on 2026-09-19 (do not re-derive, but do re-check with `00_probe_api.py`)

| Fact | Detail |
|---|---|
| Version pin | `robosuite==1.5.2` works with `mujoco==3.3.0`. With the latest MuJoCo (3.13.0 at time of writing) `suite.make(...)` fails with `AssertionError` in `binding_utils.get_joint_qpos_addr`. |
| numpy | Keep `numpy<2` (numba / robosuite stack). |
| F/T is not an observation | `obs` keys on `Wipe` contain `robot0_contact` but **no** `robot0_eef_force` / `robot0_eef_torque`. |
| How to read F/T in robosuite | `env.robots[0].ee_force["right"]`, `env.robots[0].ee_torque["right"]` (dict keyed by arm). Raw MuJoCo sensor names: `gripper0_right_force_ee`, `gripper0_right_torque_ee` (3 dims each), defined on site `ft_frame` in the gripper XML. |
| Controller config API | `from robosuite.controllers import load_composite_controller_config`; `cfg = load_composite_controller_config(controller="BASIC")`. Arm part is `cfg["body_parts"]["right"]`: `type="OSC_POSE"`, `kp=150`, `damping_ratio=1`, `impedance_mode="fixed"`. Set `impedance_mode="variable_kp"` to put stiffness in the action. |
| Sim defaults | robosuite model timestep 0.002 s, solver iterations 100, `control_freq=20` → 25 physics steps per action. |
| Built-in tasks | Single-arm peg-in-hole does **not** exist. Available contact-rich tasks: `Wipe`, `NutAssemblyRound`, `NutAssemblySquare`, `Door`, `ToolHang`, `TwoArmPegInHole` (two arms). |
| GL backend | robosuite forces `MUJOCO_GL=egl` when `macros.MUJOCO_GPU_RENDERING` is true unless `MUJOCO_GL` is already `osmesa` or `glx`. Set `MUJOCO_GL=osmesa` explicitly for CPU containers. |
| Joint torques (corrected M0, 2026-09-19) | `env.robots[0].torques` is **`None`** in robosuite 1.5.2 (never populated on the composite-controller path). Use `env.sim.data.ctrl[env.robots[0]._ref_arm_joint_actuator_indexes]` — identical to `composite_controller.part_controllers["right"].torques` and to `qfrc_actuator[:7]` (all Panda arm actuators have gear 1). Wrapped as `RobosuiteFT.joint_torques()`. |
| Custom `PegInHole` (M6 stretch, 2026-09-21) | `fvb.envs.PegInHole` is registered on `import fvb.envs`; `suite.make("PegInHole", robots="Panda", gripper_types=None, clearance=...)`. Peg = `CylinderObject` welded to `robot0_right_hand`; hole = four boxes fixed to the table (`PlateWithHoleObject` has a 100 mm square opening — useless for sub-mm clearance). F/T sensors `peg_force`/`peg_torque` on flange site `peg_ft`. **Trap:** `CylinderObject` writes `margin="0.001"` on its geom — a 1 mm collision skin that eats any sub-mm clearance; the env zeroes it. |
| Raw-scene F/T at rest (M0) | Free-hanging 0.1 kg peg, sensor reads `ft_force = [0, 0, +0.981]` N: **+Fz** for a peg pulling *down* on the wrist. See M1 for the sign derivation. |
| Raw MuJoCo scene in §6 | Runs as written. Free-space Fz ≈ +0.945 N during accelerating descent for a 0.1 kg peg (static value is m·g = 0.981 N — the difference is inertial contamination). Aligned insert bottoms out at ~21 N; 4 mm lateral offset jams on the rim at ~53 N. |

Because no single-arm peg-in-hole ships with robosuite, the plan has two tracks:

- **Track A — raw MuJoCo gantry peg-in-hole.** ~60 lines of MJCF, no robot arm. This is where F/T semantics, contamination and solver sensitivity are learned.
- **Track B — robosuite Panda + OSC.** Same measurements on a real arm model with an operational-space (impedance-like) controller, using `Wipe` first, then `NutAssemblyRound`, then (stretch) a custom single-arm `PegInHole` env.

## 3. Repo layout

```
force-vla-basics/
├── CLAUDE.md
├── PLAN.md
├── README.md                  # quickstart: make build && make m1
├── Makefile
├── docker/
│   ├── Dockerfile
│   └── compose.yaml
├── pyproject.toml             # package "fvb", ruff + pytest config
├── requirements.lock          # pip freeze from inside the built image
├── src/fvb/
│   ├── __init__.py
│   ├── scenes/
│   │   ├── peg_in_hole.xml    # Track A MJCF (seed in §6)
│   │   └── builder.py         # parametrize clearance, peg mass, kp, solver opts → XML string
│   ├── ft/
│   │   ├── read.py            # raw MuJoCo + robosuite F/T accessors, one interface
│   │   ├── frames.py          # site frame ↔ world rotation of wrench
│   │   ├── compensate.py      # gravity + inertial compensation
│   │   └── filters.py         # causal low-pass (1st-order IIR, Butterworth), spike metrics
│   ├── control/
│   │   ├── gantry.py          # Track A scripted descent / spiral search
│   │   └── osc_scripts.py     # Track B scripted EE trajectories, gain overrides
│   ├── logging/episode.py     # EpisodeLogger → .npz (+ config .json), schema in §5
│   ├── contacts.py            # mj_contactForce iteration, per-geom-pair wrench
│   └── viz/plots.py           # time-series, force-vs-depth, sweep heatmaps, mp4 writer
├── scripts/
│   ├── 00_probe_api.py
│   ├── 01_gantry_insert.py
│   ├── 02_contamination.py
│   ├── 03_solver_sweep.py
│   ├── 04_robosuite_wipe_ft.py
│   ├── 05_robosuite_gain_sweep.py
│   ├── 06_robosuite_nut_round.py
│   └── 07_record_episodes.py
├── tests/
│   ├── test_env_smoke.py
│   ├── test_ft_static.py
│   ├── test_compensation.py
│   ├── test_contact_consistency.py
│   └── test_logger_schema.py
├── docs/FINDINGS.md
├── outputs/                   # git-ignored
└── data/                      # git-ignored
```

## 4. Docker + tooling

**`docker/Dockerfile`** (CPU-default; works with GPU too):

```dockerfile
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    NUMBA_CACHE_DIR=/tmp/numba

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libosmesa6 libegl1 libgles2 libglfw3 \
      ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
      "mujoco==3.3.0" "robosuite==1.5.2" "numpy<2" \
      scipy matplotlib imageio imageio-ffmpeg h5py tqdm pytest ruff
# robosuite asks for a private macros file; create it once at build time
RUN python -c "import robosuite, os, runpy; p=os.path.join(os.path.dirname(robosuite.__file__),'scripts','setup_macros.py'); runpy.run_path(p, run_name='__main__')" || true

COPY . .
RUN pip install --no-cache-dir -e .
CMD ["bash"]
```

**`docker/compose.yaml`**: service `dev` mounting the repo at `/workspace`, `working_dir: /workspace`, `shm_size: 2gb`. Add a second service `dev-gpu` that extends `dev` with `environment: [MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl, NVIDIA_DRIVER_CAPABILITIES=all]` and `deploy.resources.reservations.devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]`. Stage 0 never needs the GPU service; it exists so later stages don't require a Docker rewrite.

**`Makefile`** targets: `build`, `shell`, `test` (`pytest -q`), `lint` (`ruff check . && ruff format --check .`), `run S=scripts/xx.py ARGS="..."`, and `m0` … `m6` which run the scripts for each milestone with default args and write to `outputs/mN/`.

After the first successful build, run `pip freeze > requirements.lock` inside the container and commit it.

Known install pitfalls to handle, not ignore:
- robosuite pulls `opencv-python`, which needs `libgl1` + `libglib2.0-0` (already in the Dockerfile).
- robosuite prints warnings about `robosuite_models` and `mink` IK. Harmless here; do not install them.
- If OSMesa rendering fails, physics and F/T still work. Rendering is only needed for the MP4s in M1/M4. Fall back to plots and note it in FINDINGS.

## 5. Episode log schema (`src/fvb/logging/episode.py`)

One `.npz` per episode plus a sibling `.json` with the full config, seed, git SHA, and package versions. Arrays are time-major, length T at **control** rate unless suffixed `_hf` (physics rate).

| key | shape | meaning |
|---|---|---|
| `t` | (T,) | sim time, s |
| `ee_pos`, `ee_quat` | (T,3), (T,4) | end-effector / peg pose, world frame, quat xyzw |
| `ee_vel` | (T,6) | linear + angular velocity, world |
| `ft_raw` | (T,6) | force(3)+torque(3) in **sensor-site frame**, unmodified |
| `ft_world` | (T,6) | same wrench rotated to world frame |
| `ft_comp` | (T,6) | after gravity + inertial compensation (sensor frame). Computed at physics rate and averaged over the control interval (M2). Sign: the wrench the **load applies to the environment** = −(contact wrench on the load). |
| `ft_raw_hf` | (T·k,6) | every physics step, for spike analysis |
| `contact_wrench` | (T,6) | ground truth: sum of `mj_contactForce` on peg/tool geoms, world frame |
| `n_contacts` | (T,) | active contacts involving the peg/tool |
| `action` | (T,A) | commanded action |
| `joint_torque` | (T,J) | Track B only: applied arm torques, `sim.data.ctrl[arm actuator idx]` (see §2) |
| `image` | (T,H,W,3) uint8 | optional, off by default |
| `success` | () | bool |

This schema intentionally matches what a later imitation-learning stage needs (synchronized vision + proprio + force + action), so Stage 1 can convert to LeRobot / robomimic HDF5 without re-collecting.

## 6. Track A seed scene (verified to run)

```xml
<mujoco model="peg_in_hole">
  <option timestep="0.002" integrator="implicitfast" cone="elliptic"/>
  <default><geom condim="4" friction="0.6 0.005 0.0001" solref="0.005 1"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <!-- square hole: opening 21 mm, depth 40 mm; peg is 20 mm → 0.5 mm clearance per side -->
    <body name="hole" pos="0 0 0.02">
      <geom name="w_px" type="box" size="0.02 0.0505 0.02" pos=" 0.0305 0 0"/>
      <geom name="w_nx" type="box" size="0.02 0.0505 0.02" pos="-0.0305 0 0"/>
      <geom name="w_py" type="box" size="0.0105 0.02 0.02" pos="0  0.0305 0"/>
      <geom name="w_ny" type="box" size="0.0105 0.02 0.02" pos="0 -0.0305 0"/>
    </body>
    <body name="gantry_x" pos="0 0 0.15">
      <joint name="jx" type="slide" axis="1 0 0" damping="20"/>
      <inertial pos="0 0 0" mass="0.5" diaginertia="1e-3 1e-3 1e-3"/>
      <body name="gantry_y">
        <joint name="jy" type="slide" axis="0 1 0" damping="20"/>
        <inertial pos="0 0 0" mass="0.5" diaginertia="1e-3 1e-3 1e-3"/>
        <body name="wrist">
          <joint name="jz" type="slide" axis="0 0 1" damping="20"/>
          <inertial pos="0 0 0" mass="0.5" diaginertia="1e-3 1e-3 1e-3"/>
          <body name="peg" pos="0 0 -0.01">
            <site name="ft" pos="0 0 0" size="0.004"/>
            <geom name="peg" type="box" size="0.010 0.010 0.03" pos="0 0 -0.03" mass="0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="jx" kp="800"/>
    <position name="ay" joint="jy" kp="800"/>
    <position name="az" joint="jz" kp="800"/>
  </actuator>
  <sensor>
    <force name="ft_force" site="ft"/>
    <torque name="ft_torque" site="ft"/>
  </sensor>
</mujoco>
```

Read with `data.sensor("ft_force").data` / `data.sensor("ft_torque").data`. Design notes Claude must understand and write up:

- A MuJoCo `force`/`torque` sensor reports the interaction wrench between the **body that owns the site and its parent**, in the site frame. The peg is its own body under `wrist` precisely so the sensor sees everything distal to the wrist: peg weight, peg inertia, and contact. That is what a real wrist F/T sensor sees.
- The position actuators are springs (`F = kp · error`). This is a 3-DoF Cartesian impedance controller in miniature; `kp` is the stiffness knob.
- Square peg + box walls avoids mesh/convex-hull collision issues. The hole has no chamfer, so small lateral offsets jam — on purpose.

## 7. Milestones

Each milestone: implement → run → save artifacts to `outputs/mN/` → add tests → append a dated section to `docs/FINDINGS.md` that answers the listed questions with numbers and a plot reference.

### M0 — Repo + container + API probe
- Create the layout in §3, Dockerfile, compose, Makefile, `pyproject.toml`, `.gitignore`, `.dockerignore`.
- `scripts/00_probe_api.py`: print package versions; list `suite.ALL_ENVIRONMENTS`; build `Wipe` with Panda headless; print obs keys, all MuJoCo sensor names and dims, timestep, solver iterations, controller type and gains, `env.robots[0].ee_force`. Write the dump to `outputs/m0/probe.json`.
- `tests/test_env_smoke.py`: raw MuJoCo scene loads and steps 100 times without NaN; robosuite `Wipe` makes, resets, steps 10 zero actions; F/T accessors return shape (3,).
- **Accept:** `make build && make test && make m0` succeed from a clean clone. The probe output matches §2; if it doesn't, update §2 and CLAUDE.md before continuing.

### M1 — Track A: first insertion and the F/T time series
- `scenes/builder.py` exposing clearance, peg mass, `kp`, damping, timestep, integrator, cone, `solref`, `solimp`, friction, `noslip_iterations` as parameters.
- `01_gantry_insert.py`: scripted linear descent with `--x-offset-mm` (0, 0.3, 1, 4), log per §5, render an MP4 if GL works.
- Plots: Fz(t) and |F|(t) with contact onset marked; Fz vs insertion depth; lateral Fx/Fy vs offset.
- **Questions:** What is the sign convention of the sensor (derive it from the free-hanging reading, don't look it up)? What does the contact-onset transient look like at physics rate vs control rate (`ft_raw_hf` vs `ft_raw`)? At which lateral offset does insertion turn into a jam, and what force does the jam settle at (compare to `kp × position error`)?
- **Tests:** `test_ft_static.py` — peg hanging at rest reads |Fz| = m·g within 1%; doubling peg mass doubles it.

### M2 — Contamination: gravity and inertia
- `02_contamination.py` runs free-space motions only (no contact): hold still; constant-velocity descent; sinusoidal z motion at 0.5 / 2 / 5 Hz; tilt experiment (add a hinge to the wrist in the builder and rotate the peg 0–90°).
- `ft/compensate.py`: `F_contact ≈ F_meas − m·Rᵀg − m·a` (and the torque analogue using the peg COM offset from the site). Identify `m` and COM from static poses by least squares rather than reading them from the XML — that is how it is done on a real robot.
- `contacts.py`: sum `mujoco.mj_contactForce` over contacts involving the peg geom, rotate from contact frame to world, take care with which geom is `geom1` vs `geom2` for the sign.
- **Questions:** How large is the free-space reading relative to typical contact forces in M1? How much does compensation reduce free-space residual (report RMS before/after)? During contact, how well does `ft_comp` match `contact_wrench` ground truth?
- **Tests:** `test_compensation.py` — free-space residual after compensation < 5% of m·g RMS for the 2 Hz motion; `test_contact_consistency.py` — static pressed-against-rim case: `ft_comp` force equals summed contact force within 2%.

### M3 — Solver and contact-parameter sensitivity
- `03_solver_sweep.py`: grid over timestep {0.0005, 0.001, 0.002, 0.005}, integrator {Euler, implicitfast}, cone {pyramidal, elliptic}, `solref` time constant {0.002, 0.005, 0.02}, `noslip_iterations` {0, 5}, stiffness `kp` {200, 800, 3000}, approach speed {10, 50, 150 mm/s}. Same scripted insertion with 0.3 mm offset, fixed seed.
- Metrics per run: peak |F|, contact-onset spike height and width, steady-state force, max penetration depth (from `data.contact[i].dist`), success, wall-clock per sim second.
- Output: tidy CSV + heatmaps + a short ranked list of "parameters that change the force signal the most".
- **Questions:** Which parameters change the *force* signal without visibly changing the *motion*? Where does the sim become unstable or let the peg tunnel? What is a defensible default for later data collection and why?
- This is the core lesson for later policy learning: a policy trained on force from one solver config is trained on that config's artifacts.

### M4 — Track B: robosuite Panda on `Wipe`
- `04_robosuite_wipe_ft.py`: `suite.make("Wipe", robots="Panda", controller_configs=cfg, has_renderer=False, has_offscreen_renderer=<GL ok>, use_camera_obs=False, control_freq=20)`. Scripted policy: move above the table, descend until |F| crosses a threshold, slide laterally while regulating downward displacement. Actions are OSC_POSE deltas (6-D here; `Wipe` uses a gripperless wiping tool).
- Log both `env.robots[0].ee_force["right"]` and the raw `gripper0_right_force_ee` sensor and assert they agree. Also log the applied joint torques (`RobosuiteFT.joint_torques()`, see §2) as the joint-torque view of the same contact (the TA-VLA-style signal).
- Rotate the wrench to world using the `ft_frame` site rotation (`env.sim.data.site_xmat`); sanity check: pressing straight down on the table gives a world-Z force.
- **Questions:** How does arm F/T in free space compare to Track A (the tool is heavier and the arm accelerates in all axes)? Does the threshold-based contact detector false-trigger during fast free-space motion before compensation? After?

### M5 — Controller stiffness vs contact force
- `05_robosuite_gain_sweep.py`: same press-and-slide script with OSC `kp` ∈ {50, 150, 400, 1000} and `damping_ratio` ∈ {0.7, 1, 2} by editing `cfg["body_parts"]["right"]`. Then repeat with `impedance_mode="variable_kp"` and lower the gains only after contact is detected.
- Plots: peak force and steady force vs `kp`; tracking error in free space vs `kp`.
- **Questions:** Quantify the stiffness trade-off (tracking accuracy vs contact force). How much does switching to low stiffness on contact reduce peak force? This is the motivation for compliance-predicting policies.

### M6 — Peg-like task on the arm + episode recorder
- `06_robosuite_nut_round.py`: `NutAssemblyRound` with a scripted grasp-move-lower sequence (uses the gripper, so action dim is 7). Log F/T through grasp, transport, and mating. Expect partial success from a script; record the failure modes.
- `07_record_episodes.py`: generic recorder CLI (`--track {A,B} --task ... --n 50 --noise-mm 2 --seed 0 --images {0,1}`) that randomizes initial offset and writes §5 episodes to `data/`. `test_logger_schema.py` validates keys, shapes, dtypes, finite values, monotonic time.
- **Stretch:** custom single-arm `PegInHole` robosuite env — subclass `ManipulationEnv`, reuse the peg/hole objects from `TwoArmPegInHole`, mount the peg on the Panda in place of the gripper, fix the hole to the table. Only attempt after M4–M5 are complete.
- **Accept:** 50 Track-A episodes and 20 Track-B episodes recorded, schema test green, a summary plot of peak force distribution across episodes.

## 8. `docs/FINDINGS.md` template

For each milestone: date, commit SHA, command line used, 3–6 bullet findings with numbers and units, the plots that back them, and "what this means for a force-aware policy" in one or two sentences. Close Stage 0 with a one-page summary covering: sensor sign/frame convention; size of gravity/inertial contamination and how well compensation works; which sim parameters the force signal is sensitive to and the chosen defaults; stiffness vs peak-force trade-off; recommended preprocessing for a learning pipeline (compensation, filtering cutoff, normalization range, history window).

## 9. Hooks for later stages (do not build now)

- Stage 1 (demos): `07_record_episodes.py` + §5 schema → converter to LeRobot / robomimic HDF5. **Started 2026-09-22 as a Tier 1 BC experiment** (`fvb.policy`, scripts 09–11, `make s1`, `docker/Dockerfile.train` adds torch): see FINDINGS “Stage 1”. Completed under §11.
- Stage 2–3 (small policies with force fusion): add `torch` to the image, use the `dev-gpu` compose service. `ft_comp` and a short `ft_raw_hf` history window are the inputs a force token would summarize.
- Stage 4 (SmolVLA / openpi LoRA on LIBERO): separate image; LIBERO has its own robosuite pin, so do not merge environments.

## 10. Working agreement for Claude

1. Start with M0. Do not skip ahead; later scripts depend on `fvb` modules from earlier milestones.
2. Before each milestone, restate its acceptance criteria; after, show the test output and list the artifacts written.
3. Prefer the smallest implementation that answers the milestone's questions. No frameworks, no config systems beyond argparse + a dataclass.
4. If a result contradicts §2 or a claim in this plan, trust the experiment, fix the doc, and say so in FINDINGS.
5. Ask before adding dependencies, changing pinned versions, or changing the §5 schema.

## 11. Stage 1 — force-token policies: arm transfer, physics robustness, data export (2026-09-24)

Stage 1 started 2026-09-22 as a Tier 1 experiment on the gantry (FINDINGS "Stage 1", "Stage 1b"):
on a hidden-hole insertion a 0.8 M-parameter ACT-lite succeeds 50/50 with the compensated wrench
and ~5/50 without it, and training on sensor noise buys robustness up to σ = 0.02 N·m. This
section finishes the stage.

### 11.1 Goals

- **G1 — Arm transfer.** The same hidden-hole behaviour-cloning experiment on the Panda
  (`fvb.envs.PegInHole`): privileged expert, ACT-lite and MLP with and without force, closed-loop
  eval. *Accept:* expert ≥ 95 % success; ACT-lite + force beats force-zeroed by ≥ 30 points.
- **G2 — Robustness to simulator parameters.** M3 showed timestep and solref change peak force
  ~2× without changing motion. Measure how the nominal gantry policy degrades over a
  timestep × solref grid, and whether training on randomised physics fixes it. *Accept:* the
  randomised-physics policy ≥ 90 % in every grid cell.
- **G3 — Data pipeline.** Export §5 episodes + policy observations/actions to a robomimic-style
  HDF5 (`h5py` only — already in the image; LeRobot would be a new dependency, so it is left
  out per §10.5). *Accept:* schema test green; both expert datasets export.
- **G4 — Report.** Findings appended to `docs/FINDINGS.md`, plus an HTML report with arm videos
  (untrained / no force / force / expert).
- **Fallback — DAgger**, only if G1 or G2 misses its bar: roll out the learned policy, relabel
  the visited states with the privileged expert (run in shadow on the same trajectory),
  aggregate, retrain.

### 11.2 Design decisions

- **Task spec registry** (`fvb.policy.spec`): obs dim, action dim and force-channel slice per
  task (`gantry`, `arm`), stored in every checkpoint's meta so training/eval/rollout code is
  task-agnostic. Old checkpoints default to `gantry`.
- **Arm observation (20):** peg-tip position relative to the episode start (3), F/T-site linear
  velocity (3), peg-axis tilt (axis x, y: 2), compensated wrench rotated to **world** (6),
  physics-rate summary of the last interval (max |F|, max |T|, std |F|: 3), previous action (3).
  Force channels = the 9 wrench + summary channels. World frame because the flange yaw varies
  with robosuite's init noise; Stage 0 showed mixing frames silently swaps axes.
- **Arm action (3):** peg-tip target delta in world, clipped ±5 mm/step. A fixed low-level layer
  turns the tip target into OSC_POSE actions: it servos the peg axis to vertical (a zero rotation
  delta does not hold orientation, M6) and integrates the xy tip error (the OSC settles ~1 mm off
  its goal, M6). The policy decides *where the tip goes*, as on the gantry.
- **Hidden hole on the arm:** the env is built once; per episode the `hole` body is moved in the
  model (`body_pos`) to N(0, 1.5 mm) per axis, clipped ±4 mm, around the nominal centre. The
  peg starts 20 mm above the rim over the nominal centre after a settle phase that is not part
  of the episode.
- **Arm expert:** the gantry state machine (descend; on |F| > jam threshold with stalled depth
  retract 3 mm, step ≤ 0.5 mm toward the true hole, descend), acting on tip targets.
- **Physics randomisation (G2):** `TaskParams` gains `solref_tc` and an optional sampler over
  timestep ∈ {0.0005, 0.001, 0.002} s and solref time constant log-uniform in [0.002, 0.02] s.
  The eval grid is timestep {0.0005, 0.001, 0.002} × solref {0.002, 0.005, 0.02}.
- **HDF5 layout (G3):** `data/demo_<i>/{actions, rewards, dones, obs/<key>}`, one group per
  episode; obs keys = the named policy-observation groups + §5 control-rate arrays + `ft_raw_hf`
  reshaped to (T, n_sub, 6); `data.attrs["env_args"]` = JSON task config; `mask/{train,valid}`.

### 11.3 Milestones

| id | what | files | tests / acceptance |
|---|---|---|---|
| S1.1 | task-spec registry; train/eval/rollout take the task from data/checkpoint | `policy/spec.py`, `data.py`, `rollout.py`, scripts 10–11 | existing tests green; old gantry checkpoints still load |
| S1.2 | HDF5 export | `policy/export.py`, `scripts/15_export_hdf5.py` | `test_export.py`: groups, shapes, masks, round-trip equality |
| S1.3 | arm hidden-hole task + expert + collection | `policy/arm_task.py`, `scripts/16_collect_arm.py` | `test_arm_task.py`: obs hides the hole (identical first obs across offsets), expert succeeds on 3 seeds; 300 episodes, expert ≥ 95 % |
| S1.4 | arm BC: ACT/MLP × force/no-force, closed-loop eval on 50 unseen seeds | reuse 10/11 via spec | G1 bar |
| S1.5 | physics randomisation on the gantry: nominal vs randomised training, 3 × 3 eval grid | `task.py`, `scripts/17_physics_sweep.py` | G2 bar |
| S1.6 | DAgger (only if S1.4 or S1.5 misses its bar) | `scripts/18_dagger.py` | the missed bar |
| S1.7 | FINDINGS + HTML report with arm videos | `docs/FINDINGS.md`, `docs/stage1/` | G4 |

### 11.4 Status (2026-09-25)

| id | status | deviation from the plan |
|---|---|---|
| S1.1 | done | spec also carries `gantry_kp`; checkpoints record `action_mode` |
| S1.2 | done | — |
| S1.3 | done | arm collection is `09_collect_expert.py --task arm` (no separate script 16); expert needs a two-speed approach; observed torque is taken about the peg tip; xy integrator frozen in contact |
| S1.4 | G1 met by the MLP (39/50 vs 4/50), not by ACT-lite (4/50 vs 4/50) | — |
| S1.5 | G2 met (9/9 cells at 100 %) | added a kp axis and kp randomisation; kp randomisation broke delta-action BC (diagnosed), `action_mode="xy_abs"` partly fixes it |
| S1.6 | not run | failures are underfitting, not compounding error |
| S1.7 | done | report in `docs/stage1/` |

Open: why ACT-lite ignores force on the arm; the source of the 8 N lateral force along the arm's
reach; a lateral action that is both timing-robust (like `xy_abs`) and precise (like `delta`),
e.g. ACT's temporal ensembling of chunks or a discrete "correct now" head.

## 12. Stage 1.5 — reliable RL with force/torque (started 2026-09-29)

TD3 on the arm hidden-hole task learns only in 1 of 5 seeds (with F/T: 49/50 unseen holes for
seed 3, the rest 0–9/50; without F/T no seed beyond chance; README "Reinforcement learning").
The other runs sit in two local optima: dive-and-abort and hover-until-timeout.

**Goal G5.** Make RL reliable enough that the force/torque comparison is a result, not a seed
lottery. *Accept:* with F/T, ≥ 4 of 5 seeds reach ≥ 80 % on the 50 unseen holes within 150k
environment steps; the no-F/T condition, run with the identical recipe, is reported alongside.

**Experiment queue** (each: 5 seeds per condition unless noted, same budget, journaled in
`docs/progress/journal.jsonl`; a variant that does not help is switched off / reverted and the
journal says so):

| id | experiment | hypothesis |
|---|---|---|
| R1 | TD3 + demonstrations: replay pre-filled with expert transitions, TD3+BC actor loss (Fujimoto & Gu 2021) with a BC weight decaying to 0 | demos put the agent past both local optima; the decaying BC term lets RL outgrow the expert's slow approach |
| R2 | asymmetric critic: the critic also sees the privileged tip→hole vector; the actor does not | an informed critic gives the actor a useful gradient early, without making the policy privileged |
| R3 | combine / tune the winners of R1–R2 | — |
| R4 | final F/T vs no-F/T comparison with the winning recipe | the no-F/T agent gets the same help; does it now search, or still fail? |
| R5 | ACT-lite ignoring force on the arm (S1.4 open issue) | — |

**Reporting.** An hourly HTML status page (`docs/progress/index.html`, built by
`docs/progress/build.py`) summarises the goal, where the work is, progress since the previous
summary, and what was tried, kept and reverted.

**Status (2026-09-29 17:05).** Queue complete. G5 not met: the best recipe (TD3 + 100 demos, BC
2.5 → floor 1.0) makes 5/10 F/T seeds reliable and 0/10 without F/T (62.2 % vs 14.8 %,
p = 0.0014). R5 solved: ACT-lite's force blindness on the arm was dropout; with dropout 0 it
scores 50/50. Details in FINDINGS "Stage 1.5" and the journal.

## 13. Stage 2 — bolt/nut sorting with a force-aware VLA (started 2026-09-29)

Full visual plan: `docs/plans/vla_bolt_nut_sorting_plan.html`. Summary:

**Task.** A Panda with a Franka Hand sorts parts from a bin: ISO 4017 M16×50 hex bolts go into a
Ø 17 mm clearance hole (ISO 273 fine, 0.5 mm radial clearance, 1 mm chamfer) and vanish 1 s after
being seated and released; ISO 4032 M16 nuts go into bucket A. Start with 3 bolts + 3 nuts.

**Robot and sensors.** Real-Panda-like: 7 joint-torque sensors (link-side τ_J from
`jointactuatorfrc` + passive, τ_ext = τ_J − `mj_inverse` model torque, F̂_ext = J^-T τ_ext, 1 kHz,
noise + bias); Franka Hand force limit raised from robosuite's 20 N to 70 N; each finger pad
replaced by a 4×4 taxel grid (separate geoms, one `touch` sensor each) plus a pad `force` sensor
for shear; cameras `agentview` + `robot0_eye_in_hand` at 256², EGL rendering.

**Decisions (2026-09-29).** Scripted stand-in demos first (privileged expert + human-like noise,
labelled synthetic); SmolVLA fine-tuned locally on the RTX 3080; few parts first; LeRobot in a
separate `Dockerfile.vla` (LeRobot needs Python ≥ 3.12 / numpy ≥ 2, robosuite 1.5.2 needs numpy
< 2, so the sim and the policy talk over TCP via `multiprocessing.connection`).

**Phase 2 (TA-VLA, arXiv 2509.07962).** Torque history (10 frames over 2 s; τ_ext 7 + tactile
summary 8) flattened into one token through an MLP and prepended to SmolVLA's action-expert
*suffix* (SmolVLA keeps state in the VLM prefix, so this is a subclass); joint loss
L = L_action + 0.1·L_torque. Also: ACT + force token (dropout 0 on that path, R5) and TD3 skills
(pick, insert with unknown in-hand offset) with torque/tactile observations.

| id | deliverable | gate |
|---|---|---|
| V0 | VLA image, policy server round trip, SmolVLA memory probe | fine-tune step fits 10 GB; round trip < 200 ms |
| V1 | M16 bolt/nut generators, fixture, bin, bucket | ISO dims ±0.1 mm, mass ±5 %, stable drops, no penetration > 0.5 mm |
| V2 | Franka Hand force, taxel pads, joint-torque sensor | τ_ext ≈ Jᵀmg ±2 %; taxel sum ≈ applied force ±5 %; CoP follows contact |
| V3 | SortBoltsNuts env (spawn, success, vanish, cameras, language, reward) | seat→vanish after 1 s; tilted/jammed/gripped bolt not seated; nut in bucket counts |
| V4 | scripted expert + human-like noise, 200 synthetic demos (with torque + tactile) | expert ≥ 90 % episodes; dataset validates |
| V5 | ACT behaviour cloning baseline | success on 50 unseen seeds (reference) |
| V6 | TD3 skills (pick, insert) with TD3+BC, scripted sequencer | skill success, 10 seeds each |
| V7 | SmolVLA fine-tune via the policy server | success vs ACT on 50 unseen seeds; latency |
| V8 | force features + seated / grasp-stable labels | offline detectors > 95 % on held-out episodes |
| V9 | ACT + force token, TD3 + F/T obs, TA-SmolVLA | each vs its no-force twin, same seeds |
| V10 | evaluation matrix, report, README | all cells with 95 % intervals; FINDINGS |

**Status (2026-10-02 08:30).** V0–V4 done (V4 on the rack presentation); V6 skills trained,
sequencer ≤ 1/10 full sorts; V8 gate not met; V5/V7 whole-task policies fail; V9 done (force helps
TD3/BC insertion, the TA-SmolVLA torque token hurts); V10 matrix + report done.

| id | status | result / notes |
|---|---|---|
| V0 | done | SmolVLA fine-tune: batch 16 in 5.9 GB, 126 ms per 50-step chunk; round trip < 200 ms |
| V1 | done | pile gate refined: transient penetration < 2 mm, resting < 0.5 mm |
| V2 | done | pads protrude 1.5 mm past the finger hull; tau_J also counts joint friction and end stops |
| V3 | done | `sortview` camera (agentview cropped the scene); vanished bolts parked on the floor 5 m away |
| V4 | done (rack) | expert 47/50 unseen seeds (noise-free reading); 200 noisy demos validate (104 complete) |
| V5 | fails (reference) | ACT on tray per-part demos 1/60 parts, on 104 rack episodes 0/36 - never reaches a part |
| V6 | insert + pick done; sequencer failing | insert TD3+BC 10 seeds: **force 74.8 % vs no force 56.2 % (p = 0.0005)**; pick 99.0 % vs 99.6 % (ceiling); sequencer 0/10: learned picks leave the bolt 3-7 mm off centre, the insert skill only saw centred grasps - retraining on randomised in-hand offsets |
| V7 | fails (reference) | SmolVLA on 104 rack episodes 0/36 parts (closest approach 30-35 mm) |
| V8 | gate not met | grasp-stable 94-95 % (MLP, tactile-driven), tip-in-hole at release 91 % (torque-driven); 5 data rounds |
| V9 | done (mixed) | force helps TD3 insert (74.8 vs 56.2 %) and BC insert (28.8 vs 6.8 %); TA-SmolVLA torque token hurts the insertion skill: raw 53 %, z-scored 43 % vs zeroed twin 66 %, plain 61 % (100 held-out set-ups each); token stays off |

V4 findings that changed the design (all journaled):

* **Bolt presentation.** With bolts dropped into the tray the scripted expert plateaued at ~58 %
  of parts (the head-up reorientation of a lying bolt with a two-pad gripper is fragile), and
  learned policies failed to locate parts. Opt-in `SortTaskParams(bolt_presentation="rack")`:
  bolts stand head-up in a narrow raised rack (a kitting presentation), nuts stay in the tray.
  Noise-free expert on 50 unseen seeds: 47/50 episodes, 297/300 parts. **Pending the user's
  decision** to make the rack the V4 design; the tray remains the env default.
* **Gate reading.** The ≥ 90 % gate is measured on the expert without its injected "human"
  noise; the recorded demos carry the noise (action jitter 0.02, hole-estimate and grasp noise,
  pauses) and training uses successful episodes / per-part segments. **Pending confirmation.**
* **Scene.** Tray 360 × 260 × 40 mm instead of the 300 × 220 × 80 mm bin (the palm hit tall
  walls); bucket A beside the tray at (−0.26, −0.46) with a box floor (a disc floor ejected
  ~1/40 dropped nuts through a mesh-vs-cylinder contact artefact).
* **Insertion.** Hover over the hole, lower the tip 8 mm, release; spiral search over missed
  entries (no contact-driven push-in: a pushed bolt pivoted in the grasp).
* **Gripper.** Finger gain 4000 (~45 N grip, under the Hand's 70 N); learned policies'
  gripper output is snapped to the expert's −1 / 0 / +1 because robosuite integrates sign(a).
