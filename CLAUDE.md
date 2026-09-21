# CLAUDE.md — force-vla-basics

Learning repo for force-aware manipulation. Current scope is **Stage 0 only**: get trustworthy force/torque (F/T) readings from a simulated peg-in-hole task and understand what contaminates them. No VLA, no policy learning yet. The full plan is in `PLAN.md`; work through its milestones in order.

## Hard rules

- **Everything runs inside the Docker container.** Never `pip install` on the host. Use `make build`, `make shell`, `make test`, `make run S=<script>`. If `make` is missing on the host, the equivalent is `docker compose -f docker/compose.yaml run --rm -T -u $(id -u):$(id -g) dev <cmd>`.
- **Pinned versions are deliberate.** `mujoco==3.3.0`, `robosuite==1.5.2`, `numpy<2`. robosuite 1.5.2 crashes on newer MuJoCo (`AssertionError` in `get_joint_qpos_addr`). Do not bump without running `make test`.
- **Headless only.** Never open a viewer window (`has_renderer=False`, no `mujoco.viewer`). Visual output is PNG plots and MP4s written to `outputs/`.
- **Verify APIs by introspection, not memory.** Before using a robosuite/MuJoCo attribute, confirm it exists (see `scripts/00_probe_api.py`). Known trap: `obs["robot0_eef_force"]` does **not** exist in robosuite 1.5.2 — use `env.robots[0].ee_force["right"]` or the raw sensor `gripper0_right_force_ee`. Second trap: `env.robots[0].torques` is `None`; joint torques are `env.sim.data.ctrl[robot._ref_arm_joint_actuator_indexes]` (`RobosuiteFT.joint_torques()`). Third trap: robosuite generated objects carry `margin="0.001"` (1 mm collision skin) — zero it for tight-clearance tasks.
- **Units and frames in every log and plot.** Force N, torque N·m, position m, time s. State the frame (sensor-site frame vs world). Never plot an unlabeled axis.
- **Determinism.** Every script takes `--seed` and writes its full config next to its outputs.
- **Small steps.** One milestone per branch/commit series. Each milestone ends with: tests green, artifacts in `outputs/<milestone>/`, and findings appended to `docs/FINDINGS.md`.

## Conventions

- Python 3.11, `src/` layout, package name `fvb`. Type hints, `ruff` for lint/format, `pytest` for tests.
- Scripts in `scripts/` are thin CLIs (argparse) over functions in `src/fvb/`. No logic in notebooks.
- Episode logs are `.npz` with keys defined in `PLAN.md §5`. Do not invent new keys without updating that section.
- `outputs/` and `data/` are git-ignored. `docs/FINDINGS.md` is committed and is the real deliverable of this stage.

## When something is surprising

Physically odd readings (nonzero force in free space, sign flips, huge spikes) are usually the lesson, not a bug. Investigate, explain it in `docs/FINDINGS.md`, then move on. If it is a bug, write a test that pins the fix.
