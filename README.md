# force-vla-basics

Force-aware manipulation in simulation (MuJoCo 3.3.0 + robosuite 1.5.2). **Stage 0:** trustworthy
force/torque readings from peg-in-hole-style tasks and what contaminates them. **Stage 1:** small
behaviour-cloning policies that use the wrist force/torque to find a hidden hole, on a gantry and
on a Panda arm. The plan is `PLAN.md`; the results are `docs/FINDINGS.md`.

## Examples

**Panda arm, hidden hole (Stage 1).** Four controllers on the same hidden hole, 0.5 mm clearance:
an untrained network, the network trained without force, the network trained with the wrist
force/torque, and the privileged expert. Only the force-trained policy and the expert insert the
peg (39/50 vs 4/50 without force over 50 unseen episodes). Each tile: 3D close-up, a to-scale
top-down schematic in mm, and the compensated force / torque about the peg tip.

![Panda arm: untrained vs trained with and without force](docs/img/arm_untrained_vs_trained.png)

**Gantry, hidden hole (Stage 1).** The same comparison on the 3-axis gantry: the untrained
network drives into the rim (60 N abort), the no-force network dithers until timeout, the
force-trained ACT-lite policy inserts as fast as the expert (50/50 vs 7/50 without force).

![Gantry: untrained vs trained with and without force](docs/img/gantry_untrained_vs_trained.png)

Full interactive reports (open locally in a browser): `docs/report/index.html` (Stage 0 +
first Stage 1 experiment) and `docs/stage1/index.html` (Stage 1 completion).

## Reinforcement learning: TD3 with vs without force/torque

The same Panda hidden-hole insertion learned from reward alone with TD3
(`src/fvb/policy/td3.py`, `scripts/18_train_td3.py`). The two agents are identical except that
one has the 9 wrist force/torque channels zeroed.
- **Networks:** actor and twin critics are 256×256 ReLU MLPs.
- **Observation:** the last 4 steps of the 20-channel arm observation.
- **Action:** a peg-tip move of ±1 mm (x, y) and ±3 mm (z) per step.
- **Reward:** depth progress, potential-based shaping toward the true hole (the reward sees the
  hole; the policy never does), a time cost, a force penalty, +50 for success and −20 for a
  60 N abort.
- **Budget:** 150k environment steps per run, 5 seeds per condition.

![TD3 training reward and evaluation success, with vs without force/torque](docs/img/td3_training_reward.png)

**Final result (best checkpoint per run, 50 unseen holes):**

| seed | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| with F/T | 9/50 | 6/50 | 0/50 | **49/50** | 0/50 | 25.6 % |
| without F/T | 4/50 | 9/50 | 1/50 | 0/50 | 8/50 | 8.8 % |

- **One F/T seed solves the task:** every hole more than 1 mm away is inserted, in a median of
  31 steps (the scripted expert needs ~114).
- **No agent without F/T gets beyond chance:** at most 9 % of holes more than 1 mm away.
- **The other runs are stuck in local optima:** diving fast into the 60 N abort, or hovering
  until timeout.
- **Verdict:** force/torque is what makes the solution learnable, but at this budget TD3 finds
  it in only 1 of 5 seeds. Behaviour cloning from the expert is far more reliable (MLP + F/T
  39/50 vs 4/50 without).

An earlier batch of these runs exploited a bug in the success check: depth alone counted a peg
lowered onto the table beside the hole block. It's fixed and covered by regression tests; the
numbers above are from the corrected re-run. Regenerate the plot with
`python scripts/19_plot_td3.py`; the full report is `docs/td3/index.html`.

## Quickstart

```bash
make build      # build the Docker image (CPU, headless)
make test       # pytest inside the container (14 tests)
make m0         # API probe             -> outputs/m0/probe.json
make m1         # Track A insertion     -> outputs/m1/
make m2         # contamination + comp  -> outputs/m2/
make m3         # solver sweep (864)    -> outputs/m3/
make m4         # Panda on Wipe         -> outputs/m4/
make m5         # stiffness sweep       -> outputs/m5/
make m6         # NutAssemblyRound, custom PegInHole, record 50 A / 20 Wipe / 10 Nut / 20 PegInHole episodes
```

### Stage 1 — Tier 1 behaviour cloning (needs the GPU image)

```bash
make build-train   # Stage 0 image + PyTorch (CUDA 12.4 wheels)
make s1            # 300 expert episodes -> train ACT-lite with/without force -> closed-loop eval
```

### Report

`docs/report/index.html` is the Peg-in-Hole Force Report: robot videos, speed/force/torque charts,
the ACT-lite topology, training curves and untrained-vs-trained policy videos. Open it in a
browser (media is in `docs/report/media/`). Regenerate with `make report` after `make all` and
`make s1` (and `scripts/12_noise_sweep.py` for the noise chart).

### Stage 1 completion (PLAN §11)

`docs/stage1/index.html` is the Peg-in-Hole Stage 1 Report: the hidden-hole task on the Panda
(`09_collect_expert.py --task arm`, `11_eval_bc.py --task arm`), physics/stiffness robustness
(`17_physics_sweep.py`), robomimic-style HDF5 export (`15_export_hdf5.py`), arm videos
(`13_policy_videos.py --task arm`). Rebuild the page with `make report-s1` once those outputs exist.

Everything runs inside Docker (`make shell` for a prompt, `make run S=scripts/xx.py ARGS="..."`
for one script). Without `make`: `docker compose -f docker/compose.yaml run --rm -T -u $(id -u):$(id -g) dev <cmd>`.

## Layout

- `src/fvb/scenes` — Track A MJCF builder (clearance, mass, kp, solver params, tilt hinge)
- `src/fvb/ft` — `read` (MuJoCo + robosuite F/T, one interface), `frames`, `compensate`
  (LSQ mass/COM identification, gravity + inertial), `filters`
- `src/fvb/control` — `gantry` (Track A) and `osc_scripts` (Track B: `ArmRig`, press-and-slide,
  variable-kp, nut grasp-and-mate)
- `src/fvb/policy/` — Stage 1: hidden-hole task + privileged expert (`task.py`), windows/normalisation (`data.py`), ACT-lite and MLP (`models.py`), closed-loop rollout (`rollout.py`)
- `src/fvb/envs/peg_in_hole.py` — custom single-arm `PegInHole` robosuite env (peg on the flange, tight hole on the table, flange F/T)
- `src/fvb/contacts.py` — ground-truth contact wrench from `mj_contactForce`
- `src/fvb/logging/episode.py` — PLAN §5 `.npz` + `.json` episode logger and validator
- `scripts/0*.py` — one thin CLI per milestone; `tests/` — pytest
