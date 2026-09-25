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
