#!/usr/bin/env python
"""M0: probe the installed MuJoCo/robosuite API and dump the facts to outputs/m0/probe.json.

Everything in PLAN.md §2 is re-checked here. Never trust memory; trust this dump.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np


def probe_versions() -> dict:
    import matplotlib
    import mujoco
    import robosuite
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "robosuite": robosuite.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
    }


def probe_raw_scene() -> dict:
    import mujoco

    from fvb.ft.read import MujocoFT
    from fvb.scenes import seed_xml_text

    m = mujoco.MjModel.from_xml_string(seed_xml_text())
    d = mujoco.MjData(m)
    ft = MujocoFT(m, d)
    # settle for 1 s so the peg hangs at rest on the position actuators
    for _ in range(500):
        mujoco.mj_step(m, d)
    return {
        "timestep": float(m.opt.timestep),
        "integrator": int(m.opt.integrator),
        "cone": int(m.opt.cone),
        "solver_iterations": int(m.opt.iterations),
        "sensors": [
            {
                "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i),
                "dim": int(m.sensor_dim[i]),
                "type": int(m.sensor_type[i]),
            }
            for i in range(m.nsensor)
        ],
        "ft_site": ft.site_name,
        "ft_raw_at_rest": ft.ft_raw().tolist(),
        "peg_mass_kg": float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "peg")]),
        "gravity": m.opt.gravity.tolist(),
        "n_contacts_at_rest": int(d.ncon),
    }


def probe_robosuite(offscreen: bool) -> dict:
    import mujoco
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    from fvb.ft.read import RobosuiteFT

    cfg = load_composite_controller_config(controller="BASIC")
    env = suite.make(
        "Wipe",
        robots="Panda",
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=offscreen,
        use_camera_obs=False,
        control_freq=20,
    )
    obs = env.reset()
    ft = RobosuiteFT(env)
    zero = np.zeros(env.action_dim)
    for _ in range(3):
        obs, _r, _done, _info = env.step(zero)
    m = env.sim.model._model
    sensors = [
        {
            "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i),
            "dim": int(m.sensor_dim[i]),
            "type": int(m.sensor_type[i]),
        }
        for i in range(m.nsensor)
    ]
    arm_cfg = cfg["body_parts"]["right"]
    out = {
        "all_environments": sorted(suite.ALL_ENVIRONMENTS),
        "all_robots": sorted(suite.ALL_ROBOTS),
        "obs_keys": sorted(obs.keys()),
        "has_robot0_eef_force_obs": "robot0_eef_force" in obs,
        "action_dim": int(env.action_dim),
        "control_freq": float(env.control_freq),
        "model_timestep": float(env.model_timestep),
        "control_timestep": float(env.control_timestep),
        "physics_steps_per_action": int(round(env.control_timestep / env.model_timestep)),
        "solver_iterations": int(m.opt.iterations),
        "integrator": int(m.opt.integrator),
        "cone": int(m.opt.cone),
        "sensors": sensors,
        "ft_site": ft.raw.site_name,
        "controller_type": cfg.get("type"),
        "arm_controller": {
            k: arm_cfg[k]
            for k in (
                "type",
                "kp",
                "damping_ratio",
                "impedance_mode",
                "input_type",
                "input_ref_frame",
            )
            if k in arm_cfg
        },
        "ee_force_right": np.asarray(env.robots[0].ee_force["right"]).tolist(),
        "ee_torque_right": np.asarray(env.robots[0].ee_torque["right"]).tolist(),
        "ft_raw_sensor": ft.ft_raw().tolist(),
        "ft_robosuite": ft.ft_robosuite().tolist(),
        "robot_torques_attr": repr(env.robots[0].torques),
        "joint_torques_via_ctrl": ft.joint_torques().tolist(),
        "osc_controller_torques": np.asarray(
            env.robots[0].composite_controller.part_controllers["right"].torques
        ).tolist(),
        "gripper_prefix": env.robots[0].gripper["right"].naming_prefix,
    }
    env.close()
    return out


def probe_gl() -> dict:
    """Can we render offscreen? Physics never needs this; MP4s do."""
    import mujoco

    from fvb.scenes import seed_xml_text

    try:
        m = mujoco.MjModel.from_xml_string(seed_xml_text())
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, height=64, width=64)
        mujoco.mj_forward(m, d)
        r.update_scene(d)
        img = r.render()
        r.close()
        return {"ok": True, "shape": list(img.shape), "backend": os.environ.get("MUJOCO_GL")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/m0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dump: dict = {"argv": sys.argv, "seed": args.seed}
    dump["versions"] = probe_versions()
    dump["gl"] = probe_gl()
    dump["raw_scene"] = probe_raw_scene()
    dump["robosuite"] = probe_robosuite(offscreen=dump["gl"]["ok"])

    (out / "probe.json").write_text(json.dumps(dump, indent=2))
    print(json.dumps(dump, indent=2))
    print(f"\nwrote {out / 'probe.json'}")


if __name__ == "__main__":
    main()
