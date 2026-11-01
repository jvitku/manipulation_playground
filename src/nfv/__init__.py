"""nfv — the nano force-VLA school's testbed and models (lessons 15-19).

Milestone M0 (this package so far): a rigid Cartesian gantry in plain MuJoCo with a wrist
F/T sensor, position and hybrid force-position controllers, tasks T0-T3, privileged scripted
experts, and the demo recorder/loader. Everything runs CPU-only (MUJOCO_GL=osmesa) inside the
fvb-lab image. See docs/plans/force_vla_school_plan.md §5 and §8 (gate M0).

Layout:
    nfv.sim    gantry_xml, sensors, controllers, env, tasks, experts
    nfv.data   record (demos -> npz shards), dataset (shards -> RAM -> torch batches)
    nfv.seeds  disjoint demo / train / eval seed ranges
"""
