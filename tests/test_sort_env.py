"""V3 gate: SortBoltsNuts success logic by scripted forcing (PLAN §13, §14 test_sort_env).

Parts are teleported into test situations; the arm stays still (zero OSC delta, gripper open).
"""

import math

import mujoco
import numpy as np
import pytest

pytest.importorskip("robosuite")

import robosuite as suite  # noqa: E402

import fvb.envs  # noqa: E402, F401
from fvb.envs.fasteners import DEFAULT_HOLE  # noqa: E402
from fvb.envs.sort_bolts_nuts import BUCKET_H, BUCKET_R, INSTRUCTION  # noqa: E402

OPEN = np.r_[np.zeros(6), -1.0]


@pytest.fixture(scope="module")
def env():
    e = suite.make(
        "SortBoltsNuts",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        seed=0,
    )
    yield e
    e.close()


def _put(env, i, pos, quat=(1, 0, 0, 0)):
    d = env.sim.data._data
    qa, da = env.part_qadr[i], env.part_dadr[i]
    d.qpos[qa : qa + 3] = pos
    d.qpos[qa + 3 : qa + 7] = quat
    d.qvel[da : da + 6] = 0
    mujoco.mj_forward(env.sim.model._model, d)


def _clear_others(env, keep):
    """Move every other part to the far side of the bin floor so it cannot interfere."""
    for j in range(len(env.parts)):
        if j not in keep:
            _put(env, j, env.bin_floor + [0.1 - 0.03 * j, 0.05, 0.03])


def _run(env, seconds):
    infos = []
    for _ in range(int(round(seconds * env.control_freq))):
        _, _, _, info = env.step(OPEN)
        infos.append(info)
    return infos


def _quat_tilt(deg):
    a = math.radians(deg) / 2
    return (math.cos(a), math.sin(a), 0.0, 0.0)  # about x


def test_reset_spawns_settled_parts(env):
    obs = env.reset()
    assert env.sim.model.opt.timestep == pytest.approx(0.001)
    assert int(env.control_timestep / env.model_timestep) == 50
    assert [p.kind for p in env.parts] == ["bolt"] * 3 + ["nut"] * 3
    for i, p in enumerate(env.parts):
        assert env._in_bin(i), p.body
        assert env._speed(i) < 0.05
        assert p.status == "in_bin"
    assert obs["torque_hist"].shape == (150,)
    assert obs["tactile"].shape == (32,)
    assert obs["parts_pos"].shape == (18,)
    assert env.instruction == INSTRUCTION


def test_seated_bolt_vanishes_after_1s(env):
    env.reset()
    _clear_others(env, keep={0})
    top = env.hole_top
    _put(env, 0, top + [0, 0, 0.0005])  # head resting on the fixture, shank in the hole
    infos = _run(env, 0.5)
    assert all(i["parts"]["bolt0"] == "seated" for i in infos[1:])
    g = env.bolt_geometry(0)
    assert g["depth"] > 0.045 and g["tilt_deg"] < 1.0
    infos = _run(env, 0.6)
    status = [i["parts"]["bolt0"] for i in infos]
    k = status.index("vanished")
    t_vanish = (10 + k + 1) / env.control_freq  # steps since the first seated step
    assert 0.95 <= t_vanish - 1 / env.control_freq <= 1.1
    assert np.linalg.norm(env.part_pose(0)[0][:2]) > 4.0  # parked far away
    _run(env, 0.5)
    assert env.parts[0].status == "vanished"
    pos = env.part_pose(0)[0]
    assert np.linalg.norm(pos[:2]) > 4.0 and pos[2] < 0.05  # and it stays parked, at rest


def test_gripped_bolt_never_vanishes(env, monkeypatch):
    env.reset()
    _clear_others(env, keep={0})
    _put(env, 0, env.hole_top + [0, 0, 0.0005])
    monkeypatch.setattr(env, "_touching_gripper", lambda i: True)
    infos = _run(env, 1.5)
    assert all(i["parts"]["bolt0"] != "vanished" for i in infos)
    assert infos[-1]["parts"]["bolt0"] == "jammed"  # in the hole, at rest, still held


def test_touching_gripper_detects_a_grasp(env):
    """Physical check of the contact test: a nut between the closing fingers."""
    env.reset()
    grip = env.sim.data._data.site_xpos[env.grip_site].copy()
    j = env.parts.index(next(p for p in env.parts if p.kind == "nut"))
    _put(env, j, grip)
    gravity = env.sim.model._model.opt.gravity.copy()
    env.sim.model._model.opt.gravity[:] = 0  # the nut floats between the closing fingers
    assert not env._touching_gripper(j)
    touching = []
    for _ in range(10):
        env.step(np.r_[np.zeros(6), 1.0])  # close
        touching.append(env._touching_gripper(j))
    assert any(touching)
    assert env.parts[j].status == "grasped"
    env.sim.model._model.opt.gravity[:] = gravity


def test_tilted_bolt_is_not_seated(env):
    env.reset()
    _clear_others(env, keep={0})
    top = env.hole_top
    # tilted 12 deg with the tip 5 mm into the chamfer: it jams or falls, never seats
    axis = np.array([0, -math.sin(math.radians(12)), math.cos(math.radians(12))])
    _put(env, 0, top + [0, 0, -0.005] + axis * 0.050, quat=_quat_tilt(12))
    infos = _run(env, 2.0)
    assert all(i["parts"]["bolt0"] not in ("seated", "vanished") for i in infos)
    # the geometric rule itself, without physics: depth fine but 6 deg tilt -> not seated
    _put(env, 0, top + [0, 0, 0.0005], quat=_quat_tilt(6))
    assert not env.bolt_seated(0)
    _put(env, 0, top + [0, 0, 0.0005], quat=_quat_tilt(4))
    assert env.bolt_seated(0)


def test_bolt_lowered_beside_the_block_is_not_success(env):
    """Stage 1.5's reward hack as a regression test: deep enough, but not in the hole."""
    env.reset()
    _clear_others(env, keep={0})
    top = env.hole_top
    beside = top + [DEFAULT_HOLE.block / 2 + 0.03, 0, 0]
    beside[2] = env.table_offset[2] + 0.050 + 0.0005  # standing upright on the table
    _put(env, 0, beside)
    assert env.bolt_geometry(0)["depth"] >= env.task.seat_depth
    assert not env.bolt_seated(0)
    infos = _run(env, 1.5)
    assert all(i["parts"]["bolt0"] not in ("seated", "vanished") for i in infos)
    assert env.reward() == 0.0


def test_nut_in_bucket_counts_and_rim_does_not(env):
    env.reset()
    j = 3
    _put(env, j, env.bucket_floor + [0, 0, 0.05])
    infos = _run(env, 1.0)
    assert infos[-1]["parts"]["nut0"] == "in_bucket"
    assert env.reward() == pytest.approx(1 / 6)
    # on the rim (above the wall top) and moving nuts do not count (predicate, no physics)
    _put(env, j, env.bucket_floor + [BUCKET_R, 0, BUCKET_H + 0.008])
    assert not env.nut_in_bucket(j)
    _put(env, j, env.bucket_floor + [0, 0, 0.02])
    env.sim.data._data.qvel[env.part_dadr[j]] = 0.1
    assert not env.nut_in_bucket(j)


def test_all_sorted_is_success(env):
    env.reset()
    top = env.hole_top
    for i in range(3):  # one bolt at a time through the hole
        _put(env, i, top + [0, 0, 0.0005])
        _run(env, 1.2)
        assert env.parts[i].status == "vanished"
    for k, j in enumerate(range(3, 6)):
        _put(env, j, env.bucket_floor + [0.04 * (k - 1), 0, 0.03])
    infos = _run(env, 1.0)
    assert infos[-1]["outcome"] == "success"
    assert env._check_success()
    assert env.reward() == pytest.approx(1.0)


def test_cameras_render_the_scene():
    """Front + wrist cameras at 256^2; the scene geoms are visible (not in hidden group 0)."""
    e = suite.make(
        "SortBoltsNuts",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        seed=1,
    )
    try:
        obs = e.reset()
        front = obs["sortview_image"]
        assert front.shape == (256, 256, 3)
        assert obs["robot0_eye_in_hand_image"].shape == (256, 256, 3)
        # bucket A is blue: a clearly blue blob must be in the front view
        f = front.astype(int)
        blue = (f[..., 2] > f[..., 0] + 40) & (f[..., 2] > f[..., 1] + 20)
        assert blue.sum() > 200
    finally:
        e.close()


def test_tau_ext_free_motion_is_zero_in_env():
    """Sensors are sampled after each full physics step. At robosuite's _pre_action (between
    mj_step1 and mj_step2 with lite_physics) qacc belonged to the previous state and tau_ext
    showed ~50 N phantom spikes (V4)."""
    e = suite.make(
        "SortBoltsNuts",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        seed=0,
        torque_noise_std=0.0,
        torque_bias_std=0.0,
    )
    try:
        e.reset()
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(20):
            e.step(np.r_[rng.uniform(-1, 1, 6) * [1, 1, 0.3, 0.5, 0.5, 0.5], -1.0])
            worst = max(worst, float(np.abs(e.tau_ext_buf.last(50)).max()))
        assert worst < 1e-3
    finally:
        e.close()


def test_rack_presentation_spawns_bolts_upright():
    """bolt_presentation='rack' (opt-in): bolts stand head-up in the rack, flats facing +-y."""
    from fvb.envs.sort_bolts_nuts import SortTaskParams

    e = suite.make(
        "SortBoltsNuts",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        seed=3,
        task=SortTaskParams(bolt_presentation="rack"),
    )
    try:
        e.reset()
        holes = e.rack_holes
        for i, p in enumerate(e.parts):
            assert e._in_bin(i), p.body
            if p.kind == "bolt":
                pos, R = e.part_pose(i)
                assert e._in_rack(i)
                assert e.bolt_geometry(i)["tilt_deg"] < 2.0
                assert np.min(np.linalg.norm(holes[:, :2] - pos[:2], axis=1)) < 0.002
                assert abs(pos[2] - holes[0, 2]) < 0.002  # head resting on the plate
            else:
                assert not e._in_rack(i)
    finally:
        e.close()
