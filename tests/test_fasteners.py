"""PLAN §13 V1: ISO M16 bolt / nut / clearance-hole fixture - dimensions, masses, contacts."""

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from fvb.envs.fasteners import (
    STEEL_DENSITY,
    STEEL_SOLIMP,
    STEEL_SOLREF,
    BoltSpec,
    HoleSpec,
    NutSpec,
    bolt_assets,
    bolt_body,
    hole_assets,
    hole_geoms,
    nut_assets,
    nut_body,
)

TOL = 1e-4  # 0.1 mm


def _model(bodies, assets, static=None, dt=0.001):
    root = ET.Element("mujoco", model="fasteners")
    ET.SubElement(root, "option", timestep=f"{dt}", integrator="implicitfast", cone="elliptic")
    a = ET.SubElement(root, "asset")
    for e in assets:
        a.append(e)
    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(
        wb,
        "geom",
        name="floor",  # a steel-like surface too: contact parameters mix from both geoms
        type="plane",
        size="1 1 0.1",
        friction="0.5 0.005 0.0001",
        solref=STEEL_SOLREF,
        solimp=STEEL_SOLIMP,
    )
    if static is not None:
        wb.append(static)
    for b in bodies:
        wb.append(b)
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return m, mujoco.MjData(m)


def _xml_vertices(elems, name):
    """The generated vertices (MuJoCo re-frames compiled meshes into their inertia frame, so
    dimensions are checked on what we generate)."""
    e = next(e for e in elems if e.get("name") == name)
    return np.array(e.get("vertex").split(), float).reshape(-1, 3)


def _widths(xy):
    """Planar widths across flats (min over directions) and across corners (max)."""
    angs = np.linspace(0, np.pi, 3600, endpoint=False)
    w = [np.ptp(xy @ np.array([np.cos(a), np.sin(a)])) for a in angs]
    return min(w), max(w)


def test_bolt_dimensions_follow_iso_4017():
    spec = BoltSpec()
    v = _xml_vertices(bolt_assets("b", spec), "b_head")
    flats, corners = _widths(v[:, :2])
    assert abs(flats - 0.024) < TOL and abs(corners - 0.024 * 2 / math.sqrt(3)) < TOL
    assert abs(np.ptp(v[:, 2]) - 0.010) < TOL  # head height k
    m, _ = _model([bolt_body("b", spec=spec)], bolt_assets("b", spec))
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "b_shank")
    assert abs(m.geom_size[g][0] - 0.008) < TOL and abs(2 * m.geom_size[g][1] - 0.050) < TOL


def test_nut_dimensions_follow_iso_4032_and_the_bore_is_real():
    spec = NutSpec()
    a = nut_assets("n", spec)
    v = np.vstack([_xml_vertices(a, f"n_w{i}") for i in range(12)])
    flats, _ = _widths(v[:, :2])
    assert abs(flats - 0.024) < TOL  # across flats
    assert abs(np.ptp(v[:, 2]) - 0.0148) < TOL  # height m
    assert abs(np.hypot(v[:, 0], v[:, 1]).min() - 0.008) < TOL  # bore radius: nothing inside


@pytest.mark.parametrize("kind", ["bolt", "nut"])
def test_masses_match_steel_within_5_percent(kind):
    if kind == "bolt":
        spec = BoltSpec()
        m, _ = _model([bolt_body("p", spec=spec)], bolt_assets("p", spec))
    else:
        spec = NutSpec()
        m, _ = _model([nut_body("p", spec=spec)], nut_assets("p", spec))
    mass = m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "p")]
    expected = STEEL_DENSITY * spec.volume()
    assert abs(mass - expected) / expected < 0.05, (mass, expected)
    assert (0.110 < mass < 0.125) if kind == "bolt" else (0.032 < mass < 0.038)


def test_hole_is_never_tighter_than_iso_273():
    h = HoleSpec()
    flat = h.r_vertex * math.cos(math.pi / h.n_seg)
    assert abs(flat - 0.0085) < 1e-9  # flats exactly at the nominal radius
    assert h.r_vertex - 0.0085 < 0.0001  # vertices at most 0.1 mm wider


def _settle(m, d, seconds):
    mujoco.mj_forward(m, d)
    worst = 0.0
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if d.ncon:
            worst = min(worst, min(d.contact[i].dist for i in range(d.ncon)))
    return worst


def test_dropped_nut_settles_flat_without_penetration():
    spec = NutSpec()
    m, d = _model([nut_body("n", pos=(0, 0, 0.05), spec=spec)], nut_assets("n", spec))
    worst = _settle(m, d, 1.5)
    assert worst > -0.0005  # including the 1 m/s impact of a 5 cm drop
    assert np.linalg.norm(d.qvel) < 1e-3
    assert abs(d.qpos[2] - spec.m / 2) < 0.0005


def test_dropped_bolt_lying_flat_settles_without_penetration():
    q = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]  # shank horizontal
    m, d = _model([bolt_body("b", pos=(0, 0, 0.05), quat=q)], bolt_assets("b"))
    worst = _settle(m, d, 2.0)
    assert worst > -0.0005
    assert np.linalg.norm(d.qvel[:3]) < 5e-3  # it may still rock slightly on the hex head


@pytest.mark.parametrize("offset,seated", [(0.0, True), (0.0012, True), (0.003, False)])
def test_bolt_seats_in_the_clearance_hole_only_when_aligned(offset, seated):
    """Vertical bolt released above the fixture: centred or inside the chamfer capture range it
    seats (head on the top face). 3 mm off it tips into the hole and jams partway - the
    "jammed, not seated" case the env must not count as a success."""
    h = HoleSpec()
    fx = ET.Element("body", name="fixture", pos="0.2 0 0")
    hole_geoms(fx, "hole", h)
    top = h.height
    b = bolt_body("b", pos=(0.2 + offset, 0, top + 0.055))
    m, d = _model([b], bolt_assets("b") + hole_assets("hole", h), static=fx)
    _settle(m, d, 1.5)
    head_z = d.qpos[2]  # body origin = underside of the head
    if seated:
        assert abs(head_z - top) < 0.001  # head on the top face
    else:
        assert head_z > top + 0.002  # not seated (jammed tilted or on the rim)


def test_mixed_parts_dropped_into_the_bin_settle_inside_without_penetration():
    """V1 gate: 3 bolts + 3 nuts dropped into the bin come to rest inside it without tunnelling
    and without resting penetration > 0.5 mm."""

    for seed in range(50):  # rejection-sample spawn poses until no two parts overlap at t=0
        m, d = _pile(np.random.default_rng(seed))
        mujoco.mj_forward(m, d)
        if all(d.contact[i].dist >= 0 for i in range(d.ncon)):
            break
    else:
        pytest.fail("no overlap-free spawn in 50 tries")
    worst = _settle(m, d, 2.5)
    # drops here are up to ~25 cm (~2 m/s): the transient must not tunnel (thinnest wall: the
    # nut's 4 mm ring, the 5 mm bin floor); at rest nothing may sink in by more than 0.5 mm.
    # (<= 5 cm drops, the env's spawn heights, stay under 0.5 mm even at impact: tests above.)
    assert worst > -0.002
    mujoco.mj_forward(m, d)
    assert min((d.contact[i].dist for i in range(d.ncon)), default=0.0) > -0.0005
    for i in range(3):
        for name in (f"bolt{i}", f"nut{i}"):
            x, y, z = d.body(name).xpos
            assert abs(x) < 0.15 and abs(y) < 0.11 and 0.0 < z < 0.08, (name, x, y, z)
    x, y, z = d.body("nut_b").xpos
    assert np.hypot(x - 0.4, y) < 0.10 and z < 0.12


def _pile(rng):
    from fvb.envs.fasteners import bin_geoms, bucket_geoms

    static = ET.Element("body", name="stat", pos="0 0 0")
    b = ET.SubElement(static, "body", name="bin", pos="0 0 0")
    bin_geoms(b, "bin")
    k = ET.SubElement(static, "body", name="bucket", pos="0.4 0 0")
    bucket_geoms(k, "bucket")
    for g in static.iter("geom"):  # the containers are stiff surfaces too
        g.set("solref", STEEL_SOLREF)
        g.set("solimp", STEEL_SOLIMP)
    bodies, assets = [], []
    for i in range(3):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        p = (rng.uniform(-0.08, 0.08), rng.uniform(-0.05, 0.05), 0.10 + 0.04 * i)
        bodies.append(bolt_body(f"bolt{i}", pos=p, quat=q))
        assets += bolt_assets(f"bolt{i}")
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        p = (rng.uniform(-0.08, 0.08), rng.uniform(-0.05, 0.05), 0.12 + 0.04 * i)
        bodies.append(nut_body(f"nut{i}", pos=p, quat=q))
        assets += nut_assets(f"nut{i}")
    bodies.append(nut_body("nut_b", pos=(0.4, 0.0, 0.15)))  # one straight into the bucket
    assets += nut_assets("nut_b")
    return _model(bodies, assets, static=static)
