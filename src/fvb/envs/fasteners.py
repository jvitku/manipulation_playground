"""ISO metric fasteners and the sorting-scene fixtures as MJCF elements (PLAN §13, V1).

Dimensions follow ISO 4017 (hex bolt), ISO 4032 (hex nut) and ISO 273 (clearance holes). Threads
are not modelled; their effect on friction is folded into the steel friction coefficient. Every
part is built from convex pieces (MuJoCo collides the convex hull of each mesh), so:

* bolt  = hex-prism head (convex mesh) + cylinder shank;
* nut   = hex ring from 12 convex wedges (each wedge spans a corner-to-flat 30 deg sector), so
          the bore is a real hole;
* hole  = block with a round through-hole from ``n_seg`` convex wedges. The hole polygon is
          *circumscribed* around the nominal circle, so the clearance is never tighter than
          ISO 273 (a 24-gon adds at most 0.07 mm on a 17 mm hole). A 45 deg entry chamfer is cut
          into the top of every wedge (a corner cut keeps the wedge convex).

Masses come from ``density`` (steel 7850 kg/m^3), i.e. MuJoCo computes them from the geometry.
Functions return ``xml.etree.ElementTree`` elements: ``*_assets`` go into ``<asset>``, the
``*_body`` / ``*_geoms`` elements into ``<worldbody>`` (the same pattern as ``envs.peg_in_hole``).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

STEEL_DENSITY = 7850.0  # kg/m^3
STEEL_FRICTION = (0.5, 0.005, 0.0001)  # dry steel on steel, sliding / torsional / rolling
# Stiff contact for steel. solref 0.002 s is the smallest MuJoCo honours at a 1 ms step (it clamps
# to 2*dt, Stage 0 M3). Contact parameters are mixed from BOTH geoms, so the surfaces the parts
# land on (table, bin floor, fixture) must use these too, or the softer side dominates: with a
# default-parameter floor a 5 cm drop penetrates 3.6 mm, with matched steel parameters 0.3 mm.
STEEL_SOLREF = "0.002 1"
STEEL_SOLIMP = "0.95 0.99 0.0005"

# ISO nominal sizes (m): width across flats s, bolt head height k (ISO 4017), nut height m
# (ISO 4032), clearance hole "fine" series (ISO 273).
ISO_METRIC = {
    "M10": {"d": 0.010, "s": 0.016, "k": 0.0064, "m": 0.0084, "hole_fine": 0.0105},
    "M12": {"d": 0.012, "s": 0.018, "k": 0.0075, "m": 0.0108, "hole_fine": 0.013},
    "M16": {"d": 0.016, "s": 0.024, "k": 0.010, "m": 0.0148, "hole_fine": 0.017},
}


@dataclass(frozen=True)
class BoltSpec:
    size: str = "M16"
    length: float = 0.050  # m, shank length under the head (ISO 4017 nominal length)

    @property
    def d(self) -> float:
        return ISO_METRIC[self.size]["d"]

    @property
    def s(self) -> float:
        return ISO_METRIC[self.size]["s"]

    @property
    def k(self) -> float:
        return ISO_METRIC[self.size]["k"]

    def volume(self) -> float:
        """Analytic steel volume: hex prism head + cylinder shank (m^3)."""
        return hex_area(self.s) * self.k + math.pi * (self.d / 2) ** 2 * self.length


@dataclass(frozen=True)
class NutSpec:
    size: str = "M16"

    @property
    def d(self) -> float:
        return ISO_METRIC[self.size]["d"]

    @property
    def s(self) -> float:
        return ISO_METRIC[self.size]["s"]

    @property
    def m(self) -> float:
        return ISO_METRIC[self.size]["m"]

    def volume(self) -> float:
        """Analytic volume with a round bore of the nominal diameter (m^3)."""
        return (hex_area(self.s) - math.pi * (self.d / 2) ** 2) * self.m


DEFAULT_BOLT = BoltSpec()
DEFAULT_NUT = NutSpec()


def hex_area(s: float) -> float:
    """Area of a regular hexagon with width across flats ``s``."""
    return math.sqrt(3) / 2 * s * s


def _fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in np.ravel(v))


def _prism_vertices(poly_xy: np.ndarray, z0: float, z1: float) -> np.ndarray:
    return np.vstack([np.column_stack([poly_xy, np.full(len(poly_xy), z)]) for z in (z0, z1)])


# ---------------------------------------------------------------------------------------------
# bolt
# ---------------------------------------------------------------------------------------------
def bolt_assets(name: str, spec: BoltSpec = DEFAULT_BOLT) -> list[ET.Element]:
    r_corner = spec.s / math.sqrt(3)  # circumradius of the hexagon
    ang = np.deg2rad(np.arange(6) * 60.0)
    hexagon = np.column_stack([r_corner * np.cos(ang), r_corner * np.sin(ang)])
    head = _prism_vertices(hexagon, 0.0, spec.k)
    return [ET.Element("mesh", name=f"{name}_head", vertex=_fmt(head))]


def bolt_body(
    name: str,
    pos=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    spec: BoltSpec = DEFAULT_BOLT,
    rgba: str = "0.62 0.64 0.67 1",
    free: bool = True,
) -> ET.Element:
    """Body origin at the underside of the head, on the axis; the shank points along -z."""
    b = ET.Element("body", name=name, pos=_fmt(pos), quat=_fmt(quat))
    if free:
        ET.SubElement(b, "freejoint", name=f"{name}_joint")
    common = {
        "density": f"{STEEL_DENSITY:g}",
        "friction": _fmt(STEEL_FRICTION),
        "rgba": rgba,
        "condim": "4",
        "solref": STEEL_SOLREF,
        "solimp": STEEL_SOLIMP,
    }
    ET.SubElement(b, "geom", name=f"{name}_head", type="mesh", mesh=f"{name}_head", **common)
    ET.SubElement(
        b,
        "geom",
        name=f"{name}_shank",
        type="cylinder",
        size=_fmt([spec.d / 2, spec.length / 2]),
        pos=_fmt([0, 0, -spec.length / 2]),
        **common,
    )
    return b


# ---------------------------------------------------------------------------------------------
# nut
# ---------------------------------------------------------------------------------------------
def nut_assets(name: str, spec: NutSpec = DEFAULT_NUT) -> list[ET.Element]:
    """12 convex wedges: each spans a hex corner (angle 60i) to a flat midpoint (60i +- 30)."""
    r_corner = spec.s / math.sqrt(3)
    r_flat = spec.s / 2
    r_in = spec.d / 2
    out = []
    for i in range(12):
        a_corner = math.radians(60.0 * ((i + 1) // 2))
        a_flat = math.radians(60.0 * (i // 2) + 30.0)
        a0, a1 = sorted([a_corner, a_flat])
        outer0 = r_corner if a0 == a_corner else r_flat
        outer1 = r_corner if a1 == a_corner else r_flat
        poly = np.array(
            [
                [outer0 * math.cos(a0), outer0 * math.sin(a0)],
                [outer1 * math.cos(a1), outer1 * math.sin(a1)],
                [r_in * math.cos(a1), r_in * math.sin(a1)],
                [r_in * math.cos(a0), r_in * math.sin(a0)],
            ]
        )
        out.append(
            ET.Element(
                "mesh",
                name=f"{name}_w{i}",
                vertex=_fmt(_prism_vertices(poly, -spec.m / 2, spec.m / 2)),
            )
        )
    return out


def nut_body(
    name: str,
    pos=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    spec: NutSpec = DEFAULT_NUT,
    rgba: str = "0.55 0.52 0.40 1",
    free: bool = True,
) -> ET.Element:
    """Body origin at the nut's centre; the bore axis is +z."""
    b = ET.Element("body", name=name, pos=_fmt(pos), quat=_fmt(quat))
    if free:
        ET.SubElement(b, "freejoint", name=f"{name}_joint")
    for i in range(12):
        ET.SubElement(
            b,
            "geom",
            name=f"{name}_w{i}",
            type="mesh",
            mesh=f"{name}_w{i}",
            density=f"{STEEL_DENSITY:g}",
            friction=_fmt(STEEL_FRICTION),
            rgba=rgba,
            condim="4",
            solref=STEEL_SOLREF,
            solimp=STEEL_SOLIMP,
        )
    return b


# ---------------------------------------------------------------------------------------------
# insertion fixture: block with a round clearance hole
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HoleSpec:
    diameter: float = ISO_METRIC["M16"]["hole_fine"]  # m, ISO 273 fine
    block: float = 0.100  # m, square block side
    height: float = 0.060  # m (deeper than the 50 mm shank, so the head seats on the top face)
    chamfer: float = 0.001  # m, 45 deg entry chamfer
    n_seg: int = 24

    @property
    def r_vertex(self) -> float:
        """Polygon vertex radius: circumscribed, so the flats sit at the nominal radius."""
        return (self.diameter / 2) / math.cos(math.pi / self.n_seg)


DEFAULT_HOLE = HoleSpec()


def _ray_to_square(a: float, half: float) -> float:
    return half / max(abs(math.cos(a)), abs(math.sin(a)))


def hole_assets(name: str, spec: HoleSpec = DEFAULT_HOLE) -> list[ET.Element]:
    out = []
    half, h, c = spec.block / 2, spec.height, spec.chamfer
    rv = spec.r_vertex
    for i in range(spec.n_seg):
        a0 = 2 * math.pi * i / spec.n_seg
        a1 = 2 * math.pi * (i + 1) / spec.n_seg
        pts = []
        for a in (a0, a1):
            ca, sa = math.cos(a), math.sin(a)
            ro = _ray_to_square(a, half)
            pts += [
                [rv * ca, rv * sa, 0.0],
                [ro * ca, ro * sa, 0.0],
                [ro * ca, ro * sa, h],
                [(rv + c) * ca, (rv + c) * sa, h],  # chamfer: wider at the top face
                [rv * ca, rv * sa, h - c],
            ]
        # the square corner lies between the two rays when a corner angle is inside [a0, a1]
        for k in range(4):
            ac = math.pi / 4 + k * math.pi / 2
            if a0 < ac < a1:
                x, y = half * np.sign(math.cos(ac)), half * np.sign(math.sin(ac))
                pts += [[x, y, 0.0], [x, y, h]]
        out.append(ET.Element("mesh", name=f"{name}_w{i}", vertex=_fmt(np.array(pts))))
    return out


def hole_geoms(parent: ET.Element, name: str, spec: HoleSpec = DEFAULT_HOLE) -> None:
    """Adds the fixture wedges to ``parent`` (a static body whose origin is the hole axis on the
    table surface) plus a site at the centre of the top face."""
    for i in range(spec.n_seg):
        ET.SubElement(
            parent,
            "geom",
            name=f"{name}_w{i}",
            type="mesh",
            mesh=f"{name}_w{i}",
            friction=_fmt(STEEL_FRICTION),
            rgba="0.35 0.38 0.42 1",
            condim="4",
            solref=STEEL_SOLREF,
            solimp=STEEL_SOLIMP,
        )
    ET.SubElement(parent, "site", name=f"{name}_top", pos=_fmt([0, 0, spec.height]), size="0.002")


# ---------------------------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------------------------
def bin_geoms(parent: ET.Element, name: str, size=(0.30, 0.22), wall_h=0.08, t=0.005) -> None:
    """Open-top bin: floor + 4 walls, origin at the floor centre on the table."""
    lx, ly = size[0] / 2, size[1] / 2
    rgba = "0.72 0.60 0.42 1"
    ET.SubElement(
        parent,
        "geom",
        name=f"{name}_floor",
        type="box",
        size=_fmt([lx, ly, t / 2]),
        pos=_fmt([0, 0, t / 2]),
        rgba=rgba,
    )
    for tag, sz, p in (
        ("px", [t / 2, ly, wall_h / 2], [lx - t / 2, 0, wall_h / 2]),
        ("nx", [t / 2, ly, wall_h / 2], [-lx + t / 2, 0, wall_h / 2]),
        ("py", [lx, t / 2, wall_h / 2], [0, ly - t / 2, wall_h / 2]),
        ("ny", [lx, t / 2, wall_h / 2], [0, -ly + t / 2, wall_h / 2]),
    ):
        ET.SubElement(
            parent, "geom", name=f"{name}_{tag}", type="box", size=_fmt(sz), pos=_fmt(p), rgba=rgba
        )


def bucket_geoms(
    parent: ET.Element,
    name: str,
    radius=0.10,
    height=0.12,
    t=0.004,
    n=16,
    floor: str = "cylinder",
) -> None:
    """Round bucket: floor + ``n`` wall boxes, origin at the floor centre on the table.

    ``floor="box"`` uses a square plate under the walls instead of a disc: mesh-vs-cylinder
    contacts occasionally threw a dropped steel nut out of the bucket (V4)."""
    rgba = "0.20 0.45 0.70 1"
    if floor == "box":
        size = [radius + t, radius + t, t / 2]
        ET.SubElement(
            parent,
            "geom",
            name=f"{name}_floor",
            type="box",
            size=_fmt(size),
            pos=_fmt([0, 0, t / 2]),
            rgba=rgba,
        )
    else:
        ET.SubElement(
            parent,
            "geom",
            name=f"{name}_floor",
            type="cylinder",
            size=_fmt([radius, t / 2]),
            pos=_fmt([0, 0, t / 2]),
            rgba=rgba,
        )
    seg = 2 * radius * math.tan(math.pi / n)
    for i in range(n):
        a = 2 * math.pi * i / n
        ET.SubElement(
            parent,
            "geom",
            name=f"{name}_wall{i}",
            type="box",
            size=_fmt([t / 2, seg / 2 + t / 2, height / 2]),
            pos=_fmt([radius * math.cos(a), radius * math.sin(a), height / 2]),
            euler=_fmt([0, 0, a]),
            rgba=rgba,
        )
