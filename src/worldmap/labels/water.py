"""Water labels (--label oceans, seas, lakes, water): which water bodies get a name, the curve each name follows,
and its glyphs laid out along it. Bodies are in projected map units; labels are in poster pixels."""

from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from worldmap.data import load_marine, project

WATER_KINDS = ("oceans", "seas", "lakes", "water")  # the --label values; water = all of them plus straits
Kind = Literal["oceans", "seas", "lakes", "straits"]
ALL_KINDS: set[Kind] = {"oceans", "seas", "lakes", "straits"}
SEA: Kind = "seas"
CLASSES: dict[str, Kind] = {"ocean": "oceans", "strait": "straits", "channel": "straits"}
CLASSES |= {c: SEA for c in ("sea", "gulf", "bay", "sound", "fjord", "lagoon", "inlet")}  # reefs, river mouths and the unnamed 'generic' areas are never labelled
MAX_BEND_DEG = 40  # the most a label's curve may turn from one end to the other
OCEAN_MAX_TILT_DEG = 20  # oceans run nearly level; seas may follow their shape up to vertical
SEA_MAX_TILT_DEG = 90
CURVE_SAMPLES = 2000  # interior points sampled to find a body's long direction
OCEAN_TEXT = {"Atlantic": "ATLANTIC OCEAN", "Pacific": "PACIFIC OCEAN", "Indian": "INDIAN OCEAN", "Arctic": "ARCTIC OCEAN", "Southern": "SOUTHERN OCEAN"}


def kinds_from_labels(label: set[str]) -> set[Kind]:
    """The water kinds asked for in --label (countries and territories are not water)."""
    if "water" in label:
        return set(ALL_KINDS)
    return {k for k in ALL_KINDS if k in label}


def classify(featurecla: str) -> Kind | None:
    """The label kind of a Natural Earth marine feature class, or None if it never gets a label."""
    return CLASSES.get(featurecla)


@dataclass
class WaterBody:
    text: str
    kind: Kind
    rank: int  # Natural Earth scalerank: 0 = oceans, 1 = major seas, ... lower is more important
    geom: Polygon  # the piece the label goes on, in map units


def largest(geom: BaseGeometry) -> Polygon:
    return max(shapely.get_parts(geom), key=lambda g: g.area)


def ocean_word(name: str) -> str:
    return next(w for w in OCEAN_TEXT if w.lower() in name.lower())


def load_water_bodies(kinds: set[Kind], lakes_projected: gpd.GeoDataFrame) -> list[WaterBody]:
    """The water bodies of `kinds`, most important first: oceans, then by scalerank, then by area. North and South
    Atlantic are one ATLANTIC; the Pacific, cut by the antimeridian, is labelled twice (left and right of x = 0).
    Every other body is labelled once, on its largest polygon."""
    marine = project(load_marine())
    marine = marine[marine.name.notna()]
    marine["geometry"] = shapely.make_valid(marine.geometry.values)  # some are self-intersecting rough areas
    marine = marine[[classify(c) in kinds for c in marine.featurecla]]
    bodies = []
    oceans = marine[marine.featurecla == "ocean"]
    for word, group in oceans.groupby(oceans.name.map(ocean_word)):
        whole = shapely.union_all(group.geometry.values)
        if word == "Pacific":
            x0, y0, x1, y1 = whole.bounds
            halves = [whole.intersection(shapely.box(x0, y0, 0, y1)), whole.intersection(shapely.box(0, y0, x1, y1))]
        else:
            halves = [whole]
        bodies += [WaterBody(OCEAN_TEXT[word], "oceans", 0, largest(h)) for h in halves]
    for row in marine[marine.featurecla != "ocean"].itertuples():
        kind = classify(row.featurecla)
        assert kind is not None
        bodies.append(WaterBody(row.name, kind, int(row.scalerank), largest(row.geometry)))
    if "lakes" in kinds:
        bodies += [WaterBody(r.name, "lakes", int(r.scalerank), largest(r.geometry)) for r in lakes_projected.itertuples() if isinstance(r.name, str)]
    bodies.sort(key=lambda b: (b.kind != "oceans", b.rank, -b.geom.area))
    seen: set[str] = set()
    unique = []
    for b in bodies:  # oceans are named alike on purpose (the two Pacifics); others once, marine before lakes
        if b.kind == "oceans" or b.text not in seen:
            unique.append(b)
            seen.add(b.text)
    return unique


def interior_points(geom: Polygon) -> np.ndarray:
    """About CURVE_SAMPLES grid points inside `geom` (a denser grid for thin bodies), shape (k, 2)."""
    x0, y0, x1, y1 = geom.bounds
    for density in (1, 4, 16, 64):
        step = np.sqrt((x1 - x0) * (y1 - y0) / (CURVE_SAMPLES * density))
        gx, gy = np.meshgrid(np.arange(x0 + step / 2, x1, step), np.arange(y0 + step / 2, y1, step))
        xs, ys = gx.ravel(), gy.ravel()
        inside = shapely.contains_xy(geom, xs, ys)
        if inside.sum() >= CURVE_SAMPLES / 4:
            break
    if inside.sum() < 3:
        return np.asarray(geom.exterior.coords)
    return np.stack([xs[inside], ys[inside]], axis=-1)


def turn(a: float, b: float, s0: float, s1: float) -> float:
    """How far (degrees) the tangent of t = a s^2 + b s turns from s0 to s1."""
    return float(np.degrees(abs(np.arctan(2 * a * s1 + b) - np.arctan(2 * a * s0 + b))))


def rotate(points: np.ndarray, centre: np.ndarray, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return centre + (points - centre) @ np.array([[c, s], [-s, c]])


def fit_curve(geom: Polygon, max_tilt_deg: float, n: int = 64) -> np.ndarray:
    """A smooth (n, 2) polyline along `geom`'s long direction through its middle: a quadratic in the frame of the
    principal axis, turning at most MAX_BEND_DEG, its chord at most max_tilt_deg from horizontal. It reads left to
    right, or bottom to top when within 1 degree of vertical, so text along it is never upside down."""
    pts = interior_points(geom)
    mean = pts.mean(axis=0)
    eigval, eigvec = np.linalg.eigh(np.cov((pts - mean).T))
    axis_x, axis_y = eigvec[:, np.argmax(eigval)]
    angle = np.clip((np.degrees(np.arctan2(axis_y, axis_x)) + 90) % 180 - 90, -max_tilt_deg, max_tilt_deg)
    d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    normal = np.array([-d[1], d[0]])
    s, t = (pts - mean) @ d, (pts - mean) @ normal
    s0, s1 = np.percentile(s, [5, 95])
    a, b, c = np.polyfit(s, t, 2)
    if turn(a, b, s0, s1) > MAX_BEND_DEG:
        lo, hi = 0.0, 1.0  # shrink the bend until it turns MAX_BEND_DEG
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if turn(a * mid, b, s0, s1) <= MAX_BEND_DEG else (lo, mid)
        a *= lo
    ss = np.linspace(s0, s1, n)
    curve = mean + ss[:, None] * d + (a * ss**2 + b * ss + c)[:, None] * normal
    chord = curve[-1] - curve[0]
    tilt = (np.degrees(np.arctan2(chord[1], chord[0])) + 90) % 180 - 90
    if abs(tilt) > max_tilt_deg:  # the linear term tilted it past the limit: turn it back about its middle
        curve = rotate(curve, curve[n // 2], np.sign(tilt) * max_tilt_deg - tilt)
        chord = curve[-1] - curve[0]
    vertical = abs(abs(np.degrees(np.arctan2(chord[1], chord[0]))) - 90) <= 1
    if (chord[1] < 0) if vertical else (chord[0] < 0):
        curve = curve[::-1]
    return curve
