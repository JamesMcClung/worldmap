"""Water labels (--label oceans, seas, lakes, water): which water bodies get a name, the curve each name follows,
and its glyphs laid out along it. Bodies are in projected map units; labels are in poster pixels."""

from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import numpy as np
import shapely
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath, TextToPath
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from worldmap.coloring import OCEAN
from worldmap.colorspace import RGB, fit_gamut, srgb_to_oklch
from worldmap.curves import arc_lengths, bend
from worldmap.data import load_marine, project
from worldmap.frame import DPI, MapFrame

WATER_KINDS = ("oceans", "seas", "lakes", "water")  # the --label values; water = all of them plus straits
Kind = Literal["oceans", "seas", "lakes", "straits"]
ALL_KINDS: set[Kind] = {"oceans", "seas", "lakes", "straits"}
SEA: Kind = "seas"
CLASSES: dict[str, Kind] = {"ocean": "oceans", "strait": "straits", "channel": "straits"}
CLASSES |= {c: SEA for c in ("sea", "gulf", "bay", "sound", "fjord", "lagoon", "inlet")}  # reefs, river mouths and the unnamed 'generic' areas are never labelled
MAX_BEND_DEG = 40  # the most a label's curve may turn from one end to the other
OCEAN_MAX_BEND_DEG = 10
OCEAN_MAX_TILT_DEG = 0  # oceans run level (along a parallel); seas may follow their shape up to vertical
SEA_MAX_TILT_DEG = 90
OCEAN_REACH = 3  # an ocean's curve follows its water within this many radii of its largest inscribed circle
CURVE_SAMPLES = 2000  # interior points sampled to find a body's long direction
OCEAN_PT = 30  # oceans: letter-spaced, sized like seas but within these (the Arctic and Southern bands are narrow)
OCEAN_MIN_PT = 18
OCEAN_TRACKING_EM = 0.6  # extra space after each letter, in ems
SEA_MAX_PT = 14  # seas, lakes, straits: sized to fill SEA_FILL of their curve, within these
WATER_MIN_PT = 6.5
SEA_FILL = 0.8
WATER_MIN_INSIDE = 0.35  # a water label with less of its glyph boxes over its body than this is dropped
WATER_HALO_PX = 2.4  # halo width (poster px); glyph boxes are padded by half of it
WATER_HALO = OCEAN  # the trim: the water colour itself
_ocean_hue = srgb_to_oklch(to_rgb(OCEAN))[2]
_fitted = fit_gamut(0.50, 0.08, _ocean_hue)
assert _fitted is not None
WATER_TEXT: RGB = _fitted[0]  # a deeper blue in the ocean's hue
PX_PER_PT = DPI / 72
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
    marine = marine[marine.name.notna() & ~marine.name.str.endswith(" River")]  # rivers come later, with river drawing
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


def fit_curve(geom: Polygon, max_tilt_deg: float, n: int = 64, max_bend_deg: float = MAX_BEND_DEG) -> np.ndarray:
    """A smooth (n, 2) polyline along `geom`'s long direction through its middle: a quadratic in the frame of the
    principal axis, turning at most max_bend_deg, its chord at most max_tilt_deg from horizontal. It reads left to
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
    if turn(a, b, s0, s1) > max_bend_deg:
        lo, hi = 0.0, 1.0  # shrink the bend until it turns max_bend_deg
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if turn(a * mid, b, s0, s1) <= max_bend_deg else (lo, mid)
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


@dataclass
class WaterLabel:
    text: str
    kind: Kind
    rank: int
    geom: BaseGeometry  # the body, in poster px
    anchor_x: float  # the curve's middle (px): where the label wants to be
    anchor_y: float
    size: float  # pt
    glyphs: Path  # the bent text, in px relative to the label's position
    parts: np.ndarray  # (k, 4) x0, y0, x1, y1 per non-space glyph, px relative to the label's position, halo-padded

    def scale(self, factor: float) -> None:
        """Shrink the label by `factor` about its position."""
        self.glyphs = Path(np.asarray(self.glyphs.vertices) * factor, self.glyphs.codes)
        self.parts = self.parts * factor
        self.size *= factor

    def inside_share(self, x: float, y: float) -> float:
        """The share of the glyph boxes' area over the body when the label is at (x, y)."""
        boxes = shapely.box(self.parts[:, 0] + x, self.parts[:, 1] + y, self.parts[:, 2] + x, self.parts[:, 3] + y)
        total = shapely.area(boxes).sum()
        return float(shapely.area(shapely.intersection(boxes, self.geom)).sum() / total) if total > 0 else 0.0


def font(kind: Kind, size: float) -> FontProperties:
    return FontProperties(size=size, style="normal" if kind == "oceans" else "italic")


def text_width(text: str, prop: FontProperties) -> float:
    """Advance width of `text` in points."""
    return TextToPath().get_text_width_height_descent(text, prop, ismath=False)[0] if text else 0.0


def extend(curve: np.ndarray, length: float) -> np.ndarray:
    """`curve` with straight continuations of `length` at both ends."""
    def step(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = b - a
        return b + d / np.hypot(*d) * length
    return np.vstack([step(curve[1], curve[0]), curve, step(curve[-2], curve[-1])])


def lay_out(body: WaterBody, curve: np.ndarray, size: float, geom_px: BaseGeometry) -> WaterLabel:
    """`body`'s name at `size` pt, centred on `curve` (px), its x-height (cap height for oceans) centred on it."""
    prop = font(body.kind, size)
    tracking = OCEAN_TRACKING_EM * size if body.kind == "oceans" else 0.0
    text = body.text
    total = text_width(text, prop) + tracking * (len(text) - 1)
    middle = TextPath((0, 0), "H" if body.kind == "oceans" else "x", prop=prop).get_extents().ymax / 2
    length = arc_lengths(curve)[-1]
    reach = max(0.0, total * PX_PER_PT - length) / 2 + size * PX_PER_PT  # extend the curve if the text is longer (+ overhang)
    long_curve = extend(curve, reach)
    centre = reach + length / 2  # distance along long_curve of the curve's middle
    h = 1.0  # px, for tangents
    anchor = bend(long_curve, np.array([centre]), np.zeros(1), h)[0]
    vertices, codes, parts = [], [], []
    for k, ch in enumerate(text):
        if ch.isspace():
            continue
        x = text_width(text[:k], prop) + tracking * k
        glyph = TextPath((x, 0), ch, prop=prop)
        if len(glyph.vertices) == 0:
            continue
        u = centre + (glyph.vertices[:, 0] - total / 2) * PX_PER_PT
        v = (glyph.vertices[:, 1] - middle) * PX_PER_PT
        bent = bend(long_curve, u, v, h) - anchor
        vertices.append(bent)
        codes.append(glyph.codes)
        pad = WATER_HALO_PX / 2
        parts.append([*(bent.min(axis=0) - pad), *(bent.max(axis=0) + pad)])
    glyphs = Path(np.vstack(vertices), np.concatenate(codes))
    return WaterLabel(body.text, body.kind, body.rank, geom_px, float(anchor[0]), float(anchor[1]), size, glyphs, np.array(parts))


def body_curve(body: WaterBody) -> np.ndarray:
    """The curve a body's name follows (map units). An ocean's is level, gently bent along its water within
    OCEAN_REACH radii of its largest inscribed circle, and centred on that circle: the middle of a whole ocean
    may be land or a narrow passage."""
    if body.kind != "oceans":
        return fit_curve(body.geom, SEA_MAX_TILT_DEG)
    circle = shapely.maximum_inscribed_circle(body.geom, np.sqrt(body.geom.area) / 1000)
    centre, radius = np.asarray(circle.coords[0]), circle.length
    area = largest(body.geom.intersection(shapely.Point(centre).buffer(OCEAN_REACH * radius)))
    curve = fit_curve(area, OCEAN_MAX_TILT_DEG, max_bend_deg=OCEAN_MAX_BEND_DEG)
    return curve + centre - curve[len(curve) // 2]


def make_water_labels(bodies: list[WaterBody], frame: MapFrame) -> list[WaterLabel]:
    """A curved label per body, in the same order, leaving out those with under WATER_MIN_INSIDE of their glyph
    boxes over their body to start with (most small bays: no room for their name)."""
    ppu = frame.px_per_unit
    labels = []
    for body in bodies:
        curve = (body_curve(body) - np.array([frame.xmin, frame.ymin])) * ppu
        width_1pt = text_width(body.text, font(body.kind, 1)) * PX_PER_PT
        if body.kind == "oceans":
            width_1pt += OCEAN_TRACKING_EM * (len(body.text) - 1) * PX_PER_PT
        low, high = (OCEAN_MIN_PT, OCEAN_PT) if body.kind == "oceans" else (WATER_MIN_PT, SEA_MAX_PT)
        size = float(np.clip(SEA_FILL * arc_lengths(curve)[-1] / width_1pt, low, high))
        label = lay_out(body, curve, size, frame.to_px(body.geom))
        if label.inside_share(label.anchor_x, label.anchor_y) >= WATER_MIN_INSIDE:
            labels.append(label)
    return labels
