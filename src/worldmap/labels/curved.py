"""Curved area labels: the curve a name follows along its feature (a sea, a mountain range...), and its glyphs bent
along it. Bodies are in projected map units; labels are in poster pixels."""

from dataclasses import dataclass, field

import numpy as np
import shapely
from matplotlib.colors import to_rgb
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from worldmap.coloring import OCEAN
from worldmap.colorspace import RGB, fit_gamut, srgb_to_oklch
from worldmap.curves import arc_lengths, bend
from worldmap.frame import MapFrame
from worldmap.labels.geometry import Coverage, coverage, on_feature
from worldmap.labels.text import PX_PER_PT, bent, font, text_width

MAX_BEND_DEG = 40  # the most a label's curve may turn from one end to the other
MAX_TILT_DEG = 90  # labels follow their body's shape up to vertical (oceans too: the Atlantic's runs pole to pole)
CURVE_SAMPLES = 2000  # interior points sampled to find a body's long direction
RING_HOLE_SHARE = 0.25  # a body round a hole holding its middle, this share of its outline or more, is labelled along it
OCEAN_PT = 30  # oceans: letter-spaced, sized like seas but within these (the Arctic and Southern bands are narrow)
OCEAN_MIN_PT = 18
OCEAN_TRACKING_EM = 0.6  # extra space after each letter, in ems
SEA_MAX_PT = 14  # seas, lakes, straits: sized to fill SEA_FILL of their curve, within these
WATER_MIN_PT = 6.5
SEA_FILL = 0.8
CURVED_MIN_INSIDE = 0.35  # a curved label with less of its glyph boxes over its body than this is dropped
CURVED_HALO_PX = 2.4  # halo width (poster px); glyph boxes are padded by half of it
WHITE_HALO = (1, 1, 1, 0.55)  # the country labels': for names on land (rivers, landforms)
WATER_HALO = OCEAN  # the trim: the water colour itself
_ocean_hue = srgb_to_oklch(to_rgb(OCEAN))[2]
_fitted = fit_gamut(0.50, 0.08, _ocean_hue)
assert _fitted is not None
WATER_TEXT: RGB = _fitted[0]  # a deeper blue in the ocean's hue
_river_text = fit_gamut(0.42, 0.10, _ocean_hue)
assert _river_text is not None
RIVER_TEXT: RGB = _river_text[0]  # darker than WATER_TEXT: it sits on land, not water
_land_text = fit_gamut(0.40, 0.07, 55)
assert _land_text is not None
LAND_TEXT: RGB = _land_text[0]  # landforms: a dark warm brown, terrain rather than water or a country's hue
LAND_MIN_PT, LAND_MAX_PT = 6.5, 14.0  # sized to fill SEA_FILL of their curve, within these


@dataclass(frozen=True)
class Style:
    """How a kind's names look: a curved label's style is its kind's (water.STYLES, land.STYLE), never its size's."""
    italic: bool
    tracking_em: float  # extra space after each letter, in ems
    cap_centred: bool  # the cap height (else the x-height) is centred on the curve
    text: RGB
    halo: tuple[float, ...] | str
    min_pt: float  # sized to fill SEA_FILL of the curve within min_pt..max_pt; it shrinks no further than min_pt
    max_pt: float


@dataclass
class Body:
    text: str
    kind: str  # its --label kind
    rank: int  # Natural Earth scalerank: 0 = oceans, 1 = major seas, ... lower is more important
    geom: Polygon | LineString  # the piece the label goes on (a river's line), in map units
    full_size: float  # km2 of the whole body, or km of the whole river, however much the frame shows: its priority
    style: Style  # its kind's


def largest(geom: BaseGeometry) -> Polygon:
    return max(shapely.get_parts(geom), key=lambda g: g.area)


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


def ring_arc(pts: np.ndarray, centre: np.ndarray, n: int, bend_deg: float) -> np.ndarray:
    """An (n, 2) arc turning bend_deg round `centre`, below it so it reads left to right, at the median distance from
    it of the points `pts` under it: along a body that surrounds its own middle."""
    half = np.radians(bend_deg) / 2
    rel = pts - centre
    under = np.abs(np.arctan2(rel[:, 1], rel[:, 0]) + np.pi / 2) <= half
    radius = float(np.median(np.hypot(*rel[under if under.any() else slice(None)].T)))
    a = -np.pi / 2 + np.linspace(-half, half, n)
    return centre + radius * np.column_stack([np.cos(a), np.sin(a)])


def fit_curve(geom: Polygon, max_tilt_deg: float, n: int = 64, max_bend_deg: float = MAX_BEND_DEG) -> np.ndarray:
    """A smooth (n, 2) polyline along `geom`'s long direction through its middle: a quadratic in the frame of the
    principal axis, turning at most max_bend_deg, its chord at most max_tilt_deg from horizontal. It reads left to
    right, or bottom to top when within 1 degree of vertical, so text along it is never upside down. A body round a
    big hole holding its middle (the Southern Ocean in a polar crop) gets an arc along it instead (`ring_arc`)."""
    pts = interior_points(geom)
    mean = pts.mean(axis=0)
    outline = Polygon(geom.exterior).area
    if any(shapely.contains_xy(h := Polygon(r), *mean) and h.area >= RING_HOLE_SHARE * outline for r in geom.interiors):
        return ring_arc(pts, mean, n, max_bend_deg)
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

    def build(a: float) -> np.ndarray:
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

    def upright(curve: np.ndarray) -> bool:
        """Every step goes the way the text reads, so no glyph leans past vertical (upside down)."""
        steps = np.diff(curve, axis=0)
        chord = curve[-1] - curve[0]
        vertical = abs(abs(np.degrees(np.arctan2(chord[1], chord[0]))) - 90) <= 1
        return bool((steps[:, 1 if vertical else 0] >= 0).all())

    if upright(build(a)):
        return build(a)
    lo, hi = 0.0, 1.0  # a steep curve: flatten its bend until both ends still read forwards
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if upright(build(a * mid)) else (lo, mid)
    return build(a * lo)


@dataclass
class CurvedLabel:
    text: str
    kind: str  # its --label kind
    geom: BaseGeometry  # the body, in poster px
    anchor_x: float  # the curve's middle (px): where the label wants to be
    anchor_y: float
    size: float  # pt
    glyphs: Path  # the bent text, in px relative to the label's position
    parts: np.ndarray  # (k, 4) x0, y0, x1, y1 per non-space glyph, px relative to the label's position, halo-padded
    style: Style  # its body's
    full_size: float = 0.0  # its body's (Body.full_size)
    cover: Coverage | None = field(default=None, repr=False, compare=False)  # geom's, once needed (geometry.coverage)

    @property
    def min_size(self) -> float:
        """The smallest it may shrink to (pt): oceans stay bigger than any sea."""
        return self.style.min_pt

    @property
    def name(self) -> str:
        return self.text

    @property
    def width(self) -> float:
        return float(self.parts[:, 2].max() - self.parts[:, 0].min())

    @property
    def height(self) -> float:
        return float(self.parts[:, 3].max() - self.parts[:, 1].min())

    @property
    def out(self) -> bool:
        return False

    @property
    def curved(self) -> bool:
        return True

    def inside_share(self, x: float, y: float) -> float:
        """The share of the glyph boxes' area on the body when the label is at (x, y), as the solver has it
        (geometry.on_feature: a box is on it if its centre is)."""
        return on_feature(coverage(self), self.parts, x, y, False, 0)[0]


def extend(curve: np.ndarray, length: float) -> np.ndarray:
    """`curve` with straight continuations of `length` at both ends."""
    def step(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = b - a
        return b + d / np.hypot(*d) * length
    return np.vstack([step(curve[1], curve[0]), curve, step(curve[-2], curve[-1])])


def lay_out(body: Body, curve: np.ndarray, size: float, geom_px: BaseGeometry, rise: float | None = None) -> CurvedLabel:
    """`body`'s name at `size` pt in its kind's style, centred on `curve` (px), its x-height (or cap height) centred
    on it; or, given `rise` (px), its baseline that far to the left of the curve (above, reading left to right)."""
    style = body.style
    prop = font(size, style.italic)
    tracking = style.tracking_em * size
    text = body.text
    total = text_width(text, prop) + tracking * (len(text) - 1)
    middle = TextPath((0, 0), "H" if style.cap_centred else "x", prop=prop).get_extents().ymax / 2
    length = arc_lengths(curve)[-1]
    reach = max(0.0, total * PX_PER_PT - length) / 2 + size * PX_PER_PT  # extend the curve if the text is longer (+ overhang)
    long_curve = extend(curve, reach)
    centre = reach + length / 2  # distance along long_curve of the curve's middle
    h = 1.0  # px, for tangents
    anchor = bend(long_curve, np.array([centre]), np.zeros(1), h)[0]
    drop, lift = (middle, 0.0) if rise is None else (0.0, rise)
    laid = bent(text, prop, long_curve, centre, drop, lift, tracking, h)
    vertices = [points - anchor for points, _ in laid]
    pad = CURVED_HALO_PX / 2
    parts = [[*(v.min(axis=0) - pad), *(v.max(axis=0) + pad)] for v in vertices]
    glyphs = Path(np.vstack(vertices), np.concatenate([codes for _, codes in laid]))
    return CurvedLabel(body.text, body.kind, geom_px, float(anchor[0]), float(anchor[1]), size, glyphs, np.array(parts), style, body.full_size)


def area_label(body: Body, frame: MapFrame) -> CurvedLabel | None:
    """`body`'s name along its polygon's curve (fit_curve), sized to fill SEA_FILL of it within its style's range;
    None if under CURVED_MIN_INSIDE of its glyph boxes are over the body at its anchor (no room for the name)."""
    assert isinstance(body.geom, Polygon)
    style = body.style
    curve = (fit_curve(body.geom, MAX_TILT_DEG) - np.array([frame.xmin, frame.ymin])) * frame.px_per_unit
    width_1pt = text_width(body.text, font(1, style.italic)) * PX_PER_PT + style.tracking_em * (len(body.text) - 1) * PX_PER_PT
    size = float(np.clip(SEA_FILL * arc_lengths(curve)[-1] / width_1pt, style.min_pt, style.max_pt))
    label = lay_out(body, curve, size, frame.to_px(body.geom))
    return label if label.inside_share(label.anchor_x, label.anchor_y) >= CURVED_MIN_INSIDE else None
