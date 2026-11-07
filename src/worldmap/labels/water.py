"""Water labels (--label oceans, seas, lakes, water, rivers): which water bodies get a name, the curve each name
follows, and its glyphs laid out along it. Bodies are in projected map units; labels are in poster pixels."""

from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import numpy as np
import shapely
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath, TextToPath
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from worldmap.coloring import OCEAN
from worldmap.colorspace import RGB, fit_gamut, srgb_to_oklch
from worldmap.curves import arc_lengths, bend
from worldmap.data import load_marine, load_rivers, project
from worldmap.frame import DPI, MapFrame

WATER_KINDS = ("oceans", "seas", "lakes", "water", "rivers")  # the --label values; water = the first three plus straits
Kind = Literal["oceans", "seas", "lakes", "straits", "rivers"]
ALL_KINDS: set[Kind] = {"oceans", "seas", "lakes", "straits"}  # what water means: rivers only when asked for
SEA: Kind = "seas"
CLASSES: dict[str, Kind] = {"ocean": "oceans", "strait": "straits", "channel": "straits"}
CLASSES |= {c: SEA for c in ("sea", "gulf", "bay", "sound", "fjord", "lagoon", "inlet")}  # reefs, river mouths and the unnamed 'generic' areas are never labelled
MAX_BEND_DEG = 40  # the most a label's curve may turn from one end to the other
MAX_TILT_DEG = 90  # labels follow their body's shape up to vertical (oceans too: the Atlantic's runs pole to pole)
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
RIVER_LABEL_MAX_RANK = 6  # rivers of Natural Earth scalerank up to this get a name
RIVER_MAX_PT, RIVER_MIN_PT = 10, 7  # river names: italic, this size at scalerank <= 1 down to RIVER_LABEL_MAX_RANK
RIVER_GAP_EM = 0.3  # baseline height over the river
RIVER_CORRIDOR_EM = 1.6  # a river name's glyph centres should be within this of its river
RIVER_JOIN_M = 30_000  # river pieces whose ends are this close are joined (lakes and reservoirs break them up)
RIVER_REPEAT_M = 1_500_000  # same-named rivers further apart than this are named separately (Colorado, Red...)
RIVER_TILT_COST = 0.4  # choosing a river's stretch: steepness (0..1 from level to vertical) against wiggliness...
RIVER_MIDDLE_COST = 0.3  # ...and distance from the river's middle (0..0.5 of its length)
RIVER_NAMES = {"Grande": "Rio Grande", "Syr Dar": "Syr Darya"}  # Natural Earth name_en (after dropping " River") -> label
NOT_RIVERS = (" branch", " Branch", " Canal", " Protoka", " Csatorna")  # name endings of delta channels and canals
_river_text = fit_gamut(0.42, 0.10, _ocean_hue)
assert _river_text is not None
RIVER_TEXT: RGB = _river_text[0]  # darker than WATER_TEXT: it sits on land, not water
RIVER_HALO = (1, 1, 1, 0.55)  # like the country labels'
PX_PER_PT = DPI / 72
OCEAN_TEXT = {"Atlantic": "ATLANTIC OCEAN", "Pacific": "PACIFIC OCEAN", "Indian": "INDIAN OCEAN", "Arctic": "ARCTIC OCEAN", "Southern": "SOUTHERN OCEAN"}


def kinds_from_labels(label: set[str]) -> set[Kind]:
    """The water kinds asked for in --label (countries and territories are not water)."""
    kinds = set(ALL_KINDS) if "water" in label else {k for k in ALL_KINDS if k in label}
    if "rivers" in label:
        kinds.add("rivers")
    return kinds


def classify(featurecla: str) -> Kind | None:
    """The label kind of a Natural Earth marine feature class, or None if it never gets a label."""
    return CLASSES.get(featurecla)


@dataclass
class WaterBody:
    text: str
    kind: Kind
    rank: int  # Natural Earth scalerank: 0 = oceans, 1 = major seas, ... lower is more important
    geom: Polygon | LineString  # the piece the label goes on (a river's line), in map units


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
    if "rivers" in kinds:
        bodies += load_river_bodies()
    bodies.sort(key=lambda b: (b.kind != "oceans", b.rank, b.kind == "rivers", -(b.geom.length if b.kind == "rivers" else b.geom.area)))
    seen: set[str] = set()
    unique = []
    for b in bodies:  # oceans and rivers may share a name (the two Pacifics, far-apart rivers); others once, marine before lakes
        if b.kind in ("oceans", "rivers") or b.text not in seen:
            unique.append(b)
            seen.add(b.text)
    return unique


def river_name(name_en: object) -> str | None:
    """A river's label from Natural Earth's English name, or None for unnamed lines, delta channels and canals."""
    if not isinstance(name_en, str) or name_en.endswith(NOT_RIVERS):
        return None
    name = name_en.removesuffix(" River")
    return RIVER_NAMES.get(name, name)


def longest_paths(lines: list[LineString], join: float) -> list[LineString]:
    """A long path through each network of `lines`, longest first. Lines meet where their ends are within `join`
    (the gap is bridged straight). The path is the longest through the network's maximum spanning tree (exact
    when the network has no loops; delta branches and braids make loops)."""
    ends = shapely.points([c for line in lines for c in (line.coords[0], line.coords[-1])])
    node = list(range(len(ends)))  # union-find: first over line ends, then over the tree's nodes

    def root(i: int) -> int:
        while node[i] != i:
            i = node[i]
        return i

    for a, b in shapely.STRtree(ends).query(ends, predicate="dwithin", distance=join).T:
        node[root(int(a))] = root(int(b))
    ends_of = [(root(2 * k), root(2 * k + 1)) for k in range(len(lines))]
    node = list(range(len(ends)))
    tree: dict[int, list[tuple[int, int, int]]] = {}  # node -> (line, other node, direction)
    for k in sorted(range(len(lines)), key=lambda k: -lines[k].length):  # Kruskal, longest lines first
        u, v = ends_of[k]
        if root(u) != root(v):
            node[root(u)] = root(v)
            tree.setdefault(u, []).append((k, v, 1))
            tree.setdefault(v, []).append((k, u, -1))

    def farthest(start: int) -> tuple[int, list[tuple[int, int]]]:
        """The tree node farthest from `start` and the (line, direction) steps to it."""
        best, best_d, best_steps = start, 0.0, []
        todo: list[tuple[int, int, float, list[tuple[int, int]]]] = [(start, -1, 0.0, [])]
        while todo:
            at, came, d, steps = todo.pop()
            if d > best_d:
                best, best_d, best_steps = at, d, steps
            todo += [(o, k, d + lines[k].length, steps + [(k, direction)]) for k, o, direction in tree.get(at, []) if k != came]
        return best, best_steps

    paths, done = [], set()
    for k in range(len(lines)):
        u = ends_of[k][0]
        if u in done:
            continue
        a, _ = farthest(u)
        b, steps = farthest(a)
        component, todo = set(), [u]
        while todo:
            n = todo.pop()
            if n not in component:
                component.add(n)
                todo += [o for _, o, _ in tree.get(n, [])]
        done |= component
        if not steps:  # a loop on its own (both ends joined): not in the tree
            steps = [(k, 1)]
        paths.append(LineString(np.vstack([np.asarray(lines[i].coords)[::d] for i, d in steps])))
    return sorted(paths, key=lambda g: -g.length)


def load_river_bodies() -> list[WaterBody]:
    """One body per river name of scalerank <= RIVER_LABEL_MAX_RANK: the longest path through its features
    (longest_paths), plus any other path at least RIVER_REPEAT_M from those kept (two rivers with one name). Rank =
    the path's most important feature."""
    rivers = project(load_rivers())
    rivers["text"] = rivers.name_en.map(river_name)
    bodies = []
    for text, group in rivers[rivers.text.notna()].groupby("text"):
        if group.scalerank.min() > RIVER_LABEL_MAX_RANK:
            continue
        parts = longest_paths(list(shapely.get_parts(shapely.line_merge(shapely.union_all(group.geometry.values)))), RIVER_JOIN_M)
        kept: list[LineString] = []
        for part in parts:
            if all(part.distance(k) >= RIVER_REPEAT_M for k in kept):
                kept.append(part)
        for part in kept:
            rank = int(group.scalerank[shapely.dwithin(group.geometry.values, part, 1.0)].min())
            if rank <= RIVER_LABEL_MAX_RANK:
                bodies.append(WaterBody(str(text), "rivers", rank, part))
    return bodies


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

    @property
    def min_size(self) -> float:
        """The smallest it may shrink to (pt): oceans stay bigger than any sea."""
        return OCEAN_MIN_PT if self.kind == "oceans" else WATER_MIN_PT

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


def font(size: float, italic: bool = False) -> FontProperties:
    return FontProperties(size=size, style="italic" if italic else "normal")


def text_width(text: str, prop: FontProperties) -> float:
    """Advance width of `text` in points."""
    return TextToPath().get_text_width_height_descent(text, prop, ismath=False)[0] if text else 0.0


def extend(curve: np.ndarray, length: float) -> np.ndarray:
    """`curve` with straight continuations of `length` at both ends."""
    def step(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = b - a
        return b + d / np.hypot(*d) * length
    return np.vstack([step(curve[1], curve[0]), curve, step(curve[-2], curve[-1])])


def lay_out(body: WaterBody, curve: np.ndarray, size: float, geom_px: BaseGeometry, rise: float | None = None) -> WaterLabel:
    """`body`'s name at `size` pt, centred on `curve` (px), its x-height (cap height for oceans) centred on it; or,
    given `rise` (px), its baseline that far to the left of the curve (above, reading left to right). Rivers in italic."""
    prop = font(size, body.kind == "rivers")
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
        v = (glyph.vertices[:, 1] - middle) * PX_PER_PT if rise is None else glyph.vertices[:, 1] * PX_PER_PT + rise
        bent = bend(long_curve, u, v, h) - anchor
        vertices.append(bent)
        codes.append(glyph.codes)
        pad = WATER_HALO_PX / 2
        parts.append([*(bent.min(axis=0) - pad), *(bent.max(axis=0) + pad)])
    glyphs = Path(np.vstack(vertices), np.concatenate(codes))
    return WaterLabel(body.text, body.kind, body.rank, geom_px, float(anchor[0]), float(anchor[1]), size, glyphs, np.array(parts))


def body_curve(body: WaterBody) -> np.ndarray:
    """The curve a body's name follows (map units): along its long direction, so the Atlantic's runs pole to pole."""
    return fit_curve(body.geom, MAX_TILT_DEG)


def river_size(rank: int) -> float:
    """A river name's size (pt) from its scalerank."""
    t = min(max(rank - 1, 0), RIVER_LABEL_MAX_RANK - 1) / (RIVER_LABEL_MAX_RANK - 1)
    return RIVER_MAX_PT + t * (RIVER_MIN_PT - RIVER_MAX_PT)


def along(pts: np.ndarray, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The points at distances `u` along the polyline `pts` (cumulative lengths `s`)."""
    return np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], axis=-1)


def best_stretch(pts: np.ndarray, length: float) -> np.ndarray:
    """The stretch of `length` along the polyline `pts` that is straightest (arc / chord), most level and nearest
    the middle, as (64, 2) points."""
    s = arc_lengths(pts)
    total = s[-1]
    centres = np.arange(length / 2, total - length / 2 + 1e-9, length / 8) if total > length else np.array([total / 2])
    best, best_score = None, np.inf
    for c in centres:
        w = along(pts, s, np.linspace(c - length / 2, c + length / 2, 64))
        dx, dy = w[-1] - w[0]
        chord = np.hypot(dx, dy)
        if chord == 0:
            continue
        tilt = abs((np.degrees(np.arctan2(dy, dx)) + 90) % 180 - 90) / 90
        score = arc_lengths(w)[-1] / chord - 1 + RIVER_TILT_COST * tilt + RIVER_MIDDLE_COST * abs(c - total / 2) / total
        if score < best_score:
            best, best_score = w, score
    return best if best is not None else along(pts, s, np.linspace(0, total, 64))


def height_above(points: np.ndarray, curve: np.ndarray) -> float:
    """How far the highest of `points` lies to the left of `curve` (0 if none does), counting only those beside it."""
    ls = LineString(curve)
    u = shapely.line_locate_point(ls, shapely.points(points))
    beside = (u > 0) & (u < ls.length)
    if not beside.any():
        return 0.0
    base = bend(curve, u[beside], np.zeros(beside.sum()), 1.0)
    normal = bend(curve, u[beside], np.ones(beside.sum()), 1.0) - base
    return max(0.0, float(((points[beside] - base) * normal).sum(axis=1).max()))


def river_label(body: WaterBody, frame: MapFrame) -> WaterLabel | None:
    """A river's name just above the best stretch of it (best_stretch) for a name its size plus half an em each side,
    clear of its wiggles there; None if the river is shorter than that."""
    size = river_size(body.rank)
    em = size * PX_PER_PT
    line = largest_line(frame.to_px(body.geom))
    pts = np.asarray(line.coords)
    need = text_width(body.text, font(size, True)) * PX_PER_PT + em
    if line.length < need:
        return None
    stretch = best_stretch(pts, need)
    curve = fit_curve(LineString(stretch).buffer(1.0), MAX_TILT_DEG)
    rise = RIVER_GAP_EM * em + height_above(stretch, curve)
    label = lay_out(body, curve, size, line.buffer(RIVER_CORRIDOR_EM * em), rise)
    return label if label.inside_share(label.anchor_x, label.anchor_y) >= WATER_MIN_INSIDE else None


def largest_line(geom: BaseGeometry) -> LineString:
    return max(shapely.get_parts(geom), key=lambda g: g.length)


def make_water_labels(bodies: list[WaterBody], frame: MapFrame) -> list[WaterLabel]:
    """A curved label per body, in the same order, leaving out those with under WATER_MIN_INSIDE of their glyph
    boxes over their body to start with (most small bays: no room for their name), and rivers too short for theirs."""
    ppu = frame.px_per_unit
    labels = []
    for body in bodies:
        if body.kind == "rivers":
            river = river_label(body, frame)
            if river is not None:
                labels.append(river)
            continue
        assert isinstance(body.geom, Polygon)
        curve = (body_curve(body) - np.array([frame.xmin, frame.ymin])) * ppu
        width_1pt = text_width(body.text, font(1)) * PX_PER_PT
        if body.kind == "oceans":
            width_1pt += OCEAN_TRACKING_EM * (len(body.text) - 1) * PX_PER_PT
        low, high = (OCEAN_MIN_PT, OCEAN_PT) if body.kind == "oceans" else (WATER_MIN_PT, SEA_MAX_PT)
        size = float(np.clip(SEA_FILL * arc_lengths(curve)[-1] / width_1pt, low, high))
        label = lay_out(body, curve, size, frame.to_px(body.geom))
        if label.inside_share(label.anchor_x, label.anchor_y) >= WATER_MIN_INSIDE:
            labels.append(label)
    return labels
