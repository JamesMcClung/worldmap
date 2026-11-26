"""Water labels (--label oceans, seas, lakes, water, rivers): which water bodies get a name, and where along a
river it goes (laid out by curved.py). Bodies are in projected map units; labels are in poster pixels."""

import functools
from typing import Literal

import geopandas as gpd
import numpy as np
import shapely
from matplotlib.colors import to_rgb
from pyproj import Geod, Transformer
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from worldmap.coloring import OCEAN
from worldmap.colorspace import RGB, fit_gamut, srgb_to_oklch
from worldmap.curves import along, arc_lengths, bend
from worldmap.data import EQUAL_AREA_CRS, load_marine, load_rivers
from worldmap.disjoint import DisjointSets
from worldmap.frame import MapFrame
from worldmap.labels.curved import CURVED_MIN_INSIDE, MAX_TILT_DEG, WHITE_HALO, Body, CurvedLabel, Style, area_label, fit_curve, largest, lay_out
from worldmap.labels.priority import UNLABELED, Unplaced
from worldmap.labels.text import PX_PER_PT, font, text_width

Kind = Literal["oceans", "seas", "lakes", "straits", "rivers"]
WATER_KINDS: tuple[Kind, ...] = ("oceans", "seas", "lakes", "straits", "rivers")  # their --label values (and --label water's order)
SEA: Kind = "seas"
CLASSES: dict[str, Kind] = {"ocean": "oceans", "strait": "straits", "channel": "straits"}
CLASSES |= {c: SEA for c in ("sea", "gulf", "bay", "sound", "fjord", "lagoon", "inlet")}  # reefs, river mouths and the unnamed 'generic' areas are never labelled
OCEAN_PT = 30  # oceans: letter-spaced, sized like seas but within these (the Arctic and Southern bands are narrow)
OCEAN_MIN_PT = 18
OCEAN_TRACKING_EM = 0.6  # extra space after each letter, in ems
SEA_MAX_PT = 14  # seas, lakes, straits: sized to fill curved.SEA_FILL of their curve, within these
WATER_MIN_PT = 6.5
WATER_HALO = OCEAN  # the trim: the water colour itself
_ocean_hue = srgb_to_oklch(to_rgb(OCEAN))[2]
_fitted = fit_gamut(0.50, 0.08, _ocean_hue)
assert _fitted is not None
WATER_TEXT: RGB = _fitted[0]  # a deeper blue in the ocean's hue
_river_text = fit_gamut(0.42, 0.10, _ocean_hue)
assert _river_text is not None
RIVER_TEXT: RGB = _river_text[0]  # darker than WATER_TEXT: it sits on land, not water
_SEA_STYLE = Style(False, 0.0, False, WATER_TEXT, WATER_HALO, WATER_MIN_PT, SEA_MAX_PT)
STYLES: dict[Kind, Style] = {
    "oceans": Style(False, OCEAN_TRACKING_EM, True, WATER_TEXT, WATER_HALO, OCEAN_MIN_PT, OCEAN_PT),
    "seas": _SEA_STYLE, "lakes": _SEA_STYLE, "straits": _SEA_STYLE,
    "rivers": Style(True, 0.0, False, RIVER_TEXT, WHITE_HALO, WATER_MIN_PT, SEA_MAX_PT),  # sized by river_size, not max_pt
}
RIVER_WIDTH_MAX, RIVER_WIDTH_MIN = 1.6, 0.3  # line widths (pt) at scalerank 0 and RIVER_THIN_RANK
RIVER_THIN_RANK = 8  # rivers of this scalerank and beyond are drawn thinnest
RIVER_MAX_PT, RIVER_MIN_PT = 8.5, 4.2  # river names: italic, linear in the river's line width, these at its widest and thinnest
RIVER_GAP_EM = 0.3  # baseline height over the river
RIVER_CORRIDOR_EM = 1.0  # a river name's glyph centres should be within this of its river (more: it slid off where it was bent to fit)
RIVER_JOIN_M = 30_000  # river pieces whose ends are this close are joined (lakes and reservoirs break them up)
GEOD = Geod(ellps="WGS84")
TO_EQUAL_AREA = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
RIVER_REPEAT_M = 1_500_000  # same-named rivers further apart than this are named separately (Colorado, Red...)
RIVER_TILT_COST = 0.4  # choosing a river's stretch: steepness (0..1 from level to vertical) against wiggliness...
RIVER_MIDDLE_COST = 0.3  # ...and distance from the river's middle (0..0.5 of its length)
RIVER_NAMES = {"Grande": "Rio Grande", "Syr Dar": "Syr Darya"}  # Natural Earth name_en (after dropping " River") -> label
OCEAN_TEXT = {"Atlantic": "ATLANTIC OCEAN", "Pacific": "PACIFIC OCEAN", "Indian": "INDIAN OCEAN", "Arctic": "ARCTIC OCEAN", "Southern": "SOUTHERN OCEAN"}


def classify(featurecla: str) -> Kind | None:
    """The label kind of a Natural Earth marine feature class, or None if it never gets a label."""
    return CLASSES.get(featurecla)


def ocean_word(name: str) -> str:
    return next(w for w in OCEAN_TEXT if w.lower() in name.lower())


def load_water_bodies(kinds: set[Kind], lakes_projected: gpd.GeoDataFrame, frame: MapFrame) -> list[Body]:
    """The water bodies of `kinds`, biggest first (full_size; of two of one name, the bigger). North and South
    Atlantic are one ATLANTIC; on the world map the Pacific, cut by the antimeridian, is labelled twice (left and right of x = 0).
    Every other body is labelled once, on its largest polygon."""
    everywhere = load_marine()
    whole_oceans = everywhere[everywhere.featurecla == "ocean"]
    ocean_km2 = whole_oceans.km2.groupby(whole_oceans.name.map(ocean_word)).sum()  # the whole of each, shown or not
    marine = frame.visible(frame.project(everywhere))
    marine = marine[~marine.geometry.is_empty]
    lakes_projected = frame.visible(lakes_projected)
    marine = marine[marine.name.notna() & ~marine.name.str.endswith(" River")]  # rivers come later, with river drawing
    marine["geometry"] = shapely.make_valid(marine.geometry.values)  # some are self-intersecting rough areas
    marine = marine[marine.featurecla.map(classify).isin(kinds)]  # not a list: an empty one would select columns
    bodies = []
    oceans = marine[marine.featurecla == "ocean"]
    for word, group in oceans.groupby(oceans.name.map(ocean_word)):
        whole = shapely.union_all(group.geometry.values)
        if word == "Pacific" and frame.globe:
            x0, y0, x1, y1 = whole.bounds
            halves = [whole.intersection(shapely.box(x0, y0, 0, y1)), whole.intersection(shapely.box(0, y0, x1, y1))]
        else:
            halves = [whole]
        bodies += [Body(OCEAN_TEXT[word], "oceans", 0, largest(h), float(ocean_km2[word]), STYLES["oceans"]) for h in halves]
    for row in marine[marine.featurecla != "ocean"].itertuples():
        kind = classify(row.featurecla)
        assert kind is not None
        bodies.append(Body(row.name, kind, int(row.scalerank), largest(row.geometry), float(row.km2), STYLES[kind]))
    if "lakes" in kinds:
        bodies += [Body(r.name, "lakes", int(r.scalerank), largest(r.geometry), float(r.km2), STYLES["lakes"]) for r in lakes_projected.itertuples() if isinstance(r.name, str) and not r.geometry.is_empty]
    if "rivers" in kinds:
        bodies += load_river_bodies(frame)
    bodies.sort(key=lambda b: -b.full_size)
    seen: set[str] = set()
    unique = []
    for b in bodies:  # oceans and rivers may share a name (the two Pacifics, far-apart rivers); others once
        if b.kind in ("oceans", "rivers") or b.text not in seen:
            unique.append(b)
            seen.add(b.text)
    return unique


def river_name(name_en: object) -> str | None:
    """A river's label from Natural Earth's English name, or None for unnamed lines."""
    if not isinstance(name_en, str):
        return None
    name = name_en.removesuffix(" River")
    return RIVER_NAMES.get(name, name)


def longest_paths(lines: list[LineString], join: float) -> list[LineString]:
    """A long path through each network of `lines`, longest first. Lines meet where their ends are within `join`
    (the gap is bridged straight). The path is the longest through the network's maximum spanning tree (exact
    when the network has no loops; delta branches and braids make loops)."""
    ends = shapely.points([c for line in lines for c in (line.coords[0], line.coords[-1])])
    joined = DisjointSets(len(ends))  # line ends that meet: one node
    for a, b in shapely.STRtree(ends).query(ends, predicate="dwithin", distance=join).T:
        joined.union(int(a), int(b))
    ends_of = [(joined.find(2 * k), joined.find(2 * k + 1)) for k in range(len(lines))]
    linked = DisjointSets(len(ends))  # the tree's nodes linked so far
    tree: dict[int, list[tuple[int, int, int]]] = {}  # node -> (line, other node, direction)
    for k in sorted(range(len(lines)), key=lambda k: -lines[k].length):  # Kruskal, longest lines first
        u, v = ends_of[k]
        if linked.find(u) != linked.find(v):
            linked.union(u, v)
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


@functools.cache
def whole_rivers() -> dict[str, list[tuple[BaseGeometry, float]]]:
    """Per river name, each river of that name: its unclipped features (EQUAL_AREA_CRS metres) and their total
    length (km). Natural Earth's pieces of one river (broken at lakes, reservoirs and junctions) come within
    RIVER_REPEAT_M of each other, as load_river_bodies names them once; rivers further apart are different rivers
    (the Colorados of the US and Argentina)."""
    rivers = load_rivers()
    text = rivers.name_en.map(river_name)
    rivers, text = rivers[text.notna()], text[text.notna()]
    km = np.array([GEOD.geometry_length(g) / 1000 for g in rivers.geometry.values])
    metres = np.asarray(rivers.to_crs(EQUAL_AREA_CRS).geometry.values)
    out: dict[str, list[tuple[BaseGeometry, float]]] = {}
    for name, rows in text.groupby(text).indices.items():
        mine = metres[rows]
        sets = DisjointSets(len(rows))
        for a, b in zip(*np.nonzero(shapely.dwithin(mine[:, None], mine[None, :], RIVER_REPEAT_M))):
            sets.union(int(a), int(b))
        out[str(name)] = [(shapely.union_all(metres[rows[g]]), float(km[rows[g]].sum())) for g in sets.groups()]
    return out


def river_length(text: str, part: LineString, frame: MapFrame) -> float:
    """The whole length (km) of the river named `text` that `part` (projected, in `frame`) lies on."""
    mid = part.interpolate(0.5, normalized=True)
    x, y = TO_EQUAL_AREA.transform(*frame.to_lonlat.transform(mid.x, mid.y))
    return min(whole_rivers()[text], key=lambda r: r[0].distance(shapely.Point(x, y)))[1]


def load_river_bodies(frame: MapFrame) -> list[Body]:
    """One body per drawn river name: the longest path through its features
    (longest_paths), plus any other path at least RIVER_REPEAT_M from those kept (two rivers with one name). Rank =
    the path's most important feature."""
    rivers = frame.visible(frame.project(load_rivers()))
    rivers = rivers[~rivers.geometry.is_empty]
    rivers["text"] = rivers.name_en.map(river_name)
    bodies = []
    for text, group in rivers[rivers.text.notna()].groupby("text"):
        parts = longest_paths(list(shapely.get_parts(shapely.line_merge(shapely.union_all(group.geometry.values)))), RIVER_JOIN_M)
        kept: list[LineString] = []
        for part in parts:
            if all(part.distance(k) >= RIVER_REPEAT_M for k in kept):
                kept.append(part)
        for part in kept:
            rank = int(group.scalerank[shapely.dwithin(group.geometry.values, part, 1.0)].min())
            bodies.append(Body(str(text), "rivers", rank, part, river_length(str(text), part, frame), STYLES["rivers"]))
    return bodies


def river_ranks(rivers_projected: gpd.GeoDataFrame, bodies: list[Body], ranks: np.ndarray) -> np.ndarray:
    """Each river line's rank: that of the nearest of the `bodies` (with their `ranks`) of its name (river_name),
    or UNLABELED if none has it (unnamed, or rivers not labelled)."""
    out = np.full(len(rivers_projected), UNLABELED)
    names = rivers_projected.name_en.map(river_name).to_numpy()
    lines = rivers_projected.geometry.to_numpy()
    for text in {b.text for b in bodies}:
        mine = [k for k, b in enumerate(bodies) if b.text == text]
        rows = np.flatnonzero((names == text) & ~shapely.is_empty(lines))
        if len(rows):
            gap = shapely.distance(lines[rows][:, None], np.array([bodies[k].geom for k in mine])[None, :])
            out[rows] = ranks[mine][gap.argmin(axis=1)]
    return out


def river_width(rank: int) -> float:
    """Line width (pt) of a river of Natural Earth scalerank `rank`, thinning linearly to RIVER_THIN_RANK."""
    t = min(max(rank, 0), RIVER_THIN_RANK) / RIVER_THIN_RANK
    return RIVER_WIDTH_MAX + t * (RIVER_WIDTH_MIN - RIVER_WIDTH_MAX)


def river_size(rank: int) -> float:
    """A river name's size (pt) from its scalerank: a function of how wide its line is drawn (river_width)."""
    t = (river_width(rank) - RIVER_WIDTH_MIN) / (RIVER_WIDTH_MAX - RIVER_WIDTH_MIN)
    return RIVER_MIN_PT + t * (RIVER_MAX_PT - RIVER_MIN_PT)


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


def river_label(body: Body, frame: MapFrame, lakes: BaseGeometry | None = None) -> CurvedLabel | None:
    """A river's name just above the best stretch of it (best_stretch) for a name its size plus half an em each side,
    clear of its wiggles there; None if the river is shorter than that. Only its longest piece outside the drawn
    `lakes` (map units) counts: Natural Earth's lines run on through lakes, hidden under them."""
    size = river_size(body.rank)
    em = size * PX_PER_PT
    dry = body.geom if lakes is None else body.geom.difference(lakes)
    if dry.is_empty:
        return None
    line = largest_line(frame.to_px(dry))
    pts = np.asarray(line.coords)
    need = text_width(body.text, font(size, True)) * PX_PER_PT + em
    if line.length < need:
        return None
    stretch = best_stretch(pts, need)
    curve = fit_curve(LineString(stretch).buffer(1.0), MAX_TILT_DEG)
    rise = RIVER_GAP_EM * em + height_above(stretch, curve)
    label = lay_out(body, curve, size, line.buffer(RIVER_CORRIDOR_EM * em), rise)
    return label if label.inside_share(label.anchor_x, label.anchor_y) >= CURVED_MIN_INSIDE else None


def largest_line(geom: BaseGeometry) -> LineString:
    return max(shapely.get_parts(geom), key=lambda g: g.length)


def make_water_labels(bodies: list[Body], frame: MapFrame, lakes: BaseGeometry | None = None) -> list[CurvedLabel | Unplaced]:
    """A curved label per body, in the same order; an Unplaced for those with under CURVED_MIN_INSIDE of their glyph
    boxes over their body to start with (most small bays: no room for their name), and rivers too short for theirs
    outside the drawn `lakes` (map units)."""
    labels = [river_label(b, frame, lakes) if b.kind == "rivers" else area_label(b, frame) for b in bodies]
    return [Unplaced(b.text, b.kind, b.full_size) if label is None else label for b, label in zip(bodies, labels)]
