"""--crop: a zoomed map of one region, in a Lambert azimuthal equal-area projection centred on it."""

import re
import sys
import unicodedata
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import shapely
from pyproj import Geod, Transformer
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from worldmap.data import is_labeled, load_countries, load_disputed, load_lakes
from worldmap.disputed import disputed_label_args, load_disputes
from worldmap.earth import Fill
from worldmap.frame import ANTIMERIDIAN_EPS, OUTLINE_POINTS, PROJECT_PAD, LonLatBox, MapFrame, lon_ranges, make_frame
from worldmap.labels.label import RENAME
from worldmap.labels.regions import PARENT_SHORT, child_regions, polygons
from worldmap.labels.water import kinds_from_labels, load_water_bodies

if TYPE_CHECKING:
    import questionary

CropBox = tuple[float, float, float, float]  # west lon, south lat, east lon, north lat; west > east crosses +-180,
# -180..180 is all the way round
CentreCrop = tuple[float, float, float, float]  # centre lon, lat; width, height in degrees of arc on the ground
CROP_MARGIN = 0.10  # a named crop's margin on each side, as a fraction of the region's projected width/height
CLIP_PAD_DEG = 5  # lon/lat clip boxes reach this far past what the canvas shows
SEGMENT_DEG = 0.5  # lon/lat edges are split this finely before projecting (they curve in the crop's projection)
EDGE_NUDGE = 1e-9  # relative: ~2 cm on the Robinson globe
CLIP_SAMPLES = 65  # points per side of the grid sampled to find the canvas's lon/lat extent
AZIMUTHAL_MAX_DEG = 60  # a region reaching further than this from its centre gets Equal Earth: azimuthal shapes would stretch
RECIPE_TOLERANCE = 0.002  # centre_recipe: how close a centre crop's size must come
CENTRE_BISECTIONS = 30  # centre_recipe: halvings of each size's range (0..120 or 360 degrees)
POLE_DEG = 0.1  # round_pole: reaching this close to a pole counts
RECENTRE_ROUNDS, RECENTRE_TOLERANCE = 8, 1e-4  # crop_frame: rounds of moving the centre to the map's middle, and how close
LABEL_SCALE_MAX = 2.5  # a crop's label_scale: the square root of its zoom over the world map, up to this
KM_PER_DEG = 6371.0088 * np.pi / 180  # a degree of arc on the mean Earth sphere
UNIT_KM = {"km": 1.0, "mi": 1.609344}  # a centre crop's size units (a bare size is in degrees)

NUM = r"\s*(-?\d+(?:\.\d*)?)?\s*"  # may be left out
BOX_RE = re.compile(f"^{NUM}:{NUM},{NUM}:{NUM}$")


def parse_box(s: str) -> CropBox | None:
    """`west:east,south:north` -> CropBox, or None if `s` isn't of that form (a name). Any edge may be left out: the
    globe's (+-180, +-90), so `:,60:` is everything north of 60N. West past east crosses +-180."""
    m = BOX_RE.match(s)
    if not m:
        return None
    west, east, south, north = (None if v is None else float(v) for v in m.groups())
    west, east = -180.0 if west is None else west, 180.0 if east is None else east
    south, north = -90.0 if south is None else south, 90.0 if north is None else north
    if not all(-180 <= lon <= 180 for lon in (west, east)) or not all(-90 <= lat <= 90 for lat in (south, north)):
        raise ValueError(f"--crop {s}: longitudes must be within +-180 and latitudes within +-90")
    if south > north:
        raise ValueError(f"--crop {s}: the south edge is north of the north edge")
    if west == east or south == north:
        raise ValueError(f"--crop {s}: the box has no area")
    return west, south, east, north


SIZE = r"\s*(\d+(?:\.\d*)?)\s*([a-z]*)\s*"
CENTRE_HEAD_RE = re.compile(r"^\s*-?\d[\d.]*\s*,\s*-?\d[\d.]*\s*/")  # lon,lat/ : meant as a centre crop
CENTRE_RE = re.compile(r"^\s*(-?\d+(?:\.\d*)?)\s*,\s*(-?\d+(?:\.\d*)?)\s*/" + f"{SIZE},{SIZE}$", re.IGNORECASE)


def parse_centre(s: str) -> CentreCrop | None:
    """`lon,lat/width,height` -> CentreCrop, or None if `s` doesn't start lon,lat/ (a name or a box). Sizes are
    degrees of arc, or with a unit (km, mi) a distance on the ground."""
    if not CENTRE_HEAD_RE.match(s):
        return None
    m = CENTRE_RE.match(s)
    if not m:
        raise ValueError(f"--crop {s}: a centre crop is lon,lat/width,height (sizes in degrees, km or mi)")
    lon, lat, width, w_unit, height, h_unit = m.groups()
    sizes = []
    for value, unit in ((width, w_unit), (height, h_unit)):
        if unit and unit.lower() not in UNIT_KM:
            raise ValueError(f"--crop {s}: unknown unit {unit!r} (use km, mi, or none for degrees)")
        sizes.append(float(value) * UNIT_KM[unit.lower()] / KM_PER_DEG if unit else float(value))
    if not (-180 <= float(lon) <= 180 and -90 <= float(lat) <= 90):
        raise ValueError(f"--crop {s}: the longitude must be within +-180 and the latitude within +-90")
    if not all(0 < size <= 360 for size in sizes):
        raise ValueError(f"--crop {s}: the width and height must be above 0 and at most 360 degrees")
    return float(lon), float(lat), sizes[0], sizes[1]


def box_region(b: CropBox) -> BaseGeometry:
    """The box as lon/lat polygon(s), split at +-180 when it crosses."""
    west, south, east, north = b
    if west <= east:
        return box(west, south, east, north)
    return shapely.union_all([box(west, south, 180, north), box(-180, south, east, north)])


def extent(geom: BaseGeometry) -> CropBox:
    """The geometry's lon/lat extent, its longitudes taken the short way round (west > east: across +-180)."""
    lon, lat = shapely.get_coordinates(shapely.segmentize(geom, SEGMENT_DEG)).T
    if lon_ranges(lon, SEGMENT_DEG) == [(-180, 180)]:  # no gap wider than the segments: all the way round
        return -180.0, float(lat.min()), 180.0, float(lat.max())
    ranges = lon_ranges(lon, 0)
    return float(ranges[0][0]), float(lat.min()), float(ranges[-1][1]), float(lat.max())


def sphere_centre(geom: BaseGeometry) -> tuple[float, float]:
    """The middle of the geometry's lon/lat extent (see `extent`). Around a pole, where the longitudes wrap the
    globe, the mean of its vertices as unit vectors."""
    west, south, east, north = extent(geom)
    if (west, east) != (-180, 180):
        mid = (west + east + (360 if east < west else 0)) / 2
        return (mid + 180) % 360 - 180, (south + north) / 2
    lon, lat = shapely.get_coordinates(shapely.segmentize(geom, SEGMENT_DEG)).T
    lon, lat = np.radians(lon), np.radians(lat)
    v = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]).mean(axis=1)
    return float(np.degrees(np.arctan2(v[1], v[0]))), float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))


def sample_lonlat(to_lonlat: Transformer, to_map: Transformer, rect: Polygon) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """The lon/lat of a CLIP_SAMPLES-square grid of points over `rect` (those on the globe), and the poles in it."""
    x0, y0, x1, y1 = rect.bounds
    mx, my = np.meshgrid(np.linspace(x0, x1, CLIP_SAMPLES), np.linspace(y0, y1, CLIP_SAMPLES))
    gx, gy = mx.ravel(), my.ravel()
    lon, lat = to_lonlat.transform(gx, gy)
    ok = np.isfinite(lon) & np.isfinite(lat)
    bx, by = to_map.transform(np.where(ok, lon, 0), np.where(ok, lat, 0))
    ok &= np.hypot(bx - gx, by - gy) < 1e-6 * (x1 - x0)  # off the edge of a whole-globe map, it doesn't map back
    poles = []
    for pole in (90, -90):
        x, y = to_map.transform(0, pole)
        if np.isfinite(x) and rect.contains(shapely.Point(x, y)):
            poles.append(pole)
    return lon[ok], lat[ok], poles


def clip_boxes(to_lonlat: Transformer, to_map: Transformer, canvas: Polygon, centre: tuple[float, float]) -> tuple[LonLatBox, ...]:
    """lon/lat boxes covering what the canvas (grown by PROJECT_PAD) shows, padded by CLIP_PAD_DEG, so data far
    round the globe is cut off before projecting (the projection is singular at the centre's antipode)."""
    x0, y0, x1, y1 = canvas.bounds
    px, py = PROJECT_PAD * (x1 - x0), PROJECT_PAD * (y1 - y0)
    lon, lat, poles = sample_lonlat(to_lonlat, to_map, box(x0 - px, y0 - py, x1 + px, y1 + py))
    south, north = max(-90.0, float(lat.min()) - CLIP_PAD_DEG), min(90.0, float(lat.max()) + CLIP_PAD_DEG)
    ranges = lon_ranges(lon, CLIP_PAD_DEG)
    for pole in poles:
        ranges = [(-180, 180)]
        south, north = (south, 90.0) if pole > 0 else (-90.0, north)
    boxes = tuple((float(w), south, float(e), north) for w, e in ranges)
    anti_lon, anti_lat = (centre[0] + 360) % 360 - 180, -centre[1]
    if any(w <= anti_lon <= e and s <= anti_lat <= n for w, s, e, n in boxes):
        raise ValueError("--crop: the region is too large for one azimuthal map (it reaches the far side of the globe)")
    return boxes


def reach(xy: np.ndarray, centre: tuple[float, float]) -> float:
    """The greatest angle (degrees) on the sphere from `centre` to any of the lon/lat points `xy`."""
    lon, lat = np.radians(xy).T
    lon0, lat0 = np.radians(centre)
    cos = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))).max())


def seam_boxes(lon0: float) -> tuple[LonLatBox, ...]:
    """Boxes covering the globe but a hair either side of the meridian opposite `lon0` (a map centred on lon0 is cut
    there), split at +-180 when that isn't the cut."""
    seam = (lon0 + 360) % 360 - 180
    if abs(seam) >= 180 - ANTIMERIDIAN_EPS:
        return ((-180, -90, 180, 90),)
    return ((seam + ANTIMERIDIAN_EPS, -90, 180, 90), (-180, -90, seam - ANTIMERIDIAN_EPS, 90))


def globe_outline(to_map: Transformer, lon0: float) -> Polygon:
    """The outline of a whole-globe map centred on lon0: its meridians a hair either side of the far one."""
    lats = np.linspace(-90, 90, OUTLINE_POINTS)
    east = np.column_stack(to_map.transform(np.full_like(lats, lon0 + 180 - ANTIMERIDIAN_EPS), lats))
    west = np.column_stack(to_map.transform(np.full_like(lats, lon0 - 180 + ANTIMERIDIAN_EPS), lats[::-1]))
    return Polygon(np.vstack([east, west]))


def crop_frame(region: BaseGeometry, margin: float) -> MapFrame:
    """A frame centred on `region` (lon/lat), fitting its projected bounds plus `margin` of their width/height on
    each side. The projection is Lambert azimuthal equal-area, or Equal Earth for a region reaching further than
    AZIMUTHAL_MAX_DEG from its centre (an ocean); there the canvas is cut to the globe's outline. The projection is
    centred on the middle of the map: the bounds aren't symmetric about the region's own centre, so it's moved there
    and the bounds refitted, up to RECENTRE_ROUNDS times. A region round a pole gets a square map centred on it."""
    if (pole := round_pole(region)) is not None:
        return with_label_scale(fit_frame(region, margin, 0.0, pole, square=True))
    lon0, lat0 = sphere_centre(region)
    for _ in range(RECENTRE_ROUNDS):
        frame = fit_frame(region, margin, lon0, lat0)
        mid_x, mid_y = (frame.xmin + frame.xmax) / 2, (frame.ymin + frame.ymax) / 2
        azimuthal = "+proj=laea" in frame.crs
        if np.hypot(mid_x, mid_y if azimuthal else 0) < RECENTRE_TOLERANCE * min(frame.width, frame.height):
            break
        lon, lat = frame.to_lonlat.transform(mid_x, mid_y if azimuthal else 0)
        lon0, lat0 = float(lon), float(lat) if azimuthal else lat0
    return with_label_scale(frame)


def centre_frame(crop: CentreCrop, azimuthal: bool | None = None) -> MapFrame:
    """A frame centred on the crop's centre, reaching half its width (height) along the ground east and west (north
    and south) from it: Lambert azimuthal equal-area, or Equal Earth if a half size is over AZIMUTHAL_MAX_DEG (or as
    `azimuthal` says). Equal Earth's height runs along the meridian, past a pole at the meridian's scale there (only
    margin: the canvas is cut to the globe)."""
    lon0, lat0, width, height = crop
    geod = Geod(ellps="WGS84")

    def ends(azimuths: tuple[float, float], half: float) -> np.ndarray:
        """The map coordinates of the points `half` degrees along the ground from the centre in each direction."""
        lon, lat, _ = geod.fwd([lon0, lon0], [lat0, lat0], azimuths, [half * KM_PER_DEG * 1000] * 2)
        return np.column_stack(to_map.transform(lon, lat))

    if max(width, height) / 2 <= AZIMUTHAL_MAX_DEG if azimuthal is None else azimuthal:
        crs = f"+proj=laea +lat_0={lat0:.6f} +lon_0={lon0:.6f} +datum=WGS84"
        to_map = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        half_x = float(np.hypot(*ends((90, -90), width / 2).T).max())
        half_y = float(np.hypot(*ends((0, 180), height / 2).T).max())
        return with_label_scale(canvas_frame(crs, lon0, lat0, box(-half_x, -half_y, half_x, half_y)))
    crs = f"+proj=eqearth +lon_0={lon0:.6f} +datum=WGS84"
    to_map = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    half_x = float(np.abs(ends((90, -90), width / 2)[:, 0]).max())
    _, yc = to_map.transform(lon0, lat0)
    lats = np.array([lat0 - height / 2, lat0 + height / 2])
    _, y = to_map.transform([lon0, lon0], np.clip(lats, -90, 90))
    _, pole_scale = np.abs(np.diff(to_map.transform([lon0, lon0], [89.0, 90.0]), axis=1)).ravel()  # per degree
    y = np.asarray(y) + np.sign(lats) * np.maximum(np.abs(lats) - 90, 0) * pole_scale
    half_y = float(np.abs(y - yc).max())
    return with_label_scale(canvas_frame(crs, lon0, lat0, box(-half_x, yc - half_y, half_x, yc + half_y)))


def with_label_scale(frame: MapFrame) -> MapFrame:
    """The crop frame with its label_scale: the square root of its zoom over the world map, 1..LABEL_SCALE_MAX."""
    zoom = frame.px_per_unit / make_frame().px_per_unit  # both in metres
    return replace(frame, label_scale=float(np.clip(np.sqrt(zoom), 1, LABEL_SCALE_MAX)))


def round_pole(region: BaseGeometry) -> float | None:
    """90 or -90 if the region surrounds that pole: it goes all the way round within one hemisphere (the Southern
    Ocean, `:,60:`), or reaches the pole and more than half way round (the Arctic Ocean, its coast cut by Greenland
    and Canada's islands). Else None (a wedge that just touches a pole)."""
    west, south, east, north = extent(region)
    span = 360.0 if (west, east) == (-180, 180) else (east - west) % 360
    if span == 360 and (south >= 0 or north <= 0):
        return 90.0 if south >= 0 else -90.0
    if span > 180 and north >= 90 - POLE_DEG:
        return 90.0
    if span > 180 and south <= -90 + POLE_DEG:
        return -90.0
    return None


def fit_frame(region: BaseGeometry, margin: float, lon0: float, lat0: float, square: bool = False) -> MapFrame:
    """crop_frame with the projection centred on (lon0, lat0); with `square`, the canvas is a square centred there."""
    xy = shapely.get_coordinates(shapely.segmentize(region, SEGMENT_DEG))
    azimuthal = reach(xy, (lon0, lat0)) <= AZIMUTHAL_MAX_DEG
    if azimuthal:
        crs = f"+proj=laea +lat_0={lat0:.6f} +lon_0={lon0:.6f} +datum=WGS84"
    else:
        crs = f"+proj=eqearth +lon_0={lon0:.6f} +datum=WGS84"
        # Measured without what touches the seam (it would wrap to the other side), and along the equator too:
        # a pseudo-cylindrical map is widest there, so a region crossing it can reach further than its outline.
        shown = region.intersection(shapely.union_all([box(*b) for b in seam_boxes(lon0)]))
        equator = shown.intersection(shapely.LineString([(-180, 0), (180, 0)]))
        xy = shapely.get_coordinates(shapely.segmentize(shapely.union_all([shown.boundary, equator]), SEGMENT_DEG))
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(xy[:, 0], xy[:, 1])
    x0, y0, x1, y1 = x.min(), y.min(), x.max(), y.max()
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    x0, y0, x1, y1 = float(x0 - mx), float(y0 - my), float(x1 + mx), float(y1 + my)
    if square:
        half = max(abs(x0), abs(y0), abs(x1), abs(y1))
        x0, y0, x1, y1 = -half, -half, half, half
    return canvas_frame(crs, lon0, lat0, box(x0, y0, x1, y1))


def canvas_frame(crs: str, lon0: float, lat0: float, canvas: Polygon) -> MapFrame:
    """A crop frame showing `canvas` (map units) in `crs`, Lambert azimuthal equal-area or Equal Earth centred on
    (lon0, lat0): clipped to what the canvas shows, and an Equal Earth canvas cut to the globe's outline."""
    to_map = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x0, y0, x1, y1 = canvas.bounds
    if "+proj=laea" in crs:
        clip = clip_boxes(to_lonlat, to_map, canvas, (lon0, lat0))
        outline = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    else:
        clip = seam_boxes(lon0)
        shown = canvas.intersection(globe_outline(to_map, lon0))
        outline = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] if shown.equals(canvas) else list(shown.exterior.coords[:-1])
    return MapFrame(to_map, to_lonlat, outline, x0, y0, x1, y1, crs=crs, globe=False, clip=clip)


def canvas_centre(frame: MapFrame) -> tuple[float, float]:
    """The lon/lat at the middle of the frame's canvas."""
    lon, lat = frame.to_lonlat.transform((frame.xmin + frame.xmax) / 2, (frame.ymin + frame.ymax) / 2)
    return float(lon), float(lat)


def centre_recipe(region: BaseGeometry, margin: float) -> CentreCrop | None:
    """A centre crop that reproduces the crop to `region` with `margin`: its canvas's middle, with each size
    bisected, in the crop's projection, until a centre crop is the same size. None if none comes close, or if one
    that size would pick the other projection."""
    target = crop_frame(region, margin)
    lon0, lat0 = canvas_centre(target)
    azimuthal = "+proj=laea" in target.crs
    top = 2.0 * AZIMUTHAL_MAX_DEG if azimuthal else 360.0  # the most a centre crop in that projection can be
    (w_lo, w_hi), (h_lo, h_hi) = (0.0, top), (0.0, top)
    for _ in range(CENTRE_BISECTIONS):
        got = centre_frame((lon0, lat0, (w_lo + w_hi) / 2, (h_lo + h_hi) / 2), azimuthal)
        w_lo, w_hi = ((w_lo + w_hi) / 2, w_hi) if got.width < target.width else (w_lo, (w_lo + w_hi) / 2)
        h_lo, h_hi = ((h_lo + h_hi) / 2, h_hi) if got.height < target.height else (h_lo, (h_lo + h_hi) / 2)
    crop = lon0, lat0, (w_lo + w_hi) / 2, (h_lo + h_hi) / 2
    got = centre_frame(crop)
    same = ("+proj=laea" in got.crs) == ("+proj=laea" in target.crs)
    return crop if same and max(abs(got.width / target.width - 1), abs(got.height / target.height - 1)) < RECIPE_TOLERANCE else None


def centre_spec(crop: CentreCrop) -> str:
    """The centre crop as --crop takes it, lon,lat/width,height with the sizes in km."""
    lon, lat, width, height = crop
    return f"{round(lon, 2):g},{round(lat, 2):g}/{width * KM_PER_DEG:.0f}km,{height * KM_PER_DEG:.0f}km"


KINDS = ("country", "territory", "disputed", "ocean", "sea", "lake", "strait", "river")  # the order a match list is in
WATER_KIND = {"oceans": "ocean", "seas": "sea", "lakes": "lake", "straits": "strait", "rivers": "river"}


@dataclass(frozen=True)
class Candidate:
    text: str  # the label's text
    kind: str  # one of KINDS
    geom: BaseGeometry  # lon/lat: what the label stands for
    aliases: tuple[str, ...] = ()  # other names it matches: a country's full name, a territory's parent's names


def candidates(label: set[str], disputed: bool) -> list[Candidate]:
    """Every label of the `label` kinds (--label) a name crop can match, with the lon/lat geometry it stands for. A
    country's excludes the parts territories label separately (French Guiana...), whether or not they're labelled."""
    frame = make_frame()
    found: list[tuple[str, str, BaseGeometry, tuple[str, ...]]] = []
    if label & {"countries", "territories"}:
        countries = load_countries()
        projected = frame.project(countries)
        children = child_regions(countries, projected, frame)
        if "countries" in label:
            relabel: dict[int, str] = {}
            extra: list[tuple[str, Polygon, Fill]] = []
            if disputed:  # the colour doesn't matter here
                relabel, extra = disputed_label_args(load_disputes(countries, frame.project(load_disputed())), lambda i, pt: "#808080")
            for i, row in projected.iterrows():
                if not (is_labeled(row) or i in relabel):
                    continue
                taken = {p.wkb for c in children if c.feature == i for p in c.polys}
                own = [p for p in polygons(row.geometry) if p.wkb not in taken]
                if own:
                    text = relabel.get(i) or RENAME.get(row.NAME_EN, row.NAME_EN)
                    found.append((text, "country", shapely.union_all(own), (row.NAME_EN,)))
            found += [(text, "disputed", poly, ()) for text, poly, _ in extra]
        if "territories" in label:
            full = {short: name for name, short in PARENT_SHORT.items()}  # "US" -> "United States of America"
            for c in children:  # so "united" finds Alaska (US) too
                name = full.get(c.parent_name, c.parent_name)
                found.append((c.text, "territory", shapely.union_all(c.polys), (RENAME.get(name, name), name)))
    kinds = kinds_from_labels(label)
    if kinds:
        bodies = load_water_bodies(kinds, frame.project(load_lakes()), frame)
        oceans: dict[str, list[BaseGeometry]] = {}
        for b in bodies:  # the world map labels the Pacific twice, either side of +-180: one candidate
            if b.kind == "oceans":
                oceans.setdefault(b.text, []).append(b.geom)
        found += [(text, "ocean", shapely.union_all(geoms), ()) for text, geoms in oceans.items()]
        found += [(b.text, WATER_KIND[b.kind], b.geom, ()) for b in bodies if b.kind != "oceans"]
    return [Candidate(text, kind, to_lonlat(g, frame), aliases) for text, kind, g, aliases in found]


def to_lonlat(geom: BaseGeometry, frame: MapFrame) -> BaseGeometry:
    """World-frame geometry back to lon/lat. Points on the globe's edge (+-180) don't invert, so x is nudged a
    hair (EDGE_NUDGE) towards the centre first."""

    def inverse(xy: np.ndarray) -> np.ndarray:
        lon, lat = frame.to_lonlat.transform(xy[:, 0] * (1 - EDGE_NUDGE), xy[:, 1])
        return np.column_stack([lon, lat])

    return shapely.transform(geom, inverse)


def fold(s: str) -> str:
    """For matching: no accents, no case."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).casefold()


def matches(name: str, cands: list[Candidate]) -> list[Candidate]:
    """The candidates whose text or an alias contains `name` (ignoring case and accents): exact matches first, then
    the rest, each in KINDS order, then by text."""
    key = fold(name)
    names = {id(c): [fold(n) for n in (c.text, *c.aliases)] for c in cands}
    found = [c for c in cands if any(key in n for n in names[id(c)])]
    return sorted(found, key=lambda c: (key not in names[id(c)], KINDS.index(c.kind), c.text))


def describe(c: Candidate) -> str:
    """'France (country) 2.62,46.33/1375km,1270km': the text, the kind, and the centre crop that crops to the same
    map (centre_recipe), as --crop takes it, to copy and adjust; without it if there's none."""
    try:
        crop = centre_recipe(c.geom, CROP_MARGIN)
    except ValueError:  # too big to crop
        crop = None
    return f"{c.text} ({c.kind})" + ("" if crop is None else f" {centre_spec(crop)}")


def checklist(message: str, lines: list[str]) -> "questionary.Question":
    """A checkbox prompt over `lines`, answering with their indices (so identical lines stay apart), without
    questionary's select-all and invert keys."""
    import questionary
    from prompt_toolkit.key_binding import KeyBindings

    q = questionary.checkbox(message, choices=[questionary.Choice(line, value=k) for k, line in enumerate(lines)],
                             instruction="(arrow keys to move, <space> to check, <enter> to confirm)",
                             validate=lambda chosen: bool(chosen) or "choose at least one")
    bindings = q.application.key_bindings
    assert isinstance(bindings, KeyBindings)
    for key in ("a", "i"):
        bindings.remove(key)
    return q


def choose(name: str, found: list[Candidate]) -> list[Candidate]:
    """The candidate(s) to crop to: the only match, or those the user checks when several match. Without a terminal
    to ask on, several matches are an error that lists them."""
    if not found:
        sys.exit(f"make_map: --crop {name}: no label matches (only the kinds in --label are searched)")
    if len(found) == 1:
        return found
    lines = [describe(c) for c in found]
    if not sys.stdin.isatty():
        sys.exit(f"make_map: --crop {name} matches several labels; run in a terminal to choose, or crop to one with its lon,lat/width,height:\n"
                 + "\n".join(lines))
    picked = checklist(f"--crop {name} matches several labels. Crop to:", lines).ask()
    if not picked:
        sys.exit("make_map: --crop: nothing chosen")
    return [found[k] for k in sorted(picked)]
