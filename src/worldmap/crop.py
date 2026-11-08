"""--crop: a zoomed map of one region, in a Lambert azimuthal equal-area projection centred on it."""

import re
import sys
import unicodedata
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from worldmap.data import is_labeled, load_countries, load_disputed, load_lakes
from worldmap.disputed import disputed_label_args, load_disputes
from worldmap.earth import Fill
from worldmap.frame import PROJECT_PAD, LonLatBox, MapFrame, make_frame
from worldmap.labels.label import RENAME
from worldmap.labels.regions import child_regions, polygons
from worldmap.labels.water import kinds_from_labels, load_water_bodies

CropBox = tuple[float, float, float, float]  # west lon, south lat, east lon, north lat; west > east crosses +-180
CROP_MARGIN = 0.10  # a named crop's margin on each side, as a fraction of the region's projected width/height
CLIP_PAD_DEG = 5  # lon/lat clip boxes reach this far past what the canvas shows
SEGMENT_DEG = 0.5  # lon/lat edges are split this finely before projecting (they curve in the crop's projection)
CLIP_SAMPLES = 65  # points per side of the grid sampled to find the canvas's lon/lat extent

NUM = r"\s*(-?\d+(?:\.\d*)?)\s*"
BOX_RE = re.compile(f"^{NUM},{NUM}:{NUM},{NUM}$")


def parse_box(s: str) -> CropBox | None:
    """`lon,lat:lon,lat` (two opposite corners, in any order) -> CropBox, or None if `s` isn't of that form (a name).
    Of the two ways round the globe between the corners' longitudes, the narrower is taken."""
    m = BOX_RE.match(s)
    if not m:
        return None
    lon_a, lat_a, lon_b, lat_b = (float(v) for v in m.groups())
    if not all(-180 <= lon <= 180 for lon in (lon_a, lon_b)) or not all(-90 <= lat <= 90 for lat in (lat_a, lat_b)):
        raise ValueError(f"--crop {s}: longitudes must be within +-180 and latitudes within +-90")
    west, east = min(lon_a, lon_b), max(lon_a, lon_b)
    if east - west > 180:  # narrower the other way round, across the antimeridian
        west, east = east, west
    return west, min(lat_a, lat_b), east, max(lat_a, lat_b)


def box_region(b: CropBox) -> BaseGeometry:
    """The box as lon/lat polygon(s), split at +-180 when it crosses."""
    west, south, east, north = b
    if west <= east:
        return box(west, south, east, north)
    return shapely.union_all([box(west, south, 180, north), box(-180, south, east, north)])


def sphere_centre(geom: BaseGeometry) -> tuple[float, float]:
    """The middle of the geometry's lon/lat extent, its longitudes taken the short way round (across +-180 if need
    be). Around a pole, where they wrap the globe, the mean of its vertices as unit vectors."""
    lon, lat = shapely.get_coordinates(shapely.segmentize(geom, SEGMENT_DEG)).T
    ranges = lon_ranges(lon, 0)
    if ranges != [(-180, 180)]:
        west, east = ranges[0][0], ranges[-1][1]
        mid = (west + east + (360 if east < west else 0)) / 2
        return (mid + 180) % 360 - 180, (lat.min() + lat.max()) / 2
    lon, lat = np.radians(lon), np.radians(lat)
    v = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]).mean(axis=1)
    return float(np.degrees(np.arctan2(v[1], v[0]))), float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))


def lon_ranges(lons: np.ndarray, pad: float) -> list[tuple[float, float]]:
    """The smallest set of +-180 longitude ranges covering `lons` (the complement of the widest gap), padded."""
    s = np.sort(lons)
    gaps = np.diff(np.append(s, s[0] + 360))
    k = int(np.argmax(gaps))
    if gaps[k] <= 2 * pad:
        return [(-180, 180)]
    start, end = s[(k + 1) % len(s)] - pad, s[k] + pad  # eastward from start to end
    if end < start:
        end += 360
    if start < -180:
        start, end = start + 360, end + 360
    if end <= 180:
        return [(start, end)]
    return [(start, 180), (-180, end - 360)]


def clip_boxes(to_lonlat: Transformer, to_map: Transformer, canvas: Polygon, centre: tuple[float, float]) -> tuple[LonLatBox, ...]:
    """lon/lat boxes covering what the canvas (grown by PROJECT_PAD) shows, padded by CLIP_PAD_DEG, so data far
    round the globe is cut off before projecting (the projection is singular at the centre's antipode)."""
    x0, y0, x1, y1 = canvas.bounds
    px, py = PROJECT_PAD * (x1 - x0), PROJECT_PAD * (y1 - y0)
    grown = box(x0 - px, y0 - py, x1 + px, y1 + py)
    gx, gy = np.meshgrid(np.linspace(x0 - px, x1 + px, CLIP_SAMPLES), np.linspace(y0 - py, y1 + py, CLIP_SAMPLES))
    lon, lat = to_lonlat.transform(gx.ravel(), gy.ravel())
    ok = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[ok], lat[ok]
    south, north = max(-90.0, float(lat.min()) - CLIP_PAD_DEG), min(90.0, float(lat.max()) + CLIP_PAD_DEG)
    ranges = lon_ranges(lon, CLIP_PAD_DEG)
    for pole in (90, -90):
        x, y = to_map.transform(0, pole)
        if np.isfinite(x) and grown.contains(shapely.Point(x, y)):
            ranges = [(-180, 180)]
            south, north = (south, 90.0) if pole > 0 else (-90.0, north)
    boxes = tuple((float(w), south, float(e), north) for w, e in ranges)
    anti_lon, anti_lat = (centre[0] + 360) % 360 - 180, -centre[1]
    if any(w <= anti_lon <= e and s <= anti_lat <= n for w, s, e, n in boxes):
        raise ValueError("--crop: the region is too large for one azimuthal map (it reaches the far side of the globe)")
    return boxes


def crop_frame(region: BaseGeometry, margin: float) -> MapFrame:
    """A frame in a Lambert azimuthal equal-area projection centred on `region` (lon/lat), fitting its projected
    bounds plus `margin` of their width/height on each side."""
    lon0, lat0 = sphere_centre(region)
    crs = f"+proj=laea +lat_0={lat0:.6f} +lon_0={lon0:.6f} +datum=WGS84"
    to_map = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xy = shapely.get_coordinates(shapely.segmentize(region, SEGMENT_DEG))
    x, y = to_map.transform(xy[:, 0], xy[:, 1])
    x0, y0, x1, y1 = x.min(), y.min(), x.max(), y.max()
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    x0, y0, x1, y1 = float(x0 - mx), float(y0 - my), float(x1 + mx), float(y1 + my)
    clip = clip_boxes(to_lonlat, to_map, box(x0, y0, x1, y1), (lon0, lat0))
    outline = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return MapFrame(to_map, to_lonlat, outline, x0, y0, x1, y1, crs=crs, globe=False, clip=clip)


KINDS = ("country", "territory", "disputed", "ocean", "sea", "lake", "strait", "river")  # the order a match list is in
WATER_KIND = {"oceans": "ocean", "seas": "sea", "lakes": "lake", "straits": "strait", "rivers": "river"}


@dataclass(frozen=True)
class Candidate:
    text: str  # the label's text
    kind: str  # one of KINDS
    geom: BaseGeometry  # lon/lat: what the label stands for


def candidates(label: set[str], disputed: bool) -> list[Candidate]:
    """Every label of the `label` kinds (--label) a name crop can match, with the lon/lat geometry it stands for. A
    country's excludes the parts territories label separately (French Guiana...), whether or not they're labelled."""
    frame = make_frame()
    found: list[tuple[str, str, BaseGeometry]] = []
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
                    found.append((relabel.get(i) or RENAME.get(row.NAME_EN, row.NAME_EN), "country", shapely.union_all(own)))
            found += [(text, "disputed", poly) for text, poly, _ in extra]
        if "territories" in label:
            found += [(c.text, "territory", shapely.union_all(c.polys)) for c in children]
    kinds = kinds_from_labels(label)
    if kinds:
        found += [(b.text, WATER_KIND[b.kind], b.geom) for b in load_water_bodies(kinds, frame.project(load_lakes()), frame)]
    lonlat = gpd.GeoSeries([g for _, _, g in found], crs=frame.crs).to_crs("EPSG:4326")
    return [Candidate(text, kind, g) for (text, kind, _), g in zip(found, lonlat)]


def fold(s: str) -> str:
    """For matching: no accents, no case."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).casefold()


def matches(name: str, cands: list[Candidate]) -> list[Candidate]:
    """The candidates whose text contains `name` (ignoring case and accents): exact matches first, then the rest,
    each in KINDS order, then by text."""
    key = fold(name)
    found = [c for c in cands if key in fold(c.text)]
    return sorted(found, key=lambda c: (fold(c.text) != key, KINDS.index(c.kind), c.text))


def describe(c: Candidate) -> str:
    """'Colorado (river) 109.5°W 36.2°N': the text, the kind, and a point on it."""
    if shapely.get_dimensions(c.geom) == 1:  # a river: the middle of its longest line
        pt = max(shapely.get_parts(c.geom), key=lambda g: g.length).interpolate(0.5, normalized=True)
    else:
        pt = c.geom.representative_point()
    lon, lat = pt.x, pt.y
    return f"{c.text} ({c.kind}) {abs(lon):.1f}°{'E' if lon >= 0 else 'W'} {abs(lat):.1f}°{'N' if lat >= 0 else 'S'}"


def choose(name: str, found: list[Candidate]) -> list[Candidate]:
    """The candidate(s) to crop to: the only match, or those the user checks when several match. Without a terminal
    to ask on, several matches are an error that lists them."""
    if not found:
        sys.exit(f"make_map: --crop {name}: no label matches (only the kinds in --label are searched)")
    if len(found) == 1:
        return found
    lines = [describe(c) for c in found]
    if not sys.stdin.isatty():
        sys.exit(f"make_map: --crop {name} matches several labels; run in a terminal to choose, or use lon,lat:lon,lat:\n"
                 + "\n".join(lines))
    import questionary

    picked = questionary.checkbox(f"--crop {name} matches several labels. Crop to:", choices=lines,
                                  validate=lambda chosen: bool(chosen) or "choose at least one").ask()
    if not picked:
        sys.exit("make_map: --crop: nothing chosen")
    return [c for c, line in zip(found, lines) if line in picked]
