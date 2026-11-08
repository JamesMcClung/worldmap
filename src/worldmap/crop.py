"""--crop: a zoomed map of one region, in a Lambert azimuthal equal-area projection centred on it."""

import re

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from worldmap.frame import PROJECT_PAD, LonLatBox, MapFrame

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
