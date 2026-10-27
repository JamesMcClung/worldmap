"""--earth mode: per-region vegetation colours from the Natural Earth I land-cover raster (NE1_LR_LC, 16200x8100,
public domain).

Each polygon (mainland, Alaska, French Guiana, Kaliningrad, islands...) is coloured independently by the mean (G - R)
of the raster's land pixels inside it (desert tan ~ -12, rainforest green ~ +20), and by its fraction of icy
(bluish-white) pixels."""

import json
import os

import geopandas as gpd
import numpy as np
from matplotlib.colors import to_rgb
from shapely.geometry import MultiPolygon, Point, Polygon, box

from worldmap.colorspace import RGB, fit_gamut
from worldmap.data import CACHE, COUNTRIES, RASTER, fetch

GR_DESERT, GR_LUSH = -12.0, 20.0  # mean (G - R) mapped to pure sand and pure green
MIN_PIX = 6  # polygons with fewer raster pixels borrow from a window around their centroid, then from their country
WINDOW_RADIUS = 3  # that window is (2 * 3 + 1) pixels square
OCEAN_MIN_CHANNEL = 250  # pixels with every channel at least this are ocean (white)
NO_DATA_VALUES = [10.0, 0.0]  # (G - R, icy fraction) for a country with no land pixels at all
ICE_START, ICE_RANGE = 0.15, 0.6  # icy fraction where the blend toward ICE starts, and over which it completes
SAND, GREEN, ICE = (0.85, 0.09, 78), (0.84, 0.095, 135), "#eef3f4"  # OKLCH (L, C, hue deg)
EARTH_RULE = "v1"  # bump when the sampling logic changes, to invalidate earth.json

Fill = RGB | str  # an RGB tuple or a matplotlib colour string
Regions = dict[int, list[tuple[Polygon, Fill]]]  # country index -> [(projected polygon, fill)]
PixelStats = tuple[float, float, int]  # (mean G - R, icy fraction, land pixel count)


def pixel_stats(pixels: np.ndarray) -> PixelStats | None:
    """Stats of the land (non-white) pixels among `pixels` (n, 3), or None if there are none."""
    land = pixels[pixels.min(1) < OCEAN_MIN_CHANNEL].astype(int)
    if len(land) == 0:
        return None
    red, green, blue = land.T
    icy = (blue >= green) & (blue >= red)  # bluish-white
    return float(np.mean(green - red)), float(icy.mean()), len(land)


def earth_values(countries: gpd.GeoDataFrame) -> list[list[list[float]]]:
    """[[(mean G-R, icy fraction) per polygon] per country], in the polygon order of `project(countries)`. Cached in earth.json."""
    cache = CACHE / "earth.json"
    raster = fetch(RASTER)
    key = {"raster": os.path.getsize(raster), "countries": os.path.getsize(fetch(COUNTRIES)), "rule": EARTH_RULE, "min_pix": MIN_PIX}
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached["key"] == key:
            return cached["vals"]
    from PIL import Image, ImageDraw

    Image.MAX_IMAGE_PIXELS = None
    image = np.asarray(Image.open(raster).convert("RGB"))
    height, width = image.shape[:2]

    def to_pixels(coords) -> list[tuple[float, float]]:
        """lon/lat -> raster pixel coordinates (plate carree)."""
        return [((lon + 180) / 360 * width, (90 - lat) / 180 * height) for lon, lat in coords]

    values = []
    for geom in countries.geometry.intersection(box(-180, -90, 180, 90)):
        polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        per_poly: list[PixelStats | None] = []
        all_pixels = []
        for poly in polys:
            if poly.is_empty or poly.geom_type != "Polygon":
                per_poly.append(None)
                continue
            lon0, lat0, lon1, lat1 = poly.bounds
            col0, col1 = max(0, int((lon0 + 180) / 360 * width)), min(width, int((lon1 + 180) / 360 * width) + 2)
            row0, row1 = max(0, int((90 - lat1) / 180 * height)), min(height, int((90 - lat0) / 180 * height) + 2)
            mask = Image.new("L", (col1 - col0, row1 - row0), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon([(x - col0, y - row0) for x, y in to_pixels(poly.exterior.coords)], fill=1)
            for ring in poly.interiors:
                draw.polygon([(x - col0, y - row0) for x, y in to_pixels(ring.coords)], fill=0)
            pixels = image[row0:row1, col0:col1][np.asarray(mask, bool)]
            all_pixels.append(pixels)
            stats = pixel_stats(pixels)
            if stats is None or stats[2] < MIN_PIX:  # tiny: look at a window around the centroid instead
                cx, cy = to_pixels([poly.representative_point().coords[0]])[0]
                r = WINDOW_RADIUS
                cx, cy = int(min(max(cx, r), width - r - 1)), int(min(max(cy, r), height - r - 1))
                stats = pixel_stats(image[cy - r : cy + r + 1, cx - r : cx + r + 1].reshape(-1, 3))
            per_poly.append(stats)
        whole = pixel_stats(np.concatenate(all_pixels)) if all_pixels else None
        fallback = list(whole[:2]) if whole else NO_DATA_VALUES
        values.append([list(stats[:2]) if stats else fallback for stats in per_poly])
    cache.write_text(json.dumps({"key": key, "vals": values}))
    return values


def earth_fill(green_minus_red: float, ice_fraction: float) -> RGB:
    """Sand -> light green by vegetation, blended toward ICE for icy land."""
    lushness = float(np.clip((green_minus_red - GR_DESERT) / (GR_LUSH - GR_DESERT), 0, 1))
    iciness = float(np.clip((ice_fraction - ICE_START) / ICE_RANGE, 0, 1))
    L, C, h = (1 - lushness) * np.array(SAND) + lushness * np.array(GREEN)  # hue 78 -> 135 through yellow-olive, no wrap
    fitted = fit_gamut(L, C, h)
    assert fitted is not None
    r, g, b = (1 - iciness) * np.array(fitted[0]) + iciness * np.array(to_rgb(ICE))
    return (r, g, b)


def build_regions(countries_projected: gpd.GeoDataFrame, countries: gpd.GeoDataFrame) -> Regions:
    """Each country's projected polygons with their land-cover fill."""
    regions: Regions = {}
    for (i, row), values in zip(countries_projected.iterrows(), earth_values(countries)):
        geom = row.geometry
        polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        regions[i] = [(p, earth_fill(*v)) for p, v in zip(polys, values) if not p.is_empty]
        if not regions[i]:
            regions[i] = [(Polygon(), "#ffffff")]
    return regions


def fill_at(regions: Regions, i: int, pt: Point | None) -> Fill:
    """Fill of country i's polygon containing (or nearest) `pt`, or of its largest polygon if pt is None."""
    regs = regions[i]
    if pt is None:
        return max(regs, key=lambda r: r[0].area)[1]
    return min(regs, key=lambda r: r[0].distance(pt))[1]
