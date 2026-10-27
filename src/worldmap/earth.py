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

GR_DESERT, GR_LUSH = -12.0, 20.0
MIN_PIX = 6  # polygons with fewer raster pixels borrow from a window around their centroid, then from their country
SAND, GREEN, ICE = (0.85, 0.09, 78), (0.84, 0.095, 135), "#eef3f4"  # OKLCH (L, C, hue deg)

Fill = RGB | str  # an RGB tuple or a matplotlib colour string
Regions = dict[int, list[tuple[Polygon, Fill]]]  # country index -> [(projected polygon, fill)]


def earth_values(countries: gpd.GeoDataFrame) -> list[list[list[float]]]:
    """[[(mean G-R, icy fraction) per polygon] per country], in the polygon order of `project(countries)`. Cached in earth.json."""
    cache = os.path.join(CACHE, "earth.json")
    raster = fetch(RASTER)
    key = {"raster": os.path.getsize(raster), "countries": os.path.getsize(fetch(COUNTRIES)), "rule": "v1", "min_pix": MIN_PIX}
    if os.path.exists(cache):
        d = json.load(open(cache))
        if d["key"] == key:
            return d["vals"]
    from PIL import Image, ImageDraw

    Image.MAX_IMAGE_PIXELS = None
    im = np.asarray(Image.open(raster).convert("RGB"))
    H, W = im.shape[:2]

    def stats(px):
        px = px[px.min(1) < 250].astype(int)  # white = ocean
        if len(px) == 0:
            return None
        R, G, B = px.T
        return float(np.mean(G - R)), float(((B >= G) & (B >= R)).mean()), len(px)

    def to_img(co):
        return [((x + 180) / 360 * W, (90 - y) / 180 * H) for x, y in co]

    vals = []
    for geom in countries.geometry.intersection(box(-180, -90, 180, 90)):
        polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        got, pool = [], []
        for p in polys:
            if p.is_empty or p.geom_type != "Polygon":
                got.append(None)
                continue
            x0, y0, x1, y1 = p.bounds
            X0, X1 = max(0, int((x0 + 180) / 360 * W)), min(W, int((x1 + 180) / 360 * W) + 2)
            Y0, Y1 = max(0, int((90 - y1) / 180 * H)), min(H, int((90 - y0) / 180 * H) + 2)
            mk = Image.new("L", (X1 - X0, Y1 - Y0), 0)
            dr = ImageDraw.Draw(mk)
            dr.polygon([(a - X0, b - Y0) for a, b in to_img(p.exterior.coords)], fill=1)
            for r in p.interiors:
                dr.polygon([(a - X0, b - Y0) for a, b in to_img(r.coords)], fill=0)
            px = im[Y0:Y1, X0:X1][np.asarray(mk, bool)]
            pool.append(px)
            st = stats(px)
            if st is None or st[2] < MIN_PIX:  # tiny: look at a window around the centroid instead
                cx, cy = to_img([p.representative_point().coords[0]])[0]
                cx, cy = int(min(max(cx, 3), W - 4)), int(min(max(cy, 3), H - 4))
                st = stats(im[cy - 3 : cy + 4, cx - 3 : cx + 4].reshape(-1, 3))
            got.append(st)
        whole = stats(np.concatenate(pool)) if pool else None
        vals.append([list(st[:2]) if st else (list(whole[:2]) if whole else [10.0, 0.0]) for st in got])
    json.dump({"key": key, "vals": vals}, open(cache, "w"))
    return vals


def earth_fill(green_minus_red: float, ice_fraction: float) -> RGB:
    g = float(np.clip((green_minus_red - GR_DESERT) / (GR_LUSH - GR_DESERT), 0, 1))
    f = float(np.clip((ice_fraction - 0.15) / 0.6, 0, 1))  # bluish-white land = ice cap
    L, C, h = (1 - g) * np.array(SAND) + g * np.array(GREEN)  # hue 78 -> 135 interpolates through yellow-olive, no wrap
    rgb, _ = fit_gamut(L, C, h)
    r, gr, b = (1 - f) * np.array(rgb) + f * np.array(to_rgb(ICE))
    return (r, gr, b)


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
