"""--type topographic: heights and depths from ETOPO 2022, hill-shaded, with contours, ice and lakes."""

import json
import os
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon

from worldmap.coloring import OCEAN
from worldmap.data import CACHE, ETOPO, fetch, load_ice, load_lakes, project
from worldmap.frame import DPI, POSTER_WIDTH_IN, MapFrame

GRID_STEP_DEG = 0.05  # 3 arc-minutes: about one poster pixel at 200 dpi
ELEVATION_RULE = "etopo2022-60s-mean-3min/v1"  # bump when the reduction changes, to rebuild the cached grid

LAND_TOP, ICE_TOP, SEA_BOTTOM = 6000, 4000, 8000  # metres at the top (bottom) of each colour scale; beyond: its end colour
LAND_RAMP = LinearSegmentedColormap.from_list("land", [(0, "#5f9e5a"), (0.08, "#a8c97f"), (0.2, "#e8dc95"), (0.4, "#c9a26b"),
                                                        (0.65, "#9a6b4a"), (0.85, "#d8cfc8"), (1, "#ffffff")])
ICE_RAMP = LinearSegmentedColormap.from_list("ice", ["#ffffff", "#cfe3f2"])
SEA_RAMP = LinearSegmentedColormap.from_list("sea", [OCEAN, "#2f5f8c"])
LIGHT_AZIMUTH, LIGHT_ALTITUDE = 315, 45  # degrees: lit from the north-west
LAND_SHADE, SEA_SHADE = 1.0, 0.4  # how strongly slopes lighten/darken land and ice, and the sea floor (--ocean)
# Contours: (spacing in metres, colour, alpha); width CONTOUR_WIDTH points.
LAND_CONTOURS = (1000, "#5a4a3a", 0.6)
ICE_CONTOURS = (1000, "#7a8fa3", 0.6)
SEA_CONTOURS = (2000, "#1f4466", 0.5)
CONTOUR_WIDTH = 0.3
VERT_EXAG = 6.0  # hill-shading exaggeration of heights (z and pixel size both in metres); tuned on crops


@dataclass
class Grid:
    z: np.ndarray  # int16 metres, row 0 = northmost; cell (i, j) is centred on (-180 + (j + .5) step, 90 - (i + .5) step)
    step: float  # degrees per cell


def elevation_grid() -> Grid:
    """ETOPO reduced to GRID_STEP_DEG cells (the mean of each block), cached in .cache/etopo_3min.npy."""
    cache, key_file = CACHE / "etopo_3min.npy", CACHE / "etopo_3min.json"
    key = {"etopo": os.path.getsize(fetch(ETOPO)), "rule": ELEVATION_RULE}
    if cache.exists() and key_file.exists() and json.loads(key_file.read_text()) == key:
        return Grid(np.load(cache), GRID_STEP_DEG)
    import rasterio
    from rasterio.enums import Resampling

    rows, cols = round(180 / GRID_STEP_DEG), round(360 / GRID_STEP_DEG)
    with rasterio.open(fetch(ETOPO)) as src:
        z = src.read(1, out_shape=(rows, cols), resampling=Resampling.average)
    z = np.round(z).astype(np.int16)
    np.save(cache, z)
    key_file.write_text(json.dumps(key))
    return Grid(z, GRID_STEP_DEG)


def sample(grid: Grid, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Heights at (lon, lat) in metres (float32), bilinear between cell centres; longitude wraps across +-180."""
    rows, cols = grid.z.shape
    fx = (np.asarray(lon, np.float64) + 180) / grid.step - 0.5
    fy = np.clip((90 - np.asarray(lat, np.float64)) / grid.step - 0.5, 0, rows - 1)
    x0, y0 = np.floor(fx).astype(np.int64), np.minimum(np.floor(fy).astype(np.int64), rows - 2)
    tx, ty = (fx - x0).astype(np.float32), (fy - y0).astype(np.float32)
    x0, x1 = x0 % cols, (x0 + 1) % cols
    z = grid.z
    top = z[y0, x0] * (1 - tx) + z[y0, x1] * tx
    bottom = z[y0 + 1, x0] * (1 - tx) + z[y0 + 1, x1] * tx
    return (top * (1 - ty) + bottom * ty).astype(np.float32)


def colorize(z: np.ndarray, ice: np.ndarray, ocean: bool) -> np.ndarray:
    """RGB (float32, z.shape + (3,)) by height: land (z > 0) on LAND_RAMP, `ice` on ICE_RAMP (ice wins over land and
    sea), sea on SEA_RAMP by depth with `ocean`, else flat OCEAN."""
    land = LAND_RAMP(np.clip(z / LAND_TOP, 0, 1))[..., :3]
    icy = ICE_RAMP(np.clip(z / ICE_TOP, 0, 1))[..., :3]
    sea = SEA_RAMP(np.clip(-z / SEA_BOTTOM, 0, 1))[..., :3] if ocean else np.broadcast_to(to_rgb(OCEAN), land.shape)
    rgb = np.where(ice[..., None], icy, np.where((z > 0)[..., None], land, sea))
    return rgb.astype(np.float32)


def relief(z: np.ndarray, pixel_m: float) -> np.ndarray:
    """How much brighter (> 0) or darker (< 0) each cell's slope is than flat ground, lit from LIGHT_AZIMUTH at
    LIGHT_ALTITUDE (Lambert). Row 0 is north. Not rescaled to the extremes, so a region looks the same on any crop."""
    dz_south, dz_east = np.gradient(z.astype(np.float32) * VERT_EXAG, pixel_m)
    az, alt = np.radians(LIGHT_AZIMUTH), np.radians(LIGHT_ALTITUDE)
    light = (np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt))  # east, north, up
    # The surface normal is (-dz/dx_east, -dz/dy_north, 1) = (-dz_east, dz_south, 1), normalised.
    lit = (-dz_east * light[0] + dz_south * light[1] + light[2]) / np.sqrt(dz_east ** 2 + dz_south ** 2 + 1)
    return (lit - light[2]).astype(np.float32)


def shade(rgb: np.ndarray, z: np.ndarray, pixel_m: float, sea: np.ndarray, ocean: bool) -> np.ndarray:
    """Hill shading: land and ice by LAND_SHADE, the sea floor by SEA_SHADE with `ocean` (a flat sea stays flat).
    `pixel_m` is the grid spacing in metres."""
    r = relief(z, pixel_m)[..., None]
    shaded = np.where(sea[..., None], rgb * (1 + SEA_SHADE * r) if ocean else rgb, rgb * (1 + LAND_SHADE * r))
    return np.clip(shaded, 0, 1).astype(np.float32)


@dataclass
class PosterGrid:
    x: np.ndarray  # pixel-centre map x, ascending (1-D)
    y: np.ndarray  # pixel-centre map y, descending (1-D): row 0 is the top of the poster
    lon: np.ndarray  # float32 (rows, cols)
    lat: np.ndarray
    inside: np.ndarray  # bool: the pixel is on the globe


def poster_grid(frame: MapFrame, dpi: float) -> PosterGrid:
    """One cell per output pixel at min(dpi, DPI) (the elevation grid has no finer detail), with each cell's lon/lat."""
    cols = round(POSTER_WIDTH_IN * min(dpi, DPI))
    rows = round(cols * frame.height / frame.width)
    pw, ph = frame.width / cols, frame.height / rows
    x = frame.xmin + (np.arange(cols) + 0.5) * pw
    y = frame.ymax - (np.arange(rows) + 0.5) * ph
    lon = np.empty((rows, cols), np.float32)
    lat = np.empty((rows, cols), np.float32)
    inside = np.empty((rows, cols), bool)
    for r0 in range(0, rows, 256):  # in bands, to keep the float64 temporaries small
        X, Y = np.meshgrid(x, y[r0:r0 + 256])
        lo, la = frame.to_lonlat.transform(X, Y)
        ok = np.isfinite(lo) & np.isfinite(la) & (np.abs(lo) <= 180) & (np.abs(la) <= 90)
        lo, la = np.where(ok, lo, 0), np.where(ok, la, 0)
        bx, by = frame.to_map.transform(lo, la)
        inside[r0:r0 + 256] = ok & (np.abs(bx - X) < pw) & (np.abs(by - Y) < ph)
        lon[r0:r0 + 256], lat[r0:r0 + 256] = lo, la
    return PosterGrid(x, y, lon, lat, inside)


def ice_mask(ice_projected: gpd.GeoDataFrame, frame: MapFrame, pg: PosterGrid) -> np.ndarray:
    """Which poster pixels are ice (glaciers, ice sheets, ice shelves), holes cleared."""
    from PIL import Image, ImageDraw

    rows, cols = pg.inside.shape
    sx, sy = cols / frame.width, rows / frame.height
    mask = Image.new("L", (cols, rows), 0)
    draw = ImageDraw.Draw(mask)

    def to_px(coords) -> list[tuple[float, float]]:
        return [((cx - frame.xmin) * sx, (frame.ymax - cy) * sy) for cx, cy in coords]

    for geom in ice_projected.geometry:
        for poly in geom.geoms if isinstance(geom, MultiPolygon) else [geom]:
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            draw.polygon(to_px(poly.exterior.coords), fill=1)
            for ring in poly.interiors:
                draw.polygon(to_px(ring.coords), fill=0)
    return np.asarray(mask, bool)


def draw_topography(ax: Axes, frame: MapFrame, dpi: float, ocean: bool) -> None:
    """Everything on the globe for --type topographic: the coloured, hill-shaded heights (sea depths with `ocean`),
    contours, the coastline and the lakes."""
    from worldmap.render import BORDER, BORDER_WIDTH, Z_BORDERS, Z_COUNTRY, Z_LAKES, holed_patches

    pg = poster_grid(frame, dpi)
    z = np.where(pg.inside, sample(elevation_grid(), pg.lon, pg.lat), 0).astype(np.float32)
    ice = ice_mask(project(load_ice()), frame, pg) & pg.inside
    sea = (z <= 0) & ~ice
    rgb = shade(colorize(z, ice, ocean), z, frame.width / len(pg.x), sea, ocean)
    rgba = np.dstack([rgb, pg.inside.astype(np.float32)])
    extent = (frame.xmin, frame.xmax, frame.ymin, frame.ymax)
    image = ax.imshow(rgba, extent=extent, interpolation="nearest", zorder=Z_COUNTRY)
    globe = MplPolygon(frame.outline, closed=True, fc="none", ec="none", transform=ax.transData)
    image.set_clip_path(globe)

    def contours(where: np.ndarray, levels: np.ndarray, color: str, alpha: float) -> None:
        if levels.size:
            ax.contour(pg.x, pg.y, np.ma.masked_where(~where, z), levels=levels, colors=color,
                       linewidths=CONTOUR_WIDTH, alpha=alpha, zorder=Z_COUNTRY + 0.1)

    land = (z > 0) & ~ice & pg.inside
    step, color, alpha = LAND_CONTOURS
    contours(land, np.arange(step, z.max() + 1, step), color, alpha)
    step, color, alpha = ICE_CONTOURS
    contours(ice, np.arange(step, z.max() + 1, step), color, alpha)
    if ocean:
        step, color, alpha = SEA_CONTOURS
        contours(sea & pg.inside, -np.arange(step, -z.min() + 1, step)[::-1], color, alpha)
    coast = np.ma.masked_where(~pg.inside, np.where(ice, np.maximum(z, 1), z))
    ax.contour(pg.x, pg.y, coast, levels=[0], colors=BORDER, linewidths=BORDER_WIDTH, zorder=Z_BORDERS)
    for g in project(load_lakes()).geometry:
        ax.add_collection(PatchCollection(list(holed_patches(g)), fc=OCEAN, ec=BORDER, lw=BORDER_WIDTH, zorder=Z_LAKES))
