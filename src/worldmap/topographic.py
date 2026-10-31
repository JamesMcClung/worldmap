"""--type topographic: heights and depths from ETOPO 2022, hill-shaded, with contours, ice and lakes."""

import json
import os
from dataclasses import dataclass

import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap, to_rgb

from worldmap.coloring import OCEAN
from worldmap.data import CACHE, ETOPO, fetch

GRID_STEP_DEG = 0.05  # 3 arc-minutes: about one poster pixel at 200 dpi
ELEVATION_RULE = "etopo2022-60s-mean-3min/v1"  # bump when the reduction changes, to rebuild the cached grid

LAND_TOP, ICE_TOP, SEA_BOTTOM = 6000, 4000, 8000  # metres at the top (bottom) of each colour scale; beyond: its end colour
LAND_RAMP = LinearSegmentedColormap.from_list("land", [(0, "#5f9e5a"), (0.08, "#a8c97f"), (0.2, "#e8dc95"), (0.4, "#c9a26b"),
                                                        (0.65, "#9a6b4a"), (0.85, "#d8cfc8"), (1, "#ffffff")])
ICE_RAMP = LinearSegmentedColormap.from_list("ice", ["#ffffff", "#cfe3f2"])
SEA_RAMP = LinearSegmentedColormap.from_list("sea", [OCEAN, "#2f5f8c"])
LIGHT = LightSource(azdeg=315, altdeg=45)  # lit from the north-west
VERT_EXAG = 0.05  # hill-shading exaggeration of heights (z and pixel size both in metres); tuned on crops


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


def shade(rgb: np.ndarray, z: np.ndarray, pixel_m: float, sea: np.ndarray, ocean: bool) -> np.ndarray:
    """Hill shading lit from the north-west: land and ice strongly, the sea floor lightly with `ocean` (a flat sea
    stays flat). `pixel_m` is the grid spacing in metres."""
    hs = LIGHT.hillshade(z, vert_exag=VERT_EXAG, dx=pixel_m, dy=pixel_m).astype(np.float32)[..., None]
    shaded = np.where(sea[..., None], rgb * (0.85 + 0.2 * hs) if ocean else rgb, rgb * (0.65 + 0.45 * hs))
    return np.clip(shaded, 0, 1).astype(np.float32)
