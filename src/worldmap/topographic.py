"""--type topographic: heights and depths from ETOPO 2022, hill-shaded, with contours, ice and lakes."""

import json
import os
from dataclasses import dataclass

import numpy as np

from worldmap.data import CACHE, ETOPO, fetch

GRID_STEP_DEG = 0.05  # 3 arc-minutes: about one poster pixel at 200 dpi
ELEVATION_RULE = "etopo2022-60s-mean-3min/v1"  # bump when the reduction changes, to rebuild the cached grid


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
