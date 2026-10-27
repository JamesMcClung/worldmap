"""The map frame: projection, globe outline, canvas extent, and map <-> poster-pixel conversion."""

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from shapely import affinity
from shapely.geometry.base import BaseGeometry

from worldmap.data import CRS

POSTER_WIDTH_IN = 36
DPI = 200  # 36 in x 200 dpi = 7200 px wide
OUTLINE_POINTS = 200  # points along each side (meridian +-180) of the globe outline
ANTIMERIDIAN_EPS = 1e-6  # degrees inside +-180, so the outline doesn't wrap
CANVAS_PAD = 0.006  # canvas margin around the outline, as a fraction of its width (room for the stroke)
PX_SIMPLIFY = 2.0  # default simplification tolerance (px) for geometry converted to pixels


@dataclass(frozen=True)
class MapFrame:
    to_map: Transformer  # lon/lat -> projected map units
    to_lonlat: Transformer  # projected map units -> lon/lat
    outline: list[tuple[float, float]]  # Robinson outline of the whole globe, poles included
    xmin: float  # canvas extent in map units: the outline's bounding box plus CANVAS_PAD
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def px_per_unit(self) -> float:
        """Poster pixels per map unit."""
        return POSTER_WIDTH_IN * DPI / self.width

    def to_px(self, geom: BaseGeometry, simplify: float = PX_SIMPLIFY) -> BaseGeometry:
        """Geometry in map units -> poster pixels (origin bottom left), simplified by `simplify` px."""
        ppu = self.px_per_unit
        matrix = [ppu, 0, 0, ppu, -self.xmin * ppu, -self.ymin * ppu]
        return affinity.affine_transform(geom, matrix).simplify(simplify, preserve_topology=False)

    def px_to_map(self, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ppu = self.px_per_unit
        return px / ppu + self.xmin, py / ppu + self.ymin


def make_frame() -> MapFrame:
    to_map = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    to_lonlat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lats = np.linspace(-90, 90, OUTLINE_POINTS)
    east = [to_map.transform(180 - ANTIMERIDIAN_EPS, lat) for lat in lats]
    west = [to_map.transform(-180 + ANTIMERIDIAN_EPS, lat) for lat in lats[::-1]]
    outline = east + west
    xs, ys = np.array(outline).T
    pad = CANVAS_PAD * (xs.max() - xs.min())
    return MapFrame(to_map, to_lonlat, outline, xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad)
