"""The map frame: projection, globe outline, canvas extent, and map <-> poster-pixel conversion."""

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from worldmap.data import CRS

POSTER_WIDTH_IN = 36
DPI = 200  # 36 in x 200 dpi = 7200 px wide
OUTLINE_POINTS = 200  # points along each side (meridian +-180) of the globe outline
ANTIMERIDIAN_EPS = 1e-6  # degrees inside +-180, so the outline doesn't wrap
CANVAS_PAD = 0.006  # canvas margin around the outline, as a fraction of its width (room for the stroke)
PX_SIMPLIFY = 2.0  # default simplification tolerance (px) for geometry converted to pixels
PROJECT_PAD = 0.05  # a crop keeps projected data this far (fraction of its width/height) outside the canvas

LonLatBox = tuple[float, float, float, float]  # lon0, lat0, lon1, lat1 with lon0 <= lon1, within +-180 / +-90
WORLD: LonLatBox = (-180, -90, 180, 90)


@dataclass(frozen=True)
class MapFrame:
    to_map: Transformer  # lon/lat -> projected map units
    to_lonlat: Transformer  # projected map units -> lon/lat
    outline: list[tuple[float, float]]  # Robinson outline of the whole globe, poles included
    xmin: float  # canvas extent in map units: the outline's bounding box plus CANVAS_PAD
    ymin: float
    xmax: float
    ymax: float
    crs: str = CRS
    globe: bool = True  # the whole globe; False for a --crop, whose outline is the canvas rectangle
    clip: tuple[LonLatBox, ...] = (WORLD,)  # lon/lat data is clipped to these boxes before projecting

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def width_in(self) -> float:
        """Poster width in inches: the long side is POSTER_WIDTH_IN."""
        return POSTER_WIDTH_IN if self.width >= self.height else POSTER_WIDTH_IN * self.width / self.height

    @property
    def height_in(self) -> float:
        return POSTER_WIDTH_IN * self.height / self.width if self.width >= self.height else POSTER_WIDTH_IN

    @property
    def px_per_unit(self) -> float:
        """Poster pixels per map unit."""
        return self.width_in * DPI / self.width

    @property
    def units_per_in(self) -> float:
        """Map units per poster inch."""
        return DPI / self.px_per_unit

    @property
    def canvas(self) -> Polygon:
        return box(self.xmin, self.ymin, self.xmax, self.ymax)

    def project(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Clip lon/lat data to the frame's boxes and project it. A crop also cuts it a little outside the canvas, so
        the cut edges are never seen."""
        gdf = gdf.copy()
        clip = box(*self.clip[0]) if len(self.clip) == 1 else shapely.union_all([box(*b) for b in self.clip])
        gdf["geometry"] = gdf.geometry.intersection(clip)
        gdf = gdf.to_crs(self.crs)
        if not self.globe:
            px, py = PROJECT_PAD * self.width, PROJECT_PAD * self.height
            gdf["geometry"] = gdf.geometry.intersection(box(self.xmin - px, self.ymin - py, self.xmax + px, self.ymax + py))
        return gdf

    def visible(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Projected data cut to the canvas (what labels may use); the whole globe is all visible."""
        if self.globe:
            return gdf
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.intersection(self.canvas)
        return gdf

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
