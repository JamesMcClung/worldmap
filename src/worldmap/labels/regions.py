"""Child regions, labelled only with --regions: dependencies that don't touch their parent (Greenland, Bermuda...)
and hand-picked outlying parts of a country's own polygons (Alaska, Tasmania, Kaliningrad...)."""

from dataclasses import dataclass

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from worldmap.data import parent_index
from worldmap.frame import MapFrame
from worldmap.labels.label import RENAME

Box = tuple[float, float, float, float]  # lon0, lat0, lon1, lat1

TOUCH_TOLERANCE_DEG = 0.01  # a dependency this close to its parent counts as attached (Hong Kong, Macau): no label

# (label name, parent NAME_EN, lon/lat boxes): the parent's polygons whose representative point lies in a box.
HAND_PICKED: list[tuple[str, str, list[Box]]] = [
    ("Kaliningrad", "Russia", [(19.5, 54.2, 23.0, 55.4)]),
    ("French Guiana", "France", [(-55, 2, -51, 6)]),
    ("Alaska", "United States of America", [(-180, 51, -129, 72), (170, 50, 180, 56)]),
    ("Hawaii", "United States of America", [(-179, 18, -154, 29)]),
    ("Svalbard", "Norway", [(10, 74, 35, 81)]),
    ("Tasmania", "Australia", [(143.5, -44, 149, -39.5)]),
    ("Galápagos", "Ecuador", [(-92.5, -1.6, -89, 1.7)]),
    ("Easter Island", "Chile", [(-109.6, -27.3, -109.1, -26.9)]),
    ("Cabinda", "Angola", [(11.5, -6, 13.2, -4.3)]),
    ("Nakhchivan", "Azerbaijan", [(44.7, 38.8, 46.7, 39.8)]),
    ("Oecusse", "East Timor", [(124, -9.5, 124.6, -9.1)]),
    ("Musandam", "Oman", [(55.9, 25.6, 56.6, 26.5)]),
    ("Canary Islands", "Spain", [(-18.5, 27.5, -13.3, 29.5)]),
    ("Azores", "Portugal", [(-31.5, 36.8, -24.9, 40)]),
    ("Madeira", "Portugal", [(-17.4, 32.3, -16.2, 33.2)]),
    ("Zanzibar", "Tanzania", [(39, -6.6, 40, -4.8)]),
    ("Socotra", "Yemen", [(52, 11.9, 54.6, 12.8)]),
    ("Andaman & Nicobar", "India", [(92, 6.5, 94, 14)]),
    ("Bioko", "Equatorial Guinea", [(8.4, 3.1, 9.0, 3.9)]),
    ("Réunion", "France", [(55.1, -21.5, 55.9, -20.8)]),
    ("Guadeloupe", "France", [(-61.9, 15.8, -61.0, 16.6)]),
    ("Martinique", "France", [(-61.3, 14.3, -60.8, 14.9)]),
]


@dataclass
class ChildRegion:
    name: str  # e.g. "Tasmania", "Falkland Islands"
    parent_name: str  # the parent's label text, e.g. "United States"
    feature: int  # row whose fill colours it: the dependency itself, or the parent for a hand-picked region
    polys: list[Polygon]  # projected polygons it labels

    @property
    def text(self) -> str:
        return f"{self.name} ({self.parent_name})"


def polygons(geom: BaseGeometry) -> list[Polygon]:
    return [p for p in (geom.geoms if isinstance(geom, MultiPolygon) else [geom]) if not p.is_empty]


def child_regions(countries: gpd.GeoDataFrame, countries_projected: gpd.GeoDataFrame, frame: MapFrame) -> list[ChildRegion]:
    """Dependencies not attached to their parent, then the HAND_PICKED regions."""

    def label_name(i: int) -> str:
        name = countries.NAME_EN[i]
        return RENAME.get(name, name)

    regions = []
    for dep, parent in parent_index(countries).items():
        if countries.geometry[dep].intersects(countries.geometry[parent].buffer(TOUCH_TOLERANCE_DEG)):
            continue
        polys = polygons(countries_projected.geometry[dep])
        if polys:
            regions.append(ChildRegion(countries.NAME_EN[dep], label_name(parent), dep, polys))
    index_of = dict(zip(countries.NAME_EN, countries.index))
    for name, parent_name, boxes in HAND_PICKED:
        parent = index_of[parent_name]
        picked = []
        for p in polygons(countries_projected.geometry[parent]):
            lon, lat = frame.to_lonlat.transform(*p.representative_point().coords[0])
            if any(lon0 <= lon <= lon1 and lat0 <= lat <= lat1 for lon0, lat0, lon1, lat1 in boxes):
                picked.append(p)
        if not picked:
            raise ValueError(f"hand-picked region {name!r} selected no polygons of {parent_name!r}: check its HAND_PICKED boxes")
        regions.append(ChildRegion(name, label_name(parent), parent, picked))
    return regions
