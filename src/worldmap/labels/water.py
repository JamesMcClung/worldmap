"""Water labels (--label oceans, seas, lakes, water): which water bodies get a name, the curve each name follows,
and its glyphs laid out along it. Bodies are in projected map units; labels are in poster pixels."""

from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import shapely
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from worldmap.data import load_marine, project

WATER_KINDS = ("oceans", "seas", "lakes", "water")  # the --label values; water = all of them plus straits
Kind = Literal["oceans", "seas", "lakes", "straits"]
ALL_KINDS: set[Kind] = {"oceans", "seas", "lakes", "straits"}
SEA: Kind = "seas"
CLASSES: dict[str, Kind] = {"ocean": "oceans", "strait": "straits", "channel": "straits"}
CLASSES |= {c: SEA for c in ("sea", "gulf", "bay", "sound", "fjord", "lagoon", "inlet")}  # reefs, river mouths and the unnamed 'generic' areas are never labelled
OCEAN_TEXT = {"Atlantic": "ATLANTIC OCEAN", "Pacific": "PACIFIC OCEAN", "Indian": "INDIAN OCEAN", "Arctic": "ARCTIC OCEAN", "Southern": "SOUTHERN OCEAN"}


def kinds_from_labels(label: set[str]) -> set[Kind]:
    """The water kinds asked for in --label (countries and territories are not water)."""
    if "water" in label:
        return set(ALL_KINDS)
    return {k for k in ALL_KINDS if k in label}


def classify(featurecla: str) -> Kind | None:
    """The label kind of a Natural Earth marine feature class, or None if it never gets a label."""
    return CLASSES.get(featurecla)


@dataclass
class WaterBody:
    text: str
    kind: Kind
    rank: int  # Natural Earth scalerank: 0 = oceans, 1 = major seas, ... lower is more important
    geom: Polygon  # the piece the label goes on, in map units


def largest(geom: BaseGeometry) -> Polygon:
    return max(shapely.get_parts(geom), key=lambda g: g.area)


def ocean_word(name: str) -> str:
    return next(w for w in OCEAN_TEXT if w.lower() in name.lower())


def load_water_bodies(kinds: set[Kind], lakes_projected: gpd.GeoDataFrame) -> list[WaterBody]:
    """The water bodies of `kinds`, most important first: oceans, then by scalerank, then by area. North and South
    Atlantic are one ATLANTIC; the Pacific, cut by the antimeridian, is labelled twice (left and right of x = 0).
    Every other body is labelled once, on its largest polygon."""
    marine = project(load_marine())
    marine = marine[marine.name.notna()]
    marine["geometry"] = shapely.make_valid(marine.geometry.values)  # some are self-intersecting rough areas
    marine = marine[[classify(c) in kinds for c in marine.featurecla]]
    bodies = []
    oceans = marine[marine.featurecla == "ocean"]
    for word, group in oceans.groupby(oceans.name.map(ocean_word)):
        whole = shapely.union_all(group.geometry.values)
        if word == "Pacific":
            x0, y0, x1, y1 = whole.bounds
            halves = [whole.intersection(shapely.box(x0, y0, 0, y1)), whole.intersection(shapely.box(0, y0, x1, y1))]
        else:
            halves = [whole]
        bodies += [WaterBody(OCEAN_TEXT[word], "oceans", 0, largest(h)) for h in halves]
    for row in marine[marine.featurecla != "ocean"].itertuples():
        kind = classify(row.featurecla)
        assert kind is not None
        bodies.append(WaterBody(row.name, kind, int(row.scalerank), largest(row.geometry)))
    if "lakes" in kinds:
        bodies += [WaterBody(r.name, "lakes", int(r.scalerank), largest(r.geometry)) for r in lakes_projected.itertuples() if isinstance(r.name, str)]
    bodies.sort(key=lambda b: (b.kind != "oceans", b.rank, -b.geom.area))
    seen: set[str] = set()
    unique = []
    for b in bodies:  # oceans are named alike on purpose (the two Pacifics); others once, marine before lakes
        if b.kind == "oceans" or b.text not in seen:
            unique.append(b)
            seen.add(b.text)
    return unique
