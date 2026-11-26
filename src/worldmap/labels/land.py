"""Landform labels (--label mountains, uplands, deserts, lowlands, land): Natural Earth's land label areas, named
along their long direction like the seas (curved.area_label). Bodies are in projected map units."""

from collections.abc import Collection
from typing import Literal

import geopandas as gpd
import shapely
from shapely.affinity import translate

from worldmap.colorspace import RGB, fit_gamut
from worldmap.data import load_regions
from worldmap.disjoint import DisjointSets
from worldmap.frame import MapFrame
from worldmap.labels.curved import WHITE_HALO, Body, CurvedLabel, Style, area_label, largest
from worldmap.labels.priority import Unplaced

Kind = Literal["mountains", "uplands", "deserts", "lowlands"]
LAND_KINDS: tuple[Kind, ...] = ("mountains", "uplands", "deserts", "lowlands")  # their --label values (and --label land's order)
CLASSES: dict[str, Kind] = {"Range/mtn": "mountains", "Plateau": "uplands", "Desert": "deserts"}
CLASSES |= {c: "lowlands" for c in ("Plain", "Basin", "Lowland", "Valley", "Gorge", "Depression")}  # islands, capes, coasts,
# continents, regions (Siberia), deltas, wetlands, foothills and tundra are never labelled
SMALL_WORDS = {"of", "de", "del", "la", "the", "and"}  # lower case inside a title-cased name
NAMES = {  # Natural Earth NAME -> label, where its NAME_EN is wrong: cut short ('Ahag'), a single peak, a part, another feature
    "AHAGGAR MTS.": "Ahaggar Mountains", "Grampian Mts.": "Grampian Mountains", "CORD. CENTRAL": "Cordillera Central",
    "Campos Sertão": "Campos Sertão", "APPENNINI": "Apennines", "NULLARBOR PLAIN": "Nullarbor Plain",
    "CRYSTAL MOUNTAINS": "Crystal Mountains", "COLUMBIA MTS.": "Columbia Mountains", "STANOVOY UPLAND": "Stanovoy Upland",
}
GENERIC_WORDS = {"mountains", "basin", "steppe", "coastal", "plain"}  # capitalised after the first word ('Amazon basin')
_land_text = fit_gamut(0.30, 0.07, 65)
assert _land_text is not None
LAND_TEXT: RGB = _land_text[0]  # a dark warm brown, terrain rather than water or a country's hue
LAND_MIN_PT, LAND_MAX_PT = 6.5, 14.0  # sized to fill curved.SEA_FILL of their curve, within these
STYLE = Style(True, 0.0, False, LAND_TEXT, WHITE_HALO, LAND_MIN_PT, LAND_MAX_PT)  # every kind's: no capitals, whatever the size
JOIN_DEG = 1.0  # rows of one name this close (degrees) are one feature (split at +-180, or in pieces); further, separate ones


def classify(featurecla: str) -> Kind | None:
    """The label kind of a Natural Earth region feature class, or None if it never gets a label."""
    return CLASSES.get(featurecla)


def kinds_from_labels(label: Collection[str]) -> set[Kind]:
    """The land kinds asked for in --label."""
    return {k for k in LAND_KINDS if k in label}


def text_of(name_en: object, name: str) -> str:
    """The label text: NAMES' for a few, else the English name, else the name; one in capitals in the data (Natural
    Earth styles some that way: CANADIAN SHIELD) in title case, small words after the first lower case; generic words
    after the first capitalised (Amazon Basin)."""
    text = NAMES.get(name) or (name_en if isinstance(name_en, str) and name_en else name)
    words = text.split(" ")
    if not text.isupper():
        return " ".join(w.capitalize() if k and w in GENERIC_WORDS else w for k, w in enumerate(words))
    return " ".join(w.lower() if k and w.lower() in SMALL_WORDS else w[:1].upper() + w[1:].lower() for k, w in enumerate(words))


def features(regions: gpd.GeoDataFrame) -> list[int]:
    """Each row's feature (an index into the rows): rows of one text within JOIN_DEG of each other, across +-180 too,
    are one feature (the Transantarctic Mountains' halves, the Brazilian Highlands' parts); further apart, separate
    ones (the Sierra Nevadas of Spain and the US)."""
    sets = DisjointSets(len(regions))
    geoms = list(regions.geometry)
    for rows in regions.groupby("text").indices.values():
        for a, i in enumerate(rows):
            for j in rows[a + 1:]:
                if min(geoms[i].distance(translate(geoms[j], shift)) for shift in (-360, 0, 360)) <= JOIN_DEG:
                    sets.union(int(j), int(i))
    return [sets.find(i) for i in range(len(regions))]


def load_land_bodies(kinds: set[Kind], frame: MapFrame) -> list[Body]:
    """The landforms of `kinds` the frame shows, biggest first (full_size: the whole feature's km2, however much is
    shown), each labelled on its largest visible polygon; a feature in several rows (`features`) is one body."""
    regions = load_regions()
    regions = regions[regions.featurecla.map(classify).isin(kinds)].reset_index(drop=True)  # not a list: an empty one would select columns
    regions["text"] = [text_of(e, n) for e, n in zip(regions.name_en, regions.name)]
    regions["feature"] = features(regions)
    shown = frame.visible(frame.project(regions))
    shown = shown[~shown.geometry.is_empty]
    shown["geometry"] = shapely.make_valid(shown.geometry.values)
    bodies = []
    for feature, rows in shown.groupby("feature"):
        whole = regions[regions.feature == feature]
        biggest = whole.loc[whole.km2.idxmax()]
        kind = classify(biggest.featurecla)
        assert kind is not None
        polygon = largest(shapely.union_all(rows.geometry.values))
        bodies.append(Body(biggest.text, kind, int(whole.scalerank.min()), polygon, float(whole.km2.sum()), STYLE))
    bodies.sort(key=lambda b: -b.full_size)
    return bodies


def make_land_labels(bodies: list[Body], frame: MapFrame) -> list[CurvedLabel | Unplaced]:
    """A curved label per body, in the same order; an Unplaced where the name has no room (curved.area_label)."""
    labels = [area_label(b, frame) for b in bodies]
    return [Unplaced(b.text, b.kind, b.full_size) if label is None else label for b, label in zip(bodies, labels)]
