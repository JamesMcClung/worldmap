"""Landform labels (--label mountains, uplands, deserts, lowlands, land): Natural Earth's land label areas, named
along their long direction like the seas (curved.area_label). Bodies are in projected map units."""

from collections.abc import Collection
from typing import Literal

import shapely

from worldmap.data import load_regions
from worldmap.frame import MapFrame
from worldmap.labels.curved import Body, CurvedLabel, area_label, largest
from worldmap.labels.priority import Unplaced

Kind = Literal["mountains", "uplands", "deserts", "lowlands"]
LAND_KINDS: tuple[Kind, ...] = ("mountains", "uplands", "deserts", "lowlands")  # their --label values (and --label land's order)
CLASSES: dict[str, Kind] = {"Range/mtn": "mountains", "Plateau": "uplands", "Desert": "deserts"}
CLASSES |= {c: "lowlands" for c in ("Plain", "Basin", "Lowland", "Valley", "Gorge", "Depression")}  # islands, capes, coasts,
# continents, regions (Siberia), deltas, wetlands, foothills and tundra are never labelled
SMALL_WORDS = {"of", "de", "del", "la", "the", "and"}  # lower case inside a title-cased name


def classify(featurecla: str) -> Kind | None:
    """The label kind of a Natural Earth region feature class, or None if it never gets a label."""
    return CLASSES.get(featurecla)


def kinds_from_labels(label: Collection[str]) -> set[Kind]:
    """The land kinds asked for in --label."""
    return {k for k in LAND_KINDS if k in label}


def text_of(name_en: object, name: str) -> str:
    """The label text: the English name, else the name; one in capitals in the data (Natural Earth styles some
    that way: CANADIAN SHIELD) in title case, small words after the first lower case."""
    text = name_en if isinstance(name_en, str) and name_en else name
    if not text.isupper():
        return text
    words = text.split(" ")
    return " ".join(w.lower() if k and w.lower() in SMALL_WORDS else w[:1].upper() + w[1:].lower() for k, w in enumerate(words))


def load_land_bodies(kinds: set[Kind], frame: MapFrame) -> list[Body]:
    """The landforms of `kinds` the frame shows, biggest first (full_size: the whole feature's km2, however much is
    shown), each labelled on its largest visible polygon; of two with one name, the bigger."""
    regions = load_regions()
    regions = regions[regions.featurecla.map(classify).isin(kinds)]  # not a list: an empty one would select columns
    shown = frame.visible(frame.project(regions))
    shown = shown[~shown.geometry.is_empty]
    shown["geometry"] = shapely.make_valid(shown.geometry.values)
    bodies = []
    for row in shown.itertuples():
        kind = classify(row.featurecla)
        assert kind is not None
        bodies.append(Body(text_of(row.name_en, row.name), kind, int(row.scalerank), largest(row.geometry), float(row.km2)))
    bodies.sort(key=lambda b: -b.full_size)
    seen: set[str] = set()
    unique = []
    for b in bodies:
        if b.text not in seen:
            unique.append(b)
            seen.add(b.text)
    return unique


def make_land_labels(bodies: list[Body], frame: MapFrame) -> list[CurvedLabel | Unplaced]:
    """A curved label per body, in the same order; an Unplaced where the name has no room (curved.area_label)."""
    labels = [area_label(b, frame) for b in bodies]
    return [Unplaced(b.text, b.kind, b.full_size) if label is None else label for b, label in zip(bodies, labels)]
