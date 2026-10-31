"""--disputed: areas claimed by more than one country (Kashmir, Crimea, Taiwan...), from Natural Earth's
disputed-areas file. Each area records every country involved: the one running it plus its claimants."""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import geopandas as gpd
import shapely
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from worldmap.earth import Fill

DISPUTED_GREY = "#cfcfcf"  # areas with 3+ countries, and the second stripe of breakaway areas
SNAP_M = 500  # metres: a disputed outline this close to an involved country counts as their shared border
STRIPE_WIDTH_IN = 0.035  # each stripe's width across, in inches on the 36 in poster

# Names used in NOTE_BRK -> countries-file NAME_EN.
ALIASES = {"China": "People's Republic of China", "U.S.A.": "United States of America", "U.S.A": "United States of America",
           "the United States": "United States of America", "U.K.": "United Kingdom", "Azer.": "Azerbaijan",
           "UAE": "United Arab Emirates", "the Philippines": "Philippines", "the Seychelles": "Seychelles",
           "Tokelau": "New Zealand"}  # Tokelau is a New Zealand territory with no feature of its own
HAND_CLAIMS = {  # (NAME_EN, ADMIN) -> countries (NAME_EN), for entries Natural Earth leaves without a note
    ("Kosovo", "Kosovo"): ["Kosovo", "Serbia"], ("Gaza Strip", "Palestine"): ["Palestine", "Israel"],
    ("West Bank", "Palestine"): ["Palestine", "Israel"], ("Belize", "Belize"): ["Belize", "Guatemala"],
    ("Dragonja", "Croatia"): ["Croatia", "Slovenia"], ("Vukovar Island", "Croatia"): ["Croatia", "Serbia"],
    ("Šarengrad Island", "Croatia"): ["Croatia", "Serbia"], ("Ilemi Triangle", "South Sudan"): ["Kenya", "South Sudan"],
    ("North Borneo", "Malaysia"): ["Malaysia", "Philippines"],
    ("Scarborough Shoal", "Scarborough Reef"): ["People's Republic of China", "Philippines", "Taiwan"]}
# Label text for self-governing areas whose Natural Earth name is an official-style one.
LABEL_TEXT = {"Turkish Republic of Northern Cyprus": "Northern Cyprus", "Donetsk People's Republic": "Donetsk",
              "Luhansk People's Republic": "Luhansk"}


class Note(NamedTuple):
    admin: str | None  # who runs it, if the note says
    claimants: list[str]
    self_admin: bool  # "Self admin.": it governs itself


@dataclass
class Dispute:
    name: str  # NAME_EN in the disputed-areas file
    admin: str  # its ADMIN
    countries: list[int]  # countries-file rows of everyone involved, the one running it first; a breakaway: only its parent
    breakaway: bool  # governs itself but has no countries-file feature (Abkhazia, Transnistria...)
    label: str | None  # label text for a self-governing area, else None
    geometry: BaseGeometry


def split_names(s: str) -> list[str]:
    """'Ireland, Denmark, and Iceland' -> ['Ireland', 'Denmark', 'Iceland']."""
    return [x.strip() for x in re.split(r",\s*and\s+|,|\s+and\s+", s) if x.strip()]


def parse_note(note: str | None) -> Note | None:
    """Who runs and who claims an area, from its NOTE_BRK; None if the note names no claimant."""
    if not isinstance(note, str):
        return None
    admin, claimants, self_admin = None, [], False
    for part in (p.strip() for p in note.split(";")):
        if m := re.fullmatch(r"(?i)admin\.? by (.+?)(?: for (.+))?", part):
            admin = (m.group(2) or m.group(1)).strip()
        elif re.match(r"(?i)self admin", part):
            self_admin = True
        elif m := re.fullmatch(r"(?i)leased to (.+?) by (.+)", part):
            admin = m.group(1).strip()
            claimants.append(m.group(2).strip())
        elif m := re.fullmatch(r"(?i)claimed by (.+)", part):
            claimants += split_names(m.group(1))
    if not claimants or claimants == ["multiple"]:
        return None
    return Note(admin, claimants, self_admin)


def load_disputes(countries: gpd.GeoDataFrame, disputed: gpd.GeoDataFrame) -> list[Dispute]:
    """Every area of `disputed` with a claim (from its note, or HAND_CLAIMS), resolved to rows of `countries`.
    The geometry is copied as is."""
    index_of = dict(zip(countries.NAME_EN, countries.index))

    def row_of(name: str, area: str) -> int:
        name = ALIASES.get(name, name)
        if name not in index_of:
            raise ValueError(f"unknown country {name!r} in the dispute over {area!r}")
        return index_of[name]

    out = []
    for _, r in disputed.iterrows():
        breakaway, label = False, None
        if (r.NAME_EN, r.ADMIN) in HAND_CLAIMS:
            names = HAND_CLAIMS[(r.NAME_EN, r.ADMIN)]
        else:
            note = parse_note(r.NOTE_BRK)
            if note is None:
                continue
            if note.self_admin:
                label = LABEL_TEXT.get(r.NAME_EN, r.NAME_EN)
                breakaway = r.NAME_EN not in index_of
                names = ([] if breakaway else [r.NAME_EN]) + note.claimants
            else:
                names = ([note.admin] if note.admin else []) + note.claimants
        rows = list(dict.fromkeys(row_of(n, r.NAME_EN) for n in names))
        out.append(Dispute(r.NAME_EN, r.ADMIN, rows, breakaway, label, r.geometry))
    return out


def dispute_edges(disputes: list[Dispute]) -> set[tuple[int, int]]:
    """Country pairs of every 2-country dispute, which the colouring keeps apart so their stripes differ."""
    return {(min(d.countries), max(d.countries)) for d in disputes if not d.breakaway and len(d.countries) == 2}


def dispute_fills(d: Dispute, fill: Callable[[int], Fill]) -> list[Fill]:
    """The stripe colours of `d`, given each country's fill: a breakaway alternates its parent's colour with grey;
    3+ countries would be too many colours, so the area is plain grey."""
    if d.breakaway:
        return [fill(d.countries[0]), DISPUTED_GREY]
    if len(d.countries) > 2:
        return [DISPUTED_GREY]
    return [fill(c) for c in d.countries]


def stripe_parts(geom: BaseGeometry, n: int, width: float) -> list[BaseGeometry]:
    """`geom` cut into `n` sets of 45-degree stripes `width` across, running like '/' (as Baikonur's hatching): part i
    holds the stripes k with k % n == i, stripe k being k*width <= (y - x)/sqrt(2) < (k+1)*width. Counted from the
    origin, so neighbouring areas line up."""
    if n == 1:
        return [geom]
    x0, y0, x1, y1 = geom.bounds
    corners = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
    us = [(y - x) / math.sqrt(2) for x, y in corners]  # across the stripes
    vs = [(x + y) / math.sqrt(2) for x, y in corners]  # along them
    v0, v1 = min(vs) - width, max(vs) + width

    def band(k: int) -> Polygon:
        a, b = k * width, (k + 1) * width
        return Polygon([((v - u) / math.sqrt(2), (u + v) / math.sqrt(2)) for u, v in [(a, v0), (b, v0), (b, v1), (a, v1)]])

    bands: list[list[Polygon]] = [[] for _ in range(n)]
    for k in range(math.floor(min(us) / width), math.floor(max(us) / width) + 1):
        bands[k % n].append(band(k))
    return [geom.intersection(shapely.union_all(b)) if b else Polygon() for b in bands]


def border_lines(countries_projected: gpd.GeoDataFrame, disputes: list[Dispute], parent_of: dict[int, int]) -> tuple[BaseGeometry, BaseGeometry]:
    """(solid, dashed) border lines, in map units. A disputed area's outline is dashed where it meets a country
    involved in it (its dependencies included), or another disputed area sharing a country; everything else on its
    outline, and every other country border, stays solid."""
    geoms = countries_projected.geometry
    dashed_parts = []
    for d in disputes:
        involved = set(d.countries)
        window = shapely.box(*d.geometry.buffer(2 * SNAP_M).bounds)  # only what's next to it matters
        near = [g for i, g in geoms.items() if i in involved or parent_of.get(i) in involved]
        near += [e.geometry for e in disputes if e is not d and involved & set(e.countries)]
        other = shapely.union_all([g.intersection(window) for g in near if g.intersects(window)]).difference(d.geometry)
        dashed_parts.append(d.geometry.boundary.intersection(other.buffer(SNAP_M)))
    dashed = shapely.union_all(dashed_parts)
    solid = shapely.MultiLineString([ring for g in geoms for ring in shapely.get_parts(shapely.boundary(g))
                                     for ring in shapely.get_parts(ring)]).difference(dashed.buffer(SNAP_M))
    return solid, dashed
