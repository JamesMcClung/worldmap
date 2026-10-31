"""--disputed: areas claimed by more than one country (Kashmir, Crimea, Taiwan...), from Natural Earth's
disputed-areas file. Each area records every country involved: the one running it plus its claimants."""

import re
from dataclasses import dataclass
from typing import NamedTuple

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

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
