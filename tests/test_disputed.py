import geopandas as gpd
import pytest
from shapely.geometry import Point

from worldmap.data import load_countries, load_disputed
from worldmap.disputed import Dispute, Note, load_disputes, parse_note


@pytest.fixture(scope="module")
def countries() -> gpd.GeoDataFrame:
    return load_countries()


@pytest.fixture(scope="module")
def disputes(countries: gpd.GeoDataFrame) -> list[Dispute]:
    return load_disputes(countries, load_disputed())


@pytest.mark.parametrize("note, expected", [
    ("Admin. by India; Claimed by Pakistan", Note("India", ["Pakistan"], False)),
    ("Admin. By Israel; Claimed by Syria", Note("Israel", ["Syria"], False)),
    ("Admin by U.K.; Claimed by Ireland, Denmark, and Iceland", Note("U.K.", ["Ireland", "Denmark", "Iceland"], False)),
    ("Admin. by American Samoa for U.S.A.; Claimed by Tokelau", Note("U.S.A.", ["Tokelau"], False)),
    ("Admin. by China; Ceded to China by Pakistan; Claimed by India", Note("China", ["India"], False)),
    ("Self admin.; Claimed by Azer.", Note(None, ["Azer."], True)),
    ("Claimed by Chile and Argentina; under survey", Note(None, ["Chile", "Argentina"], False)),
    ("Claimed by Colombia, Jamaica, Nicaragua and the United States", Note(None, ["Colombia", "Jamaica", "Nicaragua", "the United States"], False)),
    ("Leased to Russia by Kazakhstan", Note("Russia", ["Kazakhstan"], False)),
    ("Leased to U.S.A. by Cuba; Claimed by Cuba", Note("U.S.A.", ["Cuba", "Cuba"], False)),
])
def test_parse_note(note: str, expected: Note) -> None:
    assert parse_note(note) == expected


@pytest.mark.parametrize("note", [None, "Admin. by Venezuela; Claimed by multiple", "UN Patrolled Zone in Syria", "Between Egypt and Sudan"])
def test_parse_note_without_claim(note: str | None) -> None:
    assert parse_note(note) is None


def test_counts(disputes: list[Dispute]) -> None:
    two = [d for d in disputes if not d.breakaway and len(d.countries) == 2]
    grey = [d for d in disputes if not d.breakaway and len(d.countries) > 2]
    assert (len(two), len(grey), sum(d.breakaway for d in disputes)) == (72, 11, 6)


def test_examples(disputes: list[Dispute], countries: gpd.GeoDataFrame) -> None:
    by = {(d.name, d.admin): [countries.NAME_EN[i] for i in d.countries] for d in disputes}
    assert by[("Kashmir", "India")] == ["India", "Pakistan"]
    assert by[("Taiwan", "Taiwan")] == ["Taiwan", "People's Republic of China"]
    assert by[("Abkhazia", "Georgia")] == ["Georgia"]
    assert by[("Kosovo", "Kosovo")] == ["Kosovo", "Serbia"]  # hand-written
    assert by[("Baikonur", "Baykonur Cosmodrome")] == ["Russia", "Kazakhstan"]
    assert by[("Guantanamo Bay Naval Base", "US Naval Base Guantanamo Bay")] == ["United States of America", "Cuba"]
    assert by[("Swains Island", "American Samoa")] == ["United States of America", "New Zealand"]
    assert ("Israel", "Israel") not in by and ("Ceuta", "Spain") not in by and ("Isla Aves", "Venezuela") not in by


def test_labels(disputes: list[Dispute]) -> None:
    assert {d.label for d in disputes if d.label} == {"Taiwan", "Somaliland", "Northern Cyprus", "Western Sahara", "Abkhazia",
                                                      "South Ossetia", "Transnistria", "Nagorno-Karabakh", "Donetsk", "Luhansk"}


def test_unknown_claimant_is_an_error(countries: gpd.GeoDataFrame) -> None:
    bad = gpd.GeoDataFrame({"NAME_EN": ["X"], "ADMIN": ["India"], "NOTE_BRK": ["Admin. by India; Claimed by Atlantis"]}, geometry=[Point(0, 0)])
    with pytest.raises(ValueError, match="Atlantis"):
        load_disputes(countries, bad)
