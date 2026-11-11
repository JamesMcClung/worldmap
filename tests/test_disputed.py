import geopandas as gpd
import matplotlib
import pytest
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, box

from worldmap.data import CRS, load_countries, load_disputed, parent_index
from worldmap.frame import make_frame
from worldmap.labels.label import make_labels
from worldmap.render import new_figure
from worldmap.disputed import DISPUTED_GREY, Dispute, Note, border_lines, dispute_edges, dispute_fills, disputed_label_args, load_disputes, parse_note, stripe_parts


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


def test_dispute_edges(disputes: list[Dispute], countries: gpd.GeoDataFrame) -> None:
    names = {tuple(sorted((countries.NAME_EN[a], countries.NAME_EN[b]))) for a, b in dispute_edges(disputes)}
    assert ("Marshall Islands", "United States of America") in names and ("India", "Pakistan") in names
    assert not any("Georgia" in p for p in names)  # breakaways add no edge
    assert not any("Seychelles" in p for p in names)  # 3+ countries add none either


def test_stripe_parts_cover_and_balance() -> None:
    sq = box(0, 0, 100, 100)
    parts = stripe_parts(sq, 3, 2.0)
    assert len(parts) == 3
    assert abs(sum(p.area for p in parts) - sq.area) < 1e-6
    assert all(abs(p.area - sq.area / 3) < sq.area * 0.02 for p in parts)
    assert parts[0].intersection(parts[1]).area < 1e-9


def test_stripe_parts_line_up_across_shapes() -> None:
    a, b = box(0, 0, 10, 10), box(10, 0, 20, 10)
    joined = stripe_parts(a, 2, 1.0)[0].union(stripe_parts(b, 2, 1.0)[0])
    assert joined.symmetric_difference(stripe_parts(a.union(b), 2, 1.0)[0]).area < 1e-9


def test_stripe_parts_one_colour_is_the_whole_shape() -> None:
    sq = box(0, 0, 5, 5)
    assert stripe_parts(sq, 1, 1.0)[0].symmetric_difference(sq).area < 1e-9


def test_dispute_fills() -> None:
    fill = {0: "#ff0000", 1: "#00ff00", 2: "#0000ff"}.__getitem__
    assert dispute_fills(Dispute("A", "A", [0, 1], False, None, Point(0, 0)), fill) == ["#ff0000", "#00ff00"]
    assert dispute_fills(Dispute("A", "A", [0, 1, 2], False, None, Point(0, 0)), fill) == [DISPUTED_GREY]
    assert dispute_fills(Dispute("A", "A", [2], True, "A", Point(0, 0)), fill) == ["#0000ff", DISPUTED_GREY]


@pytest.fixture(scope="module")
def disputes_projected(countries: gpd.GeoDataFrame) -> list[Dispute]:
    return load_disputes(countries, make_frame().project(load_disputed()))


@pytest.fixture(scope="module")
def lines(countries: gpd.GeoDataFrame, disputes_projected: list[Dispute]) -> tuple[BaseGeometry, BaseGeometry]:
    return border_lines(make_frame().project(countries), disputes_projected, parent_index(countries))


def at(lon: float, lat: float) -> Point:
    return Point(Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lon, lat))


def test_kashmir_line_of_control_is_dashed(lines: tuple[BaseGeometry, BaseGeometry]) -> None:
    solid, dashed = lines
    p = at(73.98, 34.233)  # middle of the India/Pakistan line of control
    assert dashed.distance(p) < 1_000 and solid.distance(p) > 1_000


def test_kashmir_meets_rest_of_india_dashed(lines: tuple[BaseGeometry, BaseGeometry]) -> None:
    p = at(77.28, 32.886)  # Kashmir's edge inside India: a new line, not a country border
    assert lines[1].distance(p) < 1_000


def test_uninvolved_border_stays_solid(lines: tuple[BaseGeometry, BaseGeometry]) -> None:
    solid, dashed = lines
    p = at(92.04, 24.369)  # India/Bangladesh border
    assert solid.distance(p) < 1_000 and dashed.distance(p) > 50_000


def test_taiwan_coast_is_not_dashed(lines: tuple[BaseGeometry, BaseGeometry], disputes_projected: list[Dispute]) -> None:
    taiwan = next(d for d in disputes_projected if d.name == "Taiwan")
    assert lines[1].intersection(taiwan.geometry.buffer(1_000)).length == 0


def disputed_labels(countries: gpd.GeoDataFrame, disputes: list[Dispute], with_countries: bool) -> list[tuple[str, str]]:
    """Label texts make_labels gives with the --disputed extras, as cli.main builds them (political colours)."""
    matplotlib.use("Agg")
    frame = make_frame()
    fig, ax = new_figure(frame)
    fig.canvas.draw()
    relabel, extra = disputed_label_args(disputes, lambda i, pt: (0.5, 0.5, 0.5))
    labels = make_labels(make_frame().project(countries), frame, ax, fig.canvas.get_renderer(), lambda i, pt: (0.5, 0.5, 0.5),  # type: ignore[attr-defined]
                         countries=with_countries, relabel=relabel, extra=extra)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return [(label.name, label.text.get_fontstyle()) for label in labels]


def test_disputed_labels(countries: gpd.GeoDataFrame, disputes_projected: list[Dispute]) -> None:
    found = disputed_labels(countries, disputes_projected, True)
    labels = dict(found)
    assert {"Somaliland", "Northern Cyprus", "Abkhazia", "South Ossetia", "Transnistria", "Nagorno-Karabakh",
            "Donetsk", "Luhansk", "Taiwan", "Western Sahara", "Kosovo"} <= labels.keys()
    assert "Kashmir" not in labels and "Crimean Peninsula" not in labels and "Turkish Republic of Northern Cyprus" not in labels
    assert labels["Abkhazia"] == labels["Donetsk"] == "normal"
    assert [name for name, _ in found].count("Taiwan") == 1


def test_disputed_labels_need_countries(countries: gpd.GeoDataFrame, disputes_projected: list[Dispute]) -> None:
    assert disputed_labels(countries, disputes_projected, False) == []


def test_state_lines_below_dashed_borders() -> None:
    """With --disputed, state lines are drawn under the borders, or they hide dashes that follow a state line
    (Kashmir's edge with Himachal Pradesh)."""
    from shapely.geometry import LineString

    from worldmap.render import Z_BORDERS, Z_STRIPES, draw_states
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    draw_states(ax, gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])]), disputed=True)
    (z,) = {c.get_zorder() for c in ax.collections}
    plt.close(fig)
    assert Z_STRIPES < z < Z_BORDERS
