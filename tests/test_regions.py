from collections.abc import Callable

import geopandas as gpd
import matplotlib
import pytest
from shapely.geometry import Point

from worldmap.data import CACHE, RASTER, is_labeled, load_countries, load_lakes, project
from worldmap.earth import Fill, build_regions, fill_at
from worldmap.frame import MapFrame, make_frame
from worldmap.labels.label import RENAME, Label, make_labels
import worldmap.labels.regions as regions_module
from worldmap.labels.regions import HAND_PICKED, PARENT_SHORT, ChildRegion, child_regions
from worldmap.render import new_figure, subtract_lakes


@pytest.fixture(scope="module")
def countries() -> gpd.GeoDataFrame:
    return load_countries()


@pytest.fixture(scope="module")
def countries_projected(countries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """As cli.main passes it: projected, with the big lakes cut out."""
    return subtract_lakes(project(countries), project(load_lakes()))


@pytest.fixture(scope="module")
def frame() -> MapFrame:
    return make_frame()


@pytest.fixture(scope="module")
def regions(countries: gpd.GeoDataFrame, countries_projected: gpd.GeoDataFrame, frame: MapFrame) -> list[ChildRegion]:
    return child_regions(countries, countries_projected, frame)


def test_counts(regions: list[ChildRegion]) -> None:
    assert len(regions) == 66
    names = {r.name for r in regions}
    assert len(names) == 66
    assert not {"Hong Kong", "Macau"} & names
    assert {"Gibraltar", "Greenland", "Alaska", "Easter Island"} <= names


def test_text(regions: list[ChildRegion]) -> None:
    by = {r.name: r for r in regions}
    assert by["Tasmania"].text == "Tasmania (Australia)"
    assert by["Hawaii"].text == "Hawaii (US)"
    assert by["Bermuda"].text == "Bermuda (UK)"
    assert by["Oecusse"].text == "Oecusse (East Timor)"
    assert by["Greenland"].text == "Greenland (Denmark)"


def test_hand_picked_select_polygons_but_not_mainland(regions: list[ChildRegion], countries_projected: gpd.GeoDataFrame) -> None:
    by = {r.name: r for r in regions}
    seen: set[bytes] = set()
    for name, _, _ in HAND_PICKED:
        region = by[name]
        assert region.polys, name
        parent_geom = countries_projected.geometry[region.feature]
        largest = max(getattr(parent_geom, "geoms", [parent_geom]), key=lambda p: p.area)
        for p in region.polys:
            assert not p.equals(largest), name
            assert p.wkb not in seen, name
            seen.add(p.wkb)


def test_alaska_includes_aleutians_east_of_180(regions: list[ChildRegion], frame: MapFrame) -> None:
    alaska = next(r for r in regions if r.name == "Alaska")
    lons = [frame.to_lonlat.transform(*p.representative_point().coords[0])[0] for p in alaska.polys]
    assert max(lons) > 170 and min(lons) < -140


def test_parents_are_labeled_countries(regions: list[ChildRegion], countries: gpd.GeoDataFrame) -> None:
    labeled = {RENAME.get(r.NAME_EN, r.NAME_EN) for _, r in countries.iterrows() if is_labeled(r)}
    assert {r.parent_name for r in regions} <= labeled | set(PARENT_SHORT.values())


def build_labels(countries: gpd.GeoDataFrame, countries_projected: gpd.GeoDataFrame, frame: MapFrame, children: list[ChildRegion], fill_of: Callable[[int, Point | None], Fill]) -> list[Label]:
    matplotlib.use("Agg")
    fig, ax = new_figure(frame)
    ax.set_xlim(frame.xmin, frame.xmax)
    ax.set_ylim(frame.ymin, frame.ymax)
    fig.canvas.draw()
    return make_labels(countries_projected, frame, ax, fig.canvas.get_renderer(), fill_of, children)  # type: ignore[attr-defined]


def test_child_labels_are_italic(countries: gpd.GeoDataFrame, countries_projected: gpd.GeoDataFrame, frame: MapFrame, regions: list[ChildRegion]) -> None:
    def grey(i: int, pt: Point | None) -> Fill:
        return (0.5, 0.5, 0.5)

    with_children = build_labels(countries, countries_projected, frame, regions, grey)
    italic = [label for label in with_children if label.text.get_fontstyle() == "italic"]
    assert len(italic) == 66
    assert {label.name for label in italic} == {r.text for r in regions}
    assert not [label for label in build_labels(countries, countries_projected, frame, [], grey) if label.text.get_fontstyle() == "italic"]


def test_earth_child_label_uses_its_own_polygon(countries: gpd.GeoDataFrame, countries_projected: gpd.GeoDataFrame, frame: MapFrame, regions: list[ChildRegion]) -> None:
    if not (CACHE / RASTER).exists():
        pytest.skip("land-cover raster not downloaded")
    earth = build_regions(countries_projected, countries)
    labels = build_labels(countries, countries_projected, frame, regions, lambda i, pt: fill_at(earth, i, pt))
    color = {label.name: label.color for label in labels}
    assert color["Hawaii (US)"] != color["United States"]


def test_scattered_child_labels_sized_by_land(countries: gpd.GeoDataFrame, countries_projected: gpd.GeoDataFrame, frame: MapFrame, regions: list[ChildRegion]) -> None:
    """Archipelagos spread over an ocean are sized by their land, not their convex hull, and point at an island."""
    labels = build_labels(countries, countries_projected, frame, regions, lambda i, pt: (0.5, 0.5, 0.5))
    by = {label.name: label for label in labels}
    region = {r.text: r for r in regions}
    for name in ["French Polynesia (France)", "Saint Helena (UK)", "Cook Islands (New Zealand)"]:
        label = by[name]
        assert label.text.get_fontsize() <= 14, name
        anchor = Point(label.anchor_x / frame.px_per_unit + frame.xmin, label.anchor_y / frame.px_per_unit + frame.ymin)
        assert max(region[name].polys, key=lambda p: p.area).contains(anchor), name


def test_hand_picked_without_polygons_is_named(countries: gpd.GeoDataFrame, countries_projected: gpd.GeoDataFrame, frame: MapFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(regions_module, "HAND_PICKED", [("Atlantis", "Portugal", [(-40.0, 0.0, -39.0, 1.0)])])
    with pytest.raises(ValueError, match="Atlantis"):
        child_regions(countries, countries_projected, frame)


def test_labels_keep_their_case(countries: gpd.GeoDataFrame, countries_projected: gpd.GeoDataFrame, frame: MapFrame, regions: list[ChildRegion]) -> None:
    labels = build_labels(countries, countries_projected, frame, regions, lambda i, pt: (0.5, 0.5, 0.5))
    assert [label.name for label in labels if label.text.get_text() != label.name] == []
