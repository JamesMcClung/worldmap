import geopandas as gpd
import pytest

from worldmap.data import is_labeled, load_countries, project
from worldmap.frame import MapFrame, make_frame
from worldmap.labels.label import RENAME
from worldmap.labels.regions import HAND_PICKED, ChildRegion, child_regions


@pytest.fixture(scope="module")
def countries() -> gpd.GeoDataFrame:
    return load_countries()


@pytest.fixture(scope="module")
def countries_projected(countries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return project(countries)


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
    assert by["Hawaii"].text == "Hawaii (United States)"
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
    assert {r.parent_name for r in regions} <= labeled
