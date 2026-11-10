import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, box

from worldmap import earth
from worldmap.crop import box_region, crop_frame
from worldmap.earth import build_regions, earth_fill


def test_build_regions_keeps_values_with_polygons_after_clipping(monkeypatch: pytest.MonkeyPatch) -> None:
    countries = gpd.GeoDataFrame(geometry=[MultiPolygon([box(0, 0, 1, 1), box(10, 0, 11, 1)])], crs="EPSG:4326")
    values = [[[-30.0, 0.0], [40.0, 0.0]]]  # desert, then lush
    monkeypatch.setattr(earth, "earth_values", lambda c: values)
    regions = build_regions(countries, crop_frame(box_region((9, -1, 12, 2)), 0))
    shown = [fill for poly, fill in regions[0] if not poly.is_empty]
    assert shown == [earth_fill(40.0, 0.0)]


@pytest.mark.parametrize("spec", ["0,90/80,80", "160:-150,48:72"])  # round the pole, and a box across +-180
def test_build_regions_joins_halves_split_at_the_antimeridian(monkeypatch: pytest.MonkeyPatch, spec: str) -> None:
    from worldmap.cli import make_map_frame, parse_args

    halves = MultiPolygon([box(170, 60, 180, 65), box(-180, 60, -170, 65)])  # Chukotka, as Natural Earth has it
    countries = gpd.GeoDataFrame(geometry=[halves], crs="EPSG:4326")
    monkeypatch.setattr(earth, "earth_values", lambda c: [[[-30.0, 0.0], [40.0, 0.0]]])
    regions = build_regions(countries, make_map_frame(parse_args(["--crop", spec])))
    shown = [fill for poly, fill in regions[0] if not poly.is_empty]
    assert len(shown) == 1 and shown[0] not in (earth_fill(-30.0, 0.0), earth_fill(40.0, 0.0))  # one, of both values
