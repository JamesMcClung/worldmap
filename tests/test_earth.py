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
