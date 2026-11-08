from dataclasses import replace

import geopandas as gpd
from shapely.geometry import MultiPolygon, Point, Polygon, box

from worldmap.frame import DPI, POSTER_WIDTH_IN, make_frame


def test_world_frame_size_unchanged() -> None:
    f = make_frame()
    assert f.width_in == POSTER_WIDTH_IN and f.height_in == POSTER_WIDTH_IN * f.height / f.width
    assert f.px_per_unit == POSTER_WIDTH_IN * DPI / f.width


def test_tall_frame_long_side_is_poster_width() -> None:
    f = replace(make_frame(), xmin=0, xmax=1, ymin=0, ymax=2)
    assert f.height_in == POSTER_WIDTH_IN and f.width_in == POSTER_WIDTH_IN / 2
    assert f.px_per_unit == POSTER_WIDTH_IN * DPI / 2


def points(*xy: tuple[float, float]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[Point(p) for p in xy], crs="EPSG:4326")


def test_project_uses_the_frame_crs() -> None:
    f = replace(make_frame(), crs="+proj=laea +lat_0=0 +lon_0=10 +datum=WGS84", globe=False, xmin=-1e6, xmax=1e6, ymin=-1e6, ymax=1e6)
    projected = f.project(points((10, 0)))
    assert projected.geometry[0].distance(Point(0, 0)) < 1e-6


def test_visible_clips_only_crops() -> None:
    world = make_frame()
    gdf = world.project(points((0, 0), (100, 0)))
    assert world.visible(gdf) is gdf
    crop = replace(world, globe=False, xmin=-1e5, xmax=1e5, ymin=-1e5, ymax=1e5)
    assert list(crop.visible(gdf).geometry.is_empty) == [False, True]


def test_crop_joins_polygons_split_at_the_antimeridian() -> None:
    from worldmap.crop import box_region, crop_frame

    east = Polygon([(170, 60), (180, 60), (180, 65), (170, 65)])
    w = -179.99999999999991  # as in Natural Earth
    west = Polygon([(w, 60), (-170, 60), (-170, 65), (w, 65), (w, 63.7), (w, 61.2)])  # other seam vertices
    halves = gpd.GeoDataFrame(geometry=[MultiPolygon([east, west])], crs="EPSG:4326")
    projected = crop_frame(box_region((160, 48, -150, 72)), 0).project(halves)
    assert projected.geometry[0].geom_type == "Polygon"
