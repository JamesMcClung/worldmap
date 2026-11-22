import warnings

import geopandas as gpd
import matplotlib
import shapely
from shapely.geometry import LineString

from worldmap.frame import make_frame


def test_draw_states_skips_a_crop_with_no_state_lines() -> None:
    matplotlib.use("Agg")
    from worldmap.render import draw_states, new_figure

    fig, ax = new_figure(make_frame())
    empty = gpd.GeoDataFrame(geometry=[LineString()], crs="EPSG:4326")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        draw_states(ax, empty)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_drawable_lakes_by_poster_area() -> None:
    from worldmap.crop import box_region, crop_frame
    from worldmap.render import LAKE_MIN_IN2, drawable_lakes

    world = drawable_lakes(make_frame())
    assert (world.area / make_frame().units_per_in ** 2 >= LAKE_MIN_IN2).all() and "Lake Victoria" in set(world.name)
    assert 95 <= len(world) <= 110  # about a hundred on the world map
    alps = drawable_lakes(crop_frame(box_region((5, 44, 11, 48)), 0))
    assert "Lake Geneva" in set(alps.name)  # 580 km2


def test_drawable_lakes_may_be_none() -> None:
    from worldmap.crop import centre_frame, parse_centre
    from worldmap.data import load_countries
    from worldmap.render import drawable_lakes, subtract_lakes

    centre = parse_centre("-30,0/2,2")  # mid-Atlantic
    assert centre is not None
    sea = centre_frame(centre)
    lakes = drawable_lakes(sea)
    assert lakes.empty
    subtract_lakes(sea.project(load_countries()), shapely.union_all(lakes.geometry.values))
