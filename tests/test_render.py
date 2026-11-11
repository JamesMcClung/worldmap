import warnings

import geopandas as gpd
import matplotlib
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
