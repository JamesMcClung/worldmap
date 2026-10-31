import matplotlib
import pytest
from shapely.geometry import Polygon, box

from worldmap.colorbars import BAR_GAP_IN, bar_region, draw_colorbar
from worldmap.frame import DPI, make_frame
from worldmap.topographic import LAND_RAMP


@pytest.mark.parametrize("corner", ["br", "tr", "bl"])
def test_bar_region_clears_globe_and_canvas(corner: str) -> None:
    f = make_frame()
    r = bar_region(f, corner)  # type: ignore[arg-type]
    gap = BAR_GAP_IN * DPI / f.px_per_unit
    assert r.area > 0 and r.distance(Polygon(f.outline)) >= gap * 0.99
    assert box(f.xmin, f.ymin, f.xmax, f.ymax).contains(r)


def test_bar_region_corners() -> None:
    f = make_frame()
    assert bar_region(f, "br").centroid.x > 0 and bar_region(f, "br").centroid.y < 0
    assert bar_region(f, "tr").centroid.y > 0 and bar_region(f, "bl").centroid.x < 0


def test_labels_have_halo_and_plus() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_colorbar(ax, make_frame(), "br", LAND_RAMP, 6000, "Height")
    texts = [t.get_text() for t in ax.texts]
    plt.close(fig)
    assert texts[0] == "sea level" and "6,000 m+" in texts and "Height" in texts
    assert all(t.get_path_effects() for t in ax.texts)
