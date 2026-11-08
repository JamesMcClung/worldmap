import matplotlib
import pytest
from shapely.geometry import Polygon, box

from worldmap.colorbars import BAR_GAP_IN, bar_region, curved_title, draw_colorbar, globe_side
from worldmap.frame import make_frame
from worldmap.topographic import LAND, LAND_CONTOURS, SEA, SEA_CONTOURS, Contours, Scale


@pytest.mark.parametrize("corner", ["br", "tr", "bl"])
def test_bar_region_clears_globe_and_canvas(corner: str) -> None:
    f = make_frame()
    r = bar_region(f, corner)  # type: ignore[arg-type]
    gap = BAR_GAP_IN * f.units_per_in
    assert r.area > 0 and r.distance(Polygon(f.outline)) >= gap * 0.99
    assert box(f.xmin, f.ymin, f.xmax, f.ymax).contains(r)


def test_bar_region_corners() -> None:
    f = make_frame()
    assert bar_region(f, "br").centroid.x > 0 and bar_region(f, "br").centroid.y < 0
    assert bar_region(f, "tr").centroid.y > 0 and bar_region(f, "bl").centroid.x < 0


@pytest.mark.parametrize("corner", ["br", "tr", "bl"])
def test_title_curves_inside_the_bar_along_the_globe(corner: str) -> None:
    from shapely.geometry import LineString, MultiPoint

    f = make_frame()
    region, unit = bar_region(f, corner), f.units_per_in  # type: ignore[arg-type]
    side = globe_side(region)
    assert side[0, 0] < side[-1, 0] and LineString(side).distance(Polygon(f.outline)) < BAR_GAP_IN * unit * 1.01
    glyphs = MultiPoint(curved_title(region, "Height", unit).vertices)
    assert region.contains(glyphs)
    assert LineString(side).distance(glyphs) > 0.05 * unit  # inset from the curved side


def bar_texts_and_lines(corner: str, scale: Scale, title: str, contours: Contours) -> tuple[list[str], list]:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots()
    draw_colorbar(ax, make_frame(), corner, scale, title, contours)  # type: ignore[arg-type]
    texts = [t.get_text() for t in ax.texts]
    assert all(t.get_path_effects() for t in ax.texts)
    lines = [c for c in ax.collections if isinstance(c, LineCollection)]
    plt.close(fig)
    return texts, lines


def test_land_bar_labels_are_signed() -> None:
    texts, _ = bar_texts_and_lines("br", LAND, "Height", LAND_CONTOURS)
    assert texts[0] == "≤ −500 m" and texts[1] == "sea level" and "1,000 m" in texts
    assert "≥ 6,000 m" in texts


def test_sea_bar_labels_are_signed() -> None:
    texts, _ = bar_texts_and_lines("tr", SEA, "Depth", SEA_CONTOURS)
    assert texts[0] == "≥ 500 m" and texts[1] == "sea level" and "−1,000 m" in texts and "≤ −8,000 m" in texts


def test_bars_show_contour_lines_and_sea_level() -> None:
    _, lines = bar_texts_and_lines("br", LAND, "Height", LAND_CONTOURS)
    segments = [seg for c in lines for seg in c.get_segments()]
    ys = sorted({round(float(seg[0][1])) for seg in segments if abs(seg[0][1] - seg[-1][1]) < 1e-6})
    assert len(ys) == 6  # sea level, then 1,000 ... 5,000 m (6,000 m is the tip)
    assert all(abs(seg[0][1] - seg[-1][1]) < 1e-6 for seg in segments)  # horizontal


def alps():  # type: ignore[no-untyped-def]
    from worldmap.crop import box_region, crop_frame

    return crop_frame(box_region((5, 40, 17, 49)), 0)


def test_rect_bar_region_in_corner() -> None:
    from worldmap.colorbars import BAR_MARGIN_IN, BAR_RECT_LENGTH, BAR_RECT_WIDTH_IN

    f = alps()
    r, unit = bar_region(f, "br"), f.units_per_in
    x0, y0, x1, y1 = r.bounds
    assert r.area == pytest.approx((x1 - x0) * (y1 - y0))  # a rectangle
    assert x1 == pytest.approx(f.xmax - BAR_MARGIN_IN * unit) and y0 == pytest.approx(f.ymin + BAR_MARGIN_IN * unit)
    assert x1 - x0 == pytest.approx(BAR_RECT_WIDTH_IN * unit) and y1 - y0 == pytest.approx(BAR_RECT_LENGTH * f.height_in * unit)
    tx0, _, _, ty1 = bar_region(f, "tr").bounds
    assert ty1 == pytest.approx(f.ymax - BAR_MARGIN_IN * unit) and bar_region(f, "bl").bounds[0] == pytest.approx(f.xmin + BAR_MARGIN_IN * unit)


def test_inner_side_faces_map() -> None:
    from worldmap.colorbars import inner_side

    f = alps()
    for corner, edge in (("br", 0), ("tr", 0), ("bl", 2)):
        r = bar_region(f, corner)  # type: ignore[arg-type]
        side = inner_side(r, corner)  # type: ignore[arg-type]
        assert side[:, 0] == pytest.approx([r.bounds[edge]] * 2) and side[0, 1] < side[1, 1]


@pytest.mark.parametrize("corner", ["br", "tr", "bl"])
def test_rect_title_inside_the_bar(corner: str) -> None:
    from shapely.geometry import MultiPoint

    from worldmap.colorbars import inner_side

    f = alps()
    r = bar_region(f, corner)  # type: ignore[arg-type]
    glyphs = MultiPoint(curved_title(r, "Height", f.units_per_in, inner_side(r, corner)).vertices)  # type: ignore[arg-type]
    assert r.contains(glyphs)
