import numpy as np
import pytest
import shapely
from matplotlib.path import Path

from worldmap.frame import MapFrame, make_frame
from worldmap.labels.label import Label, new_label
from worldmap.labels.placement import clashes, place_labels
from worldmap.labels.water import WaterLabel
from worldmap.render import new_figure


@pytest.fixture(scope="module")
def frame() -> MapFrame:
    return make_frame()


def country_label(frame: MapFrame, name: str, lon: float, lat: float, half_px: float) -> Label:
    """A label for a square 'country' 2 * half_px wide around lon/lat."""
    fig, ax = new_figure(frame)
    x, y = frame.to_map.transform(lon, lat)
    half = half_px / frame.px_per_unit
    geom = shapely.box(x - half, y - half, x + half, y + half)
    return new_label(ax, fig.canvas.get_renderer(), frame, name, (x, y), geom, geom.area, (0.2, 0.2, 0.2))  # type: ignore[attr-defined]


def two_labels_copy() -> list[Label]:
    frame = make_frame()
    return [country_label(frame, "Sahara West", 0, 23, 300), country_label(frame, "Sahara East", 20, 20, 300)]


@pytest.fixture
def two_labels() -> list[Label]:
    return two_labels_copy()


@pytest.fixture
def one_label(frame: MapFrame) -> list[Label]:
    return [country_label(frame, "Atlantis", -35, 30, 500)]


def water_label(text: str, anchor: tuple[float, float], geom: shapely.Geometry, parts: np.ndarray, size: float) -> WaterLabel:
    x0, y0, x1, y1 = parts.min(axis=0)[0], parts.min(axis=0)[1], parts.max(axis=0)[2], parts.max(axis=0)[3]
    glyphs = Path([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], closed=True)
    return WaterLabel(text, "seas", 1, geom, anchor[0], anchor[1], size, glyphs, parts)


@pytest.fixture
def water_on_top_of_it(one_label: list[Label]) -> WaterLabel:
    label = one_label[0]
    w, h = label.width, label.height
    parts = np.array([[-w / 2 + k * w / 3, -h / 2, -w / 2 + (k + 1) * w / 3, h / 2] for k in range(3)])
    return water_label("Sea", (label.anchor_x, label.anchor_y), label.geom, parts, 10)


def test_no_water_is_unchanged(frame: MapFrame, two_labels: list[Label]) -> None:
    a = place_labels(two_labels, frame)
    b = place_labels(two_labels_copy(), frame, water=[])
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)


def test_water_pushes_country_but_only_water_shrinks(frame: MapFrame, one_label: list[Label], water_on_top_of_it: WaterLabel) -> None:
    size = one_label[0].text.get_fontsize()
    p = place_labels(one_label, frame, water=[water_on_top_of_it])
    assert one_label[0].text.get_fontsize() == size
    assert clashes(p) == ([], [])


def test_water_only(frame: MapFrame, water_on_top_of_it: WaterLabel) -> None:
    p = place_labels([], frame, water=[water_on_top_of_it])
    assert p.n == 1 and p.water.all()
