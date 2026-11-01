import numpy as np
import pytest
import shapely
from matplotlib.path import Path

from worldmap.frame import MapFrame, make_frame
from worldmap.labels.label import Label, new_label
from worldmap.labels.placement import clashes, place_labels, place_with_water
from worldmap.labels import placement
from worldmap.labels.water import OCEAN_MIN_PT, WATER_MIN_PT, WaterLabel
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


def water_label(text: str, anchor: tuple[float, float], geom: shapely.Geometry, parts: np.ndarray, size: float, kind: str = "seas") -> WaterLabel:
    x0, y0, x1, y1 = parts.min(axis=0)[0], parts.min(axis=0)[1], parts.max(axis=0)[2], parts.max(axis=0)[3]
    glyphs = Path([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], closed=True)
    return WaterLabel(text, kind, 0 if kind == "oceans" else 1, geom, anchor[0], anchor[1], size, glyphs, parts)  # type: ignore[arg-type]


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


def mid_atlantic_px(frame: MapFrame) -> tuple[float, float]:
    x, y = frame.to_map.transform(-35, 30)
    return (x - frame.xmin) * frame.px_per_unit, (y - frame.ymin) * frame.px_per_unit


def crossing_water_labels(frame: MapFrame) -> tuple[WaterLabel, WaterLabel]:
    """Two minimum-size labels wanting the same small water body (exactly their glyph boxes), "Sea" first."""
    x, y = mid_atlantic_px(frame)
    parts = np.array([[-30.0, -7, 0, 7], [0, -7, 30, 7]])
    geom = shapely.box(x - 30, y - 7, x + 30, y + 7)
    return water_label("Sea", (x, y), geom, parts, WATER_MIN_PT), water_label("Bay", (x, y), geom, parts.copy(), WATER_MIN_PT)


def water_label_far_from_its_geom(frame: MapFrame) -> WaterLabel:
    x, y = mid_atlantic_px(frame)
    return water_label("Lonely", (x, y), shapely.box(x + 5000, y, x + 5010, y + 10), np.array([[-30.0, -7, 30, 7]]), WATER_MIN_PT)


def test_drop_least_important(frame: MapFrame) -> None:
    sea, bay = crossing_water_labels(frame)
    p, kept = place_with_water([], [sea, bay], frame)
    assert [w.text for w in kept] == ["Sea"]
    assert clashes(p) == ([], [])


def test_off_water_is_dropped(frame: MapFrame) -> None:
    _, kept = place_with_water([], [water_label_far_from_its_geom(frame)], frame)
    assert kept == []


def test_country_labels_never_dropped(frame: MapFrame, one_label: list[Label], water_on_top_of_it: WaterLabel) -> None:
    p, _ = place_with_water(one_label, [water_on_top_of_it], frame)
    assert (~p.water).sum() == 1


def test_no_water_is_place_labels(frame: MapFrame, two_labels: list[Label]) -> None:
    a = place_labels(two_labels, frame)
    b, kept = place_with_water(two_labels_copy(), [], frame)
    assert kept == [] and np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)


def always_clashing(p: placement.Placement) -> tuple[list[int], list[int]]:
    return [], [int(i) for i in np.flatnonzero(p.water)]


def test_oceans_shrink_no_smaller_than_ocean_floor(frame: MapFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = mid_atlantic_px(frame)
    ocean = water_label("OCEAN", (x, y), shapely.box(x - 500, y - 100, x + 500, y + 100), np.array([[-200.0, -20, 200, 20]]), 20, "oceans")
    monkeypatch.setattr(placement, "clashes", always_clashing)  # every shrink round shrinks it
    place_labels([], frame, water=[ocean])
    assert ocean.size == pytest.approx(OCEAN_MIN_PT)
