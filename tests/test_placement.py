import numpy as np
import pytest
import shapely
from matplotlib.path import Path

from worldmap.frame import MapFrame, make_frame
from worldmap.labels.label import Label, new_label
from worldmap.labels.placement import FIXED, FREE, NUDGE, clashes, glyph_boxes, place_labels, subset
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
    label = new_label(ax, fig.canvas.get_renderer(), frame, name, (x, y), geom, geom.area, (0.2, 0.2, 0.2))  # type: ignore[attr-defined]
    import matplotlib.pyplot as plt
    plt.close(fig)  # the label keeps its text
    return label


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


def test_country_can_shrink_for_water(frame: MapFrame, one_label: list[Label], water_on_top_of_it: WaterLabel) -> None:
    """A sea fixed over a country's label: the country label gives way, by moving or shrinking."""
    start = (np.array([one_label[0].anchor_x, water_on_top_of_it.anchor_x]), np.array([one_label[0].anchor_y, water_on_top_of_it.anchor_y]))
    p = place_labels(one_label, frame, water=[water_on_top_of_it], start=start, freedom=np.array([FREE, FIXED]))
    assert clashes(p) == [] or one_label[0].size <= one_label[0].min_size


def test_clash_is_symmetric(frame: MapFrame, one_label: list[Label], water_on_top_of_it: WaterLabel) -> None:
    start = (np.array([one_label[0].anchor_x, water_on_top_of_it.anchor_x]), np.array([one_label[0].anchor_y, water_on_top_of_it.anchor_y]))
    p = place_labels(one_label, frame, water=[water_on_top_of_it], start=start, freedom=np.array([FIXED, FIXED]))
    assert clashes(p) == [0, 1]
    assert clashes(p, np.array([True, False])) == []


def test_leader_through_a_box_clashes(frame: MapFrame, two_labels: list[Label]) -> None:
    """An 'out' label whose leader runs through another country label's box: both clash."""
    a, b = two_labels
    p = place_labels(two_labels, frame, freedom=np.array([FIXED, FIXED]))
    p.out[0] = True
    p.anchor_x[0], p.anchor_y[0] = p.x[1] - p.width[1], p.y[1]  # dot left of b, label right of b: the leader crosses b
    p.x[0], p.y[0] = p.x[1] + p.width[1] + p.width[0], p.y[1]
    assert clashes(p) == [0, 1]


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


def always_clashing(p: placement.Placement, among: np.ndarray | None = None) -> list[int]:
    return [int(i) for i in np.flatnonzero(p.water)]


def test_oceans_shrink_no_smaller_than_ocean_floor(frame: MapFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = mid_atlantic_px(frame)
    ocean = water_label("OCEAN", (x, y), shapely.box(x - 500, y - 100, x + 500, y + 100), np.array([[-200.0, -20, 200, 20]]), 20, "oceans")
    monkeypatch.setattr(placement, "clashes", always_clashing)  # every shrink round shrinks it
    place_labels([], frame, water=[ocean])
    assert ocean.size == pytest.approx(OCEAN_MIN_PT)


def test_fixed_labels_stay_put(frame: MapFrame, one_label: list[Label], water_on_top_of_it: WaterLabel) -> None:
    start = (np.array([one_label[0].anchor_x, water_on_top_of_it.anchor_x]), np.array([one_label[0].anchor_y, water_on_top_of_it.anchor_y]))
    p = place_labels(one_label, frame, water=[water_on_top_of_it], start=start, freedom=np.array([FIXED, FREE]))
    assert p.x[0] == start[0][0] and p.y[0] == start[1][0]
    assert (p.x[1], p.y[1]) != (start[0][1], start[1][1])  # the water label made way instead


def distance_from_home(frame: MapFrame, freedom: int) -> float:
    """A lone label starting at its anchor, with its home 3 heights to the right."""
    label = country_label(frame, "Atlantis", -35, 30, 500)
    home = (np.array([label.anchor_x + 3 * label.height]), np.array([label.anchor_y]))
    p = place_labels([label], frame, start=(np.array([label.anchor_x]), np.array([label.anchor_y])), freedom=np.array([freedom]), home=home)
    return float(np.hypot(p.x[0] - home[0][0], p.y[0] - home[1][0]))


def test_nudged_label_is_pulled_home(frame: MapFrame) -> None:
    assert distance_from_home(frame, NUDGE) < distance_from_home(frame, FREE)


def test_floors_hold(frame: MapFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = mid_atlantic_px(frame)
    sea = water_label("Sea", (x, y), shapely.box(x - 100, y - 20, x + 100, y + 20), np.array([[-60.0, -10, 60, 10]]), 14)
    monkeypatch.setattr(placement, "clashes", always_clashing)  # every shrink round shrinks it
    place_labels([], frame, water=[sea], floors=np.array([12.0]))
    assert sea.size == pytest.approx(12.0)


def test_subset(frame: MapFrame, one_label: list[Label]) -> None:
    sea, bay = crossing_water_labels(frame)
    p = place_labels(one_label, frame, water=[sea, bay])
    q = subset(p, np.array([True, False, True]))
    assert q.n == 2 and set(q.owner) == {1} and q.water.tolist() == [False, True]
    assert np.array_equal(glyph_boxes(q), glyph_boxes(p)[p.owner == 2])


def test_water_floor_scales_with_the_frame() -> None:
    parts = np.array([[0.0, 0.0, 10.0, 10.0]])
    sea = water_label("Sea", (0, 0), shapely.box(0, 0, 10, 10), parts, 20)
    sea.label_scale = 2
    assert sea.min_size == 2 * WATER_MIN_PT


def test_nudge_goes_no_further_than_nudge_max(frame: MapFrame, one_label: list[Label]) -> None:
    """A tall sea fixed on top of a nudged country label (clearing it takes 3 heights): it gives way only so far."""
    label = one_label[0]
    w, h = label.width, label.height
    tall = water_label("Sea", (label.anchor_x, label.anchor_y), label.geom, np.array([[-w / 2, -2.5 * h, w / 2, 2.5 * h]]), 10)
    start = (np.array([label.anchor_x, label.anchor_x]), np.array([label.anchor_y, label.anchor_y]))
    p = place_labels(one_label, frame, water=[tall], start=start, freedom=np.array([NUDGE, FIXED]))
    assert np.hypot(p.x[0] - label.anchor_x, p.y[0] - label.anchor_y) <= 2 * h + 1e-9  # nudge_max


def test_nudged_out_label_goes_where_it_must(frame: MapFrame, one_label: list[Label]) -> None:
    """An 'out' label has no spot of its own, only its dot: nudged, it clears the tall sea however far that is."""
    label = one_label[0]
    label.out = True
    w, h = label.width, label.height
    tall = water_label("Sea", (label.anchor_x, label.anchor_y), label.geom, np.array([[-w / 2, -2.5 * h, w / 2, 2.5 * h]]), 10)
    start = (np.array([label.anchor_x, label.anchor_x]), np.array([label.anchor_y, label.anchor_y]))
    p = place_labels(one_label, frame, water=[tall], start=start, freedom=np.array([NUDGE, FIXED]))
    assert clashes(p) == []


def test_solver_keeps_a_water_label_within_overhang_max(frame: MapFrame) -> None:
    """A rock fixed over the right 40% of a lake's name: sliding it clear would hang it 40% off its water's left
    end; it may hang off by no more than OVERHANG_MAX, overlap or not."""
    from worldmap.labels.placement import OVERHANG_MAX, overhang

    x, y = mid_atlantic_px(frame)
    parts = np.array([[-50.0 + 10 * k, -7, -40 + 10 * k, 7] for k in range(10)])
    lake = water_label("Lake", (x, y), shapely.box(x - 50, y - 5, x + 50, y + 5), parts, 10, kind="lakes")
    rock = water_label("Rock", (x + 35, y), shapely.box(x + 10, y - 7, x + 60, y + 7), np.array([[-25.0, -7, 25, 7]]), 10)
    p = place_labels([], frame, water=[lake, rock], freedom=np.array([FREE, FIXED]))
    centres = parts[:, 0] + 5 + p.x[0]
    assert overhang(np.abs(centres - x) <= 51) <= OVERHANG_MAX  # on the lake, to within a pixel (Coverage)


def test_water_labels_try_fine_steps(frame: MapFrame) -> None:
    """A water label held in narrow water must be able to shift by a sliver: steps of a tenth of its extent."""
    from worldmap.labels.placement import candidates

    sea, _ = crossing_water_labels(frame)
    p = place_labels([], frame, water=[sea], freedom=np.array([FIXED]))
    cx, cy = candidates(p, [sea], 0)
    assert np.isclose(np.abs(cy - p.y[0]), 0.1 * p.height[0]).any()
