import numpy as np

from worldmap.labels.priority import label_order, ranks_per_kind, steps


def test_label_order() -> None:
    assert label_order(["countries", "water", "seas"]) == ["countries", "oceans", "seas", "lakes", "straits", "rivers"]
    assert label_order(["rivers", "countries", "rivers"]) == ["rivers", "countries"]


def test_ranks_relative_to_biggest_of_kind() -> None:
    kinds = ["seas"] * 4 + ["lakes"] * 2
    assert list(ranks_per_kind(kinds, [1000, 100.1, 99.9, 1e-9, 5, 0.04])) == [1, 1, 2, 5, 1, 3]


def test_ranks_at_the_steps() -> None:
    assert list(ranks_per_kind(["seas"] * 4, [1000, 100, 1, 0.1])) == [1, 1, 3, 4]


def test_ranks_zero_and_nan() -> None:
    assert list(ranks_per_kind(["seas"] * 3, [10, 0, np.nan])) == [1, 5, 5]


def test_steps_order() -> None:
    kinds = ["countries", "seas", "countries", "seas", "rivers"]
    assert steps(kinds, [1, 100, 1000, 1, 5], ["countries", "rivers", "seas"]) == [(1, [2]), (1, [4]), (1, [1]), (2, [3]), (3, [0])]


def test_steps_biggest_first_and_unasked_kinds_left_out() -> None:
    assert steps(["seas", "seas", "lakes"], [50, 60, 1], ["seas"]) == [(1, [1, 0])]


def test_steps_empty() -> None:
    assert steps([], [], ["countries"]) == []


# place_by_priority: the solver on small made-up maps (helpers from test_placement).
import shapely  # noqa: E402
from dataclasses import replace  # noqa: E402

from test_placement import country_label, crossing_water_labels, mid_atlantic_px, water_label, water_label_far_from_its_geom  # noqa: E402
from worldmap.frame import make_frame  # noqa: E402
from worldmap.labels.label import Label  # noqa: E402
from worldmap.labels.placement import clashes  # noqa: E402
from worldmap.labels.priority import place_by_priority  # noqa: E402
from worldmap.labels.water import WATER_MIN_PT, WaterLabel  # noqa: E402

FRAME = make_frame()


def sized(label: Label | WaterLabel, full_size: float) -> Label | WaterLabel:
    label.full_size = full_size
    return label


def atlantis(full_size: float) -> Label:
    label = country_label(FRAME, "Atlantis", -35, 30, 500)
    label.full_size = full_size
    return label


def boxed_in() -> tuple[WaterLabel, object]:
    """A sea whose one glyph box (at its smallest size: it can't shrink) fills a crop's outline but for 5 px all
    round: no room for another label anywhere. (sea, frame)"""
    x, y = mid_atlantic_px(FRAME)
    sea = water_label("Sea", (x, y), shapely.box(x - 1100, y - 1100, x + 1100, y + 1100), np.array([[-1095.0, -1095, 1095, 1095]]), WATER_MIN_PT)
    cx, cy = FRAME.px_to_map(np.array(x), np.array(y))
    half = 1100 / FRAME.px_per_unit
    frame = replace(FRAME, globe=False, outline=[(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)])
    return sea, frame


def test_smaller_of_a_clashing_pair_is_skipped() -> None:
    sea, bay = crossing_water_labels(FRAME)
    p, labels, kept = place_by_priority([], [sized(sea, 100), sized(bay, 10)], FRAME, ["seas"])
    assert [w.text for w in kept] == ["Sea"] and labels == []
    assert clashes(p) == []


def test_off_water_is_skipped() -> None:
    _, _, kept = place_by_priority([], [sized(water_label_far_from_its_geom(FRAME), 1)], FRAME, ["seas"])
    assert kept == []


def test_country_can_lose_to_a_bigger_sea() -> None:
    sea, frame = boxed_in()
    _, labels, kept = place_by_priority([atlantis(1)], [sized(sea, 1e6)], frame, ["countries", "seas"],
                                        rank=lambda kinds, sizes: np.array([3, 1]))  # the sea two ranks up: fixed
    assert labels == [] and [w.text for w in kept] == ["Sea"]


def test_label_order_breaks_ties_within_a_rank() -> None:
    sea, frame = boxed_in()
    _, labels, kept = place_by_priority([atlantis(1)], [sized(sea, 1)], frame, ["countries", "seas"])
    assert [label.name for label in labels] == ["Atlantis"] and kept == []
    sea, frame = boxed_in()
    _, labels, kept = place_by_priority([atlantis(1)], [sized(sea, 1)], frame, ["seas", "countries"])
    assert labels == [] and [w.text for w in kept] == ["Sea"]


def test_two_ranks_down_is_fixed() -> None:
    alone, _, _ = place_by_priority([atlantis(1)], [], FRAME, ["countries", "seas"])
    country = atlantis(1)
    w, h = country.width, country.height
    small = water_label("Small", (country.anchor_x, country.anchor_y), country.geom, np.array([[-w / 2, -h / 2, w / 2, h / 2]]), 10)
    x, y = FRAME.to_px(shapely.Point(*FRAME.to_map.transform(150, -30))).coords[0]
    big = water_label("Big", (x, y), shapely.box(x - 200, y - 50, x + 200, y + 50), np.array([[-100.0, -20, 100, 20]]), 10)
    p, labels, _ = place_by_priority([country], [sized(big, 1e6), sized(small, 1e3)], FRAME, ["countries", "seas"])  # Small: rank 3
    assert labels == [country] and (p.x[0], p.y[0]) == (alone.x[0], alone.y[0])


def test_crop_skips_water_labels_outside_the_outline() -> None:
    sea, _ = crossing_water_labels(FRAME)
    x, y = FRAME.px_to_map(np.array(mid_atlantic_px(FRAME)[0]), np.array(mid_atlantic_px(FRAME)[1]))
    half = 2000 / FRAME.px_per_unit  # an outline that leaves out the sea, far off to one side
    cropped = replace(FRAME, globe=False, outline=[(x + half, y - half), (x + 3 * half, y - half), (x + 3 * half, y + half), (x + half, y + half)])
    assert place_by_priority([], [sea], cropped, ["seas"])[2] == []
    assert [w.text for w in place_by_priority([], [crossing_water_labels(FRAME)[0]], FRAME, ["seas"])[2]] == ["Sea"]


def test_nothing_to_place() -> None:
    p, labels, water = place_by_priority([], [], FRAME, ["countries"])
    assert p.n == 0 and labels == [] and water == []
