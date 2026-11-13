from dataclasses import replace
from typing import TypeVar

import numpy as np
import shapely
from test_placement import country_label, crossing_water_labels, mid_atlantic_px, of_kind, water_label

from worldmap.frame import make_frame
from worldmap.labels.label import Label
from worldmap.labels.placement import Placement, clashes, overhang
from worldmap.labels.priority import UNLABELED, Outcome, Unplaced, check_step, place_by_priority, ranks_by_quantile, saturated, steps
from worldmap.labels.water import WATER_MIN_PT, WaterLabel


def test_ranks_are_fifths_of_every_kind_together() -> None:
    kinds = ["seas", "lakes"] * 5
    assert list(ranks_by_quantile(kinds, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_rivers_rank_by_a_power_of_their_length() -> None:
    assert list(ranks_by_quantile(["seas", "seas", "rivers"], [400, 600, 130])) == [4, 1, 2]  # the river as 1.6 * 130 ** 1.19 = 525


def test_ranks_zero_and_nan() -> None:
    assert list(ranks_by_quantile(["seas"] * 3, [10, 0, np.nan])) == [1, 5, 5]


def test_steps_order() -> None:
    kinds = ["countries", "seas", "countries", "seas", "rivers"]
    assert steps(kinds, [1, 100, 1000, 1, 5], ["countries", "rivers", "seas"]) == [(1, [2]), (2, [1]), (3, [4]), (4, [0]), (5, [3])]


def test_steps_biggest_first_and_unasked_kinds_left_out() -> None:
    assert steps(["seas", "seas", "lakes"], [50, 60, 1], ["seas"], rank=lambda kinds, sizes: np.ones(3, int)) == [(1, [1, 0])]


def test_overhang_is_the_longer_end_off() -> None:
    assert overhang([False, True, True, True, False, False]) == 2 / 6
    assert overhang([True, False, True]) == 0  # a gap in the middle doesn't count
    assert overhang([False, False]) == 1


# place_by_priority: the solver on small made-up maps (helpers from test_placement).

FRAME = make_frame()
T = TypeVar("T", Label, WaterLabel)


def sized(label: T, full_size: float) -> T:
    label.full_size = full_size
    return label


def atlantis(full_size: float) -> Label:
    return sized(country_label(FRAME, "Atlantis", -35, 30, 500), full_size)


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
    p = place_by_priority([sized(sea, 100), sized(bay, 10)], FRAME, ["seas"]).placement
    assert [w.text for w in of_kind(p, WaterLabel)] == ["Sea"] and of_kind(p, Label) == []
    assert clashes(p) == []


def test_country_can_lose_to_a_bigger_sea() -> None:
    sea, frame = boxed_in()
    p = place_by_priority([atlantis(1), sized(sea, 1e6)], frame, ["countries", "seas"],
                          rank=lambda kinds, sizes: np.array([3, 1])).placement  # the sea two ranks up: fixed
    assert of_kind(p, Label) == [] and [w.text for w in of_kind(p, WaterLabel)] == ["Sea"]


def test_label_order_breaks_ties_within_a_rank() -> None:
    sea, frame = boxed_in()
    same = lambda kinds, sizes: np.array([1, 1])  # noqa: E731
    p = place_by_priority([atlantis(1), sized(sea, 1)], frame, ["countries", "seas"], rank=same).placement
    assert [label.name for label in of_kind(p, Label)] == ["Atlantis"] and of_kind(p, WaterLabel) == []
    sea, frame = boxed_in()
    p = place_by_priority([atlantis(1), sized(sea, 1)], frame, ["seas", "countries"], rank=same).placement
    assert of_kind(p, Label) == [] and [w.text for w in of_kind(p, WaterLabel)] == ["Sea"]


def test_two_ranks_down_is_fixed() -> None:
    alone = place_by_priority([atlantis(1)], FRAME, ["countries", "seas"]).placement  # rank 1 too
    country = atlantis(1)
    w, h = country.width, country.height
    small = water_label("Small", (country.anchor_x, country.anchor_y), country.geom, np.array([[-w / 2, -h / 2, w / 2, h / 2]]), 10)
    x, y = FRAME.to_px(shapely.Point(*FRAME.to_map.transform(150, -30))).coords[0]
    big = water_label("Big", (x, y), shapely.box(x - 200, y - 50, x + 200, y + 50), np.array([[-100.0, -20, 100, 20]]), 10)
    p = place_by_priority([country, sized(big, 1e6), sized(small, 1e3)], FRAME, ["countries", "seas"], rank=lambda kinds, sizes: np.array([1, 1, 3])).placement
    assert of_kind(p, Label) == [country] and (p.x[0], p.y[0]) == (alone.x[0], alone.y[0])


def test_crop_skips_water_labels_outside_the_outline() -> None:
    sea, _ = crossing_water_labels(FRAME)
    x, y = FRAME.px_to_map(np.array(mid_atlantic_px(FRAME)[0]), np.array(mid_atlantic_px(FRAME)[1]))
    half = 2000 / FRAME.px_per_unit  # an outline that leaves out the sea, far off to one side
    cropped = replace(FRAME, globe=False, outline=[(x + half, y - half), (x + 3 * half, y - half), (x + 3 * half, y + half), (x + half, y + half)])
    assert of_kind(place_by_priority([sea], cropped, ["seas"]).placement, WaterLabel) == []
    assert [w.text for w in of_kind(place_by_priority([crossing_water_labels(FRAME)[0]], FRAME, ["seas"]).placement, WaterLabel)] == ["Sea"]


def test_nothing_to_place() -> None:
    p = place_by_priority([], FRAME, ["countries"]).placement
    assert p.n == 0 and of_kind(p, Label) == [] and of_kind(p, WaterLabel) == []


def test_water_label_mostly_off_its_water_is_skipped() -> None:
    """Its body is 40% of its name's width wherever it goes."""
    x, y = mid_atlantic_px(FRAME)
    parts = np.array([[-30.0 + 10 * k, -7, -20 + 10 * k, 7] for k in range(6)])  # wherever it goes, 2 of 6 glyphs off one end
    sea = water_label("Sea", (x, y), shapely.box(x - 12, y - 7, x + 12, y + 7), parts, WATER_MIN_PT)
    assert of_kind(place_by_priority([sized(sea, 1)], FRAME, ["seas"]).placement, WaterLabel) == []


def test_country_with_no_room_gets_a_leader() -> None:
    """A sea fixed over all of a small country and just round it: its name goes beside it, with a leader to it
    (whether the solver keeps it on the country, clashing, or pushes it off)."""
    country = sized(country_label(FRAME, "Atlantis", -35, 30, 120), 1)
    x, y = country.anchor_x, country.anchor_y
    sea = water_label("Sea", (x, y), shapely.box(x - 130, y - 130, x + 130, y + 130), np.array([[-130.0, -130, 130, 130]]), WATER_MIN_PT)
    p = place_by_priority([country, sized(sea, 1e6)], FRAME, ["countries", "seas"], rank=lambda kinds, sizes: np.array([3, 1])).placement
    assert of_kind(p, Label) == [country] and bool(p.out[0])
    assert [w.text for w in of_kind(p, WaterLabel)] == ["Sea"] and clashes(p) == []


def test_centred_label_may_hang_off_both_ends() -> None:
    """A sea 48% of its name's length: centred, 26% off each end (most of it off, but neither end over 30%)."""
    x, y = mid_atlantic_px(FRAME)
    parts = np.array([[-50.0 + 5 * k, -7, -45 + 5 * k, 7] for k in range(20)])
    sea = water_label("Sea", (x, y), shapely.box(x - 24, y - 20, x + 24, y + 20), parts, WATER_MIN_PT)
    assert [w.text for w in of_kind(place_by_priority([sized(sea, 1)], FRAME, ["seas"]).placement, WaterLabel)] == ["Sea"]


def test_water_label_on_a_narrow_lake_is_kept() -> None:
    """Every glyph's centre is over the water, though under half of each (halo-padded) glyph box is."""
    x, y = mid_atlantic_px(FRAME)
    parts = np.array([[-50.0 + 10 * k, -7, -40 + 10 * k, 7] for k in range(10)])
    lake = water_label("Lake", (x, y), shapely.box(x - 52, y - 2.5, x + 52, y + 2.5), parts, WATER_MIN_PT, kind="lakes")
    assert [w.text for w in of_kind(place_by_priority([sized(lake, 1)], FRAME, ["lakes"]).placement, WaterLabel)] == ["Lake"]


def test_check_step_puts_a_clashing_country_out_without_touching_the_placement() -> None:
    country = atlantis(1)
    x, y, w, h = country.anchor_x, country.anchor_y, country.width, country.height
    sea = water_label("Sea", (x, y), country.geom, np.array([[-w / 2, -h / 2, w / 2, h / 2]]), WATER_MIN_PT)
    p = Placement.of([country, sea])
    verdict = check_step(p, [0], np.array([True, True]), FRAME)
    assert verdict.out == [0] and verdict.rival is None and verdict.skip == []
    assert not p.out.any()


def test_check_step_names_the_smaller_label_at_a_water_label_s_anchor() -> None:
    """A sea pushed off its water, a smaller bay of the step sitting where it belongs: the bay is the rival."""
    x, y = mid_atlantic_px(FRAME)
    glyphs = np.array([[-30.0, -7, 30, 7]])
    sea = water_label("Sea", (x, y), shapely.box(x - 40, y - 10, x + 40, y + 10), glyphs, WATER_MIN_PT)
    bay = water_label("Bay", (x, y), shapely.box(x - 40, y - 10, x + 40, y + 10), glyphs.copy(), WATER_MIN_PT)
    p = Placement.of([sea, bay])
    p.x[0] += 500  # the sea, off its water
    verdict = check_step(p, [0, 1], np.array([True, True]), FRAME)
    assert verdict.rival == 1 and verdict.out == []


# The saturation cut.

def unfit(n: int, kind: str = "seas") -> list[Unplaced]:
    return [Unplaced(f"Bay {k}", kind, 1.0) for k in range(n)]


def test_saturated() -> None:
    assert saturated(5, 16) and not saturated(5, 17)  # 31% and 29%
    assert not saturated(4, 4) and saturated(5, 5)  # CUT_MIN_DROPS


def test_cut_drops_the_whole_group_and_later_ranks() -> None:
    sea, bay = crossing_water_labels(FRAME)
    out = place_by_priority([sized(sea, 100), sized(bay, 10), *unfit(5)], FRAME, ["seas"], rank=lambda k, s: np.array([1, 2] + [1] * 5))
    assert out.cut == {"seas": 1} and not out.kept.any() and out.placement.n == 0
    assert out.cut_fit == {"seas": 1}  # the sea fit before the cut undid it
    assert list(out.shown("seas", np.array([1, 2, UNLABELED]))) == [False, False, False]


def test_no_cut_below_the_minimum_drops() -> None:
    sea, _ = crossing_water_labels(FRAME)
    out = place_by_priority([sized(sea, 100), *unfit(4)], FRAME, ["seas"], rank=lambda k, s: np.ones(5, int))
    assert out.cut == {} and out.kept[0] and list(out.shown("seas", np.array([5, UNLABELED]))) == [True, True]


def test_countries_are_cut_too() -> None:
    out = place_by_priority([atlantis(1), *unfit(5, "countries")], FRAME, ["countries"], rank=lambda k, s: np.ones(6, int))
    assert out.cut == {"countries": 1} and not out.kept[0]


def sea_on(label: Label) -> WaterLabel:
    """A sea wanting to sit right on `label`, its body a wide box round it."""
    x, y = label.anchor_x, label.anchor_y
    return water_label("Sea", (x, y), shapely.box(x - 400, y - 400, x + 400, y + 400), np.array([[-60.0, -10, 60, 10]]), WATER_MIN_PT)


def test_cut_rolls_back_what_the_group_nudged() -> None:
    nudged = place_by_priority([atlantis(1), sized(sea_on(atlantis(1)), 1)], FRAME, ["countries", "seas"], rank=lambda k, s: np.array([1, 2])).placement
    alone = place_by_priority([atlantis(1)], FRAME, ["countries"], rank=lambda k, s: np.array([1])).placement
    assert (nudged.x[0], nudged.y[0]) != (alone.x[0], alone.y[0])  # the sea moves Atlantis, so the test means something
    p = place_by_priority([atlantis(1), sized(sea_on(atlantis(1)), 1), *unfit(5)], FRAME, ["countries", "seas"],
                          rank=lambda k, s: np.array([1, 2] + [2] * 5)).placement
    assert p.n == 1 and (p.x[0], p.y[0], p.size[0]) == (alone.x[0], alone.y[0], alone.size[0])


def test_cut_rivers_are_not_drawn() -> None:
    cut = Outcome([], Placement.of([]), [], np.zeros(0, bool), np.zeros(0, int), {"rivers": 2}, {"rivers": 0})
    assert list(cut.shown("rivers", np.array([1, 2, UNLABELED]))) == [True, False, False]
    assert cut.shown("seas", np.array([5, UNLABELED])).all()


def test_cut_counts_failures_across_ranks() -> None:
    """3 of rank 1's 4 fail, then all 3 of rank 2: under 5 in either, but 6 of the 7 together."""
    sea, _ = crossing_water_labels(FRAME)
    out = place_by_priority([sized(sea, 100), *unfit(6)], FRAME, ["seas"], rank=lambda k, s: np.array([1, 1, 1, 1, 2, 2, 2]))
    assert out.cut == {"seas": 2} and out.kept[0]


def test_cut_share_is_of_every_rank() -> None:
    """Rank 1's 5 failures are 5 of 26: under 30% of the kind, so the sea stays; rank 2 tips it."""
    sea, _ = crossing_water_labels(FRAME)
    out = place_by_priority([sized(sea, 100), *unfit(25)], FRAME, ["seas"], rank=lambda k, s: np.array([1] * 6 + [2] * 20))
    assert out.cut == {"seas": 2} and out.kept[0]


def test_progress_counts_the_steps_a_cut_skips() -> None:
    """Seas are cut at rank 1, so rank 2's seas step is skipped: its labels still count as done, or the bar never fills."""
    calls: list[tuple[int, int, int, str]] = []
    rank = lambda k, s: np.array([1] * 5 + [2] * 2 + [2])  # noqa: E731
    place_by_priority([*unfit(7), *unfit(1, "rivers")], FRAME, ["seas", "rivers"], rank=rank, on_step=lambda *a: calls.append(a))
    assert calls == [(1, 0, 5, "seas"), (2, 2, 3, "rivers")]


def test_entries_keep_the_callers_order() -> None:
    """Unplaced entries may come anywhere: kept and ranks follow the entries as given."""
    sea, _ = crossing_water_labels(FRAME)
    out = place_by_priority([Unplaced("Cove", "seas", 1.0), sized(sea, 100), Unplaced("Creek", "rivers", 1.0)], FRAME, ["seas", "rivers"],
                            rank=lambda k, s: np.array([2, 1, 3]))
    assert list(out.kept) == [False, True, False] and list(out.ranks) == [2, 1, 3]
    assert list(out.ranks_of([out.entries[2], out.entries[0]])) == [3, 2] and len(out.ranks_of([])) == 0
    assert [w.text for w in of_kind(out.placement, WaterLabel)] == ["Sea"]
