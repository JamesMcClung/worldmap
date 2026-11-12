import numpy as np

from worldmap.labels.placement import overhang
from worldmap.labels.priority import label_order, rank_table, ranks_by_quantile, steps


def test_label_order() -> None:
    assert label_order(["countries", "water", "seas"]) == ["countries", "oceans", "seas", "lakes", "straits", "rivers"]
    assert label_order(["rivers", "countries", "rivers"]) == ["rivers", "countries"]


def test_ranks_are_fifths_of_every_kind_together() -> None:
    kinds = ["seas", "lakes"] * 5
    assert list(ranks_by_quantile(kinds, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_rivers_rank_by_ten_times_their_length() -> None:
    assert list(ranks_by_quantile(["seas", "seas", "rivers"], [400, 600, 50])) == [4, 1, 2]  # the river as 500


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


PLAN = [(1, [0, 1]), (1, [2]), (3, [3])]
KINDS = ["seas", "seas", "rivers", "seas"]


def test_rank_table() -> None:
    """Placed of each rank's candidates, by kind (in `order`, those with any), then of all, and as a percentage; on the
    right, of every rank, and as a percentage."""
    assert rank_table(PLAN, KINDS, ["rivers", "seas", "lakes"], placed={0, 2}, color=False) == (
        "rank         │       1       2       3       4       5 │     all       %\n"
        "─────────────┼─────────────────────────────────────────┼────────────────\n"
        "rivers       │     1/1       -       -       -       - │     1/1    100%\n"
        "seas         │     1/2       -     0/1       -       - │     1/3     33%\n"
        "─────────────┼─────────────────────────────────────────┼────────────────\n"
        "all          │     2/3       -     0/1       -       - │     2/4     50%\n"
        "%            │     67%       -      0%       -       - │     50%")


def test_rank_table_names_the_biggest_dropped() -> None:
    """A kind's in plan order (by size), cut to 20 characters, '...' included, '-' if none; none for all kinds together."""
    plan = [(1, [0, 1, 2, 3, 4]), (1, [5]), (2, [6])]
    kinds = ["seas"] * 5 + ["rivers", "lakes"]
    assert rank_table(plan, kinds, ["seas", "rivers", "lakes"], placed={1, 6}, color=False, names=["Aaaa", "B", "Cccc", "Ddddd", "Eeeeeeee", "F", "G"]) == (
        "rank         │       1       2       3       4       5 │     all       % │ dropped\n"
        "─────────────┼─────────────────────────────────────────┼─────────────────┼─────────────────────\n"
        "seas         │     1/5       -       -       -       - │     1/5     20% │ Aaaa, Cccc, Ddddd...\n"
        "rivers       │     0/1       -       -       -       - │     0/1      0% │ F\n"
        "lakes        │       -     1/1       -       -       - │     1/1    100% │ -\n"
        "─────────────┼─────────────────────────────────────────┼─────────────────┼─────────────────────\n"
        "all          │     1/6     1/1       -       -       - │     2/7     29% │\n"
        "%            │     17%    100%       -       -       - │     29%         │")


def test_rank_table_in_colour() -> None:
    """Green: all placed, red: none, yellow: some; only a yellow rank cell needs its placed count (the totals always show it)."""
    green, yellow, red, end = "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[0m"
    rows = rank_table(PLAN, KINDS, ["rivers", "seas"], placed={0, 2}, color=True).split("\n")
    assert rows[2] == f"rivers       │{green}       1{end}       -       -       -       - │{green}     1/1{end}{green}    100%{end}"
    assert rows[3] == f"seas         │{yellow}     1/2{end}       -{red}       1{end}       -       - │{yellow}     1/3{end}{yellow}     33%{end}"
    assert rows[5] == f"all          │{yellow}     2/3{end}       -{red}     0/1{end}       -       - │{yellow}     2/4{end}{yellow}     50%{end}"
    assert rows[6] == f"%            │{yellow}     67%{end}       -{red}      0%{end}       -       - │{yellow}     50%{end}"


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
    same = lambda kinds, sizes: np.array([1, 1])  # noqa: E731
    _, labels, kept = place_by_priority([atlantis(1)], [sized(sea, 1)], frame, ["countries", "seas"], rank=same)
    assert [label.name for label in labels] == ["Atlantis"] and kept == []
    sea, frame = boxed_in()
    _, labels, kept = place_by_priority([atlantis(1)], [sized(sea, 1)], frame, ["seas", "countries"], rank=same)
    assert labels == [] and [w.text for w in kept] == ["Sea"]


def test_two_ranks_down_is_fixed() -> None:
    alone, _, _ = place_by_priority([atlantis(1)], [], FRAME, ["countries", "seas"])  # rank 1 too
    country = atlantis(1)
    w, h = country.width, country.height
    small = water_label("Small", (country.anchor_x, country.anchor_y), country.geom, np.array([[-w / 2, -h / 2, w / 2, h / 2]]), 10)
    x, y = FRAME.to_px(shapely.Point(*FRAME.to_map.transform(150, -30))).coords[0]
    big = water_label("Big", (x, y), shapely.box(x - 200, y - 50, x + 200, y + 50), np.array([[-100.0, -20, 100, 20]]), 10)
    p, labels, _ = place_by_priority([country], [sized(big, 1e6), sized(small, 1e3)], FRAME, ["countries", "seas"], rank=lambda kinds, sizes: np.array([1, 1, 3]))
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


def test_water_label_mostly_off_its_water_is_skipped() -> None:
    """Its body is 40% of its name's width wherever it goes."""
    x, y = mid_atlantic_px(FRAME)
    parts = np.array([[-30.0 + 10 * k, -7, -20 + 10 * k, 7] for k in range(6)])  # wherever it goes, 2 of 6 glyphs off one end
    sea = water_label("Sea", (x, y), shapely.box(x - 12, y - 7, x + 12, y + 7), parts, WATER_MIN_PT)
    assert place_by_priority([], [sized(sea, 1)], FRAME, ["seas"])[2] == []


def test_country_pushed_off_its_country_gets_a_leader() -> None:
    """A sea fixed over all of Atlantis and round it: its name goes outside, with a leader to it."""
    country = atlantis(1)
    x, y = mid_atlantic_px(FRAME)
    sea = water_label("Sea", (x, y), shapely.box(x - 700, y - 700, x + 700, y + 700), np.array([[-700.0, -700, 700, 700]]), WATER_MIN_PT)
    p, labels, kept = place_by_priority([country], [sized(sea, 1e6)], FRAME, ["countries", "seas"], rank=lambda kinds, sizes: np.array([3, 1]))
    assert labels == [country] and country.out and bool(p.out[0])
    assert [w.text for w in kept] == ["Sea"] and clashes(p) == []


def test_centred_label_may_hang_off_both_ends() -> None:
    """A sea 48% of its name's length: centred, 26% off each end (most of it off, but neither end over 30%)."""
    x, y = mid_atlantic_px(FRAME)
    parts = np.array([[-50.0 + 5 * k, -7, -45 + 5 * k, 7] for k in range(20)])
    sea = water_label("Sea", (x, y), shapely.box(x - 24, y - 20, x + 24, y + 20), parts, WATER_MIN_PT)
    assert [w.text for w in place_by_priority([], [sized(sea, 1)], FRAME, ["seas"])[2]] == ["Sea"]



def test_water_label_on_a_narrow_lake_is_kept() -> None:
    """Every glyph's centre is over the water, though under half of each (halo-padded) glyph box is."""
    x, y = mid_atlantic_px(FRAME)
    parts = np.array([[-50.0 + 10 * k, -7, -40 + 10 * k, 7] for k in range(10)])
    lake = water_label("Lake", (x, y), shapely.box(x - 52, y - 2.5, x + 52, y + 2.5), parts, WATER_MIN_PT, kind="lakes")
    assert [w.text for w in place_by_priority([], [sized(lake, 1)], FRAME, ["lakes"])[2]] == ["Lake"]
