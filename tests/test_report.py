import numpy as np
import pytest
from test_priority import FRAME, sized
from test_placement import crossing_water_labels

from worldmap.labels import report
from worldmap.labels.priority import Unplaced, place_by_priority
from worldmap.labels.report import progress, rank_table, summary


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


NAMES_PLAN = [(1, [0, 1, 2, 3, 4]), (1, [5]), (2, [6])]
NAMES_KINDS = ["seas"] * 5 + ["rivers", "lakes"]
NAMES = ["Aaaa", "B", "Cccc", "Ddddd", "Eeeeeeee", "F", "G"]


def test_rank_table_names_the_biggest_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A kind's in plan order (by size), whole names that fit in DROPPED_WIDTH with ', N more', '-' if none; none for all kinds together."""
    monkeypatch.setattr(report, "DROPPED_WIDTH", 20)
    assert rank_table(NAMES_PLAN, NAMES_KINDS, ["seas", "rivers", "lakes"], placed={1, 6}, color=False, names=NAMES) == (
        "rank         │       1       2       3       4       5 │     all       % │ dropped\n"
        "─────────────┼─────────────────────────────────────────┼─────────────────┼───────────────────\n"
        "seas         │     1/5       -       -       -       - │     1/5     20% │ Aaaa, Cccc, 2 more\n"
        "rivers       │     0/1       -       -       -       - │     0/1      0% │ F\n"
        "lakes        │       -     1/1       -       -       - │     1/1    100% │ -\n"
        "─────────────┼─────────────────────────────────────────┼─────────────────┼───────────────────\n"
        "all          │     1/6     1/1       -       -       - │     2/7     29% │\n"
        "%            │     17%    100%       -       -       - │     29%         │")


def test_rank_table_dropped_fills_the_terminal() -> None:
    """On a terminal the dropped column takes the rest of the line, and is left out if narrower than DROPPED_MIN."""
    rows = rank_table(NAMES_PLAN, NAMES_KINDS, ["seas", "rivers", "lakes"], placed={1, 6}, color=False, names=NAMES,
                      columns=75 + 30).split("\n")
    assert rows[2].endswith("│ Aaaa, Cccc, Ddddd, Eeeeeeee")
    assert max(map(len, rows)) <= 105
    narrow = rank_table(NAMES_PLAN, NAMES_KINDS, ["seas", "rivers", "lakes"], placed={1, 6}, color=False, names=NAMES,
                        columns=75 + report.DROPPED_MIN - 1)
    assert narrow == rank_table(NAMES_PLAN, NAMES_KINDS, ["seas", "rivers", "lakes"], placed={1, 6}, color=False)


def test_rank_table_in_colour() -> None:
    """Green: all placed, red: none, yellow: some; only a yellow rank cell needs its placed count (the totals always show it)."""
    green, yellow, red, end = "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[0m"
    rows = rank_table(PLAN, KINDS, ["rivers", "seas"], placed={0, 2}, color=True).split("\n")
    assert rows[2] == f"rivers       │{green}       1{end}       -       -       -       - │{green}     1/1{end}{green}    100%{end}"
    assert rows[3] == f"seas         │{yellow}     1/2{end}       -{red}       1{end}       -       - │{yellow}     1/3{end}{yellow}     33%{end}"
    assert rows[5] == f"all          │{yellow}     2/3{end}       -{red}     0/1{end}       -       - │{yellow}     2/4{end}{yellow}     50%{end}"
    assert rows[6] == f"%            │{yellow}     67%{end}       -{red}      0%{end}       -       - │{yellow}     50%{end}"


def test_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rank, a bar of its labels done, the step's kind; never as wide as the terminal."""
    monkeypatch.setattr(report, "PROGRESS_WIDTH", 4)
    assert progress(2, 1, 4, "seas", 80) == "placing rank 2/5 █░░░ 1/4 seas"
    assert progress(2, 1, 4, "seas", 20) == "placing rank 2/5 █░"


def test_summary_lists_the_skipped_then_the_rank_table() -> None:
    sea, bay = crossing_water_labels(FRAME)
    outcome = place_by_priority([sized(sea, 100), sized(bay, 10)], FRAME, ["seas"])
    lines = summary([sea, bay], [], outcome, ["seas"], color=False, columns=None).split("\n")
    assert lines[0] == "labels skipped: ['Bay']"
    assert lines[3].startswith("seas") and lines[3].endswith("│ Bay")


def test_summary_shows_unfit_and_cuts() -> None:
    sea, _ = crossing_water_labels(FRAME)
    bays = [Unplaced(f"Bay {k}", "seas", 1.0) for k in range(5)]
    outcome = place_by_priority([sized(sea, 100)], FRAME, ["seas"], rank=lambda k, s: np.ones(6, int), unfit=bays)
    lines = summary([sea], bays, outcome, ["seas"], color=False, columns=None).split("\n")
    assert lines[0] == "labels skipped: []"  # what a cut drops is only counted: thousands on the world map
    assert lines[1] == "cut: seas from rank 1 (6 dropped)"
    assert lines[4].startswith("seas") and "(1/)6" in lines[4]  # the sea fit before the cut


def test_summary_lists_neither_the_unfit_nor_the_cut() -> None:
    sea, bay = crossing_water_labels(FRAME)
    rivers = [Unplaced(f"Creek {k}", "rivers", 1.0) for k in range(5)]
    outcome = place_by_priority([sized(sea, 100), sized(bay, 10)], FRAME, ["seas", "rivers"], rank=lambda k, s: np.array([1, 1] + [2] * 5 + [1]),
                                unfit=rivers + [Unplaced("Cove", "seas", 1.0)])
    lines = summary([sea, bay], [*rivers, Unplaced("Cove", "seas", 1.0)], outcome, ["seas", "rivers"], color=False, columns=None).split("\n")
    assert lines[0] == "labels skipped: ['Bay']"
    assert lines[1] == "cut: rivers from rank 2 (5 dropped)"


def test_rank_table_shows_what_fit_in_the_cut_rank() -> None:
    """The rank a kind was cut at: '(fit/)of', fit being its labels placed before the cut undid them."""
    rows = rank_table(PLAN, KINDS, ["rivers", "seas"], placed={0, 2}, color=False, cut={"seas": (3, 1)}).split("\n")
    assert rows[3] == "seas         │     1/2       -   (1/)1       -       - │     1/3     33%"
    red, end = "\x1b[31m", "\x1b[0m"
    coloured = rank_table(PLAN, KINDS, ["rivers", "seas"], placed={0, 2}, color=True, cut={"seas": (3, 1)}).split("\n")
    assert f"{red}   (1/)1{end}" in coloured[3]


def test_rank_table_widens_the_ranks_for_a_long_cut_cell() -> None:
    plan = [(1, [0]), (2, list(range(1, 1001)))]
    rows = rank_table(plan, ["seas"] * 1001, ["seas"], placed={0}, color=False, cut={"seas": (2, 999)}).split("\n")
    assert rows[0] == "rank         │          1          2          3          4          5 │     all       %"
    assert rows[2] == "seas         │        1/1 (999/)1000          -          -          - │  1/1001      0%"
    assert len(rows[1]) == len(rows[0])  # the rule spans the wider columns
