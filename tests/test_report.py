import pytest
from test_priority import FRAME, sized
from test_placement import crossing_water_labels

from worldmap.labels import report
from worldmap.labels.priority import place_by_priority
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
    p = place_by_priority([sized(sea, 100), sized(bay, 10)], FRAME, ["seas"])
    lines = summary([sea, bay], p, ["seas"], color=False, columns=None).split("\n")
    assert lines[0] == "labels skipped: ['Bay']"
    assert lines[3].startswith("seas") and lines[3].endswith("│ Bay")
