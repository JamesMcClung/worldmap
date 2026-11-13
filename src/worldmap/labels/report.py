"""What placing by priority shows: a progress line on a terminal while it runs, then a table of what it placed."""

from collections.abc import Collection, Sequence

import numpy as np

from worldmap.labels.placeable import Placeable
from worldmap.labels.priority import N_RANKS, Outcome

DROPPED_WIDTH = 40  # the rank table's dropped labels of a kind fit in this many characters off a terminal
DROPPED_MIN = 20  # on a terminal, the dropped column fills the rest of the line, left out if narrower than this

GREEN, YELLOW, RED, PLAIN = "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[0m"


def rank_table(plan: list[tuple[int, list[int]]], kinds: Sequence[str], order: Sequence[str], placed: Collection[int], color: bool,
               names: Sequence[str] = (), columns: int | None = None) -> str:
    """How many of each rank's labels were placed, as text: a row per kind (in `order`, those with any), then of
    all kinds, then that as a percentage; on the right, of every rank, then as a percentage, then (with `names`) a
    kind's dropped labels, biggest first ('-' if none), as many whole names as fit, then ', N more': in
    DROPPED_WIDTH characters, or on a terminal `columns` wide the rest of the line (no column if under
    DROPPED_MIN). In colour a cell is green if all were placed, red if none, else yellow; of a kind's rank cells,
    only a yellow one shows how many ('28/36')."""
    total = {kind: np.zeros(N_RANKS, int) for kind in [*order, "all"]}
    done = {kind: np.zeros(N_RANKS, int) for kind in total}
    dropped: dict[str, list[str]] = {kind: [] for kind in order}
    for r, members in plan:
        for i in members:
            for kind in (kinds[i], "all"):
                total[kind][r - 1] += 1
                done[kind][r - 1] += i in placed
            if i not in placed and names:
                dropped[kinds[i]].append(names[i])

    def cell(n: int, of: int, percent: bool, short: bool) -> str:
        if of == 0:
            return f"{'-':>8}"
        text = f"{round(100 * n / of)}%" if percent else f"{n}/{of}"
        if not color:
            return f"{text:>8}"
        hue = GREEN if n == of else RED if n == 0 else YELLOW
        return f"{hue}{str(of) if short and hue != YELLOW else text:>8}{PLAIN}"

    room = DROPPED_WIDTH if columns is None else columns - (12 + 8 * N_RANKS + 16 + 3 * len(" │") + 1)
    if room < DROPPED_MIN:
        names = ()

    def gone(kind: str) -> str:
        lost = dropped[kind]
        if not lost:
            return " -"
        for k in range(len(lost), -1, -1):
            text = ", ".join([*lost[:k], *[f"{len(lost) - k} more"] * (k < len(lost))])
            if len(text) <= room:
                break
        return " " + text

    def line(parts: list[str]) -> str:
        return " │".join(parts).rstrip()

    def row(name: str, kind: str, percent: bool = False) -> str:
        n, of = done[kind], total[kind]
        parts = [f"{name:<12}", "".join(cell(a, b, percent, kind != "all") for a, b in zip(n, of)),
                 cell(n.sum(), of.sum(), percent, False) + (" " * 8 if percent else cell(n.sum(), of.sum(), True, False))]
        if names:
            parts.append("" if kind == "all" else gone(kind))
        return line(parts)

    shown = [kind for kind in order if total[kind].any()]
    header = [f"{'rank':<12}", "".join(f"{r:>8}" for r in range(1, N_RANKS + 1)), f"{'all':>8}{'%':>8}"]
    widths = [12, 8 * N_RANKS, 16]
    if names:
        header.append(" dropped")
        widths.append(max(len(text) for text in [header[-1], *map(gone, shown)]))
    rule = "─┼".join("─" * w for w in widths)
    rows = [line(header), rule, *(row(kind, kind) for kind in shown), rule]
    return "\n".join([*rows, row("all", "all"), row("%", "all", percent=True)])


PROGRESS_WIDTH = 30  # characters of the progress bar


def progress(r: int, done: int, of: int, kind: str, columns: int) -> str:
    """The progress line shown on a terminal while placing: the rank, a bar of its labels placed so far (`done` of
    `of`: those of its finished steps), the step's kind; cut to `columns` - 1 so it never wraps."""
    full = round(PROGRESS_WIDTH * done / of) if of else PROGRESS_WIDTH
    text = f"placing rank {r}/{N_RANKS} {'█' * full}{'░' * (PROGRESS_WIDTH - full)} {done}/{of} {kind}"
    return text[:columns - 1]


def summary(items: Sequence[Placeable], outcome: Outcome, order: Sequence[str], color: bool, columns: int | None) -> str:
    """What place_by_priority did with `items`: the labels skipped, biggest first, then the rank_table."""
    placed = set(np.flatnonzero(outcome.kept).tolist())
    skipped = [items[i].name for _, members in outcome.plan for i in members if i not in placed]
    return f"labels skipped: {skipped}\n" + rank_table(outcome.plan, [item.kind for item in items], order, placed, color,
                                                       [item.name for item in items], columns)
