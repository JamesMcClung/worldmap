"""What placing by priority shows: a progress line on a terminal while it runs, then a table of what it placed."""

from collections.abc import Collection, Mapping, Sequence

import numpy as np

from worldmap.labels.priority import N_RANKS, Outcome, Unplaced

DROPPED_WIDTH = 40  # the rank table's dropped labels of a kind fit in this many characters off a terminal
DROPPED_MIN = 20  # on a terminal, the dropped column fills the rest of the line, left out if narrower than this

GREEN, YELLOW, RED, PLAIN = "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[0m"


def rank_table(plan: list[tuple[int, list[int]]], kinds: Sequence[str], order: Sequence[str], placed: Collection[int], color: bool, names: Sequence[str] = (), columns: int | None = None, cut: Mapping[str, tuple[int, int]] = {}) -> str:
    """How many of each rank's labels were placed, as text: a row per kind (in `order`, those with any), then of
    all kinds, then that as a percentage; on the right, of every rank, then as a percentage, then (with `names`) a
    kind's dropped labels, biggest first ('-' if none), as many whole names as fit, then ', N more': in
    DROPPED_WIDTH characters, or on a terminal `columns` wide the rest of the line (no column if under
    DROPPED_MIN). In colour a cell is green if all were placed, red if none, else yellow; of a kind's rank cells,
    only a yellow one shows how many ('28/36'). The rank a kind was `cut` at (kind -> (rank, labels that fit
    before the cut undid them)) shows '(fit/)of', in red; the rank columns widen to fit it."""
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

    def fitted(kind: str, r: int) -> str | None:
        """The text of kind's cell of rank r if it was cut there."""
        at, fit = cut.get(kind, (0, 0))
        return f"({fit}/){total[kind][r - 1]}" if at == r else None

    rank_w = max([8, *(1 + len(text) for kind in order for r in range(1, N_RANKS + 1) if (text := fitted(kind, r)))])

    def cell(n: int, of: int, percent: bool, short: bool, width: int = 8, cut_text: str | None = None) -> str:
        if of == 0:
            return f"{'-':>{width}}"
        text = cut_text or (f"{round(100 * n / of)}%" if percent else f"{n}/{of}")
        if not color:
            return f"{text:>{width}}"
        hue = GREEN if n == of else RED if n == 0 else YELLOW
        return f"{hue}{str(of) if short and hue != YELLOW and not cut_text else text:>{width}}{PLAIN}"

    widths = [12, rank_w * N_RANKS, 16]  # the columns before the dropped one
    room = DROPPED_WIDTH if columns is None else columns - (sum(widths) + len(widths) * len(" │") + 1)
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
        parts = [f"{name:<12}", "".join(cell(a, b, percent, kind != "all", rank_w, None if percent else fitted(kind, r)) for r, (a, b) in enumerate(zip(n, of), 1)), cell(n.sum(), of.sum(), percent, False) + (" " * 8 if percent else cell(n.sum(), of.sum(), True, False))]
        if names:
            parts.append("" if kind == "all" else gone(kind))
        return line(parts)

    shown = [kind for kind in order if total[kind].any()]
    header = [f"{'rank':<12}", "".join(f"{r:>{rank_w}}" for r in range(1, N_RANKS + 1)), f"{'all':>8}{'%':>8}"]
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
    return text[: columns - 1]


def summary(outcome: Outcome, order: Sequence[str], color: bool, columns: int | None) -> str:
    """What place_by_priority did with its entries: the labels skipped in the steps it kept, biggest first, the
    kinds cut, where and how many that dropped, then the rank_table (which counts the Unplaced too)."""
    entries = outcome.entries
    placed = set(np.flatnonzero(outcome.kept).tolist())
    dropped = dict.fromkeys(outcome.cut, 0)
    skipped = []
    for r, members in outcome.plan:
        kind = entries[members[0]].kind
        if r >= outcome.cut.get(kind, r + 1):
            dropped[kind] += len(members)
        else:
            skipped += [entries[i].name for i in members if not isinstance(entries[i], Unplaced) and i not in placed]
    cut = ", ".join(f"{kind} from rank {r} ({dropped[kind]} dropped)" for kind, r in outcome.cut.items())
    return f"labels skipped: {skipped}\n" + (f"cut: {cut}\n" if cut else "") + rank_table(outcome.plan, [e.kind for e in entries], order, placed, color, [e.name for e in entries], columns, {kind: (r, outcome.cut_fit[kind]) for kind, r in outcome.cut.items()})
