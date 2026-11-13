"""Priority: labels are placed in steps, most important first. Labels are ranked by full size (area, or a river's
length) into fifths of all those in the frame; rank by rank, the kinds follow the --label order."""

import os
import sys
from collections.abc import Callable, Collection, Sequence

import numpy as np

from worldmap.frame import MapFrame
from worldmap.labels.geometry import Covered, coverage
from worldmap.labels.label import Label
from worldmap.labels.placeable import Placeable
from worldmap.labels.placement import FIXED, FREE, NUDGE, OVERHANG_MAX, Placement, clash, outside_outline, overhang, place_labels, rects, subset
from worldmap.labels.water import WATER_KINDS, WaterLabel

N_RANKS = 5
RIVER_KM2_PER_KM = 10.0  # a river ranks as an area of this many km2 per km of its length

OVERHANG_SLICES = 16  # a country label's box is judged in this many upright slices; a water label by its glyphs
NUDGE_MIN_SCALE = 0.9  # a nudged label shrinks no further than this share of the size it settled at

Ranker = Callable[[Sequence[str], Sequence[float]], np.ndarray]


def label_order(label: Sequence[str]) -> list[str]:
    """The --label kinds in order, `water` standing for each water kind in its place, without repeats."""
    water = [k for k in WATER_KINDS if k != "water"]
    return list(dict.fromkeys(k for kind in label for k in (water if kind == "water" else [kind])))


def ranks_by_quantile(kinds: Sequence[str], sizes: Sequence[float]) -> np.ndarray:
    """Each label's rank, 1..N_RANKS: which fifth (for 5 ranks) of every label, whatever its kind, biggest first, it
    falls in, a river's length (km) counting as RIVER_KM2_PER_KM times it in km2. A size of 0 (or unknown) is the
    last rank."""
    size = np.array([RIVER_KM2_PER_KM * v if k == "rivers" else v for k, v in zip(kinds, sizes)], float)
    ranks = np.full(len(size), N_RANKS)
    known = np.flatnonzero(size > 0)  # NaN compares False
    biggest_first = known[np.argsort(-size[known], kind="stable")]
    ranks[biggest_first] = 1 + np.arange(len(known)) * N_RANKS // max(len(known), 1)
    return ranks


def steps(kinds: Sequence[str], sizes: Sequence[float], order: Sequence[str], rank: Ranker = ranks_by_quantile) -> list[tuple[int, list[int]]]:
    """The placement steps: (rank, label indices, biggest first) for each rank, then each kind of `order` in turn.
    Empty steps, and kinds not in `order`, are left out."""
    ranks = rank(kinds, sizes)
    out = []
    for r in range(1, N_RANKS + 1):
        for kind in order:
            members = [i for i in range(len(kinds)) if kinds[i] == kind and ranks[i] == r]
            if members:
                out.append((r, sorted(members, key=lambda i: -np.nan_to_num(sizes[i]))))
    return out


DROPPED_WIDTH = 30  # the rank table's dropped labels of a kind are cut to this many characters, "..." included

GREEN, YELLOW, RED, PLAIN = "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[0m"


def rank_table(plan: list[tuple[int, list[int]]], kinds: Sequence[str], order: Sequence[str], placed: Collection[int], color: bool,
               names: Sequence[str] = ()) -> str:
    """How many of each rank's labels were placed, as text: a row per kind (in `order`, those with any), then of
    all kinds, then that as a percentage; on the right, of every rank, then as a percentage, then (with `names`) a
    kind's dropped labels, biggest first, cut to DROPPED_WIDTH ('-' if none; the plan has a kind's labels by size). In
    colour a cell is green if all were placed, red if none, else yellow; of the rank cells of a kind, only a yellow
    one shows how many ('28/36'). Lines set off the header and the totals, and the rank, total and dropped columns."""
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

    def gone(kind: str) -> str:
        text = ", ".join(dropped[kind]) or "-"
        return " " + (text if len(text) <= DROPPED_WIDTH else text[:DROPPED_WIDTH - 3].rstrip(", ") + "...")

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


def overhang_of(p: Placement, k: int, item: Covered) -> float:
    """overhang of label k: of its glyphs (a water label), each on its water if its box's centre is, as the solver
    has it (cost_batch); or of OVERHANG_SLICES slices of its box (a country), each on it if at least half over it."""
    if p.water[k]:
        g = rects(p)[p.owner == k]
        return overhang(coverage(item).contains((g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2))
    x0, y0, w, h = p.x[k] - p.width[k] / 2, p.y[k] - p.height[k] / 2, p.width[k], p.height[k]
    edges = x0 + w * np.arange(OVERHANG_SLICES + 1) / OVERHANG_SLICES
    slices = np.column_stack([edges[:-1], np.full(OVERHANG_SLICES, y0), edges[1:], np.full(OVERHANG_SLICES, y0 + h)])
    return overhang(coverage(item).area(slices) >= w * h / OVERHANG_SLICES / 2)


def place_by_priority(labels: list[Label], water: list[WaterLabel], frame: MapFrame, order: Sequence[str],
                      rank: Ranker = ranks_by_quantile) -> tuple[Placement, list[Label], list[WaterLabel]]:
    """Place the labels step by step (steps, with `rank`), returning the placement of those kept (country labels first, then
    water labels, as the draw functions take them) and the kept labels. In each step the step's labels are free,
    the labels already placed of its rank and the rank before are nudged (they pay for leaving where they settled
    and shrink to no less than NUDGE_MIN_SCALE of their settled size), and the rest stay put. Then the step's
    labels are checked, smallest first: one clashing with a label still kept, or a water label hanging
    off its water at either end by over OVERHANG_MAX (overhang_of) (or, in a crop, outside the outline), is skipped for good; a country label hanging off its
    country by more than that becomes an 'out' label, with a leader, and the step is solved again."""
    items: list[Placeable] = [*labels, *water]
    kinds = [item.kind for item in items]
    plan = steps(kinds, [item.full_size for item in items], order, rank)
    ranks = np.zeros(len(items), int)
    for r, members in plan:
        ranks[members] = r
    p = Placement.of(labels, water)  # one placement of every label, solved step by step over the active ones
    active = np.zeros(p.n, bool)  # the labels kept so far, and the step's
    settled = np.zeros(p.n, bool)  # kept, and where (home) and at what size each settled
    home_x, home_y, settled_size = p.x.copy(), p.y.copy(), np.zeros(p.n)
    min_size = np.array([item.min_size for item in items])
    skipped: list[str] = []

    def solve(r: int, new: list[int]) -> None:
        freedom = np.where(ranks >= r - 1, NUDGE, FIXED)
        freedom[new] = FREE
        home = (np.where(settled, home_x, p.x), np.where(settled, home_y, p.y))
        floors = np.where(settled, np.maximum(min_size, NUDGE_MIN_SCALE * settled_size), min_size)
        place_labels(p, items, frame, active=active, freedom=freedom, home=home, floors=floors)

    def rival(k: int, smaller: list[int]) -> int | None:
        """The smallest of the `smaller` labels whose rects overlap label k's were k at its anchor."""
        placed = rects(p)
        a = placed[p.owner == k] + np.tile([p.anchor_x[k] - p.x[k], p.anchor_y[k] - p.y[k]], 2)
        for j in reversed(smaller):
            b = placed[p.owner == j]
            ox = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
            oy = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
            if ((ox > 0) & (oy > 0)).any():
                return j
        return None

    for r, step in plan:
        new = list(step)
        while True:
            active[new] = True
            solve(r, new)
            among = active.copy()
            outside = set() if frame.globe else outside_outline(p, frame)
            retry, failed, to_leader = False, [], False
            for n, i in reversed(list(enumerate(new))):  # smallest first: of two clashing, the smaller goes
                if not p.water[i] and not p.out[i] and overhang_of(p, i, items[i]) > OVERHANG_MAX:
                    p.out[i] = True  # hanging off its country: beside it, with a leader, from the next solve
                    to_leader = True
                    continue
                off = bool(p.water[i]) and (overhang_of(p, i, items[i]) > OVERHANG_MAX or i in outside)
                if off:  # pushed off its water by a smaller label of this step? then that one goes, and again
                    loser = rival(i, [m for m in new[n + 1:] if among[m]])
                    if loser is not None:
                        skipped.append(items[loser].name)
                        new.remove(loser)
                        active[loser] = False
                        retry = True
                        break
                if off or clash(p, i, among):
                    among[i] = False
                    failed.append(items[i].name)
            if not retry and not to_leader:
                skipped.extend(failed)
                break
        active = among
        for i in new:
            if active[i]:
                settled[i] = True
                home_x[i], home_y[i], settled_size[i] = p.x[i], p.y[i], p.size[i]
    print("labels skipped:", skipped)
    print(rank_table(plan, kinds, order, set(np.flatnonzero(active).tolist()), sys.stdout.isatty() and "NO_COLOR" not in os.environ,
                     [item.name for item in items]))
    n_country = len(labels)
    return subset(p, active), [label for i, label in enumerate(labels) if active[i]], [w for k, w in enumerate(water) if active[n_country + k]]
