"""Priority: labels are placed in steps, most important first. Labels are ranked by full size (area, or a river's
length) into fifths of all those in the frame; rank by rank, the kinds follow the --label order."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from worldmap.frame import MapFrame
from worldmap.labels.geometry import coverage, on_feature
from worldmap.labels.placeable import Placeable
from worldmap.labels.placement import FIXED, FREE, NUDGE, OVERHANG_MAX, OVERHANG_SLICES, Placement, clash, outside_outline, overhang, place_labels, rects, subset

N_RANKS = 5
RIVER_KM2_PER_KM = 10.0  # a river ranks as an area of this many km2 per km of its length

NUDGE_MIN_SCALE = 0.9  # a nudged label shrinks no further than this share of the size it settled at

Ranker = Callable[[Sequence[str], Sequence[float]], np.ndarray]


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


def overhang_of(p: Placement, k: int) -> float:
    """overhang of label k where it is, its pieces judged as the solver does (geometry.on_feature)."""
    _, on = on_feature(coverage(p.items[k]), p.parts[p.owner == k], p.x[k], p.y[k], not p.water[k], OVERHANG_SLICES)
    return overhang(on)


def rival(p: Placement, k: int, smaller: list[int]) -> int | None:
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


@dataclass
class Verdict:
    """What check_step decided. Unless `out` and `rival` are empty, the step is solved again."""
    out: list[int]  # country labels to put 'out', beside their country with a leader
    rival: int | None  # a label of the step to skip, so that a bigger water label can keep to its water
    skip: list[int]  # labels to skip for good (once nothing else is left to try)


def check_step(p: Placement, new: list[int], active: np.ndarray, frame: MapFrame) -> Verdict:
    """Check a solved step's labels (`new`, biggest first) smallest first, so of two clashing the smaller goes,
    against the `active` labels not skipped so far. A country label clashing, or hanging off its country by over
    OVERHANG_MAX (overhang_of), goes 'out'. One already out and clashing, or a water label clashing, hanging off
    its water or (in a crop) outside the outline, is skipped; but a water label off its water first looks for a
    smaller label of the step overlapping it at its anchor (rival): that one goes instead, and the check stops.
    Each check sees the labels checked before it as decided. `p` is left as it is."""
    q = replace(p, out=p.out.copy())
    among = active.copy()
    outside = set() if frame.globe else outside_outline(q, frame)
    verdict = Verdict([], None, [])
    for n, i in reversed(list(enumerate(new))):
        if not q.water[i] and not q.out[i] and (overhang_of(q, i) > OVERHANG_MAX or clash(q, i, among)):
            q.out[i] = True
            verdict.out.append(i)
            continue
        off = bool(q.water[i]) and (overhang_of(q, i) > OVERHANG_MAX or i in outside)
        if off:
            verdict.rival = rival(q, i, [m for m in new[n + 1:] if among[m]])
            if verdict.rival is not None:
                return verdict
        if off or clash(q, i, among):
            among[i] = False
            verdict.skip.append(i)
    return verdict


@dataclass
class Outcome:
    """What place_by_priority did with its labels."""
    placement: Placement  # the kept labels'
    plan: list[tuple[int, list[int]]]  # its steps (steps), indices into its labels
    kept: np.ndarray  # bool, per label


def place_by_priority(items: Sequence[Placeable], frame: MapFrame, order: Sequence[str],
                      rank: Ranker = ranks_by_quantile, on_step: Callable[[int, int, int, str], None] | None = None) -> Outcome:
    """Place the labels (`items`) step by step (steps, with `rank`), returning the Outcome. Before each
    step, on_step(rank, labels of the rank's steps done so far, of all its steps, the step's kind). In each step the step's labels are free,
    the labels already placed of its rank and the rank before are nudged (they pay for leaving where they settled
    and shrink to no less than NUDGE_MIN_SCALE of their settled size), and the rest stay put. Then check_step
    decides which labels go 'out' or are skipped, and the step is solved again until it puts none out and finds
    no rival to skip."""
    kinds = [item.kind for item in items]
    plan = steps(kinds, [item.full_size for item in items], order, rank)
    ranks = np.zeros(len(items), int)
    for r, members in plan:
        ranks[members] = r
    p = Placement.of(items)  # one placement of every label, solved step by step over the active ones
    active = np.zeros(p.n, bool)  # the labels kept so far, and the step's
    settled = np.zeros(p.n, bool)  # kept, and where (home) and at what size each settled
    home_x, home_y, settled_size = p.x.copy(), p.y.copy(), np.zeros(p.n)
    min_size = np.array([item.min_size for item in items])

    def solve(r: int, new: list[int]) -> None:
        freedom = np.where(ranks >= r - 1, NUDGE, FIXED)
        freedom[new] = FREE
        home = (np.where(settled, home_x, p.x), np.where(settled, home_y, p.y))
        floors = np.where(settled, np.maximum(min_size, NUDGE_MIN_SCALE * settled_size), min_size)
        place_labels(p, frame, active=active, freedom=freedom, home=home, floors=floors)

    in_rank = {r: sum(len(step) for q, step in plan if q == r) for r, _ in plan}
    done = {r: 0 for r in in_rank}
    for r, step in plan:
        if on_step is not None:
            on_step(r, done[r], in_rank[r], kinds[step[0]])
        done[r] += len(step)
        new = list(step)
        while True:
            active[new] = True
            solve(r, new)
            verdict = check_step(p, new, active, frame)
            p.out[verdict.out] = True
            if verdict.rival is not None:
                new.remove(verdict.rival)
                active[verdict.rival] = False
            elif not verdict.out:
                active[verdict.skip] = False
                break
        for i in new:
            if active[i]:
                settled[i] = True
                home_x[i], home_y[i], settled_size[i] = p.x[i], p.y[i], p.size[i]
    return Outcome(subset(p, active), plan, active)

