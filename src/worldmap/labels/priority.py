"""Priority: labels are placed in steps, most important first. Labels are ranked by full size (area, or a river's
length) into fifths of all those in the frame; rank by rank, the kinds follow the --label order. A kind is cut at
the first rank by which too many of all its labels have failed (saturated): that rank's labels and every later rank
of the kind are dropped."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np

from worldmap.frame import MapFrame
from worldmap.labels.geometry import coverage, on_feature
from worldmap.labels.placeable import Placeable
from worldmap.labels.placement import FIXED, FREE, NUDGE, OVERHANG_MAX, OVERHANG_SLICES, Placement, clash, outside_outline, overhang, place_labels, rects, subset

N_RANKS = 5
# A river ranks as an area of RIVER_KM2 * length (km) ** RIVER_POWER km2. The power is the trend of 751 rivers'
# Wikidata drainage basins (P2053) against their Natural Earth length: basin ~ 37 * L ** 1.19 (log-log fit, r 0.70,
# scatter x3). The true 37 moved rivers up about a rank; 1.6 puts 80 rivers in rank 1 of the world map with
# --label water (with 10 * L, 100 were).
RIVER_KM2 = 1.6
RIVER_POWER = 1.19

UNLABELED = N_RANKS + 1  # the rank of a feature with no label: drawn only if its kind was never cut
CUT_SHARE = 0.3  # a kind is cut at the first rank by which this share of all its labels (every rank) have failed...
CUT_MIN_DROPS = 5  # ...and at least this many do
NUDGE_MIN_SCALE = 0.9  # a nudged label shrinks no further than this share of the size it settled at

Ranker = Callable[[Sequence[str], Sequence[float]], np.ndarray]


class Unplaced(NamedTuple):
    """A feature of a labelled kind whose label doesn't fit at all (a river shorter than its name): ranked with the
    labels, it counts as one that failed."""
    name: str
    kind: str
    full_size: float


def saturated(failed: int, total: int) -> bool:
    """Whether a kind with `failed` of its `total` labels (every rank) failed so far is cut: CUT_MIN_DROPS and CUT_SHARE."""
    return failed >= CUT_MIN_DROPS and failed / total >= CUT_SHARE


def ranks_by_quantile(kinds: Sequence[str], sizes: Sequence[float]) -> np.ndarray:
    """Each label's rank, 1..N_RANKS: which fifth (for 5 ranks) of every label, whatever its kind, biggest first, it
    falls in, a river's length L (km) counting as RIVER_KM2 * L ** RIVER_POWER km2. A size of 0 (or unknown) is the
    last rank."""
    size = np.array([RIVER_KM2 * v ** RIVER_POWER if k == "rivers" else v for k, v in zip(kinds, sizes)], float)
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
    """What place_by_priority did with its entries."""
    entries: Sequence[Placeable | Unplaced]  # as given
    placement: Placement  # the kept labels'
    plan: list[tuple[int, list[int]]]  # its steps (steps), indices into its entries
    kept: np.ndarray  # bool, per entry (an Unplaced never is)
    ranks: np.ndarray  # int, per entry; UNLABELED for kinds not placed
    cut: dict[str, int]  # kind -> the rank it was cut at
    cut_fit: dict[str, int]  # kind -> its labels placed in that rank before the cut undid them

    def shown(self, kind: str, ranks: np.ndarray) -> np.ndarray:
        """Which features of `kind`, of these `ranks`, are drawn: those ranked before its cut, if it was cut."""
        return np.asarray(ranks) < self.cut.get(kind, UNLABELED + 1)

    def ranks_of(self, some: Sequence[Placeable | Unplaced]) -> np.ndarray:
        """The ranks of `some` of its entries (the same objects, in any order)."""
        at = {id(e): k for k, e in enumerate(self.entries)}
        return self.ranks[[at[id(e)] for e in some]]


@dataclass
class Settled:
    """What placing has decided so far: one placement of every label, solved step by step over the `active` ones
    (kept so far, and the step's), and where (home_x/y) and at what `size` each kept one settled (0 until it has)."""
    p: Placement
    active: np.ndarray
    home_x: np.ndarray
    home_y: np.ndarray
    size: np.ndarray

    @classmethod
    def of(cls, items: Sequence[Placeable]) -> "Settled":
        p = Placement.of(items)
        return cls(p, np.zeros(p.n, bool), p.x.copy(), p.y.copy(), np.zeros(p.n))

    def copy(self) -> "Settled":
        return Settled(self.p.copy(), self.active.copy(), self.home_x.copy(), self.home_y.copy(), self.size.copy())


@dataclass
class Cuts:
    """Each kind's labels failed so far, of all its labels (`of_kind`, every rank), and the kinds cut: at which rank,
    and how many of that rank's labels had placed (`fit`)."""
    of_kind: dict[str, int]
    failed: dict[str, int]
    at: dict[str, int] = field(default_factory=dict)
    fit: dict[str, int] = field(default_factory=dict)

    @classmethod
    def of(cls, plan: list[tuple[int, list[int]]], kinds: Sequence[str]) -> "Cuts":
        of_kind: dict[str, int] = {}
        for _, step in plan:
            of_kind[kinds[step[0]]] = of_kind.get(kinds[step[0]], 0) + len(step)
        return cls(of_kind, dict.fromkeys(of_kind, 0))

    def tally(self, kind: str, r: int, total: int, kept: int) -> bool:
        """Count a step of `kind`, rank r, that kept `kept` of its `total`; whether that cuts the kind (saturated)."""
        self.failed[kind] += total - kept
        if not saturated(self.failed[kind], self.of_kind[kind]):
            return False
        self.at[kind], self.fit[kind] = r, kept
        return True


def place_step(s: Settled, r: int, new: list[int], ranks: np.ndarray, min_size: np.ndarray, frame: MapFrame) -> None:
    """Place a step's labels (`new`, of rank r) among those settled: they are free, the settled ones of `ranks` r
    and r - 1 nudged (paying for leaving home, shrinking to no less than NUDGE_MIN_SCALE of their size), the rest
    fixed. Then check_step decides which go 'out' or are skipped, and the step is solved again until it puts none
    out and finds no rival to skip. Those kept settle where they are."""
    new = list(new)
    while new:
        s.active[new] = True
        freedom = np.where(ranks >= r - 1, NUDGE, FIXED)
        freedom[new] = FREE
        floors = np.maximum(min_size, NUDGE_MIN_SCALE * s.size)  # an unsettled label's size is 0
        place_labels(s.p, frame, active=s.active, freedom=freedom, home=(s.home_x, s.home_y), floors=floors)  # only settled labels are nudged
        verdict = check_step(s.p, new, s.active, frame)
        s.p.out[verdict.out] = True
        if verdict.rival is not None:
            new.remove(verdict.rival)
            s.active[verdict.rival] = False
        elif not verdict.out:
            s.active[verdict.skip] = False
            break
    kept = [i for i in new if s.active[i]]
    s.home_x[kept], s.home_y[kept], s.size[kept] = s.p.x[kept], s.p.y[kept], s.p.size[kept]


def place_by_priority(entries: Sequence[Placeable | Unplaced], frame: MapFrame, order: Sequence[str],
                      rank: Ranker = ranks_by_quantile, on_step: Callable[[int, int, int, str], None] | None = None) -> Outcome:
    """Place the labels among the `entries` step by step (steps, with `rank`; place_step), returning the Outcome.
    Before each step, on_step(rank, labels of the rank's steps done so far, of all its steps, the step's kind).
    An Unplaced entry is ranked with the labels and counts as failed in its step. A step after which its kind's
    failures so far saturate all its labels (Cuts.tally) is undone, and the kind cut there: its later steps are
    skipped."""
    kinds = [e.kind for e in entries]
    plan = steps(kinds, [e.full_size for e in entries], order, rank)
    ranks = np.full(len(entries), UNLABELED)
    for r, members in plan:
        ranks[members] = r
    labelled = [k for k, e in enumerate(entries) if not isinstance(e, Unplaced)]
    items = [e for e in entries if not isinstance(e, Unplaced)]
    label_of = np.full(len(entries), -1)  # entry -> its label's index in the placement
    label_of[labelled] = np.arange(len(labelled))
    s = Settled.of(items)
    min_size = np.array([item.min_size for item in items])
    cuts = Cuts.of(plan, kinds)
    in_rank = {r: sum(len(step) for q, step in plan if q == r) for r, _ in plan}
    done = dict.fromkeys(in_rank, 0)
    for r, step in plan:
        kind = kinds[step[0]]
        if on_step is not None and kind not in cuts.at:
            on_step(r, done[r], in_rank[r], kind)
        done[r] += len(step)  # a step a cut skips is done too: else the rank's progress never fills
        if kind in cuts.at:
            continue
        before = s.copy()
        members = [int(label_of[k]) for k in step if label_of[k] >= 0]
        place_step(s, r, members, ranks[labelled], min_size, frame)
        if cuts.tally(kind, r, len(step), int(s.active[members].sum())):
            s = before
    kept = np.zeros(len(entries), bool)
    kept[labelled] = s.active
    return Outcome(entries, subset(s.p, s.active), plan, kept, ranks, cuts.at, cuts.fit)
