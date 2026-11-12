"""Priority: labels are placed in steps, most important first. Each kind ranks its labels by full size (area, or a
river's length) against the biggest of that kind in the frame; rank by rank, the kinds follow the --label order."""

from collections.abc import Callable, Sequence

import numpy as np

from worldmap.frame import MapFrame
from worldmap.labels.label import Label
from worldmap.labels.placement import FIXED, FREE, NUDGE, SHRINK_MIN_PT, Placement, clash, clipped_area, outside_outline, place_labels, rects_of, subset
from worldmap.labels.water import WATER_KINDS, WaterLabel

N_RANKS = 5
RANK_STEP = 10.0  # rank k holds the labels down to 1 / RANK_STEP^k of their kind's biggest; the last, the rest

OVERHANG_MAX = 0.3  # a label on its feature (not 'out') may hang off either end of it by no more than this share of its length
OVERHANG_SLICES = 16  # a country label's box is judged in this many upright slices; a water label by its glyphs
NUDGE_MIN_SCALE = 0.9  # a nudged label shrinks no further than this share of the size it settled at

Ranker = Callable[[Sequence[str], Sequence[float]], np.ndarray]


def label_order(label: Sequence[str]) -> list[str]:
    """The --label kinds in order, `water` standing for each water kind in its place, without repeats."""
    water = [k for k in WATER_KINDS if k != "water"]
    return list(dict.fromkeys(k for kind in label for k in (water if kind == "water" else [kind])))


def ranks_per_kind(kinds: Sequence[str], sizes: Sequence[float]) -> np.ndarray:
    """Each label's rank, 1..N_RANKS: how many factors of RANK_STEP it is below the biggest label of its kind.
    A size of 0 (or unknown) is the last rank."""
    size = np.asarray(sizes, float)
    kind_of = np.asarray(kinds, object)
    ranks = np.full(len(size), N_RANKS)
    for kind in set(kinds):
        mine = (kind_of == kind) & (size > 0)  # NaN compares False
        if mine.any():  # rank k: at least biggest / RANK_STEP^k (compared, not logged: exact at the steps)
            floors = size[mine].max() / RANK_STEP ** np.arange(1, N_RANKS)
            ranks[mine] = 1 + (size[mine][:, None] < floors).sum(1)
    return ranks


def steps(kinds: Sequence[str], sizes: Sequence[float], order: Sequence[str], rank: Ranker = ranks_per_kind) -> list[tuple[int, list[int]]]:
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


def overhang(on: Sequence[bool]) -> float:
    """How far a label hangs off its feature at its worse end: the share of its equal pieces (`on`: each one over
    the feature or not, in reading order) before the first one on it, or after the last; 1 if none is. A gap in
    the middle doesn't count, so a centred label may be mostly off as long as neither end is far off."""
    flags = np.asarray(on, bool)
    if not flags.any():
        return 1.0
    return max(int(np.argmax(flags)), int(np.argmax(flags[::-1]))) / len(flags)


def overhang_of(p: Placement, k: int, item: "Label | WaterLabel") -> float:
    """overhang of label k: its glyph boxes (a water label) or OVERHANG_SLICES slices of its box (a country), each
    on its feature if at least half over it."""
    if p.water[k]:
        rects, owner = rects_of(p)
        pieces = rects[owner == k]
    else:
        x0, y0, w, h = p.x[k] - p.width[k] / 2, p.y[k] - p.height[k] / 2, p.width[k], p.height[k]
        edges = x0 + w * np.arange(OVERHANG_SLICES + 1) / OVERHANG_SLICES
        pieces = np.column_stack([edges[:-1], np.full(OVERHANG_SLICES, y0), edges[1:], np.full(OVERHANG_SLICES, y0 + h)])
    area = (pieces[:, 2] - pieces[:, 0]) * (pieces[:, 3] - pieces[:, 1])
    return overhang([clipped_area(item.geom, piece) >= a / 2 for piece, a in zip(pieces, area)])


def place_by_priority(labels: list[Label], water: list[WaterLabel], frame: MapFrame, order: Sequence[str],
                      rank: Ranker = ranks_per_kind) -> tuple[Placement, list[Label], list[WaterLabel]]:
    """Place the labels step by step (steps, with `rank`), returning the placement of those kept (country labels first, then
    water labels, as the draw functions take them) and the kept labels. In each step the step's labels are free,
    the labels already placed of its rank and the rank before are nudged (they pay for leaving where they settled
    and shrink to no less than NUDGE_MIN_SCALE of their settled size), and the rest stay put. Then the step's
    labels are checked, smallest first: one clashing with a label still kept, or a water label hanging
    off its water at either end by over OVERHANG_MAX (overhang_of) (or, in a crop, outside the outline), is skipped for good; a country label hanging off its
    country by more than that becomes an 'out' label, with a leader, and the step is solved again."""
    items: list[Label | WaterLabel] = [*labels, *water]
    n_country = len(labels)
    plan = steps([item.kind for item in items], [item.full_size for item in items], order, rank)
    rank_of = {i: r for r, members in plan for i in members}
    x = {i: item.anchor_x for i, item in enumerate(items)}
    y = dict((i, item.anchor_y) for i, item in enumerate(items))
    settled: dict[int, tuple[float, float, float]] = {}  # x, y, size where each kept label settled
    kept: list[int] = []
    skipped: list[str] = []

    def size(i: int) -> float:
        item = items[i]
        return item.size if isinstance(item, WaterLabel) else float(item.text.get_fontsize())

    def name(i: int) -> str:
        item = items[i]
        return item.name if isinstance(item, Label) else item.text

    def floor(i: int) -> float:
        item = items[i]
        normal = item.min_size if isinstance(item, WaterLabel) else SHRINK_MIN_PT * frame.label_scale
        return max(normal, NUDGE_MIN_SCALE * settled[i][2]) if i in settled else normal

    def solve(r: int, new: list[int]) -> tuple[Placement, np.ndarray]:
        at = np.array(sorted([*kept, *new], key=lambda i: (i >= n_country, i)))  # placement index -> item
        freedom = np.array([FREE if i in new else NUDGE if rank_of[i] >= r - 1 else FIXED for i in at])
        p = place_labels([labels[i] for i in at if i < n_country], frame, water=[water[i - n_country] for i in at if i >= n_country],
                         start=(np.array([x[i] for i in at]), np.array([y[i] for i in at])), freedom=freedom,
                         home=(np.array([settled[i][0] if i in settled else x[i] for i in at]), np.array([settled[i][1] if i in settled else y[i] for i in at])),
                         floors=np.array([floor(i) for i in at]), seed_out=True)
        for k, i in enumerate(at):
            x[i], y[i] = float(p.x[k]), float(p.y[k])
            if i < n_country:  # its box follows its shrunk text, for the next placements
                labels[i].width, labels[i].height = float(p.width[k]), float(p.height[k])
        return p, at

    def rival(p: Placement, k: int, smaller: list[int]) -> int | None:
        """The smallest of the `smaller` placement indices whose rects overlap label k's were k at its anchor."""
        rects, owner = rects_of(p)
        a = rects[owner == k] + np.tile([p.anchor_x[k] - p.x[k], p.anchor_y[k] - p.y[k]], 2)
        for j in reversed(smaller):
            b = rects[owner == j]
            ox = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
            oy = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
            if ((ox > 0) & (oy > 0)).any():
                return j
        return None

    p, at = place_labels([], frame), np.zeros(0, int)
    for r, step in plan:
        new = list(step)
        while True:
            p, at = solve(r, new)
            among = np.ones(len(at), bool)
            outside = set() if frame.globe else outside_outline(p, frame)
            index = {int(i): k for k, i in enumerate(at)}
            retry, failed, to_leader = False, [], False
            for n, i in reversed(list(enumerate(new))):  # smallest first: of two clashing, the smaller goes
                k, item = index[i], items[i]
                if isinstance(item, Label) and not item.out and overhang_of(p, k, item) > OVERHANG_MAX:
                    item.out = True  # hanging off its country: beside it, with a leader, from the next solve
                    to_leader = True
                    continue
                off = isinstance(item, WaterLabel) and (overhang_of(p, k, item) > OVERHANG_MAX or k in outside)
                if off:  # pushed off its water by a smaller label of this step? then that one goes, and again
                    j = rival(p, k, [index[m] for m in new[n + 1:] if among[index[m]]])
                    if j is not None:
                        loser = int(at[j])
                        skipped.append(name(loser))
                        new.remove(loser)
                        retry = True
                        break
                if off or clash(p, k, among):
                    among[k] = False
                    failed.append(name(i))
            if not retry and not to_leader:
                skipped.extend(failed)
                break
        kept = [int(i) for i in at[among]]
        for i in new:
            if i in kept:
                settled[i] = (x[i], y[i], size(i))
        p, at = subset(p, among), at[among]
    print("labels skipped:", skipped)
    return p, [labels[i] for i in at if i < n_country], [water[i - n_country] for i in at if i >= n_country]
