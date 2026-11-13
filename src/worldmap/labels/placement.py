"""Label placement: coordinate descent over candidate positions, minimising a cost that penalises overlaps,
leader-line clutter, drifting off the country and leaving the globe. All units are poster pixels."""

import functools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

from worldmap.frame import MapFrame
from worldmap.labels import kernels
from worldmap.labels.geometry import Coverage, coverage, nearest_point_on_rect
from worldmap.labels.label import Label
from worldmap.labels.placeable import Placeable
from worldmap.labels.water import PX_PER_PT, WaterLabel

SEED = 7
OUTLINE_SIMPLIFY_PX = 0.5
OUTLINE_MAX_CELLS = 4_000_000  # the outline's Coverage: cells of ~2.6 px on the world poster

# Candidate positions around a label's current centre: steps of (dx in widths, dy in heights), at several scales.
STEPS = [(a * 0.25, b) for a in (-2, -1, 0, 1, 2) for b in (-1, 0, 1) if (a, b) != (0, 0)]
STEP_SCALES = (0.4, 1, 2, 4)
WATER_STEP_SCALES = (0.1, 0.2, *STEP_SCALES)  # water labels also take small steps: in narrow water a big one leaves it
# 'out' labels also try rings around their anchor: 15 degree spacing, at several radii.
RING_ANGLES = np.radians(np.arange(0, 360, 15))
RING_SCALES = (1.0, 1.4, 2.0, 3.0, 4.5)
RING_X_GAP = 1.3  # ring half-width = label half-width + this many label heights
RING_Y = 1.8  # ring half-height, in label heights
# 'in' labels also scan a grid over their country in the first passes, so they can jump to where they fit.
GRID_NX, GRID_NY = 14, 9
STEP_ARRAY, SCALE_ARRAY, WATER_SCALE_ARRAY = np.array(STEPS), np.array(STEP_SCALES, float), np.array(WATER_STEP_SCALES, float)  # as the kernels take them
RING_COS, RING_SIN, RING_SCALE_ARRAY, NO_BOUNDS = np.cos(RING_ANGLES), np.sin(RING_ANGLES), np.array(RING_SCALES), np.zeros(4)
GRID_PASSES = 3

PASSES = 40  # coordinate-descent passes (stops early when nothing moves)
SHRINK_ROUNDS = 12  # rounds of shrinking still-overlapping labels...
SHRINK_PASSES = 15  # ...each followed by this many passes
OVERHANG_MAX = 0.3  # a label on its feature (not 'out') may hang off either end of it by no more than this share of its length
OVERHANG_SLICES = 16  # a country label's box is judged in this many upright slices; a water label by its glyphs
STAY_BIAS = 1e-6  # ties go to the current position
SHRINK_FACTOR = 0.94  # a label still clashing after a descent shrinks by this (down to its floor)
FIXED, NUDGE, FREE = 0, 1, 2  # a label's freedom: stays put; moves (and shrinks) a little, at a cost; moves freely


@dataclass(frozen=True)
class CostWeights:
    overlap: float = 40  # label-label overlap area / height^2
    dot_cover: float = 15  # covering another 'out' label's dot
    dot_radius: float = 0.35  # dot keep-clear radius, in that label's heights
    leader_hits_label: float = 20  # a leader line (ours or another's) crossing a label box
    leader_crosses_leader: float = 6
    leader_length: float = 1.0  # squared deviation of our leader's length from leader_target...
    leader_target: float = 1.3  # ...in label heights
    leader_too_short: float = 10  # leader shorter than leader_min heights
    leader_min: float = 0.9
    off_feature: float = 20  # share of a label (not 'out') off its feature: box area, or glyph boxes by their centres
    drift: float = 0.15  # a label's (not 'out') distance from its anchor, in units (unit_px: heights, or ems)
    outside_globe: float = 1000  # any part of the label outside the globe outline
    overhang: float = 100  # a label hanging off its feature by over OVERHANG_MAX: it would be skipped, or put 'out'
    nudge: float = 2.0  # a nudged label's distance from its home, in heights (water labels: ems; 'out' labels: none)...
    nudge_max: float = 2.0  # ...which can't be more than this


@dataclass
class Placement:
    """Per-label arrays: the country labels in their order, then the water labels. x/y are box centres for country
    labels; for a water label they are where its anchor goes, and width/height are its glyph boxes' extent (only
    used to scale its moves). Every label's rects are in `parts` (offsets from x/y), `owner` their label
    (ascending): a country label's box, a water label's glyph boxes. The labels themselves stay as laid out:
    shrinking changes only `size` (pt) and the extents here, which the draw functions read."""

    x: np.ndarray
    y: np.ndarray
    width: np.ndarray
    height: np.ndarray
    size: np.ndarray
    anchor_x: np.ndarray
    anchor_y: np.ndarray
    out: np.ndarray  # bool
    water: np.ndarray
    parts: np.ndarray
    owner: np.ndarray

    @property
    def n(self) -> int:
        return len(self.x)

    @classmethod
    def of(cls, labels: Sequence[Label], water: Sequence[WaterLabel] = ()) -> "Placement":
        """The labels at their anchors, at their sizes: the country labels in their order, then the water labels."""
        items: list[Placeable] = [*labels, *water]
        anchor_x = np.array([label.anchor_x for label in items], float)
        anchor_y = np.array([label.anchor_y for label in items], float)
        extent = [(w.parts[:, 2].max() - w.parts[:, 0].min(), w.parts[:, 3].max() - w.parts[:, 1].min()) for w in water]
        boxes = [np.array([[-label.width / 2, -label.height / 2, label.width / 2, label.height / 2]]) for label in labels]
        return cls(
            anchor_x.copy(), anchor_y.copy(),
            np.array([label.width for label in labels] + [e[0] for e in extent], float),
            np.array([label.height for label in labels] + [e[1] for e in extent], float),
            np.array([label.size for label in labels] + [w.size for w in water], float),
            anchor_x, anchor_y, np.array([label.out for label in labels] + [False] * len(water), bool),
            np.array([False] * len(labels) + [True] * len(water), bool),
            np.vstack([np.zeros((0, 4)), *boxes, *[w.parts for w in water]]),
            np.concatenate([np.arange(len(labels)), *[np.full(len(w.parts), len(labels) + k) for k, w in enumerate(water)]]),
        )


def overhang(on: Sequence[bool] | np.ndarray) -> float:
    """How far a label hangs off its feature at its worse end: the share of its equal pieces (`on`: each one over
    the feature or not, in reading order) before the first one on it, or after the last; 1 if none is. A gap in
    the middle doesn't count, so a centred label may be mostly off as long as neither end is far off."""
    return float(kernels.hang(np.asarray(on, bool)))


def unit_px(p: Placement, i: int) -> float:
    """What label i's moves are measured in: a country label's height, a water label's em."""
    return float(p.size[i] * PX_PER_PT if p.water[i] else p.height[i])


def shrink_step(size: float, floor: float) -> float:
    """The factor a label of `size` (pt) shrinks by: SHRINK_FACTOR, or less so as to stop at `floor`; 1 at or under it."""
    return 1.0 if size <= floor else max(SHRINK_FACTOR, floor / size)


def leader_ends(p: Placement) -> tuple[np.ndarray, np.ndarray]:
    """Where each label's leader line meets its box (the box point nearest its anchor)."""
    return nearest_point_on_rect(p.anchor_x, p.anchor_y, p.x, p.y, p.width, p.height)


def rects(p: Placement) -> np.ndarray:
    """Every label's rects (parts) where its label is now, (parts, 4)."""
    x, y = p.x[p.owner], p.y[p.owner]
    return p.parts + np.stack([x, y, x, y], -1)


@functools.cache
def weight_array(weights: CostWeights) -> np.ndarray:
    """The weights the cost kernels take, in their order (kernels.OVERLAP...)."""
    return np.array([weights.overlap, weights.dot_cover, weights.dot_radius, weights.leader_hits_label, weights.leader_crosses_leader,
                     weights.leader_length, weights.leader_target, weights.leader_too_short, weights.leader_min, weights.off_feature,
                     weights.drift, weights.outside_globe, weights.overhang], float)


def cost_batch(p: Placement, labels: Sequence[Placeable], i: int, cx: np.ndarray, cy: np.ndarray, weights: CostWeights, outline: Coverage,
               placed: np.ndarray | None = None, active: np.ndarray | None = None) -> np.ndarray:
    """Cost of placing label i at each candidate (cx[k], cy[k]): a country label's box centre, a water label's anchor.
    Every label is its rects (parts: a country label's box, a water label's glyph boxes), in units of unit_px:
    their overlap with other labels' rects; covering another 'out' label's dot, or its leader crossing them; any
    corner outside the outline. An 'out' label: its leader's length, and its leader crossing other rects (unless
    one covers its dot) and leaders. Any other: its share off its feature (geometry.Coverage: a country label's box
    area; a water label's glyph boxes whose centres are off), hanging off it by over OVERHANG_MAX (a country label
    in OVERHANG_SLICES slices of its box, a water label by glyph), and its drift from its anchor. Compiled: kernels.label_cost. `placed`: rects(p), if the caller keeps them up to date
    (place_labels does). `active` (a mask, default all): the labels it sees."""
    placed = rects(p) if placed is None else placed
    active = np.ones(p.n, bool) if active is None else active
    cover = coverage(labels[i])
    first, end = np.searchsorted(p.owner, [i, i + 1])  # its rects' rows (owner ascends)
    return kernels.label_cost(i, np.asarray(cx, float), np.asarray(cy, float), p.parts[first:end], unit_px(p, i), not p.water[i],
                              p.x, p.y, p.width, p.height, p.anchor_x, p.anchor_y, p.out, active, placed, p.owner,
                              cover.table, cover.x0, cover.y0, cover.step, outline.table, outline.x0, outline.y0, outline.step,
                              OVERHANG_MAX, OVERHANG_SLICES, weight_array(weights))


def candidates(p: Placement, labels: Sequence[Placeable], i: int, grid: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Candidate centres for label i; the first is its current position. Then STEPS at each of its step scales;
    for an 'out' label, rings round its anchor; else its anchor and (`grid`, country labels) a grid over its
    country. Compiled: kernels.candidates."""
    grid = grid and not p.water[i] and not p.out[i]
    return kernels.candidates(p.x[i], p.y[i], p.width[i], p.height[i], p.anchor_x[i], p.anchor_y[i], bool(p.out[i]), STEP_ARRAY,
                              WATER_SCALE_ARRAY if p.water[i] else SCALE_ARRAY, RING_COS, RING_SIN, RING_SCALE_ARRAY, RING_X_GAP, RING_Y,
                              grid, np.array(labels[i].geom.bounds) if grid else NO_BOUNDS, GRID_NX, GRID_NY)


def clash(p: Placement, i: int, among: np.ndarray) -> bool:
    """Whether label i clashes with another label of `among` (a mask), whatever their kinds: their boxes or glyph
    boxes overlap, or a leader line of one crosses the other (not a box covering that leader's own dot). Compiled:
    kernels.clash."""
    end_x, end_y = leader_ends(p)
    return bool(kernels.clash(i, rects(p), p.owner, among, p.out, p.water, p.anchor_x, p.anchor_y, end_x, end_y))


def clashes(p: Placement, among: np.ndarray | None = None) -> list[int]:
    """The labels of `among` (a mask, default all) that clash with another of them (clash)."""
    among = np.ones(p.n, bool) if among is None else among
    end_x, end_y = leader_ends(p)
    return [int(i) for i in np.flatnonzero(kernels.clashes(rects(p), p.owner, among, p.out, p.water, p.anchor_x, p.anchor_y, end_x, end_y))]


def place_labels(p: Placement, items: Sequence[Placeable], frame: MapFrame, seed: int = SEED, weights: CostWeights = CostWeights(),
                 active: np.ndarray | None = None, freedom: np.ndarray | None = None, home: tuple[np.ndarray, np.ndarray] | None = None,
                 floors: np.ndarray | None = None) -> None:
    """Place the `active` labels of `p` (a mask, default all; the rest are ignored), `items` being its labels in its
    order, from where they are, moving p. Free 'out' labels first move, in order, to their best spot, so they start
    separated; then coordinate descent. Labels still clashing after it (clashes) are shrunk a step (shrink_step:
    in p, not the labels) and the descent repeated. `freedom` (per label, default FREE): FIXED labels stay where they are, at their
    size; NUDGE labels pay `weights.nudge` for their distance from `home` (default: where they start). `floors`:
    the size (pt) each label shrinks no lower than (default: each label's min_size)."""
    active = np.ones(p.n, bool) if active is None else active
    freedom = np.full(p.n, FREE) if freedom is None else np.asarray(freedom)
    home_x, home_y = (p.x.copy(), p.y.copy()) if home is None else (np.asarray(home[0]), np.asarray(home[1]))
    if floors is None:
        floors = np.array([item.min_size for item in items])
    movable = np.flatnonzero(active & (freedom != FIXED))
    outline = outline_coverage(frame)
    rng = np.random.default_rng(seed)
    placed = rects(p)  # kept up to date as labels move and shrink: rebuilding them per cost was ~5% of a run
    first = np.searchsorted(p.owner, np.arange(p.n + 1))  # label k's rects: rows first[k]:first[k + 1] (owner ascends)

    def place_rects(i: int) -> None:
        rows = slice(first[i], first[i + 1])
        placed[rows] = p.parts[rows] + np.array([p.x[i], p.y[i], p.x[i], p.y[i]])

    def move_to_best(i: int, grid: bool = False, stay_bias: float = 0.0) -> bool:
        """Move label i to its cheapest candidate; True if it moved."""
        cx, cy = candidates(p, items, i, grid)
        cost = cost_batch(p, items, i, cx, cy, weights, outline, placed, active)
        if freedom[i] == NUDGE and not p.out[i]:  # an 'out' label has no spot of its own to keep, only its dot
            away = np.hypot(cx - home_x[i], cy - home_y[i]) / unit_px(p, i)
            cost += np.where(away <= weights.nudge_max, weights.nudge * away, np.inf)
        cost[0] -= stay_bias
        j = int(np.argmin(cost))
        p.x[i], p.y[i] = cx[j], cy[j]
        place_rects(i)
        return j != 0

    def optimize(passes: int) -> None:
        for pass_no in range(passes):
            moved = [move_to_best(int(i), grid=pass_no < GRID_PASSES, stay_bias=STAY_BIAS) for i in movable[rng.permutation(len(movable))]]
            if not any(moved):
                break

    for k in np.flatnonzero(active & p.out & (freedom == FREE)).tolist():
        move_to_best(k)
    optimize(PASSES)
    for _ in range(SHRINK_ROUNDS):
        clashing = [i for i in clashes(p, active) if freedom[i] != FIXED]
        if not clashing:
            break
        for i in clashing:
            factor = shrink_step(p.size[i], floors[i])
            p.size[i] *= factor
            p.width[i] *= factor
            p.height[i] *= factor
            p.parts[first[i]:first[i + 1]] *= factor
            place_rects(i)
        optimize(SHRINK_PASSES)


def subset(p: Placement, keep: np.ndarray) -> Placement:
    """The placement of the `keep` labels (boolean mask) only, in the same order."""
    new_index = np.cumsum(keep) - 1
    glyph_kept = keep[p.owner]
    return Placement(p.x[keep], p.y[keep], p.width[keep], p.height[keep], p.size[keep], p.anchor_x[keep], p.anchor_y[keep], p.out[keep],
                     p.water[keep], p.parts[glyph_kept], new_index[p.owner[glyph_kept]])


_outline: list[tuple[MapFrame, Coverage]] = []  # the last frame's outline_coverage (one: each is ~32 MB)


def outline_coverage(frame: MapFrame) -> Coverage:
    """The frame's outline in poster px as a Coverage (cells of OUTLINE_MAX_CELLS at most), made once per frame."""
    if not _outline or _outline[0][0] is not frame:
        _outline[:] = [(frame, Coverage.of(frame.to_px(Polygon(frame.outline), simplify=OUTLINE_SIMPLIFY_PX), OUTLINE_MAX_CELLS))]
    return _outline[0][1]


def outside_outline(p: Placement, frame: MapFrame) -> set[int]:
    """Water labels with a glyph box not wholly inside the frame's outline."""
    inside = outline_coverage(frame).holds(rects(p))
    return {int(i) for i in p.owner[~inside & p.water[p.owner]]}
