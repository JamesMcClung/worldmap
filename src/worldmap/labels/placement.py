"""Label placement: coordinate descent over candidate positions, minimising a cost that penalises overlaps,
leader-line clutter, drifting off the country and leaving the globe. All units are poster pixels."""

import functools
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon

from worldmap.frame import MapFrame
from worldmap.labels import kernels
from worldmap.labels.geometry import Coverage, coverage, nearest_point_on_rect, segment_hits_rects
from worldmap.labels.label import Label
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
GRID_PASSES = 3

PASSES = 40  # coordinate-descent passes (stops early when nothing moves)
SHRINK_ROUNDS = 12  # rounds of shrinking still-overlapping labels...
SHRINK_PASSES = 15  # ...each followed by this many passes
SHRINK_FACTOR = 0.94
SHRINK_MIN_PT = 5
OVERHANG_MAX = 0.3  # a label on its feature (not 'out') may hang off either end of it by no more than this share of its length
STAY_BIAS = 1e-6  # ties go to the current position
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
    outside_country: float = 20  # fraction of an 'in' label's box off its country
    drift: float = 0.15  # an 'in' label's distance from its anchor, in heights
    outside_globe: float = 1000  # any part of the label outside the globe outline
    outside_water: float = 20  # share of a water label's glyph boxes off its water body
    overhang: float = 100  # a water label hanging off its water by over OVERHANG_MAX (overhang): it would be skipped
    water_drift: float = 0.15  # a water label's distance from its anchor, in ems
    nudge: float = 2.0  # a nudged label's distance from its home, in heights (water labels: ems; 'out' labels: none)...
    nudge_max: float = 2.0  # ...which can't be more than this


@dataclass
class Placement:
    """Per-label arrays: the country labels in their order, then the water labels. x/y are box centres for country
    labels; for a water label they are where its anchor goes, and width/height are its glyph boxes' extent (only
    used to scale its moves). Water labels' glyph boxes are in `parts` (offsets from x/y), `owner` their label."""

    x: np.ndarray
    y: np.ndarray
    width: np.ndarray
    height: np.ndarray
    anchor_x: np.ndarray
    anchor_y: np.ndarray
    out: np.ndarray  # bool
    water: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    parts: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    owner: np.ndarray = field(default_factory=lambda: np.zeros(0, int))

    @property
    def n(self) -> int:
        return len(self.x)


def overhang(on: Sequence[bool] | np.ndarray) -> float:
    """How far a label hangs off its feature at its worse end: the share of its equal pieces (`on`: each one over
    the feature or not, in reading order) before the first one on it, or after the last; 1 if none is. A gap in
    the middle doesn't count, so a centred label may be mostly off as long as neither end is far off."""
    flags = np.asarray(on, bool)
    if not flags.any():
        return 1.0
    return max(int(np.argmax(flags)), int(np.argmax(flags[::-1]))) / len(flags)


def leader_ends(p: Placement) -> tuple[np.ndarray, np.ndarray]:
    """Where each label's leader line meets its box (the box point nearest its anchor)."""
    return nearest_point_on_rect(p.anchor_x, p.anchor_y, p.x, p.y, p.width, p.height)


def glyph_boxes(p: Placement) -> np.ndarray:
    """Every water glyph box where its label is now, (parts, 4)."""
    x, y = p.x[p.owner], p.y[p.owner]
    return p.parts + np.stack([x, y, x, y], -1)


def boxes_of(p: Placement, keep: np.ndarray) -> np.ndarray:
    return np.stack([p.x[keep] - p.width[keep] / 2, p.y[keep] - p.height[keep] / 2, p.x[keep] + p.width[keep] / 2, p.y[keep] + p.height[keep] / 2], -1)


def near(rects: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """The rects (n, 4) that touch the envelope [x0, y0, x1, y1]."""
    return rects[(rects[:, 0] < envelope[2]) & (rects[:, 2] > envelope[0]) & (rects[:, 1] < envelope[3]) & (rects[:, 3] > envelope[1])]


@functools.cache
def weight_array(weights: CostWeights) -> np.ndarray:
    """The weights the cost kernels take, in their order (kernels.OVERLAP...)."""
    return np.array([weights.overlap, weights.dot_cover, weights.dot_radius, weights.leader_hits_label, weights.leader_crosses_leader,
                     weights.leader_length, weights.leader_target, weights.leader_too_short, weights.leader_min, weights.outside_country,
                     weights.drift, weights.outside_globe, weights.outside_water, weights.overhang, weights.water_drift], float)


def cost_batch(p: Placement, labels: Sequence[Label | WaterLabel], i: int, cx: np.ndarray, cy: np.ndarray, weights: CostWeights, outline: Coverage) -> np.ndarray:
    """Cost of placing label i at each candidate (cx[k], cy[k]): a country label's box centre, a water label's anchor.

    Country labels: overlap with other boxes and water glyph boxes; covering another 'out' label's dot, or its
    leader crossing our box; for an 'out' label, its leader's length, and its leader crossing other boxes (unless
    one covers its dot), glyph boxes and leaders; for an 'in' label, the share of its box off its country
    (geometry.Coverage) and its drift from its anchor; any corner outside the outline.
    Water labels: their glyph boxes' overlap with every box and other glyph box; covering a dot, or a leader
    crossing them; the share of glyph boxes whose centres are off their water, hanging off it by over
    OVERHANG_MAX; drift from the anchor in ems; any glyph corner outside the outline. Compiled: kernels."""
    label = labels[i]
    cover = coverage(label)
    o = (outline.table, outline.x0, outline.y0, outline.step)
    cx, cy = np.asarray(cx, float), np.asarray(cy, float)
    if isinstance(label, WaterLabel):
        return kernels.water_cost(i, cx, cy, p.parts[p.owner == i], label.size * PX_PER_PT, (float(label.anchor_x), float(label.anchor_y)),
                                  p.x, p.y, p.width, p.height, p.anchor_x, p.anchor_y, p.out, p.water, glyph_boxes(p), p.owner,
                                  cover.table, cover.x0, cover.y0, cover.step, *o, OVERHANG_MAX, weight_array(weights))
    return kernels.country_cost(i, cx, cy, p.x, p.y, p.width, p.height, p.anchor_x, p.anchor_y, p.out, p.water, glyph_boxes(p),
                                cover.table, cover.x0, cover.y0, cover.step, *o, weight_array(weights))


def candidates(p: Placement, labels: Sequence[Label | WaterLabel], i: int, grid: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Candidate centres for label i; the first is its current position."""
    w, h = p.width[i], p.height[i]
    cx, cy = [p.x[i]], [p.y[i]]
    for scale in WATER_STEP_SCALES if p.water[i] else STEP_SCALES:
        for a, b in STEPS:
            cx.append(p.x[i] + a * w * scale)
            cy.append(p.y[i] + b * h * scale)
    if p.out[i]:
        for k in RING_SCALES:
            cx += list(p.anchor_x[i] + np.cos(RING_ANGLES) * (w / 2 + RING_X_GAP * h) * k)
            cy += list(p.anchor_y[i] + np.sin(RING_ANGLES) * RING_Y * h * k)
    else:
        cx.append(p.anchor_x[i])
        cy.append(p.anchor_y[i])
        if grid and not p.water[i]:
            gx0, gy0, gx1, gy1 = labels[i].geom.bounds
            us = np.linspace(gx0 + w / 2, max(gx1 - w / 2, gx0 + w / 2), GRID_NX)
            vs = np.linspace(gy0 + h / 2, max(gy1 - h / 2, gy0 + h / 2), GRID_NY)
            for u in us:
                for v in vs:
                    cx.append(u)
                    cy.append(v)
    return np.array(cx), np.array(cy)


def rects_of(p: Placement) -> tuple[np.ndarray, np.ndarray]:
    """Every label's rects: a country label's box, a water label's glyph boxes; (rects (r, 4), owner (r,))."""
    country = np.flatnonzero(~p.water)
    return np.vstack([boxes_of(p, ~p.water), glyph_boxes(p)]), np.concatenate([country, p.owner])


def clash(p: Placement, i: int, among: np.ndarray, rects: tuple[np.ndarray, np.ndarray] | None = None) -> bool:
    """Whether label i clashes with another label of `among` (a mask), whatever their kinds: their boxes or glyph
    boxes overlap, or a leader line of one crosses the other (not a box covering that leader's own dot). `rects`:
    rects_of(p), if already at hand."""
    every, owner = rects_of(p) if rects is None else rects
    mine, theirs = owner == i, (owner != i) & among[owner]
    a = every[mine]
    envelope = np.array([a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()])
    b = near(every[theirs], envelope)
    ox = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    oy = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    if ((ox > 0) & (oy > 0)).any():
        return True
    leaders = np.flatnonzero(p.out & ~p.water & among)
    if not len(leaders):
        return False
    end_x, end_y = leader_ends(p)

    def crossed(dots: np.ndarray, ends: np.ndarray, hit: np.ndarray) -> bool:
        """Any of the leaders (dots (l, 2) to ends) crossing any of the rects `hit`, not those over its own dot."""
        if not len(dots) or not len(hit):
            return False
        r = hit[None]  # (1, m, 4) against (l, 1, 2)
        d, e = dots[:, None, :], ends[:, None, :]
        lo, hi = np.minimum(d, e), np.maximum(d, e)
        near_box = (r[..., 0] <= hi[..., 0]) & (r[..., 2] >= lo[..., 0]) & (r[..., 1] <= hi[..., 1]) & (r[..., 3] >= lo[..., 1])
        on_dot = (r[..., 0] <= d[..., 0]) & (d[..., 0] <= r[..., 2]) & (r[..., 1] <= d[..., 1]) & (d[..., 1] <= r[..., 3])
        check = near_box & ~on_dot
        return bool(check.any() and (segment_hits_rects(d, e, r) & check).any())

    others = leaders[leaders != i]
    dots = np.stack([p.anchor_x[others], p.anchor_y[others]], -1)
    if crossed(dots, np.stack([end_x[others], end_y[others]], -1), a):
        return True
    if p.out[i] and not p.water[i]:
        return crossed(np.array([[p.anchor_x[i], p.anchor_y[i]]]), np.array([[end_x[i], end_y[i]]]), every[theirs])
    return False


def clashes(p: Placement, among: np.ndarray | None = None) -> list[int]:
    """The labels of `among` (a mask, default all) that clash with another of them (clash)."""
    among = np.ones(p.n, bool) if among is None else among
    rects = rects_of(p)
    return [int(i) for i in np.flatnonzero(among) if clash(p, int(i), among, rects)]


def place_labels(labels: list[Label], frame: MapFrame, seed: int = SEED, weights: CostWeights = CostWeights(),
                 water: Sequence[WaterLabel] = (), start: tuple[np.ndarray, np.ndarray] | None = None,
                 freedom: np.ndarray | None = None, home: tuple[np.ndarray, np.ndarray] | None = None,
                 floors: np.ndarray | None = None, seed_out: bool | None = None) -> Placement:
    """Place the labels (largest first in `labels`) and the water labels (most important first). Labels still
    clashing after descent (clashes: with any label) are shrunk slightly (their text too) and the descent repeated. `start`: positions to start from, for all labels.
    `freedom` (per label, default FREE): FIXED labels stay where they start, at their size; NUDGE labels pay
    `weights.nudge` for their distance from `home` (default: where they start). `floors`: the size (pt) each label
    shrinks no lower than (default: SHRINK_MIN_PT, scaled, for country labels; a water label's min_size; a country
    label above it may still take one last step under it). `seed_out` (default: no `start`): first move the free
    'out' labels, in order, to their best spot, so they start separated."""
    items: list[Label | WaterLabel] = [*labels, *water]
    anchor_x = np.array([label.anchor_x for label in items])
    anchor_y = np.array([label.anchor_y for label in items])
    extent = [(w.parts[:, 2].max() - w.parts[:, 0].min(), w.parts[:, 3].max() - w.parts[:, 1].min()) for w in water]
    p = Placement(
        anchor_x.copy(), anchor_y.copy(),
        np.array([label.width for label in labels] + [e[0] for e in extent]),
        np.array([label.height for label in labels] + [e[1] for e in extent]),
        anchor_x, anchor_y, np.array([label.out for label in labels] + [False] * len(water), bool),
        np.array([False] * len(labels) + [True] * len(water), bool),
        np.vstack([np.zeros((0, 4)), *[w.parts for w in water]]),
        np.concatenate([np.zeros(0, int), *[np.full(len(w.parts), len(labels) + k) for k, w in enumerate(water)]]),
    )
    if start is not None:
        p.x, p.y = start[0].copy(), start[1].copy()
    freedom = np.full(p.n, FREE) if freedom is None else np.asarray(freedom)
    home_x, home_y = (p.x.copy(), p.y.copy()) if home is None else (np.asarray(home[0]), np.asarray(home[1]))
    if floors is None:
        floors = np.array([SHRINK_MIN_PT * frame.label_scale] * len(labels) + [w.min_size for w in water])
    movable = np.flatnonzero(freedom != FIXED)
    outline = outline_coverage(frame)
    rng = np.random.default_rng(seed)

    def move_to_best(i: int, grid: bool = False, stay_bias: float = 0.0) -> bool:
        """Move label i to its cheapest candidate; True if it moved."""
        cx, cy = candidates(p, items, i, grid)
        cost = cost_batch(p, items, i, cx, cy, weights, outline)
        if freedom[i] == NUDGE and not p.out[i]:  # an 'out' label has no spot of its own to keep, only its dot
            item = items[i]
            unit = item.size * PX_PER_PT if isinstance(item, WaterLabel) else p.height[i]
            away = np.hypot(cx - home_x[i], cy - home_y[i]) / unit
            cost += np.where(away <= weights.nudge_max, weights.nudge * away, np.inf)
        cost[0] -= stay_bias
        j = int(np.argmin(cost))
        p.x[i], p.y[i] = cx[j], cy[j]
        return j != 0

    def optimize(passes: int) -> None:
        for pass_no in range(passes):
            moved = [move_to_best(int(i), grid=pass_no < GRID_PASSES, stay_bias=STAY_BIAS) for i in movable[rng.permutation(len(movable))]]
            if not any(moved):
                break

    # Seed 'out' labels in largest-first order so they start separated, then descend.
    if start is None if seed_out is None else seed_out:
        for i in range(p.n):
            if p.out[i] and freedom[i] == FREE:
                move_to_best(i)
    optimize(PASSES)
    for _ in range(SHRINK_ROUNDS):
        clashing = [i for i in clashes(p) if freedom[i] != FIXED]
        if not clashing:
            break
        bad, water_bad = [i for i in clashing if not p.water[i]], [i for i in clashing if p.water[i]]
        for i in bad:
            size = float(labels[i].text.get_fontsize())
            if size > floors[i]:
                labels[i].text.set_fontsize(size * SHRINK_FACTOR)
                p.width[i] *= SHRINK_FACTOR
                p.height[i] *= SHRINK_FACTOR
        for i in water_bad:
            w = water[i - len(labels)]
            if w.size > floors[i]:
                factor = max(SHRINK_FACTOR, floors[i] / w.size)  # stop at its floor
                w.scale(factor)
                p.parts[p.owner == i] = w.parts
                p.width[i] *= factor
                p.height[i] *= factor
        optimize(SHRINK_PASSES)
    print("passes done; unresolved overlaps:", [labels[i].name if i < len(labels) else water[i - len(labels)].text for i in clashes(p)])
    return p


def subset(p: Placement, keep: np.ndarray) -> Placement:
    """The placement of the `keep` labels (boolean mask) only, in the same order."""
    new_index = np.cumsum(keep) - 1
    glyph_kept = keep[p.owner]
    return Placement(p.x[keep], p.y[keep], p.width[keep], p.height[keep], p.anchor_x[keep], p.anchor_y[keep], p.out[keep],
                     p.water[keep], p.parts[glyph_kept], new_index[p.owner[glyph_kept]])


_outline: list[tuple[MapFrame, Coverage]] = []  # the last frame's outline_coverage (one: each is ~32 MB)


def outline_coverage(frame: MapFrame) -> Coverage:
    """The frame's outline in poster px as a Coverage (cells of OUTLINE_MAX_CELLS at most), made once per frame."""
    if not _outline or _outline[0][0] is not frame:
        _outline[:] = [(frame, Coverage.of(frame.to_px(Polygon(frame.outline), simplify=OUTLINE_SIMPLIFY_PX), OUTLINE_MAX_CELLS))]
    return _outline[0][1]


def outside_outline(p: Placement, frame: MapFrame) -> set[int]:
    """Water labels with a glyph box not wholly inside the frame's outline."""
    inside = outline_coverage(frame).holds(glyph_boxes(p))
    return {int(i) for i in p.owner[~inside]}
