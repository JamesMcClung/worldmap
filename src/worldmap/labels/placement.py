"""Label placement: coordinate descent over candidate positions, minimising a cost that penalises overlaps,
leader-line clutter, drifting off the country and leaving the globe. All units are poster pixels."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from worldmap.frame import MapFrame
from worldmap.labels.geometry import nearest_point_on_rect, segment_hits_rects, segments_cross
from worldmap.labels.label import Label
from worldmap.labels.water import PX_PER_PT, WaterLabel

SEED = 7
OUTLINE_SIMPLIFY_PX = 0.5

# Candidate positions around a label's current centre: steps of (dx in widths, dy in heights), at several scales.
STEPS = [(a * 0.25, b) for a in (-2, -1, 0, 1, 2) for b in (-1, 0, 1) if (a, b) != (0, 0)]
STEP_SCALES = (0.4, 1, 2, 4)
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


def clipped_area(geom: BaseGeometry, b: Sequence[float]) -> float:
    """The area of `geom` inside the box b = [x0, y0, x1, y1]. clip_by_rect, not shapely.intersection: GEOS's overlay
    costs ~80 us a box, even against a small country; but clip_by_rect raises on a box edge a hair off a vertex."""
    try:
        return float(shapely.area(shapely.clip_by_rect(geom, *b)))
    except shapely.errors.GEOSException:
        return float(shapely.intersection(geom, shapely.box(*b)).area)


def overhang(on: Sequence[bool]) -> float:
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


def overlap_area(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Overlap areas of rects a (..., 1, 4) and b (m, 4), summed over b."""
    ox = np.clip(np.minimum(a[..., 2], b[:, 2]) - np.maximum(a[..., 0], b[:, 0]), 0, None)
    oy = np.clip(np.minimum(a[..., 3], b[:, 3]) - np.maximum(a[..., 1], b[:, 1]), 0, None)
    return (ox * oy).sum(-1)


def out_leaders(p: Placement) -> tuple[np.ndarray, np.ndarray]:
    """The 'out' labels' leader lines: (dots (d, 2), ends (d, 2))."""
    out = p.out & ~p.water
    end_x, end_y = nearest_point_on_rect(p.anchor_x[out], p.anchor_y[out], p.x[out], p.y[out], p.width[out], p.height[out])
    return np.stack([p.anchor_x[out], p.anchor_y[out]], -1), np.stack([end_x, end_y], -1)


def water_cost(p: Placement, label: WaterLabel, i: int, cx: np.ndarray, cy: np.ndarray, weights: CostWeights, outline_px: Polygon) -> np.ndarray:
    """Cost of putting water label i's anchor at each candidate (cx[k], cy[k])."""
    em = label.size * PX_PER_PT
    mine = p.owner == i
    g = p.parts[mine][None] + np.stack([cx, cy, cx, cy], -1)[:, None, :]  # (candidates, glyphs, 4)
    envelope = np.array([g[..., 0].min(), g[..., 1].min(), g[..., 2].max(), g[..., 3].max()])
    obstacles = np.vstack([near(boxes_of(p, ~p.water), envelope), near(glyph_boxes(p)[~mine], envelope)])
    cost = weights.overlap * overlap_area(g[..., None, :], obstacles).sum(1) / (em * em)

    dots, ends = out_leaders(p)
    if len(dots):
        dot_r = weights.dot_radius * p.height[p.out & ~p.water]
        gap_x = np.clip(np.maximum(g[..., None, 0] - dots[:, 0], dots[:, 0] - g[..., None, 2]), 0, None)
        gap_y = np.clip(np.maximum(g[..., None, 1] - dots[:, 1], dots[:, 1] - g[..., None, 3]), 0, None)
        gap = np.hypot(gap_x, gap_y).min(1)  # (candidates, dots): nearest glyph
        cost += weights.dot_cover * (np.clip(dot_r - gap, 0, None) / dot_r).sum(1)
        lo, hi = np.minimum(dots, ends), np.maximum(dots, ends)
        close = (lo[:, 0] <= envelope[2]) & (hi[:, 0] >= envelope[0]) & (lo[:, 1] <= envelope[3]) & (hi[:, 1] >= envelope[1])
        if close.any():
            cost += weights.leader_hits_label * segment_hits_rects(dots[close], ends[close], g[:, :, None, :]).any(1).sum(1)

    mid_x, mid_y = (g[..., 0] + g[..., 2]) / 2, (g[..., 1] + g[..., 3]) / 2
    area = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    on = shapely.contains_xy(label.geom, mid_x, mid_y)  # (candidates, glyphs)
    inside = (area * on).sum(1) / area.sum(1)
    ends = np.where(on.any(1), np.maximum(np.argmax(on, 1), np.argmax(on[:, ::-1], 1)) / on.shape[1], 1.0)  # overhang, per candidate
    cost += weights.overhang * (ends > OVERHANG_MAX)
    cost += weights.outside_water * (1 - inside) + weights.water_drift * np.hypot(cx - label.anchor_x, cy - label.anchor_y) / em
    corners_in = [shapely.contains_xy(outline_px, g[..., a], g[..., b]) for a, b in ((0, 1), (2, 1), (0, 3), (2, 3))]
    cost += weights.outside_globe * ~np.logical_and.reduce(corners_in).all(1)
    return cost


def cost_batch(p: Placement, labels: Sequence[Label | WaterLabel], i: int, cx: np.ndarray, cy: np.ndarray, weights: CostWeights, outline_px: Polygon) -> np.ndarray:
    """Cost of placing label i at each candidate centre (cx[k], cy[k])."""
    label = labels[i]
    if isinstance(label, WaterLabel):
        return water_cost(p, label, i, cx, cy, weights, outline_px)
    n_candidates = len(cx)
    w, h = p.width[i], p.height[i]
    anchor_x, anchor_y = p.anchor_x[i], p.anchor_y[i]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)  # (candidates, 4)
    others = (np.arange(p.n) != i) & ~p.water  # water labels are only seen through their glyph boxes, below
    ox, oy = p.x[others], p.y[others]
    ow, oh = p.width[others], p.height[others]
    other_boxes = np.stack([ox - ow / 2, oy - oh / 2, ox + ow / 2, oy + oh / 2], -1)  # (others, 4)
    overlap_x = np.clip(np.minimum(boxes[:, None, 2], other_boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], other_boxes[None, :, 0]), 0, None)
    overlap_y = np.clip(np.minimum(boxes[:, None, 3], other_boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], other_boxes[None, :, 1]), 0, None)
    cost = weights.overlap * (overlap_x * overlap_y).sum(1) / (h * h)

    # Other labels' leader dots and lines.
    other_out = p.out[others]
    if other_out.any():
        dot_x, dot_y = p.anchor_x[others][other_out], p.anchor_y[others][other_out]
        dot_r = weights.dot_radius * p.height[others][other_out]
        gap_x = np.clip(np.abs(cx[:, None] - dot_x[None]) - w / 2, 0, None)
        gap_y = np.clip(np.abs(cy[:, None] - dot_y[None]) - h / 2, 0, None)
        gap = np.hypot(gap_x, gap_y)
        cost += weights.dot_cover * (np.clip(dot_r[None] - gap, 0, None) / dot_r[None]).sum(1)
        end_x, end_y = nearest_point_on_rect(dot_x, dot_y, ox[other_out], oy[other_out], ow[other_out], oh[other_out])
        their_start = np.stack([dot_x, dot_y], -1)[None]
        their_end = np.stack([end_x, end_y], -1)[None]
        cost += weights.leader_hits_label * segment_hits_rects(their_start, their_end, boxes[:, None, :]).sum(1)

    if p.out[i]:
        start = np.array([anchor_x, anchor_y])[None, None, :]
        nx, ny = nearest_point_on_rect(anchor_x, anchor_y, cx, cy, w, h)
        end = np.stack([nx, ny], -1)[:, None, :]
        length = np.hypot(nx - anchor_x, ny - anchor_y)
        cost += weights.leader_length * ((length - weights.leader_target * h) / h) ** 2 + weights.leader_too_short * np.clip(weights.leader_min * h - length, 0, None) / h
        # Our leader vs other label boxes (unless that box covers our anchor: unavoidable).
        keep = ~((other_boxes[:, 0] <= anchor_x) & (anchor_x <= other_boxes[:, 2]) & (other_boxes[:, 1] <= anchor_y) & (anchor_y <= other_boxes[:, 3]))
        cost += weights.leader_hits_label * segment_hits_rects(start, end, other_boxes[None, keep, :]).sum(1)
        if other_out.any():
            cost += weights.leader_crosses_leader * segments_cross(np.broadcast_to(start, (n_candidates, 1, 2)), end, their_start, their_end).sum(1)
    else:
        inside = np.array([clipped_area(label.geom, b) for b in boxes]) / (w * h)
        cost += weights.outside_country * (1 - inside) + weights.drift * np.hypot(cx - anchor_x, cy - anchor_y) / h
    cost += weights.outside_globe * ~shapely.contains(outline_px, shapely.box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]))
    if p.water.any():  # water glyphs: overlap, and our own leader crossing them
        envelope = np.array([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()])
        cost += weights.overlap * overlap_area(boxes[:, None, :], near(glyph_boxes(p), envelope)) / (h * h)
        if p.out[i]:
            reach = np.array([min(anchor_x, end[..., 0].min()), min(anchor_y, end[..., 1].min()), max(anchor_x, end[..., 0].max()), max(anchor_y, end[..., 1].max())])
            g = near(glyph_boxes(p), reach + [-1, -1, 1, 1])
            g = g[~((g[:, 0] <= anchor_x) & (anchor_x <= g[:, 2]) & (g[:, 1] <= anchor_y) & (anchor_y <= g[:, 3]))]
            cost += weights.leader_hits_label * segment_hits_rects(start, end, g[None]).sum(1)
    return cost


def candidates(p: Placement, labels: Sequence[Label | WaterLabel], i: int, grid: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Candidate centres for label i; the first is its current position."""
    w, h = p.width[i], p.height[i]
    cx, cy = [p.x[i]], [p.y[i]]
    for scale in STEP_SCALES:
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
    for w in water:
        shapely.prepare(w.geom)
    if start is not None:
        p.x, p.y = start[0].copy(), start[1].copy()
    freedom = np.full(p.n, FREE) if freedom is None else np.asarray(freedom)
    home_x, home_y = (p.x.copy(), p.y.copy()) if home is None else (np.asarray(home[0]), np.asarray(home[1]))
    if floors is None:
        floors = np.array([SHRINK_MIN_PT * frame.label_scale] * len(labels) + [w.min_size for w in water])
    movable = np.flatnonzero(freedom != FIXED)
    outline_px = frame.to_px(Polygon(frame.outline), simplify=OUTLINE_SIMPLIFY_PX)
    shapely.prepare(outline_px)
    rng = np.random.default_rng(seed)

    def move_to_best(i: int, grid: bool = False, stay_bias: float = 0.0) -> bool:
        """Move label i to its cheapest candidate; True if it moved."""
        cx, cy = candidates(p, items, i, grid)
        cost = cost_batch(p, items, i, cx, cy, weights, outline_px)
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


def outside_outline(p: Placement, frame: MapFrame) -> set[int]:
    """Water labels with a glyph box not wholly inside the frame's outline."""
    outline_px = frame.to_px(Polygon(frame.outline), simplify=OUTLINE_SIMPLIFY_PX)
    g = glyph_boxes(p)
    inside = shapely.contains(outline_px, shapely.box(g[:, 0], g[:, 1], g[:, 2], g[:, 3]))
    return {int(i) for i in p.owner[~inside]}
