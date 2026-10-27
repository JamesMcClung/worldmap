"""Label placement: coordinate descent over candidate positions, minimising a cost that penalises overlaps,
leader-line clutter, drifting off the country and leaving the globe. All units are poster pixels."""

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon

from worldmap.frame import MapFrame
from worldmap.labels.geometry import nearest_point_on_rect, segment_hits_rects, segments_cross
from worldmap.labels.label import Label

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
STAY_BIAS = 1e-6  # ties go to the current position


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


@dataclass
class Placement:
    """Per-label arrays, in the order of the labels list. x/y are box centres."""

    x: np.ndarray
    y: np.ndarray
    width: np.ndarray
    height: np.ndarray
    anchor_x: np.ndarray
    anchor_y: np.ndarray
    out: np.ndarray  # bool

    @property
    def n(self) -> int:
        return len(self.x)


def leader_ends(p: Placement) -> tuple[np.ndarray, np.ndarray]:
    """Where each label's leader line meets its box (the box point nearest its anchor)."""
    return nearest_point_on_rect(p.anchor_x, p.anchor_y, p.x, p.y, p.width, p.height)


def cost_batch(p: Placement, labels: list[Label], i: int, cx: np.ndarray, cy: np.ndarray, weights: CostWeights, outline_px: Polygon) -> np.ndarray:
    """Cost of placing label i at each candidate centre (cx[k], cy[k])."""
    n_candidates = len(cx)
    w, h = p.width[i], p.height[i]
    anchor_x, anchor_y = p.anchor_x[i], p.anchor_y[i]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)  # (candidates, 4)
    others = np.arange(p.n) != i
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
        shapes = shapely.box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
        inside = shapely.area(shapely.intersection(shapes, labels[i].geom)) / (w * h)
        cost += weights.outside_country * (1 - inside) + weights.drift * np.hypot(cx - anchor_x, cy - anchor_y) / h
    cost += weights.outside_globe * ~shapely.contains(outline_px, shapely.box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]))
    return cost


def candidates(p: Placement, labels: list[Label], i: int, grid: bool = False) -> tuple[np.ndarray, np.ndarray]:
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
        if grid:
            gx0, gy0, gx1, gy1 = labels[i].geom.bounds
            us = np.linspace(gx0 + w / 2, max(gx1 - w / 2, gx0 + w / 2), GRID_NX)
            vs = np.linspace(gy0 + h / 2, max(gy1 - h / 2, gy0 + h / 2), GRID_NY)
            for u in us:
                for v in vs:
                    cx.append(u)
                    cy.append(v)
    return np.array(cx), np.array(cy)


def overlapping(p: Placement) -> list[int]:
    """Indices of labels whose boxes overlap another's."""
    bad = []
    for i in range(p.n):
        others = np.arange(p.n) != i
        ox = np.minimum(p.x[i] + p.width[i] / 2, p.x[others] + p.width[others] / 2) - np.maximum(p.x[i] - p.width[i] / 2, p.x[others] - p.width[others] / 2)
        oy = np.minimum(p.y[i] + p.height[i] / 2, p.y[others] + p.height[others] / 2) - np.maximum(p.y[i] - p.height[i] / 2, p.y[others] - p.height[others] / 2)
        if ((ox > 0) & (oy > 0)).any():
            bad.append(i)
    return bad


def place_labels(labels: list[Label], frame: MapFrame, seed: int = SEED, weights: CostWeights = CostWeights()) -> Placement:
    """Place the labels (largest first in `labels`). Labels still overlapping after descent are shrunk slightly
    (their text too) and the descent repeated."""
    anchor_x = np.array([label.anchor_x for label in labels])
    anchor_y = np.array([label.anchor_y for label in labels])
    p = Placement(
        anchor_x.copy(), anchor_y.copy(),
        np.array([label.width for label in labels]), np.array([label.height for label in labels]),
        anchor_x, anchor_y, np.array([label.out for label in labels]),
    )
    outline_px = frame.to_px(Polygon(frame.outline), simplify=OUTLINE_SIMPLIFY_PX)
    shapely.prepare(outline_px)
    rng = np.random.default_rng(seed)

    def move_to_best(i: int, grid: bool = False, stay_bias: float = 0.0) -> bool:
        """Move label i to its cheapest candidate; True if it moved."""
        cx, cy = candidates(p, labels, i, grid)
        cost = cost_batch(p, labels, i, cx, cy, weights, outline_px)
        cost[0] -= stay_bias
        j = int(np.argmin(cost))
        p.x[i], p.y[i] = cx[j], cy[j]
        return j != 0

    def optimize(passes: int) -> None:
        for pass_no in range(passes):
            moved = [move_to_best(i, grid=pass_no < GRID_PASSES, stay_bias=STAY_BIAS) for i in rng.permutation(p.n)]
            if not any(moved):
                break

    # Seed 'out' labels in largest-first order so they start separated, then descend.
    for i in range(p.n):
        if p.out[i]:
            move_to_best(i)
    optimize(PASSES)
    for _ in range(SHRINK_ROUNDS):
        bad = overlapping(p)
        if not bad:
            break
        for i in bad:
            size = labels[i].text.get_fontsize()
            if size > SHRINK_MIN_PT:
                labels[i].text.set_fontsize(size * SHRINK_FACTOR)
                p.width[i] *= SHRINK_FACTOR
                p.height[i] *= SHRINK_FACTOR
        optimize(SHRINK_PASSES)
    print("passes done; unresolved overlaps:", [labels[i].name for i in overlapping(p)])
    return p
