"""Compiled (numba) cost of a label at each of its candidate positions: the solver's inner loop, as plain loops.
placement.cost_batch calls label_cost; the rules (and CostWeights' meanings) are documented there."""

import math

import numpy as np
from numba import njit

PARALLEL_EPS = 1e-9  # segment direction components below this count as parallel to that axis

# Indices into the weights array (placement.weight_array, in CostWeights' field order).
OVERLAP, DOT_COVER, DOT_RADIUS, LEADER_HITS_LABEL, LEADER_CROSSES_LEADER, LEADER_LENGTH, LEADER_TARGET, LEADER_TOO_SHORT, LEADER_MIN, \
    OFF_FEATURE, DRIFT, OUTSIDE_GLOBE, OVERHANG = range(13)


@njit(cache=True)
def segment_hits_rect(ax: float, ay: float, bx: float, by: float, x0: float, y0: float, x1: float, y1: float) -> bool:
    """Whether segment a-b touches rect [x0, y0, x1, y1] (slab clipping; a coordinate within PARALLEL_EPS of
    constant counts as parallel to that axis)."""
    tmin, tmax = 0.0, 1.0
    for p, d, lo, hi in ((ax, bx - ax, x0, x1), (ay, by - ay, y0, y1)):
        if abs(d) < PARALLEL_EPS:
            if p < lo or p > hi:
                return False
        else:
            t1, t2 = (lo - p) / d, (hi - p) / d
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
    return tmax >= tmin


@njit(cache=True)
def cross(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float:
    return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)


@njit(cache=True)
def segments_cross(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> bool:
    """Proper intersection of segments a-b and c-d (touching doesn't count)."""
    return cross(ax, ay, bx, by, cx, cy) * cross(ax, ay, bx, by, dx, dy) < 0 and cross(cx, cy, dx, dy, ax, ay) * cross(cx, cy, dx, dy, bx, by) < 0


@njit(cache=True)
def covered(table: np.ndarray, x0: float, y0: float, step: float, x: float, y: float) -> bool:
    """Coverage.contains for one point."""
    rows, cols = table.shape[0] - 1, table.shape[1] - 1
    u, v = math.floor((x - x0) / step), math.floor((y - y0) / step)
    if u < 0 or u >= cols or v < 0 or v >= rows:
        return False
    i, j = int(u), int(v)
    return table[j + 1, i + 1] - table[j, i + 1] - table[j + 1, i] + table[j, i] > 0


@njit(cache=True)
def below_left(table: np.ndarray, x0: float, y0: float, step: float, x: float, y: float) -> float:
    """Coverage.below_left for one point."""
    rows, cols = table.shape[0] - 1, table.shape[1] - 1
    u, v = min(max((x - x0) / step, 0.0), cols), min(max((y - y0) / step, 0.0), rows)
    i, j = min(int(u), cols - 1), min(int(v), rows - 1)
    fu, fv = u - i, v - j
    return ((1 - fv) * ((1 - fu) * table[j, i] + fu * table[j, i + 1]) + fv * ((1 - fu) * table[j + 1, i] + fu * table[j + 1, i + 1])) * step * step


@njit(cache=True)
def holds(table: np.ndarray, x0: float, y0: float, step: float, bx0: float, by0: float, bx1: float, by1: float) -> bool:
    """Coverage.holds for one box: all four corners covered."""
    return (covered(table, x0, y0, step, bx0, by0) and covered(table, x0, y0, step, bx1, by0)
            and covered(table, x0, y0, step, bx0, by1) and covered(table, x0, y0, step, bx1, by1))


@njit(cache=True)
def overlap(a0: float, a1: float, a2: float, a3: float, b0: float, b1: float, b2: float, b3: float) -> float:
    return max(min(a2, b2) - max(a0, b0), 0.0) * max(min(a3, b3) - max(a1, b1), 0.0)


@njit(cache=True)
def envelope(cx: np.ndarray, cy: np.ndarray, parts: np.ndarray, out: bool, ax: float, ay: float) -> tuple[float, float, float, float]:
    """The box round a label's rects (`parts`, relative) at every candidate (cx, cy) and, if it is 'out', its dot
    (ax, ay): its leaders too lie within it. Nothing beyond it can touch them."""
    ex0, ey0 = cx.min() + parts[:, 0].min(), cy.min() + parts[:, 1].min()
    ex1, ey1 = cx.max() + parts[:, 2].max(), cy.max() + parts[:, 3].max()
    if out:
        ex0, ey0, ex1, ey1 = min(ex0, ax), min(ey0, ay), max(ex1, ax), max(ey1, ay)
    return ex0, ey0, ex1, ey1


@njit(cache=True)
def nearby(i: int, ex0: float, ey0: float, ex1: float, ey1: float, rects: np.ndarray, owner: np.ndarray, active: np.ndarray,
           x: np.ndarray, y: np.ndarray, width: np.ndarray, height: np.ndarray, anchor_x: np.ndarray, anchor_y: np.ndarray,
           out: np.ndarray, dot_radius: float, obstacles: np.ndarray, leaders: np.ndarray) -> tuple[int, int]:
    """What can add to label i's cost within the envelope: into `obstacles` (len(rects), 4) the rects of other
    active labels that meet it, into `leaders` (len(x)) the other active 'out' labels whose dot (with its keep-clear
    radius) or leader meets it; returns how many of each. (The caller allocates them: the cost loops ran half as
    fast again over arrays this returned.)"""
    n_obs = 0
    for g in range(len(rects)):
        if owner[g] != i and active[owner[g]] and rects[g, 0] <= ex1 and rects[g, 2] >= ex0 and rects[g, 1] <= ey1 and rects[g, 3] >= ey0:
            obstacles[n_obs] = rects[g]
            n_obs += 1
    n_lead = 0
    for j in range(len(x)):
        if j == i or not out[j] or not active[j]:
            continue
        dx, dy = anchor_x[j], anchor_y[j]
        lx = min(max(dx, x[j] - width[j] / 2), x[j] + width[j] / 2)
        ly = min(max(dy, y[j] - height[j] / 2), y[j] + height[j] / 2)
        r = dot_radius * height[j]
        if min(dx - r, lx) <= ex1 and max(dx + r, lx) >= ex0 and min(dy - r, ly) <= ey1 and max(dy + r, ly) >= ey0:
            leaders[n_lead] = j
            n_lead += 1
    return n_obs, n_lead


@njit(cache=True)
def hang(on: np.ndarray) -> float:
    """placement.overhang: the share of `on` before its first True, or after its last, whichever is more; 1 if none."""
    first, last = -1, -1
    for p in range(len(on)):
        if on[p]:
            if first < 0:
                first = p
            last = p
    return 1.0 if first < 0 else max(first, len(on) - 1 - last) / len(on)


@njit(cache=True)
def label_cost(i: int, cx: np.ndarray, cy: np.ndarray, parts: np.ndarray, unit: float, by_area: bool,
               x: np.ndarray, y: np.ndarray, width: np.ndarray, height: np.ndarray, anchor_x: np.ndarray, anchor_y: np.ndarray,
               out: np.ndarray, active: np.ndarray, rects: np.ndarray, owner: np.ndarray,
               table: np.ndarray, tx0: float, ty0: float, tstep: float,
               outline: np.ndarray, ox0: float, oy0: float, ostep: float, overhang_max: float, slices: int, wt: np.ndarray) -> np.ndarray:
    """placement.cost_batch for label i (its rects `parts`, relative to its position; distances in `unit` px) at
    each candidate position (cx[k], cy[k]). How much of it is on its feature, and how far it hangs off it: by_area
    (a country), the area of the box round its rects over it, the box judged in `slices` upright slices, each on
    it if at least half over it; else (a water label) the area of the rects whose centres are on it, by rect.
    rects/owner: every label's rects where it is now, and whose (placement.rects); active: the labels being
    placed (the rest are ignored); table...: its feature's Coverage; outline...: the frame's."""
    n, q = len(cx), len(parts)
    ax, ay = anchor_x[i], anchor_y[i]
    cost = np.zeros(n)
    ex0, ey0, ex1, ey1 = envelope(cx, cy, parts, out[i], ax, ay)
    obstacles, leaders = np.empty((len(rects), 4)), np.empty(len(x), np.int64)
    n_obs, n_lead = nearby(i, ex0, ey0, ex1, ey1, rects, owner, active, x, y, width, height, anchor_x, anchor_y, out, wt[DOT_RADIUS], obstacles, leaders)
    p0, p1, p2, p3 = parts[:, 0].min(), parts[:, 1].min(), parts[:, 2].max(), parts[:, 3].max()  # the box round its rects
    on = np.zeros(slices if by_area else q, np.bool_)
    for k in range(n):
        c, inside, total = 0.0, 0.0, 0.0
        all_in = True
        nx, ny = min(max(ax, cx[k] + p0), cx[k] + p2), min(max(ay, cy[k] + p1), cy[k] + p3)  # its leader's end
        if out[i]:
            length = math.hypot(nx - ax, ny - ay)
            c += wt[LEADER_LENGTH] * ((length - wt[LEADER_TARGET] * unit) / unit) ** 2 + wt[LEADER_TOO_SHORT] * max(wt[LEADER_MIN] * unit - length, 0.0) / unit
        for p in range(q):
            g0, g1, g2, g3 = parts[p, 0] + cx[k], parts[p, 1] + cy[k], parts[p, 2] + cx[k], parts[p, 3] + cy[k]
            for o in range(n_obs):
                c += wt[OVERLAP] * overlap(g0, g1, g2, g3, obstacles[o, 0], obstacles[o, 1], obstacles[o, 2], obstacles[o, 3]) / (unit * unit)
            area = (g2 - g0) * (g3 - g1)
            total += area
            if not by_area:
                on[p] = covered(table, tx0, ty0, tstep, (g0 + g2) / 2, (g1 + g3) / 2)
                if on[p]:
                    inside += area
            if all_in and not holds(outline, ox0, oy0, ostep, g0, g1, g2, g3):
                all_in = False
        if out[i]:  # our leader through another's rect (unless that covers our dot: unavoidable)
            for o in range(n_obs):
                r0, r1, r2, r3 = obstacles[o, 0], obstacles[o, 1], obstacles[o, 2], obstacles[o, 3]
                if not (r0 <= ax <= r2 and r1 <= ay <= r3) and segment_hits_rect(ax, ay, nx, ny, r0, r1, r2, r3):
                    c += wt[LEADER_HITS_LABEL]
        for li in range(n_lead):  # other 'out' labels: their dots under our rects, their leaders through them or ours
            j = leaders[li]
            dx, dy = anchor_x[j], anchor_y[j]
            lx = min(max(dx, x[j] - width[j] / 2), x[j] + width[j] / 2)
            ly = min(max(dy, y[j] - height[j] / 2), y[j] + height[j] / 2)
            r = wt[DOT_RADIUS] * height[j]
            gap, hit = math.inf, False
            for p in range(q):
                g0, g1, g2, g3 = parts[p, 0] + cx[k], parts[p, 1] + cy[k], parts[p, 2] + cx[k], parts[p, 3] + cy[k]
                gap = min(gap, math.hypot(max(g0 - dx, dx - g2, 0.0), max(g1 - dy, dy - g3, 0.0)))
                if not hit and segment_hits_rect(dx, dy, lx, ly, g0, g1, g2, g3):
                    hit = True
            c += wt[DOT_COVER] * max(r - gap, 0.0) / r
            if hit:
                c += wt[LEADER_HITS_LABEL]
            if out[i] and segments_cross(ax, ay, nx, ny, dx, dy, lx, ly):
                c += wt[LEADER_CROSSES_LEADER]
        if not out[i]:  # an 'out' label is beside its feature by design
            if by_area:  # the area over it left of each slice edge, differenced
                bx0, by0, bx1, by1 = cx[k] + p0, cy[k] + p1, cx[k] + p2, cy[k] + p3
                half = (bx1 - bx0) * (by1 - by0) / slices / 2
                before = below_left(table, tx0, ty0, tstep, bx0, by1) - below_left(table, tx0, ty0, tstep, bx0, by0)
                start = before
                for e in range(slices):
                    xe = bx0 + (bx1 - bx0) * (e + 1) / slices
                    upto = below_left(table, tx0, ty0, tstep, xe, by1) - below_left(table, tx0, ty0, tstep, xe, by0)
                    on[e] = upto - before >= half
                    before = upto
                inside = before - start
            c += wt[OFF_FEATURE] * (1 - inside / total) + wt[DRIFT] * math.hypot(cx[k] - ax, cy[k] - ay) / unit
            if hang(on) > overhang_max:
                c += wt[OVERHANG]
        if not all_in:
            c += wt[OUTSIDE_GLOBE]
        cost[k] = c
    return cost


@njit(cache=True)
def candidates(x: float, y: float, w: float, h: float, ax: float, ay: float, out: bool, steps: np.ndarray, scales: np.ndarray,
               ring_cos: np.ndarray, ring_sin: np.ndarray, ring_scales: np.ndarray, ring_x_gap: float, ring_y: float,
               grid: bool, bounds: np.ndarray, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """placement.candidates: the current position, steps (a, b) in widths and heights at each scale, then for an
    'out' label rings round its anchor, else its anchor and (grid) an nx x ny grid over `bounds`."""
    n = 1 + len(scales) * len(steps) + (len(ring_scales) * len(ring_cos) if out else 1 + (nx * ny if grid else 0))
    cx, cy = np.empty(n), np.empty(n)
    cx[0], cy[0] = x, y
    k = 1
    for scale in scales:
        for s in range(len(steps)):
            cx[k], cy[k] = x + steps[s, 0] * w * scale, y + steps[s, 1] * h * scale
            k += 1
    if out:
        for r in ring_scales:
            for a in range(len(ring_cos)):
                cx[k], cy[k] = ax + ring_cos[a] * (w / 2 + ring_x_gap * h) * r, ay + ring_sin[a] * ring_y * h * r
                k += 1
    else:
        cx[k], cy[k] = ax, ay
        k += 1
        if grid:  # as np.linspace: start + i * step, the last exactly at the end
            u0, v0 = bounds[0] + w / 2, bounds[1] + h / 2
            u1, v1 = max(bounds[2] - w / 2, u0), max(bounds[3] - h / 2, v0)
            for iu in range(nx):
                u = u1 if iu == nx - 1 else iu * ((u1 - u0) / (nx - 1)) + u0
                for iv in range(ny):
                    cx[k], cy[k] = u, v1 if iv == ny - 1 else iv * ((v1 - v0) / (ny - 1)) + v0
                    k += 1
    return cx, cy


@njit(cache=True)
def leader_hits(dx: float, dy: float, ex: float, ey: float, r0: float, r1: float, r2: float, r3: float) -> bool:
    """Whether the leader from dot d to end e crosses rect r, unless r covers the dot (as placement.clash)."""
    if r0 <= dx <= r2 and r1 <= dy <= r3:
        return False
    if r0 > max(dx, ex) or r2 < min(dx, ex) or r1 > max(dy, ey) or r3 < min(dy, ey):
        return False
    return segment_hits_rect(dx, dy, ex, ey, r0, r1, r2, r3)


@njit(cache=True)
def clash(i: int, rects: np.ndarray, owner: np.ndarray, among: np.ndarray, out: np.ndarray,
          anchor_x: np.ndarray, anchor_y: np.ndarray, end_x: np.ndarray, end_y: np.ndarray) -> bool:
    """placement.clash: label i's rects overlap those of another label of `among`, or another's leader crosses
    them, or (an 'out' label) its leader crosses another's rects; rects/owner: placement.rects, p.owner; end_x/y: leader_ends."""
    mine = np.flatnonzero(owner == i)
    e0, e1, e2, e3 = rects[mine, 0].min(), rects[mine, 1].min(), rects[mine, 2].max(), rects[mine, 3].max()
    for b in range(len(rects)):
        j = owner[b]
        if j == i or not among[j] or rects[b, 0] >= e2 or rects[b, 2] <= e0 or rects[b, 1] >= e3 or rects[b, 3] <= e1:
            continue
        for a in mine:
            if min(rects[a, 2], rects[b, 2]) - max(rects[a, 0], rects[b, 0]) > 0 and min(rects[a, 3], rects[b, 3]) - max(rects[a, 1], rects[b, 1]) > 0:
                return True
    for j in range(len(out)):
        if j == i or not among[j] or not out[j]:
            continue
        for a in mine:
            if leader_hits(anchor_x[j], anchor_y[j], end_x[j], end_y[j], rects[a, 0], rects[a, 1], rects[a, 2], rects[a, 3]):
                return True
    if out[i]:
        for b in range(len(rects)):
            j = owner[b]
            if j != i and among[j] and leader_hits(anchor_x[i], anchor_y[i], end_x[i], end_y[i], rects[b, 0], rects[b, 1], rects[b, 2], rects[b, 3]):
                return True
    return False


@njit(cache=True)
def clashes(rects: np.ndarray, owner: np.ndarray, among: np.ndarray, out: np.ndarray,
            anchor_x: np.ndarray, anchor_y: np.ndarray, end_x: np.ndarray, end_y: np.ndarray) -> np.ndarray:
    """clash for every label of `among`: a mask."""
    hit = np.zeros(len(among), np.bool_)
    for i in range(len(among)):
        if among[i]:
            hit[i] = clash(i, rects, owner, among, out, anchor_x, anchor_y, end_x, end_y)
    return hit
