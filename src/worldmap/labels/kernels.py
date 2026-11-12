"""Compiled (numba) cost of a label at each of its candidate positions: the solver's inner loop, as plain loops.
placement.cost_batch calls these; the rules (and CostWeights' meanings) are documented there."""

import math

import numpy as np
from numba import njit

PARALLEL_EPS = 1e-9  # segment direction components below this count as parallel to that axis

# Indices into the weights array (placement.weight_array, in CostWeights' field order).
OVERLAP, DOT_COVER, DOT_RADIUS, LEADER_HITS_LABEL, LEADER_CROSSES_LEADER, LEADER_LENGTH, LEADER_TARGET, LEADER_TOO_SHORT, LEADER_MIN, \
    OUTSIDE_COUNTRY, DRIFT, OUTSIDE_GLOBE, OUTSIDE_WATER, OVERHANG, WATER_DRIFT = range(15)


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
def covered_area(table: np.ndarray, x0: float, y0: float, step: float, bx0: float, by0: float, bx1: float, by1: float) -> float:
    """Coverage.area for one box."""
    return (below_left(table, x0, y0, step, bx1, by1) - below_left(table, x0, y0, step, bx0, by1)
            - below_left(table, x0, y0, step, bx1, by0) + below_left(table, x0, y0, step, bx0, by0))


@njit(cache=True)
def holds(table: np.ndarray, x0: float, y0: float, step: float, bx0: float, by0: float, bx1: float, by1: float) -> bool:
    """Coverage.holds for one box: all four corners covered."""
    return (covered(table, x0, y0, step, bx0, by0) and covered(table, x0, y0, step, bx1, by0)
            and covered(table, x0, y0, step, bx0, by1) and covered(table, x0, y0, step, bx1, by1))


@njit(cache=True)
def overlap(a0: float, a1: float, a2: float, a3: float, b0: float, b1: float, b2: float, b3: float) -> float:
    return max(min(a2, b2) - max(a0, b0), 0.0) * max(min(a3, b3) - max(a1, b1), 0.0)


@njit(cache=True)
def country_cost(i: int, cx: np.ndarray, cy: np.ndarray, x: np.ndarray, y: np.ndarray, width: np.ndarray, height: np.ndarray,
                 anchor_x: np.ndarray, anchor_y: np.ndarray, out: np.ndarray, water: np.ndarray, glyphs: np.ndarray,
                 table: np.ndarray, tx0: float, ty0: float, tstep: float,
                 outline: np.ndarray, ox0: float, oy0: float, ostep: float, wt: np.ndarray) -> np.ndarray:
    """placement.cost_batch for country label i at each candidate centre (cx[k], cy[k]). glyphs: every water glyph
    box where it is now (glyph_boxes); table...: its country's Coverage; outline...: the frame's."""
    n, m = len(cx), len(x)
    w, h = width[i], height[i]
    ax, ay = anchor_x[i], anchor_y[i]
    cost = np.zeros(n)
    # Our candidate boxes' envelope: glyph boxes beyond it can't touch them.
    ex0, ey0, ex1, ey1 = cx.min() - w / 2, cy.min() - h / 2, cx.max() + w / 2, cy.max() + h / 2
    if out[i]:  # our leaders reach from the dot to the candidate boxes: glyphs beyond that can't meet them either
        ex0, ey0, ex1, ey1 = min(ex0, ax), min(ey0, ay), max(ex1, ax), max(ey1, ay)
    near = np.empty(len(glyphs), np.int64)
    n_near = 0
    for g in range(len(glyphs)):
        if glyphs[g, 0] <= ex1 and glyphs[g, 2] >= ex0 and glyphs[g, 1] <= ey1 and glyphs[g, 3] >= ey0:
            near[n_near] = g
            n_near += 1
    for k in range(n):
        b0, b1, b2, b3 = cx[k] - w / 2, cy[k] - h / 2, cx[k] + w / 2, cy[k] + h / 2
        c = 0.0
        if out[i]:
            nx, ny = min(max(ax, b0), b2), min(max(ay, b1), b3)  # our leader's end
            length = math.hypot(nx - ax, ny - ay)
            c += wt[LEADER_LENGTH] * ((length - wt[LEADER_TARGET] * h) / h) ** 2 + wt[LEADER_TOO_SHORT] * max(wt[LEADER_MIN] * h - length, 0.0) / h
        for j in range(m):
            if j == i or water[j]:
                continue
            o0, o1, o2, o3 = x[j] - width[j] / 2, y[j] - height[j] / 2, x[j] + width[j] / 2, y[j] + height[j] / 2
            c += wt[OVERLAP] * overlap(b0, b1, b2, b3, o0, o1, o2, o3) / (h * h)
            if out[j]:  # its dot under our box, its leader through it
                dx, dy = anchor_x[j], anchor_y[j]
                r = wt[DOT_RADIUS] * height[j]
                gap = math.hypot(max(abs(cx[k] - dx) - w / 2, 0.0), max(abs(cy[k] - dy) - h / 2, 0.0))
                c += wt[DOT_COVER] * max(r - gap, 0.0) / r
                lx, ly = min(max(dx, o0), o2), min(max(dy, o1), o3)
                if segment_hits_rect(dx, dy, lx, ly, b0, b1, b2, b3):
                    c += wt[LEADER_HITS_LABEL]
                if out[i] and segments_cross(ax, ay, nx, ny, dx, dy, lx, ly):
                    c += wt[LEADER_CROSSES_LEADER]
            if out[i] and not (o0 <= ax <= o2 and o1 <= ay <= o3) and segment_hits_rect(ax, ay, nx, ny, o0, o1, o2, o3):
                c += wt[LEADER_HITS_LABEL]  # our leader through its box (unless that covers our dot: unavoidable)
        if not out[i]:
            inside = covered_area(table, tx0, ty0, tstep, b0, b1, b2, b3) / (w * h)
            c += wt[OUTSIDE_COUNTRY] * (1 - inside) + wt[DRIFT] * math.hypot(cx[k] - ax, cy[k] - ay) / h
        if not holds(outline, ox0, oy0, ostep, b0, b1, b2, b3):
            c += wt[OUTSIDE_GLOBE]
        for gi in range(n_near):
            g = near[gi]
            g0, g1, g2, g3 = glyphs[g, 0], glyphs[g, 1], glyphs[g, 2], glyphs[g, 3]
            c += wt[OVERLAP] * overlap(b0, b1, b2, b3, g0, g1, g2, g3) / (h * h)
            if out[i] and not (g0 <= ax <= g2 and g1 <= ay <= g3) and segment_hits_rect(ax, ay, nx, ny, g0, g1, g2, g3):
                c += wt[LEADER_HITS_LABEL]
        cost[k] = c
    return cost


@njit(cache=True)
def water_cost(i: int, cx: np.ndarray, cy: np.ndarray, parts: np.ndarray, em: float, anchor: tuple[float, float],
               x: np.ndarray, y: np.ndarray, width: np.ndarray, height: np.ndarray, anchor_x: np.ndarray, anchor_y: np.ndarray,
               out: np.ndarray, water: np.ndarray, glyphs: np.ndarray, owner: np.ndarray,
               table: np.ndarray, tx0: float, ty0: float, tstep: float,
               outline: np.ndarray, ox0: float, oy0: float, ostep: float, overhang_max: float, wt: np.ndarray) -> np.ndarray:
    """placement.water_cost for water label i (its glyph boxes `parts`, relative) with its anchor at each candidate
    (cx[k], cy[k]). glyphs/owner: every water glyph box where it is now, and whose; table...: its water's
    Coverage; outline...: the frame's."""
    n, q = len(cx), len(parts)
    cost = np.zeros(n)
    ex0, ey0 = cx.min() + parts[:, 0].min(), cy.min() + parts[:, 1].min()
    ex1, ey1 = cx.max() + parts[:, 2].max(), cy.max() + parts[:, 3].max()
    # Obstacles near our glyphs' envelope: country boxes, other water labels' glyph boxes.
    obstacles = np.empty((len(x) + len(glyphs), 4))
    n_obs = 0
    for j in range(len(x)):
        if not water[j]:
            o0, o1, o2, o3 = x[j] - width[j] / 2, y[j] - height[j] / 2, x[j] + width[j] / 2, y[j] + height[j] / 2
            if o0 < ex1 and o2 > ex0 and o1 < ey1 and o3 > ey0:
                obstacles[n_obs] = (o0, o1, o2, o3)
                n_obs += 1
    for g in range(len(glyphs)):
        if owner[g] != i and glyphs[g, 0] < ex1 and glyphs[g, 2] > ex0 and glyphs[g, 1] < ey1 and glyphs[g, 3] > ey0:
            obstacles[n_obs] = glyphs[g]
            n_obs += 1
    leaders = np.empty(len(x), np.int64)  # 'out' country labels whose dot (with its radius) or leader nears our envelope
    n_lead = 0
    for j in range(len(x)):
        if water[j] or not out[j]:
            continue
        dx, dy = anchor_x[j], anchor_y[j]
        lx = min(max(dx, x[j] - width[j] / 2), x[j] + width[j] / 2)
        ly = min(max(dy, y[j] - height[j] / 2), y[j] + height[j] / 2)
        r = wt[DOT_RADIUS] * height[j]
        if min(dx - r, lx) <= ex1 and max(dx + r, lx) >= ex0 and min(dy - r, ly) <= ey1 and max(dy + r, ly) >= ey0:
            leaders[n_lead] = j
            n_lead += 1
    on = np.zeros(q, np.bool_)
    for k in range(n):
        c, inside, total = 0.0, 0.0, 0.0
        all_in = True
        for p in range(q):
            g0, g1, g2, g3 = parts[p, 0] + cx[k], parts[p, 1] + cy[k], parts[p, 2] + cx[k], parts[p, 3] + cy[k]
            for o in range(n_obs):
                c += wt[OVERLAP] * overlap(g0, g1, g2, g3, obstacles[o, 0], obstacles[o, 1], obstacles[o, 2], obstacles[o, 3]) / (em * em)
            area = (g2 - g0) * (g3 - g1)
            on[p] = covered(table, tx0, ty0, tstep, (g0 + g2) / 2, (g1 + g3) / 2)
            total += area
            if on[p]:
                inside += area
            if all_in and not holds(outline, ox0, oy0, ostep, g0, g1, g2, g3):
                all_in = False
        for li in range(n_lead):  # 'out' country labels: their dots under our glyphs, their leaders through them
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
        first, last = -1, -1
        for p in range(q):
            if on[p]:
                if first < 0:
                    first = p
                last = p
        hang = 1.0 if first < 0 else max(first, q - 1 - last) / q
        if hang > overhang_max:
            c += wt[OVERHANG]
        c += wt[OUTSIDE_WATER] * (1 - inside / total) + wt[WATER_DRIFT] * math.hypot(cx[k] - anchor[0], cy[k] - anchor[1]) / em
        if not all_in:
            c += wt[OUTSIDE_GLOBE]
        cost[k] = c
    return cost


@njit(cache=True)
def leader_hits(dx: float, dy: float, ex: float, ey: float, r0: float, r1: float, r2: float, r3: float) -> bool:
    """Whether the leader from dot d to end e crosses rect r, unless r covers the dot (as placement.clash)."""
    if r0 <= dx <= r2 and r1 <= dy <= r3:
        return False
    if r0 > max(dx, ex) or r2 < min(dx, ex) or r1 > max(dy, ey) or r3 < min(dy, ey):
        return False
    return segment_hits_rect(dx, dy, ex, ey, r0, r1, r2, r3)


@njit(cache=True)
def clash(i: int, rects: np.ndarray, owner: np.ndarray, among: np.ndarray, out: np.ndarray, water: np.ndarray,
          anchor_x: np.ndarray, anchor_y: np.ndarray, end_x: np.ndarray, end_y: np.ndarray) -> bool:
    """placement.clash: label i's rects overlap those of another label of `among`, or another's leader crosses
    them, or (an 'out' country label) its leader crosses another's rects; rects/owner: rects_of; end_x/y: leader_ends."""
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
        if j == i or not among[j] or not out[j] or water[j]:
            continue
        for a in mine:
            if leader_hits(anchor_x[j], anchor_y[j], end_x[j], end_y[j], rects[a, 0], rects[a, 1], rects[a, 2], rects[a, 3]):
                return True
    if out[i] and not water[i]:
        for b in range(len(rects)):
            j = owner[b]
            if j != i and among[j] and leader_hits(anchor_x[i], anchor_y[i], end_x[i], end_y[i], rects[b, 0], rects[b, 1], rects[b, 2], rects[b, 3]):
                return True
    return False


@njit(cache=True)
def clashes(rects: np.ndarray, owner: np.ndarray, among: np.ndarray, out: np.ndarray, water: np.ndarray,
            anchor_x: np.ndarray, anchor_y: np.ndarray, end_x: np.ndarray, end_y: np.ndarray) -> np.ndarray:
    """clash for every label of `among`: a mask."""
    hit = np.zeros(len(among), np.bool_)
    for i in range(len(among)):
        if among[i]:
            hit[i] = clash(i, rects, owner, among, out, water, anchor_x, anchor_y, end_x, end_y)
    return hit
