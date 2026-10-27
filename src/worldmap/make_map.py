"""Render a poster-sized world map (country + large-country state borders, non-overlapping labels).

Data: Natural Earth 10m (public domain), downloaded into .cache/ on first use. Run: make_map [out.png]
Pins: --pick pins.txt (click the rendered map to add pins) and --pins pins.txt [--pin-size inches] (draw them).
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.patheffects as pe
import numpy as np
import shapely
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch, Polygon as MplPolygon
from matplotlib.path import Path
from shapely import maximum_inscribed_circle
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from worldmap.data import is_labeled, load_countries, load_lakes, load_states, project
from worldmap.coloring import OCEAN, load_or_compute_coloring, make_palette, spot_check
from worldmap.colorspace import darken
from worldmap.earth import build_regions, fill_at
from worldmap.labels.geometry import nearest_point_on_rect, rect, segment_hits_rects, segments_cross, two_line
from worldmap.frame import DPI, POSTER_WIDTH_IN, make_frame
from worldmap.pins import draw_pins, load_pins, pick_pins, pin_sprite


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="make_map", description=__doc__.split("\n")[0])
    p.add_argument("out", nargs="?", help="output PNG (default worldmap.png, or worldmap_pins.png with --pins)")
    p.add_argument("--recolor", action="store_true", help="rebuild the cached graph colouring")
    p.add_argument("--earth", action="store_true",
                   help="colour by land cover as seen from space (sandy desert -> light green) instead of graph colours")
    p.add_argument("--pins", metavar="FILE", help="draw the pins listed in FILE (one lon,lat per line)")
    p.add_argument("--pin-size", metavar="INCHES", type=float, default=0.24,
                   help="pin diameter in inches on the 36 in poster (default %(default)s)")
    p.add_argument("--pick", metavar="FILE",
                   help="click the rendered map to add pins; they are appended to FILE when the window closes")
    return p.parse_args(argv)


BORDER, STATE = "#3a3a3a", "#7d7d7d"




def patches(geom):
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if p.geom_type == "Polygon" and not p.is_empty:
            yield MplPolygon(np.asarray(p.exterior.coords), closed=True)
            for r in p.interiors:
                pass


def holed_patches(geom):
    """Like patches(), but keeping holes (e.g. islands in a lake)."""
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if p.geom_type == "Polygon" and not p.is_empty:
            p = orient(p)  # holes wind opposite to the exterior, so the nonzero fill rule leaves them empty
            yield PathPatch(Path.make_compound_path(*(Path(np.asarray(ring.coords), closed=True) for ring in [p.exterior, *p.interiors])))


def biggest(geom):
    return max(geom.geoms, key=lambda g: g.area) if isinstance(geom, MultiPolygon) else geom


RENAME = {"People's Republic of China": "China", "United States of America": "United States", "Democratic Republic of the Congo": "DR Congo", "Republic of the Congo": "Congo", "Central African Republic": "C.A.R.", "Bosnia and Herzegovina": "Bosnia & Herz.", "United Arab Emirates": "UAE", "Dominican Republic": "Dominican Rep."}
MIN_PT, MAX_PT = 6, 30
FORCE_ISLAND = {"Greece"}

halo = [pe.withStroke(linewidth=2.2, foreground=(1, 1, 1, 0.55))]


class Label:
    pass


TARGET = 1.3


STEPS = [(a * 0.25, b) for a in (-2, -1, 0, 1, 2) for b in (-1, 0, 1) if (a, b) != (0, 0)]
ANG = np.radians(np.arange(0, 360, 15))


def main(argv=None):
    args = parse_args(argv)
    PINS, PICK = args.pins, args.pick  # pin-list files: --pins draws them, --pick adds to one by clicking
    PIN_IN = args.pin_size  # pin diameter in inches on the 36-inch-wide poster

    if not PICK:  # --pick needs an interactive backend
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT = args.out or ("worldmap_pins.png" if PINS else "worldmap.png")
    # --earth: colour countries by how they look from space (sandy desert -> light green vegetation) instead of graph colours.
    EARTH = args.earth
    if PICK and not os.path.exists(OUT):
        sys.exit("--pick shows the rendered map: run make_map first to create %s" % OUT)

    countries = load_countries()
    major = [is_labeled(r) for _, r in countries.iterrows()]
    coloring = load_or_compute_coloring(countries, major, args.recolor)

    PALETTE = make_palette(coloring.k, coloring.class_pairs)
    COLOR_OF = dict(zip(countries.index, coloring.color_class))
    # Baikonur Cosmodrome: Kazakh territory leased to Russia -> Russia's fill, striped with Kazakhstan's colour.
    _idx = dict(zip(countries.NAME_EN, countries.index))
    COLOR_OF[_idx["Baikonur"]] = COLOR_OF[_idx["Russia"]]
    print("same-colour pairs among spot checks:", spot_check(countries, [COLOR_OF[i] for i in countries.index]))

    def fill_of(i, pt=None):
        """Country fill; in --earth mode, the fill of the country's polygon containing (or nearest) `pt`, else its largest."""
        if not EARTH:
            return PALETTE[COLOR_OF[i]]
        return fill_at(REGIONS, i, pt)

    states, lakes = load_states(), load_lakes()
    countries_p, states_p, lakes_p = project(countries), project(states), project(lakes)
    REGIONS = build_regions(countries_p, countries) if EARTH else {}

    frame = make_frame()
    if PICK:
        pick_pins(PICK, OUT, frame, PIN_IN, DPI)
        return

    fig_w = POSTER_WIDTH_IN
    fig_h = fig_w * frame.height / frame.width
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(frame.xmin, frame.xmax)
    ax.set_ylim(frame.ymin, frame.ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0)  # transparent outside the globe outline
    ax.add_patch(MplPolygon(frame.outline, closed=True, fc=OCEAN, ec="none", zorder=0))
    ax.add_patch(MplPolygon(frame.outline, closed=True, fc="none", ec=BORDER, lw=2.2, zorder=3))

    for _, row in countries_p.iterrows():
        col = "#ffffff" if row.NAME_EN == "Antarctica" else fill_of(_)  # Antarctica: hardcoded white, unlabelled
        if EARTH and row.NAME_EN != "Antarctica":
            for p, c in REGIONS[_]:
                ax.add_collection(PatchCollection(list(patches(p)), fc=c, ec=BORDER, lw=0.6, zorder=1))
        else:
            ax.add_collection(PatchCollection(list(patches(row.geometry)), fc=col, ec=BORDER, lw=0.6, zorder=1))
        if row.NAME_EN == "Baikonur" and not EARTH:
            stripe = fill_of(_idx["Kazakhstan"])
            plt.rcParams["hatch.linewidth"] = 1.6
            ax.add_collection(PatchCollection(list(patches(row.geometry)), fc="none", ec=stripe, lw=0, hatch="////", zorder=1.5))
    states_p.plot(ax=ax, color=STATE, linewidth=0.35, zorder=2)
    # The country polygons include the lakes; paint the big ones back as water (borders through them then run on the shore),
    # above the state lines, which also cross them. Labels then treat the lakes as water too.
    for g in lakes_p.geometry:
        ax.add_collection(PatchCollection(list(holed_patches(g)), fc=OCEAN, ec=BORDER, lw=0.6, zorder=2.5))
    countries_p["geometry"] = countries_p.geometry.difference(unary_union(list(lakes_p.geometry)))
    ax.set_xlim(frame.xmin, frame.xmax)
    ax.set_ylim(frame.ymin, frame.ymax)

    # Graticule-free; labels ----------------------------------------------------
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    # Everything below works in poster pixels (frame.to_px).
    PPU = frame.px_per_unit
    to_px = frame.to_px

    labels, island_names = [], []
    gx, gy = frame.to_map.transform(-53, 4)  # French Guiana
    kx, ky = frame.to_map.transform(21, 54.7)  # Kaliningrad
    for _, row in countries_p.iterrows():
        if not is_labeled(row):
            continue
        polys = [p for p in (row.geometry.geoms if isinstance(row.geometry, MultiPolygon) else [row.geometry]) if not p.is_empty]
        if not polys:
            continue
        big = max(polys, key=lambda p: p.area)
        pieces = []
        sig = [p for p in polys if p.area >= 0.02 * big.area]
        hull = unary_union(sig).convex_hull
        hb = hull.bounds
        island = len(sig) >= 2 and big.area / sum(p.area for p in sig) < 0.65 and (hb[2] - hb[0]) < 0.25 * frame.width
        if row.NAME_EN in FORCE_ISLAND and len(sig) >= 2:
            island = True
        if island:
            island_names.append(row.NAME_EN)
            pieces.append((row.NAME_EN, hull.centroid.coords[0], unary_union(sig), hull, max(sum(p.area for p in sig), hull.area)))
        else:
            c = maximum_inscribed_circle(big, tolerance=frame.width / 5000).coords[0]
            pieces.append((row.NAME_EN, c, big, big, big.area))
        extra = []
        if row.NAME_EN == "France":
            extra = [("French Guiana", p) for p in polys if p.contains(Point(gx, gy))]
        if row.NAME_EN == "Russia":
            extra = [("Kaliningrad (Russia)", p) for p in polys if p.contains(Point(kx, ky))]
        for nm, p in extra:
            c = maximum_inscribed_circle(p, tolerance=frame.width / 5000).coords[0]
            pieces.append((nm, c, p, p, p.area))
        for nm, c, geom, home, area in pieces:
            nm = RENAME.get(nm, nm)
            size = float(np.clip(1.6 * np.sqrt(area) * PPU / DPI * 72 / 10, MIN_PT, MAX_PT))
            L = Label()
            L.name, L.area = nm, area
            L.color = darken(fill_of(_, geom.representative_point() if geom.geom_type == "Polygon" else None))
            L.txt = ax.text(c[0], c[1], nm.upper() if size > 14 else nm, fontsize=size, ha="center", va="center", color=L.color, zorder=6, path_effects=halo)
            ext = L.txt.get_window_extent(rend)
            L.w, L.h = ext.width * 1.06, ext.height * 1.15
            L.ax, L.ay = (c[0] - frame.xmin) * PPU, (c[1] - frame.ymin) * PPU
            L.x, L.y = L.ax, L.ay
            L.geom, L.home = to_px(geom), to_px(home)
            labels.append(L)
    labels.sort(key=lambda l: -l.area)
    N = len(labels)
    print("island-heavy (hull-centred):", island_names)

    # Mode: 'in' = sits on its country; 'out' = would hide its country, so it moves aside with a leader line.
    def covers(L):
        return rect(L.x, L.y, L.w, L.h).intersection(L.geom).area / L.geom.area if L.geom.area > 0 else 1.0

    for L in labels:
        gb = L.geom.bounds
        L.out = covers(L) >= 0.6 or (L.w > 1.6 * (gb[2] - gb[0]) and (gb[3] - gb[1]) < 6 * L.h)
    OUT_ = np.array([L.out for L in labels])

    # Wrap long names onto two lines when a label sitting on its country is wider than the country.
    for L in labels:
        gb = L.geom.bounds
        if not L.out and " " in L.txt.get_text() and L.w > 0.9 * (gb[2] - gb[0]) and (gb[3] - gb[1]) > 2.5 * L.h:
            L.txt.set_multialignment("center")
            L.txt.set_linespacing(1.0)
            L.txt.set_text(two_line(L.txt.get_text()))
            ext = L.txt.get_window_extent(rend)
            L.w, L.h = ext.width * 1.06, ext.height * 1.15

    W_ = np.array([L.w for L in labels])
    H_ = np.array([L.h for L in labels])
    AX = np.array([L.ax for L in labels])
    AY = np.array([L.ay for L in labels])
    X = AX.copy()
    Y = AY.copy()
    OUTLINE_PX = to_px(Polygon(frame.outline), simplify=0.5)
    shapely.prepare(OUTLINE_PX)
    def leader_ends():
        nx, ny = nearest_point_on_rect(AX, AY, X, Y, W_, H_)
        return nx, ny

    def cost_batch(i, CX, CY):
        """Cost of placing label i at each candidate centre (arrays of length c)."""
        c = len(CX)
        w, h = W_[i], H_[i]
        R = np.stack([CX - w / 2, CY - h / 2, CX + w / 2, CY + h / 2], -1)  # (c,4)
        mask = np.arange(N) != i
        ox_, oy_ = X[mask], Y[mask]
        ow, oh = W_[mask], H_[mask]
        O = np.stack([ox_ - ow / 2, oy_ - oh / 2, ox_ + ow / 2, oy_ + oh / 2], -1)  # (m,4)
        ovx = np.clip(np.minimum(R[:, None, 2], O[None, :, 2]) - np.maximum(R[:, None, 0], O[None, :, 0]), 0, None)
        ovy = np.clip(np.minimum(R[:, None, 3], O[None, :, 3]) - np.maximum(R[:, None, 1], O[None, :, 1]), 0, None)
        cost = 40 * (ovx * ovy).sum(1) / (h * h)
        # other leaders' dots and lines
        omode = OUT_[mask]
        if omode.any():
            dax, day = AX[mask][omode], AY[mask][omode]
            dr = 0.35 * H_[mask][omode]
            ddx = np.clip(np.abs(CX[:, None] - dax[None]) - w / 2, 0, None)
            ddy = np.clip(np.abs(CY[:, None] - day[None]) - h / 2, 0, None)
            dd = np.hypot(ddx, ddy)
            cost += 15 * (np.clip(dr[None] - dd, 0, None) / dr[None]).sum(1)
            nx_, ny_ = nearest_point_on_rect(dax, day, ox_[omode], oy_[omode], ow[omode], oh[omode])
            p0 = np.stack([dax, day], -1)[None]
            p1 = np.stack([nx_, ny_], -1)[None]
            cost += 20 * segment_hits_rects(p0, p1, R[:, None, :]).sum(1)
        if OUT_[i]:
            q0 = np.array([AX[i], AY[i]])[None, None, :]
            nx, ny = nearest_point_on_rect(AX[i], AY[i], CX, CY, w, h)
            q1 = np.stack([nx, ny], -1)[:, None, :]
            d = np.hypot(nx - AX[i], ny - AY[i])
            cost += 1.0 * ((d - TARGET * h) / h) ** 2 + 10 * np.clip(0.9 * h - d, 0, None) / h
            # own leader vs other label boxes (unless that box covers our anchor: unavoidable)
            keep = ~((O[:, 0] <= AX[i]) & (AX[i] <= O[:, 2]) & (O[:, 1] <= AY[i]) & (AY[i] <= O[:, 3]))
            cost += 20 * segment_hits_rects(q0, q1, O[None, keep, :]).sum(1)
            # own leader vs other leaders
            if omode.any():
                cost += 6 * segments_cross(q0[:, None, :, :] if False else np.broadcast_to(q0, (c, 1, 2)), q1, np.stack([dax, day], -1)[None], np.stack([nx_, ny_], -1)[None]).sum(1)
        else:
            boxes = shapely.box(R[:, 0], R[:, 1], R[:, 2], R[:, 3])
            inside = shapely.area(shapely.intersection(boxes, labels[i].geom)) / (w * h)
            cost += 20 * (1 - inside) + 0.15 * np.hypot(CX - AX[i], CY - AY[i]) / h
        cost += 1000 * ~shapely.contains(OUTLINE_PX, shapely.box(R[:, 0], R[:, 1], R[:, 2], R[:, 3]))  # labels stay inside the globe
        return cost

    def candidates(i, grid=False):
        w, h = W_[i], H_[i]
        cx, cy = [X[i]], [Y[i]]
        for sc in (0.4, 1, 2, 4):
            for a, b in STEPS:
                cx.append(X[i] + a * w * sc)
                cy.append(Y[i] + b * h * sc)
        if OUT_[i]:
            for k in (1.0, 1.4, 2.0, 3.0, 4.5):
                cx += list(AX[i] + np.cos(ANG) * (w / 2 + 1.3 * h) * k)
                cy += list(AY[i] + np.sin(ANG) * 1.8 * h * k)
        else:
            cx.append(AX[i])
            cy.append(AY[i])
            if grid:  # coarse scan of the whole country so a label can jump to where it fits
                gx0, gy0, gx1, gy1 = labels[i].geom.bounds
                nxg, nyg = 14, 9
                us = np.linspace(gx0 + w / 2, max(gx1 - w / 2, gx0 + w / 2), nxg)
                vs = np.linspace(gy0 + h / 2, max(gy1 - h / 2, gy0 + h / 2), nyg)
                for u in us:
                    for v in vs:
                        cx.append(u)
                        cy.append(v)
        return np.array(cx), np.array(cy)

    rng = np.random.default_rng(7)

    def optimize(passes, grid_passes=3):
        for p in range(passes):
            changed = 0
            for i in rng.permutation(N):
                CX, CY = candidates(i, grid=p < grid_passes)
                cst = cost_batch(i, CX, CY)
                cst[0] -= 1e-6
                j = int(np.argmin(cst))
                if j != 0:
                    X[i], Y[i] = CX[j], CY[j]
                    changed += 1
            if changed == 0:
                break

    def label_overlaps():
        bad = []
        for i in range(N):
            m = np.arange(N) != i
            ox = np.minimum(X[i] + W_[i] / 2, X[m] + W_[m] / 2) - np.maximum(X[i] - W_[i] / 2, X[m] - W_[m] / 2)
            oy = np.minimum(Y[i] + H_[i] / 2, Y[m] + H_[m] / 2) - np.maximum(Y[i] - H_[i] / 2, Y[m] - H_[m] / 2)
            if ((ox > 0) & (oy > 0)).any():
                bad.append(i)
        return bad

    # seed 'out' labels on a ring in largest-first order so they start separated, then descend.
    for i in range(N):
        if OUT_[i]:
            CX, CY = candidates(i)
            j = int(np.argmin(cost_batch(i, CX, CY)))
            X[i], Y[i] = CX[j], CY[j]
    optimize(40)
    for attempt in range(12):
        bad = label_overlaps()
        if not bad:
            break
        for i in bad:  # still colliding: shrink the text slightly and retry
            if labels[i].txt.get_fontsize() > 5:
                f = 0.94
                labels[i].txt.set_fontsize(labels[i].txt.get_fontsize() * f)
                W_[i] *= f
                H_[i] *= f
        optimize(15)
    bad = label_overlaps()
    print("passes done; unresolved overlaps:", [labels[i].name for i in bad])

    nx, ny = leader_ends()
    for i, L in enumerate(labels):
        L.txt.set_position(frame.px_to_map(X[i], Y[i]))
        if OUT_[i]:
            x0_, y0_ = frame.px_to_map(AX[i], AY[i])
            x1_, y1_ = frame.px_to_map(nx[i], ny[i])
            ax.plot([x0_, x1_], [y0_, y1_], color=L.color, lw=0.7, zorder=4, solid_capstyle="round")
            ax.plot([x0_], [y0_], marker="o", ms=1.3, color=L.color, zorder=4)
    if PINS:
        _pins = load_pins(PINS)
        draw_pins(ax, frame, _pins, pin_sprite(PIN_IN * DPI), PIN_IN)
        ax.set_xlim(frame.xmin, frame.xmax)
        ax.set_ylim(frame.ymin, frame.ymax)
        print("pins:", len(_pins))
    fig.savefig(OUT, dpi=DPI, transparent=True)
    print("wrote", OUT, "labels:", N, "leaders:", int(OUT_.sum()))


if __name__ == "__main__":
    main()
