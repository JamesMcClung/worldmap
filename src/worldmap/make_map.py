"""Render a poster-sized world map (country + large-country state borders, non-overlapping labels).

Data: Natural Earth 10m (public domain), downloaded into .cache/ on first use. Run: make_map [out.png]
Pins: --pick pins.txt (click the rendered map to add pins) and --pins pins.txt [--pin-size inches] (draw them).
"""

import argparse
import os
import sys

import matplotlib
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch, Polygon as MplPolygon
from matplotlib.path import Path
from shapely.geometry import MultiPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from worldmap.data import is_labeled, load_countries, load_lakes, load_states, project
from worldmap.coloring import OCEAN, load_or_compute_coloring, make_palette, spot_check
from worldmap.earth import build_regions, fill_at
from worldmap.labels.placement import leader_ends, place_labels
from worldmap.labels.label import decide_out_and_wrap, make_labels
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

    labels = make_labels(countries_p, frame, ax, rend, fill_of)
    N = len(labels)
    decide_out_and_wrap(labels, rend)

    placement = place_labels(labels, frame)

    nx, ny = leader_ends(placement)
    for i, L in enumerate(labels):
        L.text.set_position(frame.px_to_map(placement.x[i], placement.y[i]))
        if placement.out[i]:
            x0_, y0_ = frame.px_to_map(placement.anchor_x[i], placement.anchor_y[i])
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
    print("wrote", OUT, "labels:", N, "leaders:", int(placement.out.sum()))


if __name__ == "__main__":
    main()
