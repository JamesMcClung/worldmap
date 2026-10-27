"""Render a poster-sized world map (country + large-country state borders, non-overlapping labels).

Data: Natural Earth 10m (public domain), downloaded into .cache/ on first use. Run: make_map [out.png]
Pins: --pick pins.txt (click the rendered map to add pins) and --pins pins.txt [--pin-size inches] (draw them).
"""

import argparse
import os
import sys

import matplotlib

from worldmap.data import is_labeled, load_countries, load_lakes, load_states, project
from worldmap.coloring import load_or_compute_coloring, make_palette, spot_check
from worldmap.earth import build_regions, fill_at
from worldmap.labels.placement import place_labels
from worldmap.labels.label import decide_out_and_wrap, make_labels
from worldmap.frame import DPI, make_frame
from worldmap.pins import draw_pins, load_pins, pick_pins, pin_sprite
from worldmap.render import draw_countries, draw_labels, draw_lakes, draw_states, new_figure, subtract_lakes


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


def main(argv=None):
    args = parse_args(argv)
    PINS, PICK = args.pins, args.pick  # pin-list files: --pins draws them, --pick adds to one by clicking
    PIN_IN = args.pin_size  # pin diameter in inches on the 36-inch-wide poster

    if not PICK:  # --pick needs an interactive backend
        matplotlib.use("Agg")
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

    fig, ax = new_figure(frame)
    draw_countries(ax, countries_p, fill_of, REGIONS if EARTH else None)
    draw_states(ax, states_p)
    draw_lakes(ax, lakes_p)
    countries_p = subtract_lakes(countries_p, lakes_p)
    ax.set_xlim(frame.xmin, frame.xmax)
    ax.set_ylim(frame.ymin, frame.ymax)

    # Labels are measured in rendered pixels, so draw once first.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    labels = make_labels(countries_p, frame, ax, rend, fill_of)
    N = len(labels)
    decide_out_and_wrap(labels, rend)

    placement = place_labels(labels, frame)

    draw_labels(ax, labels, placement, frame)
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
