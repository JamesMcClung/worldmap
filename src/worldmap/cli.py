"""Render a poster-sized world map (country + large-country state borders, non-overlapping labels).

Data: Natural Earth 10m (public domain), downloaded into .cache/ on first use. Run: make_map [out.png]
Pins: --pick pins.txt (click the rendered map to add pins) and --pins pins.txt [--pin-size inches] (draw them).
"""

import argparse
import os
import sys

import geopandas as gpd
import matplotlib
from shapely.geometry import Point

from worldmap.coloring import Coloring, load_or_compute_coloring, make_palette, spot_check
from worldmap.colorspace import RGB
from worldmap.data import is_labeled, load_countries, load_lakes, load_states, project
from worldmap.earth import Fill, build_regions, fill_at
from worldmap.frame import DPI, make_frame
from worldmap.labels.label import decide_out_and_wrap, make_labels
from worldmap.labels.placement import place_labels
from worldmap.pins import draw_pins, load_pins, pick_pins, pin_sprite
from worldmap.render import draw_countries, draw_labels, draw_lakes, draw_states, new_figure, subtract_lakes


DEFAULT_OUT, DEFAULT_PINS_OUT = "worldmap.png", "worldmap_pins.png"
DEFAULT_PIN_SIZE_IN = 0.24


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="make_map", description=(__doc__ or "").split("\n")[0])
    p.add_argument("out", nargs="?", help=f"output PNG (default {DEFAULT_OUT}, or {DEFAULT_PINS_OUT} with --pins)")
    p.add_argument("--recolor", action="store_true", help="rebuild the cached graph colouring")
    p.add_argument("--earth", action="store_true",
                   help="colour by land cover as seen from space (sandy desert -> light green) instead of graph colours")
    p.add_argument("--pins", metavar="FILE", help="draw the pins listed in FILE (one lon,lat per line)")
    p.add_argument("--pin-size", metavar="INCHES", type=float, default=DEFAULT_PIN_SIZE_IN,
                   help="pin diameter in inches on the 36 in poster (default %(default)s)")
    p.add_argument("--pick", metavar="FILE",
                   help="click the rendered map to add pins; they are appended to FILE when the window closes")
    return p.parse_args(argv)


def graph_fills(countries: gpd.GeoDataFrame, coloring: Coloring) -> dict[int, RGB]:
    """Each country's fill from its colour class. Baikonur (Kazakh land leased to Russia) takes Russia's fill;
    render.draw_countries hatches it in Kazakhstan's."""
    palette = make_palette(coloring.k, coloring.class_pairs)
    color_class = dict(zip(countries.index, coloring.color_class))
    index_of = dict(zip(countries.NAME_EN, countries.index))
    color_class[index_of["Baikonur"]] = color_class[index_of["Russia"]]
    print("same-colour pairs among spot checks:", spot_check(countries, [color_class[i] for i in countries.index]))
    return {i: palette[c] for i, c in color_class.items()}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.pick:  # --pick needs an interactive backend
        matplotlib.use("Agg")
    out = args.out or (DEFAULT_PINS_OUT if args.pins else DEFAULT_OUT)
    if args.pick and not os.path.exists(out):
        sys.exit("--pick shows the rendered map: run make_map first to create %s" % out)

    countries = load_countries()
    major = [is_labeled(row) for _, row in countries.iterrows()]
    fills = graph_fills(countries, load_or_compute_coloring(countries, major, args.recolor))
    states, lakes = load_states(), load_lakes()
    countries_p, states_p, lakes_p = project(countries), project(states), project(lakes)
    regions = build_regions(countries_p, countries) if args.earth else None

    def fill_of(i: int, pt: Point | None = None) -> Fill:
        """Country i's fill; in --earth mode, that of its polygon containing (or nearest) `pt`, else its largest."""
        return fills[i] if regions is None else fill_at(regions, i, pt)

    frame = make_frame()
    if args.pick:
        pick_pins(args.pick, out, frame, args.pin_size, DPI)
        return

    fig, ax = new_figure(frame)
    draw_countries(ax, countries_p, fill_of, regions)
    draw_states(ax, states_p)
    draw_lakes(ax, lakes_p)
    countries_p = subtract_lakes(countries_p, lakes_p)
    ax.set_xlim(frame.xmin, frame.xmax)
    ax.set_ylim(frame.ymin, frame.ymax)

    # Labels are measured in rendered pixels, so draw once first.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = make_labels(countries_p, frame, ax, renderer, fill_of)
    decide_out_and_wrap(labels, renderer)
    placement = place_labels(labels, frame)
    draw_labels(ax, labels, placement, frame)

    if args.pins:
        pins = load_pins(args.pins)
        draw_pins(ax, frame, pins, pin_sprite(args.pin_size * DPI), args.pin_size)
        ax.set_xlim(frame.xmin, frame.xmax)
        ax.set_ylim(frame.ymin, frame.ymax)
        print("pins:", len(pins))
    fig.savefig(out, dpi=DPI, transparent=True)
    print("wrote", out, "labels:", len(labels), "leaders:", int(placement.out.sum()))


if __name__ == "__main__":
    main()
