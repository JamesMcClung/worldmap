"""Render a poster-sized world map (country + large-country state borders, non-overlapping labels).

Data: Natural Earth 10m (public domain), downloaded into .cache/ on first use. Run: make_map [out.png]
Pins: --pick pins.txt (click the rendered map to add pins) and --pins pins.txt [--pin-size inches] (draw them).
"""

import argparse
import os
import sys
import time

import geopandas as gpd
import matplotlib
from matplotlib.axes import Axes
from shapely.geometry import Point

from worldmap.colorbars import draw_colorbar
from worldmap.coloring import Coloring, load_or_compute_coloring, make_palette, spot_check
from worldmap.colorspace import RGB
from worldmap.data import is_labeled, load_countries, load_disputed, load_lakes, load_states, parent_index, project
from worldmap.disputed import DISPUTED_GREY, STRIPE_WIDTH_IN, border_lines, dispute_edges, disputed_label_args, load_disputes
from worldmap.earth import Fill, build_regions, fill_at
from worldmap.frame import DPI, MapFrame, make_frame
from worldmap.labels.label import decide_out_and_wrap, make_labels
from worldmap.labels.placement import place_with_water
from worldmap.labels.regions import child_regions
from worldmap.labels.water import WATER_KINDS, kinds_from_labels, load_water_bodies, make_water_labels
from worldmap.pins import draw_pins, load_pins, pick_pins, pin_sprite
from worldmap.render import draw_borders, draw_countries, draw_labels, draw_lakes, draw_states, draw_stripes, draw_water_labels, new_figure, subtract_lakes
from worldmap.topographic import ICE, ICE_CONTOURS, LAND, LAND_CONTOURS, SEA, SEA_CONTOURS, draw_topography


DEFAULT_OUT, DEFAULT_PINS_OUT = "worldmap.png", "worldmap_pins.png"
DEFAULT_PIN_SIZE_IN = 0.24


def positive_float(s: str) -> float:
    v = float(s)
    if not v > 0:
        raise argparse.ArgumentTypeError("must be > 0, got %r" % s)
    return v


LABEL_KINDS = ("countries", "territories", *WATER_KINDS)


def label_kinds(s: str) -> set[str]:
    """A comma-separated list of LABEL_KINDS, or "none"."""
    if s == "none":
        return set()
    kinds = s.split(",")
    unknown = [k for k in kinds if k not in LABEL_KINDS]
    if unknown:
        raise argparse.ArgumentTypeError("unknown label kind %r: choose from %s, or none" % (unknown[0], ",".join(LABEL_KINDS)))
    return set(kinds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="make_map", description=(__doc__ or "").split("\n")[0])
    p.add_argument("out", nargs="?", help=f"output PNG (default {DEFAULT_OUT}, or {DEFAULT_PINS_OUT} with --pins)")
    p.add_argument("--type", choices=["political", "topographic"], default="political",
                   help="political = countries, borders and labels (default); topographic = heights, hill shading and "
                        "contours, with no borders, and only water labels if --label asks for them (--color, --disputed, "
                        "--recolor and --label countries, territories are ignored)")
    p.add_argument("--ocean", action="store_true",
                   help="with --type topographic, colour the sea by depth (ignored with --type political)")
    p.add_argument("--recolor", action="store_true", help="rebuild the cached graph colouring")
    p.add_argument("--color", choices=["colorblind", "earth"], default="colorblind",
                   help="how countries are filled: colorblind = distinct colours that red/green colour-blind viewers "
                        "can tell apart (default); earth = land cover as seen from space (sandy desert -> light green)")
    p.add_argument("--label", metavar="LIST", type=label_kinds, default={"countries"},
                   help="what to label, comma-separated, or none: countries (default); territories = dependencies and "
                        "outlying regions (Alaska, Greenland, Kaliningrad...) with their parent country; oceans, seas "
                        "(and gulfs, bays...), lakes; water = oceans, seas, lakes, straits and channels")
    p.add_argument("--disputed", action="store_true",
                   help="stripe disputed areas in the colours of the countries involved and dash their borders")
    p.add_argument("--pins", metavar="FILE", help="draw the pins listed in FILE (one lon,lat per line)")
    p.add_argument("--pin-size", metavar="INCHES", type=float, default=DEFAULT_PIN_SIZE_IN,
                   help="pin diameter in inches on the 36 in poster (default %(default)s)")
    p.add_argument("--dpi", metavar="DPI", type=positive_float, default=DPI,
                   help="output dots per inch on the 36 in poster (default %(default)s); "
                        "the layout is the same at any resolution, only the pixel count changes")
    p.add_argument("--pick", metavar="FILE",
                   help="click the rendered map to add pins; they are appended to FILE when the window closes")
    return p.parse_args(argv)


def graph_fills(countries: gpd.GeoDataFrame, coloring: Coloring) -> dict[int, RGB]:
    """Each country's fill from its colour class. Baikonur (Kazakh land leased to Russia) takes Russia's fill;
    render.draw_stripes stripes it with Kazakhstan's."""
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

    frame = make_frame()
    if args.pick:
        pick_pins(args.pick, out, frame, args.pin_size, DPI)
        return
    if args.type == "topographic":
        draw_topographic(args, out, frame)
        return

    countries = load_countries()
    major = [is_labeled(row) for _, row in countries.iterrows()]
    disputes = load_disputes(countries, project(load_disputed()))
    fills = graph_fills(countries, load_or_compute_coloring(countries, major, args.recolor, parent_index(countries), dispute_edges(disputes)))
    states, lakes = load_states(), load_lakes()
    countries_p, states_p, lakes_p = project(countries), project(states), project(lakes)
    regions = build_regions(countries_p, countries) if args.color == "earth" else None

    def fill_of(i: int, pt: Point | None = None) -> Fill:
        """Country i's fill; with --color earth, that of its polygon containing (or nearest) `pt`, else its largest."""
        return fills[i] if regions is None else fill_at(regions, i, pt)

    fig, ax = new_figure(frame)
    draw_countries(ax, countries_p, fill_of, regions, args.disputed)
    if regions is None:  # natural colours don't stand for countries, so no stripes with --color earth
        striped = disputes if args.disputed else [d for d in disputes if d.name == "Baikonur"]
        draw_stripes(ax, striped, fill_of, STRIPE_WIDTH_IN * frame.units_per_in)
    if args.disputed:
        draw_borders(ax, *border_lines(countries_p, disputes, parent_index(countries)))
    draw_states(ax, states_p, args.disputed)
    draw_lakes(ax, lakes_p)
    countries_p = subtract_lakes(countries_p, lakes_p)
    ax.set_xlim(frame.xmin, frame.xmax)
    ax.set_ylim(frame.ymin, frame.ymax)

    # Labels are measured in rendered pixels, so draw once first.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # Agg canvas
    children = child_regions(countries, countries_p, frame) if "territories" in args.label else []
    relabel, extra = disputed_label_args(disputes, lambda i, pt: fill_of(i, pt) if regions else DISPUTED_GREY) if args.disputed else ({}, [])
    labels = make_labels(countries_p, frame, ax, renderer, fill_of, children, "countries" in args.label, relabel, extra)
    decide_out_and_wrap(labels, renderer)
    water = make_water_labels(load_water_bodies(kinds_from_labels(args.label), lakes_p), frame) if kinds_from_labels(args.label) else []
    placement, water = place_with_water(labels, water, frame)
    draw_labels(ax, labels, placement, frame)
    draw_water_labels(ax, water, placement, frame)

    add_pins(ax, frame, args)
    # Layout above is measured at DPI; saving at another resolution only rescales it.
    fig.savefig(out, dpi=args.dpi, transparent=True)
    print("wrote", out, "labels:", len(labels), "leaders:", int(placement.out.sum()), "water labels:", len(water))


def add_pins(ax: Axes, frame: MapFrame, args: argparse.Namespace) -> None:
    """The --pins, if any, above everything else."""
    if args.pins:
        pins = load_pins(args.pins)
        draw_pins(ax, frame, pins, pin_sprite(args.pin_size * args.dpi), args.pin_size)
        ax.set_xlim(frame.xmin, frame.xmax)
        ax.set_ylim(frame.ymin, frame.ymax)
        print("pins:", len(pins))


def draw_topographic(args: argparse.Namespace, out: str, frame: MapFrame) -> None:
    """--type topographic: heights (and sea depths with --ocean), colour bars in the corners, pins; nothing political."""
    start = time.perf_counter()
    fig, ax = new_figure(frame)
    draw_topography(ax, frame, args.dpi, args.ocean)
    draw_colorbar(ax, frame, "br", LAND, "Height", LAND_CONTOURS)
    if args.ocean:
        draw_colorbar(ax, frame, "tr", SEA, "Depth", SEA_CONTOURS)
    draw_colorbar(ax, frame, "bl", ICE, "Ice", ICE_CONTOURS)
    kinds = kinds_from_labels(args.label)
    if kinds:  # without water kinds no label stage runs
        placement, water = place_with_water([], make_water_labels(load_water_bodies(kinds, project(load_lakes())), frame), frame)
        draw_water_labels(ax, water, placement, frame)
    ax.set_xlim(frame.xmin, frame.xmax)
    ax.set_ylim(frame.ymin, frame.ymax)
    add_pins(ax, frame, args)
    fig.savefig(out, dpi=args.dpi, transparent=True)
    print("wrote", out, "in %.1f s" % (time.perf_counter() - start))


if __name__ == "__main__":
    main()
