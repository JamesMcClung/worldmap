"""Render a poster-sized world map (country + large-country state borders, non-overlapping labels).

Data: Natural Earth 10m (public domain), downloaded into .cache/ on first use. Run: make_map [out.png]
Pins: --pick pins.txt (click the rendered map to add pins) and --pins pins.txt [--pin-size inches] (draw them).
"""

import argparse
import dataclasses
import os
import shutil
import sys
import time
from collections.abc import Sequence

import geopandas as gpd
import matplotlib
import shapely
from matplotlib.axes import Axes
from shapely.geometry import Point

from worldmap.colorbars import Corner, draw_colorbar, label_outline
from worldmap.coloring import Coloring, load_or_compute_coloring, make_palette, spot_check
from worldmap.colorspace import RGB
from worldmap.data import is_labeled, load_countries, load_disputed, load_states, parent_index
from worldmap.disputed import DISPUTED_GREY, STRIPE_WIDTH_IN, border_lines, dispute_edges, disputed_label_args, load_disputes
from worldmap.crop import (CROP_MARGIN, box_region, candidates, centre_frame, centre_recipe, centre_spec, choose, crop_frame, matches,
                           parse_box, parse_centre)
from worldmap.earth import Fill, build_regions, fill_at
from worldmap.frame import DPI, MapFrame, make_frame
from worldmap.labels.label import decide_out_and_wrap, make_labels
from worldmap.labels.placeable import Placeable
from worldmap.labels.priority import Outcome, Unplaced, place_by_priority
from worldmap.labels.regions import child_regions
from worldmap.labels.report import progress, summary
from worldmap.labels.curved import Body, CurvedLabel
from worldmap.labels.water import WATER_KINDS, Kind, kinds_from_labels, load_water_bodies, make_water_labels, river_ranks
from worldmap.pins import draw_pins, fit_to_render, load_pins, pick_pins, pin_sprite
from worldmap.render import draw_borders, draw_countries, draw_labels, draw_lakes, drawable_lakes, draw_rivers, draw_states, draw_stripes, draw_water_labels, Lakes, new_figure, subtract_lakes
from worldmap.rivers import river_flow
from worldmap.topographic import ICE, ICE_CONTOURS, LAND, LAND_CONTOURS, SEA, SEA_CONTOURS, draw_topography


DEFAULT_OUT, DEFAULT_PINS_OUT = "worldmap.png", "worldmap_pins.png"
DEFAULT_PIN_SIZE_IN = 0.24


def positive_float(s: str) -> float:
    v = float(s)
    if not v > 0:
        raise argparse.ArgumentTypeError("must be > 0, got %r" % s)
    return v


LABEL_KINDS = ("countries", "territories", *WATER_KINDS, "water")  # water = every water kind


def label_kinds(s: str) -> list[str]:
    """A comma-separated list of LABEL_KINDS, or "none", in order (the order labels are placed in), without repeats;
    `water` stands for every water kind, in its place."""
    if s == "none":
        return []
    kinds = s.split(",")
    unknown = [k for k in kinds if k not in LABEL_KINDS]
    if unknown:
        raise argparse.ArgumentTypeError("unknown label kind %r: choose from %s, or none" % (unknown[0], ",".join(LABEL_KINDS)))
    return list(dict.fromkeys(k for kind in kinds for k in (WATER_KINDS if kind == "water" else [kind])))


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
    p.add_argument("--label", metavar="LIST", type=label_kinds, default=["countries"],
                   help="what to label, comma-separated, or none: countries (default); territories = dependencies and "
                        "outlying regions (Alaska, Greenland, Kaliningrad...) with their parent country; oceans, seas "
                        "(and gulfs, bays...), lakes, straits (and channels), rivers (draws them too); water = all of these")
    p.add_argument("--disputed", action="store_true",
                   help="stripe disputed areas in the colours of the countries involved and dash their borders")
    p.add_argument("--rivers", action="store_true", help="draw the major rivers")
    p.add_argument("--pins", metavar="FILE", help="draw the pins listed in FILE (one lon,lat per line)")
    p.add_argument("--pin-size", metavar="INCHES", type=float, default=DEFAULT_PIN_SIZE_IN,
                   help="pin diameter in inches on the poster (default %(default)s)")
    p.add_argument("--size", metavar="IN", type=positive_float,
                   help="the poster's long side in inches (default 36 for the world map; a crop's grows with its "
                        "region, at least 6); text sizes are fixed, so a bigger poster fits more labels")
    p.add_argument("--dpi", metavar="DPI", type=positive_float, default=DPI,
                   help="output dots per inch on the poster (default %(default)s); "
                        "the layout is the same at any resolution, only the pixel count changes")
    p.add_argument("--crop", metavar="SPEC",
                   help="zoom in on a region, redrawn in a projection centred on it: label names, comma-separated (or parts of "
                        "them; only the --label kinds are searched), edges west:east,south:north (any may be left out, "
                        "e.g. :,60: is everything north of 60N; west past east crosses +-180), or a centre and size "
                        "lon,lat/width,height (sizes along the ground, in degrees, or with km or mi, e.g. 2,46/1000km,800km)")
    p.add_argument("--pick", metavar="FILE",
                   help="click the rendered map to add pins; they are appended to FILE when the window closes")
    return p.parse_args(argv)


def draws_rivers(args: argparse.Namespace) -> bool:
    """--rivers, or --label rivers or water (a name needs its river)."""
    return bool(args.rivers or "rivers" in kinds_from_labels(args.label))


def make_map_frame(args: argparse.Namespace) -> MapFrame:
    """The whole globe, or the --crop region, --size inches on its long side if given."""
    return dataclasses.replace(region_frame(args), size_in=args.size)


def region_frame(args: argparse.Namespace) -> MapFrame:
    """The whole globe, or the --crop region, at its default size."""
    if not args.crop:
        return make_frame()
    try:
        centre = parse_centre(args.crop)
        if centre is not None:
            return centre_frame(centre)
        crop = parse_box(args.crop)
        if crop is not None:
            return crop_frame(box_region(crop), 0)
        if not args.label:
            sys.exit(f"make_map: --crop {args.crop}: a name is looked up among the labels, but --label is none")
        cands = candidates(args.label, args.disputed)
        names = [n.strip() for n in args.crop.split(",") if n.strip()]  # no label has a comma in it
        chosen = [c for name in names for c in choose(name, matches(name, cands))]
        region = shapely.union_all([c.geom for c in chosen])
        centre = centre_recipe(region, CROP_MARGIN)
        same = "" if centre is None else f"; the same as --crop {centre_spec(centre)}"
        print(f"cropping to {', '.join(c.text for c in chosen)}{same}")
        return crop_frame(region, CROP_MARGIN)
    except ValueError as e:
        sys.exit(f"make_map: {e}")


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

    frame = make_map_frame(args)
    if args.pick:
        pick_pins(args.pick, out, fit_to_render(out, frame), args.pin_size, DPI)  # its preview pins the render's size
        return
    if args.type == "topographic":
        draw_topographic(args, out, frame)
        return

    countries = load_countries()
    major = [is_labeled(row) for _, row in countries.iterrows()]
    disputes = load_disputes(countries, frame.project(load_disputed()))
    fills = graph_fills(countries, load_or_compute_coloring(countries, major, args.recolor, parent_index(countries), dispute_edges(disputes)))
    disputes = [d for d in disputes if not d.geometry.is_empty]  # a crop leaves out those it doesn't show
    countries_p, states_p, lakes = frame.project(countries), frame.project(load_states()), drawable_lakes(frame)
    regions = build_regions(countries, frame) if args.color == "earth" else None

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
    draw_lakes(ax, lakes)
    countries_p = subtract_lakes(countries_p, lakes)
    ax.set_xlim(frame.xmin, frame.xmax)
    ax.set_ylim(frame.ymin, frame.ymax)

    shown = frame.visible(countries_p)  # labels go on what the frame shows
    children = child_regions(countries, shown, frame) if "territories" in args.label else []
    if not frame.globe:
        disputes = [dataclasses.replace(d, geometry=d.geometry.intersection(frame.canvas)) for d in disputes]
    relabel, extra = disputed_label_args(disputes, lambda i, pt: fill_of(i, pt) if regions else DISPUTED_GREY) if args.disputed else ({}, [])
    labels = make_labels(shown, frame, fill_of, children, "countries" in args.label, relabel, extra)
    decide_out_and_wrap(labels)
    bodies, water = water_labels(kinds_from_labels(args.label), lakes, frame)
    outcome = place([*labels, *water], frame, args.label)
    if draws_rivers(args):
        draw_rivers(ax, shown_rivers(frame, bodies, water, outcome))
    placement = outcome.placement
    draw_labels(ax, placement, frame)
    draw_water_labels(ax, placement, frame)

    add_pins(ax, frame, args)
    # Layout above is measured at DPI; saving at another resolution only rescales it.
    fig.savefig(out, dpi=args.dpi, transparent=True)
    print("wrote", out, "labels:", int((~placement.water).sum()), "leaders:", int(placement.out.sum()), "water labels:", int(placement.water.sum()))


def water_labels(kinds: set[Kind], lakes: Lakes, frame: MapFrame) -> tuple[list[Body], list[CurvedLabel | Unplaced]]:
    """The water bodies of `kinds` and their labels, one per body (make_water_labels: an Unplaced where the name
    doesn't fit; river names kept off the drawn `lakes`)."""
    if not kinds:
        return [], []
    bodies = load_water_bodies(kinds, lakes.shapes, frame)
    return bodies, make_water_labels(bodies, frame, lakes.merged)


def shown_rivers(frame: MapFrame, bodies: list[Body], water: list[CurvedLabel | Unplaced], outcome: Outcome | None) -> gpd.GeoDataFrame:
    """The river pieces to draw, projected: those whose rank (water.river_ranks; the bodies' are those of their
    labels, `water`, in `outcome`), or that of a piece upstream (RiverFlow.downstream), rivers weren't cut at;
    every one if no labels were placed at all (`outcome` None)."""
    flow = river_flow()
    rivers = frame.project(flow.pieces)
    if outcome is None:
        return rivers
    ranks = outcome.ranks_of(water)
    named = [k for k, b in enumerate(bodies) if b.kind == "rivers"]
    return rivers[outcome.shown("rivers", flow.downstream(river_ranks(rivers, [bodies[k] for k in named], ranks[named])))]


def place(entries: Sequence[Placeable | Unplaced], frame: MapFrame, order: list[str]) -> Outcome:
    """place_by_priority, showing on a terminal the rank being placed, then printing its summary."""
    tty, columns = sys.stdout.isatty(), shutil.get_terminal_size().columns

    def show(r: int, done: int, of: int, kind: str) -> None:
        print(f"\r\x1b[K{progress(r, done, of, kind, columns)}", end="", flush=True)

    outcome = place_by_priority(entries, frame, order, on_step=show if tty else None)
    if tty:
        print("\r\x1b[K", end="")
    print(summary(outcome, order, tty and "NO_COLOR" not in os.environ, columns if tty else None))
    return outcome


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
    lakes = drawable_lakes(frame)
    draw_topography(ax, frame, args.dpi, args.ocean, lakes)
    draw_colorbar(ax, frame, "br", LAND, "Height", LAND_CONTOURS)
    if args.ocean:
        draw_colorbar(ax, frame, "tr", SEA, "Depth", SEA_CONTOURS)
    draw_colorbar(ax, frame, "bl", ICE, "Ice", ICE_CONTOURS)
    kinds = kinds_from_labels(args.label)
    bodies, water = water_labels(kinds, lakes, frame)
    outcome = None
    if kinds:  # without water kinds no label stage runs
        # A crop's bars sit over the map: keep the labels off them.
        bars: list[Corner] = ["br", "tr", "bl"] if args.ocean else ["br", "bl"]
        room = frame if frame.globe else dataclasses.replace(frame, outline=label_outline(frame, bars))
        outcome = place(water, room, args.label)
        draw_water_labels(ax, outcome.placement, frame)
    if draws_rivers(args):
        draw_rivers(ax, shown_rivers(frame, bodies, water, outcome))
    ax.set_xlim(frame.xmin, frame.xmax)
    ax.set_ylim(frame.ymin, frame.ymax)
    add_pins(ax, frame, args)
    fig.savefig(out, dpi=args.dpi, transparent=True)
    print("wrote", out, "in %.1f s" % (time.perf_counter() - start))


if __name__ == "__main__":
    main()
