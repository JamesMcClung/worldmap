"""Render a poster-sized world map (country + large-country state borders, non-overlapping labels).

Data: Natural Earth 10m (public domain), downloaded into .cache/ on first use. Run: make_map [out.png] [--format png|svg]
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
from matplotlib.figure import Figure
from shapely.geometry import Point

from worldmap.colorbars import Corner, draw_colorbar, label_outline
from worldmap.coloring import Coloring, load_or_compute_coloring, make_palette, spot_check
from worldmap.colorspace import RGB
from worldmap.crop import CROP_MARGIN, box_region, candidates, centre_frame, centre_recipe, centre_spec, choose, crop_frame, matches, parse_box, parse_centre
from worldmap.data import is_labeled, load_countries, load_disputed, load_states, parent_index
from worldmap.disputed import DISPUTED_GREY, STRIPE_WIDTH_IN, border_lines, dispute_edges, disputed_label_args, load_disputes
from worldmap.earth import Fill, build_regions, fill_at
from worldmap.frame import DPI, MapFrame, make_frame
from worldmap.labels import land
from worldmap.labels.curved import Body, CurvedLabel, kinds_in
from worldmap.labels.label import decide_out_and_wrap, make_labels
from worldmap.labels.placeable import Placeable
from worldmap.labels.priority import Outcome, Unplaced, place_by_priority
from worldmap.labels.regions import child_regions
from worldmap.labels.report import progress, summary
from worldmap.labels.water import WATER_KINDS, Kind, load_water_bodies, make_water_labels, river_ranks
from worldmap.pins import draw_pin_circles, draw_pins, fit_to_render, load_pins, pick_pins, pin_sprite
from worldmap.render import Lakes, draw_borders, draw_countries, draw_curved_labels, draw_labels, draw_lakes, draw_rivers, draw_states, draw_stripes, drawable_lakes, new_figure, subtract_lakes
from worldmap.rivers import river_flow
from worldmap.topographic import ICE, ICE_CONTOURS, LAND, LAND_CONTOURS, SEA, SEA_CONTOURS, draw_topography

DEFAULT_OUT, DEFAULT_PINS_OUT = "worldmap", "worldmap_pins"  # + "." + the format
FORMATS = ("png", "svg")
DEFAULT_PIN_SIZE_IN = 0.24


def positive_float(s: str) -> float:
    v = float(s)
    if not v > 0:
        raise argparse.ArgumentTypeError("must be > 0, got %r" % s)
    return v


LABEL_KINDS = ("countries", "territories", *WATER_KINDS, "water", *land.LAND_KINDS, "land")
GROUPS: dict[str, tuple[str, ...]] = {"water": WATER_KINDS, "land": land.LAND_KINDS}  # each stands for its kinds, in its place


def label_kinds(s: str) -> list[str]:
    """A comma-separated list of LABEL_KINDS, or "none", in order (the order labels are placed in), without repeats;
    `water` and `land` stand for every water or land kind (GROUPS), in their place."""
    if s == "none":
        return []
    kinds = s.split(",")
    unknown = [k for k in kinds if k not in LABEL_KINDS]
    if unknown:
        raise argparse.ArgumentTypeError("unknown label kind %r: choose from %s, or none" % (unknown[0], ",".join(LABEL_KINDS)))
    return list(dict.fromkeys(k for kind in kinds for k in GROUPS.get(kind, (kind,))))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="make_map", description=(__doc__ or "").split("\n")[0])
    p.add_argument("out", nargs="?", help=f"output file (default {DEFAULT_OUT}.png, or {DEFAULT_PINS_OUT}.png with --pins; " f"the extension follows --format)")
    p.add_argument("--format", choices=FORMATS, help="output format (default: from the output's extension, else png); svg is vector, for --type " "political only, with pins drawn as flat circles, and ignores --dpi")
    p.add_argument("--type", choices=["political", "topographic"], default="political", help="political = countries, borders and labels (default); topographic = heights, hill shading and " "contours, with no borders, and only water and landform labels if --label asks for them (--color, --disputed, " "--recolor and --label countries, territories are ignored)")
    p.add_argument("--ocean", action="store_true", help="with --type topographic, colour the sea by depth (ignored with --type political)")
    p.add_argument("--ice", action="store_true", help="with --type topographic, show the ice sheets and glaciers on their own scale, at the top of the ice; without, " "Greenland and Antarctica are the rock under their ice, coloured like any land, and the ice shelves are sea (ignored " "with --type political)")
    p.add_argument("--recolor", action="store_true", help="rebuild the cached graph colouring")
    p.add_argument("--color", choices=["colorblind", "earth"], default="colorblind", help="how countries are filled: colorblind = distinct colours that red/green colour-blind viewers " "can tell apart (default); earth = land cover as seen from space (sandy desert -> light green)")
    p.add_argument("--label", metavar="LIST", type=label_kinds, default=["countries"], help="what to label, comma-separated, or none: countries (default); territories = dependencies and " "outlying regions (Alaska, Greenland, Kaliningrad...) with their parent country; oceans, seas " "(and gulfs, bays...), lakes, straits (and channels), rivers (draws them too); water = all of these; " "mountains, uplands (plateaus), deserts, lowlands (plains, basins, valleys); land = all of these")
    p.add_argument("--disputed", action="store_true", help="stripe disputed areas in the colours of the countries involved and dash their borders")
    p.add_argument("--rivers", action="store_true", help="draw the major rivers")
    p.add_argument("--pins", metavar="FILE", help="draw the pins listed in FILE (one lon,lat per line)")
    p.add_argument("--pin-size", metavar="INCHES", type=float, default=DEFAULT_PIN_SIZE_IN, help="pin diameter in inches on the poster (default %(default)s)")
    p.add_argument("--size", metavar="IN", type=positive_float, help="the poster's long side in inches (default 36 for the world map; a crop's grows with its " "region, at least 6); text sizes are fixed, so a bigger poster fits more labels")
    p.add_argument("--dpi", metavar="DPI", type=positive_float, default=DPI, help="output dots per inch on the poster (default %(default)s); " "the layout is the same at any resolution, only the pixel count changes")
    p.add_argument("--crop", metavar="SPEC", help="zoom in on a region, redrawn in a projection centred on it: label names, comma-separated (or parts of " "them; only the --label kinds are searched), edges west:east,south:north (any may be left out, " "e.g. :,60: is everything north of 60N; west past east crosses +-180), or a centre and size " "lon,lat/width,height (sizes along the ground, in degrees, or with km or mi, e.g. 2,46/1000km,800km)")
    p.add_argument("--pick", metavar="FILE", help="click the rendered map to add pins; they are appended to FILE when the window closes")
    args = p.parse_args(argv)
    ext = os.path.splitext(args.out or "")[1].lower().lstrip(".")
    if args.format is None:
        args.format = ext if ext in FORMATS else "png"
    elif ext in FORMATS and ext != args.format:
        p.error(f"--format {args.format}, but {args.out} ends in .{ext}")
    if args.out is None:
        args.out = f"{DEFAULT_PINS_OUT if args.pins else DEFAULT_OUT}.{args.format}"
    if args.format == "svg" and args.pick:
        p.error("--pick shows the rendered PNG: drop --format svg")
    if args.format == "svg" and args.type == "topographic":
        p.error("--format svg is for --type political only: a topographic map is a raster")
    return args


def draws_rivers(args: argparse.Namespace) -> bool:
    """--rivers, or --label rivers or water (a name needs its river)."""
    return bool(args.rivers or "rivers" in args.label)


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
    out = args.out
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
    bodies, water = water_labels(kinds_in(args.label, WATER_KINDS), lakes, frame)
    outcome = place([*labels, *water, *land_labels(kinds_in(args.label, land.LAND_KINDS), frame)], frame, args.label)
    if draws_rivers(args):
        draw_rivers(ax, shown_rivers(frame, bodies, water, outcome))
    placement = outcome.placement
    draw_labels(ax, placement, frame)
    draw_curved_labels(ax, placement, frame)

    add_pins(ax, frame, args, vector=args.format == "svg")
    save(fig, out, args.format, args.dpi)
    print("wrote", out, "labels:", int((~placement.curved).sum()), "leaders:", int(placement.out.sum()), "curved labels:", int(placement.curved.sum()))


def water_labels(kinds: set[Kind], lakes: Lakes, frame: MapFrame) -> tuple[list[Body], list[CurvedLabel | Unplaced]]:
    """The water bodies of `kinds` and their labels, one per body (make_water_labels: an Unplaced where the name
    doesn't fit; river names kept off the drawn `lakes`)."""
    if not kinds:
        return [], []
    bodies = load_water_bodies(kinds, lakes.shapes, frame)
    return bodies, make_water_labels(bodies, frame, lakes.merged)


def land_labels(kinds: set[land.Kind], frame: MapFrame) -> list[CurvedLabel | Unplaced]:
    """The landforms of `kinds` the frame shows, a label (or an Unplaced) each."""
    return land.make_land_labels(land.load_land_bodies(kinds, frame), frame) if kinds else []


def area_kinds(label: Sequence[str]) -> bool:
    """Whether --label asks for any curved area labels (water or land): without, --type topographic places none."""
    return bool(kinds_in(label, WATER_KINDS) or kinds_in(label, land.LAND_KINDS))


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


def save(fig: Figure, out: str, fmt: str, dpi: float) -> None:
    """Save `fig` as a PNG at `dpi`, or an SVG (`fmt`). The layout is measured at DPI: saving at another resolution
    only rescales it. An SVG leaves out the date and takes its ids from a fixed salt, so a render is the same file
    every time."""
    if fmt == "png":
        fig.savefig(out, format=fmt, dpi=dpi, transparent=True)
        return
    with matplotlib.rc_context({"svg.hashsalt": "worldmap"}):
        fig.savefig(out, format=fmt, transparent=True, metadata={"Date": None})


def add_pins(ax: Axes, frame: MapFrame, args: argparse.Namespace, vector: bool = False) -> None:
    """The --pins, if any, above everything else: sprites, or flat circles for `vector` output."""
    if args.pins:
        pins = load_pins(args.pins)
        if vector:
            draw_pin_circles(ax, frame, pins, args.pin_size)
        else:
            draw_pins(ax, frame, pins, pin_sprite(args.pin_size * args.dpi), args.pin_size)
        ax.set_xlim(frame.xmin, frame.xmax)
        ax.set_ylim(frame.ymin, frame.ymax)
        print("pins:", len(pins))


def draw_topographic(args: argparse.Namespace, out: str, frame: MapFrame) -> None:
    """--type topographic: heights (and sea depths with --ocean, ice with --ice), colour bars in the corners, pins;
    nothing political."""
    start = time.perf_counter()
    fig, ax = new_figure(frame)
    lakes = drawable_lakes(frame)
    draw_topography(ax, frame, args.dpi, args.ocean, args.ice, lakes)
    draw_colorbar(ax, frame, "br", LAND, "Height", LAND_CONTOURS)
    if args.ocean:
        draw_colorbar(ax, frame, "tr", SEA, "Depth", SEA_CONTOURS)
    if args.ice:
        draw_colorbar(ax, frame, "bl", ICE, "Ice", ICE_CONTOURS)
    bodies, water = water_labels(kinds_in(args.label, WATER_KINDS), lakes, frame)
    outcome = None
    if area_kinds(args.label):  # without water or land kinds no label stage runs
        # A crop's bars sit over the map: keep the labels off them.
        shown: dict[Corner, bool] = {"br": True, "tr": args.ocean, "bl": args.ice}
        bars = [c for c, on in shown.items() if on]
        room = frame if frame.globe else dataclasses.replace(frame, outline=label_outline(frame, bars))
        outcome = place([*water, *land_labels(kinds_in(args.label, land.LAND_KINDS), frame)], room, args.label)
        draw_curved_labels(ax, outcome.placement, frame)
    if draws_rivers(args):
        draw_rivers(ax, shown_rivers(frame, bodies, water, outcome))
    ax.set_xlim(frame.xmin, frame.xmax)
    ax.set_ylim(frame.ymin, frame.ymax)
    add_pins(ax, frame, args)
    save(fig, out, args.format, args.dpi)
    print("wrote", out, "in %.1f s" % (time.perf_counter() - start))


if __name__ == "__main__":
    main()
