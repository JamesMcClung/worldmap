"""Drawing the map with matplotlib: globe, country fills and borders, state lines, lakes, labels."""

from collections.abc import Callable, Iterator

import geopandas as gpd
import matplotlib
import numpy as np
import shapely
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch, Polygon as MplPolygon
from matplotlib.path import Path
from shapely.geometry import MultiPolygon, Point
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from worldmap.coloring import OCEAN
from worldmap.disputed import Dispute, dispute_fills, stripe_parts
from worldmap.earth import Fill, Regions
from worldmap.frame import DPI, POSTER_WIDTH_IN, MapFrame
from worldmap.labels.label import Label
from worldmap.labels.placement import Placement, leader_ends

BORDER, STATE_LINE = "#3a3a3a", "#7d7d7d"
ANTARCTICA_FILL = "#ffffff"  # Antarctica is drawn white and left unlabelled
GLOBE_STROKE = 2.2
BORDER_WIDTH = 0.6
STATE_LINE_WIDTH = 0.35
BAIKONUR_HATCH_WIDTH = 1.6
LEADER_WIDTH = 0.7
LEADER_DOT_SIZE = 1.3
BORDER_DASH = (0, (2.5, 1.5))  # --disputed: borders between a disputed area and the countries involved
# Drawing order, bottom to top (pins are at 10, labels at 6). Borders sit below the lakes so they don't cross them.
Z_OCEAN, Z_COUNTRY, Z_BAIKONUR, Z_STRIPES, Z_BAIKONUR_EDGE, Z_BORDERS, Z_STATES, Z_LAKES, Z_GLOBE, Z_LEADERS = 0, 1, 1.5, 1.5, 1.6, 1.7, 2, 2.5, 3, 4


def patches(geom: BaseGeometry) -> Iterator[MplPolygon]:
    """Exterior rings of a (Multi)Polygon as patches; holes are ignored."""
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if p.geom_type == "Polygon" and not p.is_empty:
            yield MplPolygon(np.asarray(p.exterior.coords), closed=True)


def holed_patches(geom: BaseGeometry) -> Iterator[PathPatch]:
    """Like patches(), but keeping holes (e.g. islands in a lake)."""
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if p.geom_type == "Polygon" and not p.is_empty:
            p = orient(p)  # holes wind opposite to the exterior, so the nonzero fill rule leaves them empty
            yield PathPatch(Path.make_compound_path(*(Path(np.asarray(ring.coords), closed=True) for ring in [p.exterior, *p.interiors])))


def new_figure(frame: MapFrame) -> tuple[Figure, Axes]:
    """Poster-sized figure, transparent outside the globe, with the ocean and the globe's outline drawn."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(POSTER_WIDTH_IN, POSTER_WIDTH_IN * frame.height / frame.width), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(frame.xmin, frame.xmax)
    ax.set_ylim(frame.ymin, frame.ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.add_patch(MplPolygon(frame.outline, closed=True, fc=OCEAN, ec="none", zorder=Z_OCEAN))
    ax.add_patch(MplPolygon(frame.outline, closed=True, fc="none", ec=BORDER, lw=GLOBE_STROKE, zorder=Z_GLOBE))
    return fig, ax


def draw_countries(ax: Axes, countries_projected: gpd.GeoDataFrame, fill_of: Callable[[int, Point | None], Fill], regions: Regions | None, disputed: bool = False) -> None:
    """Country fills with borders. With `regions` (--color earth), each polygon gets its own fill; otherwise Baikonur
    (Kazakh land leased to Russia) is hatched in Kazakhstan's colour. With `disputed`, fills only: draw_borders draws
    the borders and draw_stripes stripes Baikonur with the other disputed areas."""
    kazakhstan = countries_projected.index[countries_projected.NAME_EN == "Kazakhstan"][0]
    ec = "none" if disputed else BORDER
    for i, row in countries_projected.iterrows():
        if row.NAME_EN == "Antarctica":
            ax.add_collection(PatchCollection(list(patches(row.geometry)), fc=ANTARCTICA_FILL, ec=ec, lw=BORDER_WIDTH, zorder=Z_COUNTRY))
        elif regions is not None:
            for poly, fill in regions[i]:
                ax.add_collection(PatchCollection(list(patches(poly)), fc=fill, ec=ec, lw=BORDER_WIDTH, zorder=Z_COUNTRY))
        elif row.NAME_EN == "Baikonur" and not disputed:
            # Hatched in Kazakhstan's colour, with its border drawn above the hatching.
            ax.add_collection(PatchCollection(list(patches(row.geometry)), fc=fill_of(i, None), ec="none", zorder=Z_COUNTRY))
            matplotlib.rcParams["hatch.linewidth"] = BAIKONUR_HATCH_WIDTH
            stripe = fill_of(kazakhstan, None)
            ax.add_collection(PatchCollection(list(patches(row.geometry)), fc="none", ec=stripe, lw=0, hatch="////", zorder=Z_BAIKONUR))
            ax.add_collection(PatchCollection(list(patches(row.geometry)), fc="none", ec=BORDER, lw=BORDER_WIDTH, zorder=Z_BAIKONUR_EDGE))
        else:
            ax.add_collection(PatchCollection(list(patches(row.geometry)), fc=fill_of(i, None), ec=ec, lw=BORDER_WIDTH, zorder=Z_COUNTRY))


def draw_stripes(ax: Axes, disputes: list[Dispute], fill: Callable[[int], Fill], width: float) -> None:
    """Each disputed area in equal stripes, `width` map units across, of the colours of the countries involved."""
    for d in disputes:
        fills = dispute_fills(d, fill)
        for part, color in zip(stripe_parts(d.geometry, len(fills), width), fills):
            polys = [patch for g in shapely.get_parts(part) if g.geom_type == "Polygon" for patch in holed_patches(g)]
            ax.add_collection(PatchCollection(polys, fc=color, ec="none", zorder=Z_STRIPES))


def line_coords(geom: BaseGeometry) -> list[np.ndarray]:
    return [np.asarray(line.coords) for line in shapely.get_parts(shapely.line_merge(geom)) if not line.is_empty]


def draw_borders(ax: Axes, solid: BaseGeometry, dashed: BaseGeometry) -> None:
    """Country borders as lines (--disputed), above the stripes; see disputed.border_lines."""
    ax.add_collection(LineCollection(line_coords(solid), colors=BORDER, linewidths=BORDER_WIDTH, zorder=Z_BORDERS))
    ax.add_collection(LineCollection(line_coords(dashed), colors=BORDER, linewidths=BORDER_WIDTH, linestyles=[BORDER_DASH], capstyle="butt", zorder=Z_BORDERS))


def draw_states(ax: Axes, states_projected: gpd.GeoDataFrame) -> None:
    states_projected.plot(ax=ax, color=STATE_LINE, linewidth=STATE_LINE_WIDTH, zorder=Z_STATES)


def draw_lakes(ax: Axes, lakes_projected: gpd.GeoDataFrame) -> None:
    """The country polygons include the lakes; paint the big ones back as water (borders through them then run on
    the shore), above the state lines, which also cross them."""
    for g in lakes_projected.geometry:
        ax.add_collection(PatchCollection(list(holed_patches(g)), fc=OCEAN, ec=BORDER, lw=BORDER_WIDTH, zorder=Z_LAKES))


def subtract_lakes(countries_projected: gpd.GeoDataFrame, lakes_projected: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Countries minus the drawn lakes, so label placement treats the lakes as water."""
    out = countries_projected.copy()
    out["geometry"] = countries_projected.geometry.difference(unary_union(list(lakes_projected.geometry)))
    return out


def draw_labels(ax: Axes, labels: list[Label], placement: Placement, frame: MapFrame) -> None:
    """Move each label's text to its placed position, and draw leader lines and dots for 'out' labels."""
    end_x, end_y = leader_ends(placement)
    for i, label in enumerate(labels):
        x, y = frame.px_to_map(placement.x[i], placement.y[i])
        label.text.set_position((float(x), float(y)))
        if placement.out[i]:
            x0, y0 = frame.px_to_map(placement.anchor_x[i], placement.anchor_y[i])
            x1, y1 = frame.px_to_map(end_x[i], end_y[i])
            ax.plot([x0, x1], [y0, y1], color=label.color, lw=LEADER_WIDTH, zorder=Z_LEADERS, solid_capstyle="round")
            ax.plot([x0], [y0], marker="o", ms=LEADER_DOT_SIZE, color=label.color, zorder=Z_LEADERS)
