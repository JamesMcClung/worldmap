"""Drawing the map with matplotlib: globe, country fills and borders, state lines, lakes, labels."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property

import geopandas as gpd
import matplotlib.patheffects as pe
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

from worldmap.coloring import OCEAN
from worldmap.data import load_lakes
from worldmap.disputed import DISPUTED_GREY, Dispute, dispute_fills, stripe_parts
from worldmap.earth import Fill, Regions
from worldmap.frame import DPI, MapFrame
from worldmap.labels.label import Label
from worldmap.labels.placement import Placement, leader_ends
from worldmap.labels.curved import CURVED_HALO_PX, CurvedLabel
from worldmap.labels.water import RIVER_HALO, RIVER_TEXT, river_width, WATER_HALO, WATER_TEXT

BORDER, STATE_LINE = "#3a3a3a", "#7d7d7d"
ANTARCTICA_FILL = "#ffffff"  # Antarctica is drawn white and left unlabelled
FIXED_FILLS = {"Antarctica": ANTARCTICA_FILL, "Bir Tawil": DISPUTED_GREY}  # Bir Tawil: claimed by no one
GLOBE_STROKE = 2.2
BORDER_WIDTH = 0.6
STATE_LINE_WIDTH = 0.35
LEADER_WIDTH = 0.7
LEADER_DOT_SIZE = 1.3
RIVER = "#2d76a3"  # OKLCH (0.54, 0.10, 240): near the ocean's hue, dark enough to show on every fill
BORDER_DASH = (0, (2.5, 1.5))  # --disputed: borders between a disputed area and the countries involved
# Drawing order, bottom to top (pins are at 10, labels at 6). Borders sit below the lakes so they don't cross them.
Z_OCEAN, Z_COUNTRY, Z_STRIPES, Z_BAIKONUR_EDGE, Z_BORDERS, Z_STATES, Z_LAKES, Z_GLOBE, Z_LEADERS = 0, 1, 1.5, 1.6, 1.7, 2, 2.5, 3, 4
LAKE_MIN_IN2 = 0.002  # lakes covering at least this many square inches of the poster are drawn as water (~100 on the world map)
Z_RIVERS = 2.2  # above the state lines, below the lakes (which hide the lake centrelines)
Z_STATES_UNDER_BORDERS = 1.65  # --disputed: state lines above the stripes, below the (dashed) borders
LABEL_ZORDER = 6
HALO = [pe.withStroke(linewidth=2.2, foreground=(1, 1, 1, 0.55))]  # the country labels'


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

    fig = plt.figure(figsize=(frame.width_in, frame.height_in), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(frame.xmin, frame.xmax)
    ax.set_ylim(frame.ymin, frame.ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.add_patch(MplPolygon(frame.outline, closed=True, fc=OCEAN, ec="none", zorder=Z_OCEAN))
    if frame.globe:  # a crop is filled to its edges
        ax.add_patch(MplPolygon(frame.outline, closed=True, fc="none", ec=BORDER, lw=GLOBE_STROKE, zorder=Z_GLOBE))
    return fig, ax


def draw_countries(ax: Axes, countries_projected: gpd.GeoDataFrame, fill_of: Callable[[int, Point | None], Fill], regions: Regions | None, disputed: bool = False) -> None:
    """Country fills with borders. With `regions` (--color earth), each polygon gets its own fill. Baikonur (Kazakh
    land leased to Russia) is striped by draw_stripes, so its border is drawn again above the stripes. With `disputed`,
    fills only: draw_borders draws the borders."""
    ec = "none" if disputed else BORDER
    for i, row in countries_projected.iterrows():
        if row.NAME_EN in FIXED_FILLS:
            ax.add_collection(PatchCollection(list(patches(row.geometry)), fc=FIXED_FILLS[row.NAME_EN], ec=ec, lw=BORDER_WIDTH, zorder=Z_COUNTRY))
        elif regions is not None:
            for poly, fill in regions[i]:
                ax.add_collection(PatchCollection(list(patches(poly)), fc=fill, ec=ec, lw=BORDER_WIDTH, zorder=Z_COUNTRY))
        elif row.NAME_EN == "Baikonur" and not disputed:
            ax.add_collection(PatchCollection(list(patches(row.geometry)), fc=fill_of(i, None), ec="none", zorder=Z_COUNTRY))
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


def draw_states(ax: Axes, states_projected: gpd.GeoDataFrame, disputed: bool = False) -> None:
    """State/province lines. With `disputed` they go under the borders, which would otherwise lose their dashes where
    a disputed area's edge follows a state line (Kashmir and Himachal Pradesh)."""
    states_projected = states_projected[~states_projected.geometry.is_empty]  # a crop may show none
    if len(states_projected):
        states_projected.plot(ax=ax, color=STATE_LINE, linewidth=STATE_LINE_WIDTH, zorder=Z_STATES_UNDER_BORDERS if disputed else Z_STATES)


@dataclass
class Lakes:
    """The drawn lakes (drawable_lakes)."""
    shapes: gpd.GeoDataFrame  # projected, one row per lake

    @cached_property
    def merged(self) -> BaseGeometry:
        """As one shape: what subtract_lakes cuts out and river names keep off."""
        return shapely.union_all(self.shapes.geometry.values)


def drawable_lakes(frame: MapFrame) -> Lakes:
    """The lakes, projected, that cover at least LAKE_MIN_IN2 of the poster in the frame."""
    lakes = frame.project(load_lakes())
    return Lakes(lakes[lakes.area / frame.units_per_in ** 2 >= LAKE_MIN_IN2])


def draw_lakes(ax: Axes, lakes: Lakes) -> None:
    """The country polygons include the lakes; paint the big ones back as water (borders through them then run on
    the shore), above the state lines, which also cross them."""
    for g in lakes.shapes.geometry:
        ax.add_collection(PatchCollection(list(holed_patches(g)), fc=OCEAN, ec=BORDER, lw=BORDER_WIDTH, zorder=Z_LAKES))


def draw_rivers(ax: Axes, rivers_projected: gpd.GeoDataFrame) -> None:
    """--rivers: centrelines, widest for the most important. Thin ones go first, so big rivers run over their tributaries."""
    rivers = rivers_projected.sort_values("scalerank", ascending=False, kind="stable")
    lines, widths = [], []
    for rank, geom in zip(rivers.scalerank, rivers.geometry):
        for line in line_coords(geom):
            lines.append(line)
            widths.append(river_width(rank))
    ax.add_collection(LineCollection(lines, colors=RIVER, linewidths=widths, capstyle="round", joinstyle="round", zorder=Z_RIVERS))


def subtract_lakes(countries_projected: gpd.GeoDataFrame, lakes: Lakes) -> gpd.GeoDataFrame:
    """Countries minus the drawn lakes, so label placement treats the lakes as water."""
    out = countries_projected.copy()
    out["geometry"] = countries_projected.geometry.difference(lakes.merged)
    return out


def glyph_patch(glyphs: Path, scale: float, x: float, y: float, frame: MapFrame, **style: object) -> PathPatch:
    """A label's glyphs (px, relative to its position) scaled by `scale` (shrunk in placing), at (x, y) px. Round
    joins: the halo stroke (path_effects) would grow spikes at the glyphs' sharp corners."""
    vertices = np.asarray(glyphs.vertices) * scale + [x, y]
    mx, my = frame.px_to_map(vertices[:, 0], vertices[:, 1])
    return PathPatch(Path(np.stack([mx, my], -1), glyphs.codes), ec="none", lw=0, joinstyle="round", **style)  # type: ignore[arg-type]


def draw_labels(ax: Axes, placement: Placement, frame: MapFrame) -> None:
    """Each country label's glyphs at its placed position and size, and leader lines and dots for 'out' labels."""
    end_x, end_y = leader_ends(placement)
    for i, label in placement.each(Label):
        ax.add_patch(glyph_patch(label.glyphs, placement.size[i] / label.size, placement.x[i], placement.y[i], frame,
                                 fc=label.color, zorder=LABEL_ZORDER, path_effects=HALO))
        if placement.out[i]:
            x0, y0 = frame.px_to_map(placement.anchor_x[i], placement.anchor_y[i])
            x1, y1 = frame.px_to_map(end_x[i], end_y[i])
            ax.plot([x0, x1], [y0, y1], color=label.color, lw=LEADER_WIDTH, zorder=Z_LEADERS, solid_capstyle="round")
            ax.plot([x0], [y0], marker="o", ms=LEADER_DOT_SIZE, color=label.color, zorder=Z_LEADERS)


def draw_water_labels(ax: Axes, placement: Placement, frame: MapFrame) -> None:
    """Each water label's bent glyphs at its placed position and size, blue with a water-coloured halo (rivers:
    darker, with the country labels' halo), just under the country labels."""
    halo = [pe.withStroke(linewidth=CURVED_HALO_PX * 72 / DPI, foreground=WATER_HALO)]
    river_halo = [pe.withStroke(linewidth=CURVED_HALO_PX * 72 / DPI, foreground=RIVER_HALO)]
    for i, label in placement.each(CurvedLabel):
        river = label.kind == "rivers"
        ax.add_patch(glyph_patch(label.glyphs, placement.size[i] / label.size, placement.x[i], placement.y[i], frame,
                                 fc=RIVER_TEXT if river else WATER_TEXT, zorder=LABEL_ZORDER - 0.1, path_effects=river_halo if river else halo))
