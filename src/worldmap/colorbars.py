"""--type topographic: colour bars in the empty canvas corners outside the globe. Each is a right triangle with its
right angle in the corner and its third side following the globe; tick labels sit in the gap between bar and globe,
the title inside the bar, curved along that third side."""

from typing import Literal

import numpy as np
import shapely
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from shapely.geometry import LineString, Polygon, box

from worldmap.curves import arc_lengths, bend
from worldmap.frame import MapFrame
from worldmap.render import BORDER, BORDER_WIDTH, Z_GLOBE
from worldmap.topographic import CONTOUR_WIDTH, Contours, Scale

Corner = Literal["br", "tr", "bl"]  # bottom-right, top-right, bottom-left
BAR_GAP_IN = 0.7  # gap between the bar and the globe, for the tick labels (poster inches)
BAR_MARGIN_IN = 0.2  # from the canvas edges
TICK_STEP = 1000  # metres
TIP = 0.97  # the top value sits this far along from the canvas edge to the tip; beyond it, the top colour
TEXT_COLOR = "#3a3a3a"
TICK_SIZE, TITLE_SIZE = 9, 24  # points
TITLE_INSET_IN = 0.08  # from the bar's curved side to the title
HALO = [patheffects.withStroke(linewidth=2.5, foreground="white")]  # readable on a transparent or dark background
TITLE_HALO = [patheffects.withStroke(linewidth=3, foreground="white")]


def bar_region(frame: MapFrame, corner: Corner) -> Polygon:
    """The corner's quadrant of the canvas, minus the globe grown by the gap, minus a margin from the canvas edges."""
    unit = frame.units_per_in
    m = BAR_MARGIN_IN * unit
    xs = (0, frame.xmax - m) if corner in ("br", "tr") else (frame.xmin + m, 0)
    ys = (frame.ymin + m, 0) if corner in ("br", "bl") else (0, frame.ymax - m)
    region = box(xs[0], ys[0], xs[1], ys[1]).difference(Polygon(frame.outline).buffer(BAR_GAP_IN * unit))
    return max(shapely.get_parts(region), key=lambda g: g.area)


def metres(v: float) -> str:
    """Signed metres with a real minus sign: '1,000 m', '−500 m'."""
    return ("−" if v < 0 else "") + f"{abs(int(v)):,} m"


def multiples(scale: Scale, step: float) -> list[float]:
    """The multiples of `step` strictly between the scale's ends, ascending."""
    lo, hi = sorted((scale.start, scale.end))
    return [float(v) for v in np.arange(np.floor(lo / step) + 1, np.ceil(hi / step)) * step]


def tick_values(scale: Scale) -> list[float]:
    """The scale's two ends, and every multiple of TICK_STEP between them, from the canvas edge to the tip."""
    inner = multiples(scale, TICK_STEP)
    values = [scale.start, *inner, scale.end]
    return values if scale.start < scale.end else [scale.start, *inner[::-1], scale.end]


def tick_label(v: float, scale: Scale) -> str:
    """'sea level' at 0; the ends say that heights beyond them take the end colour ('≥ 6,000 m', '≤ −8,000 m')."""
    if v == 0:
        return "sea level"
    if v in (scale.start, scale.end):
        beyond = v == max(scale.start, scale.end)
        return ("≥ " if beyond else "≤ ") + metres(v)
    return metres(v)


def globe_side(region: Polygon) -> np.ndarray:
    """The bar's curved side (the one following the globe) as (n, 2) vertices, left to right."""
    x0, y0, x1, y1 = region.bounds
    edges = box(x0, y0, x1, y1).exterior.buffer(1e-6 * (x1 - x0))
    curve = shapely.line_merge(region.exterior.difference(edges))
    side = max(shapely.get_parts(curve), key=lambda g: g.length)
    xy = np.asarray(side.coords)
    return xy if xy[0, 0] < xy[-1, 0] else xy[::-1]


def curved_title(region: Polygon, title: str, unit: float) -> Path:
    """`title`'s glyphs bent along the bar's curved side, centred on it, inside the bar TITLE_INSET_IN from it.
    `unit` is map units per poster inch."""
    side = globe_side(region)
    s = arc_lengths(side)
    glyphs = TextPath((0, 0), title, size=TITLE_SIZE, prop=FontProperties(weight="bold"))
    scale = unit / 72  # map units per point
    u, v = glyphs.vertices[:, 0] * scale, glyphs.vertices[:, 1] * scale
    extents = glyphs.get_extents()
    width, top, bottom = extents.xmax * scale, extents.ymax * scale, extents.ymin * scale
    u = u + s[-1] / 2 - width / 2  # distance along the side
    h = 0.05 * unit
    k = [len(u) // 2]
    mid = bend(side, np.array([s[-1] / 2]), np.zeros(1), h)[0]
    mid_normal = bend(side, u[k], np.ones(1), h)[0] - bend(side, u[k], np.zeros(1), h)[0]
    inset = TITLE_INSET_IN * unit
    inside_left = region.contains(shapely.Point(mid + mid_normal * inset))
    v = v + inset - bottom if inside_left else v - inset - top  # descenders above the side, or cap tops below it
    return Path(bend(side, u, v, h), glyphs.codes)


def draw_colorbar(ax: Axes, frame: MapFrame, corner: Corner, scale: Scale, title: str, contours: Contours) -> None:
    """A colour bar in `corner`: scale.start at the canvas edge (the bottom for br/bl, the top for tr) to scale.end near
    the tip, a labelled tick every TICK_STEP on the globe side, the map's contour lines across it (and sea level in
    the coastline's style), and `title` along the curved side."""
    region = bar_region(frame, corner)
    x0, y0, x1, y1 = region.bounds
    up = corner != "tr"  # the canvas edge is at the bottom
    edge, tip = (y0, y1) if up else (y1, y0)
    t = np.clip(np.linspace(0, 1, 512) / TIP, 0, 1)[:, None]  # rows from the edge to the tip
    gradient = scale.cmap(np.repeat(t, 2, axis=1))
    image = ax.imshow(gradient, extent=(x0, x1, edge, tip), origin="lower", aspect="auto", interpolation="bilinear", zorder=Z_GLOBE)
    outline = MplPolygon(np.asarray(region.exterior.coords), closed=True, fc="none", ec=BORDER, lw=BORDER_WIDTH, zorder=Z_GLOBE + 0.1)
    ax.add_patch(outline)
    image.set_clip_path(outline)

    def y_of(v: float) -> float:
        return edge + (tip - edge) * TIP * float(scale.position(v))

    def across(v: float) -> list[np.ndarray]:
        """The bar's width at height v, as line segments."""
        y = y_of(v)
        cut = LineString([(x0 - 1, y), (x1 + 1, y)]).intersection(region)
        return [np.asarray(part.coords) for part in shapely.get_parts(cut) if part.geom_type == "LineString"]

    levels = [v for v in multiples(scale, contours.step) if v != 0]
    ax.add_collection(LineCollection([seg for v in levels for seg in across(v)], colors=contours.color, linewidths=CONTOUR_WIDTH,
                                     alpha=contours.alpha, linestyles=contours.style, zorder=Z_GLOBE + 0.05))
    if 0 < scale.position(0) < 1:  # sea level is on the bar
        ax.add_collection(LineCollection(across(0), colors=BORDER, linewidths=BORDER_WIDTH, zorder=Z_GLOBE + 0.05))

    unit = frame.units_per_in
    tick, pad = 0.08 * unit, 0.12 * unit
    left = corner in ("br", "tr")  # labels left of the bar (the globe is to the left)
    for v in tick_values(scale):
        segments = across(v)
        if not segments:
            continue
        xs = np.concatenate(segments)[:, 0]
        x = xs.min() if left else xs.max()
        y = y_of(v)
        sign = -1 if left else 1
        ax.plot([x, x + sign * tick], [y, y], color=BORDER, lw=BORDER_WIDTH, zorder=Z_GLOBE + 0.1, solid_capstyle="butt",
                path_effects=HALO)
        ax.text(x + sign * pad, y, tick_label(v, scale), fontsize=TICK_SIZE, color=TEXT_COLOR, ha="right" if left else "left",
                va="center", path_effects=HALO, zorder=Z_GLOBE + 0.2)
    ax.add_patch(PathPatch(curved_title(region, title, unit), fc=TEXT_COLOR, ec="none", path_effects=TITLE_HALO,
                           zorder=Z_GLOBE + 0.2))
