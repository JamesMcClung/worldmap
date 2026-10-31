"""--type topographic: colour bars in the empty canvas corners outside the globe. Each is a right triangle with its
right angle in the corner and its third side following the globe; tick labels sit in the gap between bar and globe."""

from typing import Literal

import numpy as np
import shapely
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import LineString, Polygon, box

from worldmap.frame import DPI, MapFrame
from worldmap.topographic import Contours, Scale

Corner = Literal["br", "tr", "bl"]  # bottom-right, top-right, bottom-left
BAR_GAP_IN = 0.7  # gap between the bar and the globe, for the tick labels (poster inches)
BAR_MARGIN_IN = 0.2  # from the canvas edges
TICK_STEP = 1000  # metres
TIP = 0.97  # the top value sits this far along from the canvas edge to the tip; beyond it, the top colour
TEXT_COLOR = "#3a3a3a"
TICK_SIZE, TITLE_SIZE = 9, 12  # points
HALO = [patheffects.withStroke(linewidth=2.5, foreground="white")]  # readable on a transparent or dark background


def bar_region(frame: MapFrame, corner: Corner) -> Polygon:
    """The corner's quadrant of the canvas, minus the globe grown by the gap, minus a margin from the canvas edges."""
    unit = DPI / frame.px_per_unit  # map units per poster inch
    m = BAR_MARGIN_IN * unit
    xs = (0, frame.xmax - m) if corner in ("br", "tr") else (frame.xmin + m, 0)
    ys = (frame.ymin + m, 0) if corner in ("br", "bl") else (0, frame.ymax - m)
    region = box(xs[0], ys[0], xs[1], ys[1]).difference(Polygon(frame.outline).buffer(BAR_GAP_IN * unit))
    return max(shapely.get_parts(region), key=lambda g: g.area)


def metres(v: float) -> str:
    """Signed metres with a real minus sign: '1,000 m', '−500 m'."""
    return ("−" if v < 0 else "") + f"{abs(int(v)):,} m"


def tick_values(scale: Scale) -> list[float]:
    """The scale's two ends, and every multiple of TICK_STEP between them, from the canvas edge to the tip."""
    lo, hi = sorted((scale.start, scale.end))
    inner = [float(v) for v in np.arange(np.floor(lo / TICK_STEP) + 1, np.ceil(hi / TICK_STEP)) * TICK_STEP]
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


def draw_colorbar(ax: Axes, frame: MapFrame, corner: Corner, scale: Scale, title: str, contours: Contours) -> None:
    """A colour bar in `corner`: scale.start at the canvas edge (the bottom for br/bl, the top for tr) to scale.end near
    the tip, a labelled tick every TICK_STEP on the globe side, the map's contour lines across it (and sea level in
    the coastline's style), and `title` at the tip."""
    from worldmap.render import BORDER, BORDER_WIDTH, Z_GLOBE
    from worldmap.topographic import CONTOUR_WIDTH

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

    lo, hi = sorted((scale.start, scale.end))
    levels = [v for v in np.arange(np.floor(lo / contours.step) + 1, np.ceil(hi / contours.step)) * contours.step if v != 0]
    ax.add_collection(LineCollection([seg for v in levels for seg in across(v)], colors=contours.color, linewidths=CONTOUR_WIDTH,
                                     alpha=contours.alpha, linestyles=contours.style, zorder=Z_GLOBE + 0.05))
    if lo < 0 < hi:
        ax.add_collection(LineCollection(across(0), colors=BORDER, linewidths=BORDER_WIDTH, zorder=Z_GLOBE + 0.05))

    unit = DPI / frame.px_per_unit
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
    title_y = tip + (pad if up else -pad)
    title_x = x1 if left else x0
    ax.text(title_x, title_y, title, fontsize=TITLE_SIZE, fontweight="bold", color=TEXT_COLOR, ha="right" if left else "left",
            va="bottom" if up else "top", path_effects=HALO, zorder=Z_GLOBE + 0.2)
