"""--type topographic: colour bars in the empty canvas corners outside the globe. Each is a right triangle with its
right angle in the corner and its third side following the globe; tick labels sit in the gap between bar and globe."""

from typing import Literal

import numpy as np
import shapely
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import LineString, Polygon, box

from worldmap.frame import DPI, MapFrame

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


def draw_colorbar(ax: Axes, frame: MapFrame, corner: Corner, cmap: Colormap, top: float, title: str) -> None:
    """A colour bar in `corner`: 0 m at the canvas edge (the bottom for br/bl, the top for tr) rising to `top` near
    the tip, a tick every TICK_STEP on the globe side, and `title` at the tip."""
    from worldmap.render import BORDER, BORDER_WIDTH, Z_GLOBE

    region = bar_region(frame, corner)
    x0, y0, x1, y1 = region.bounds
    up = corner != "tr"  # values grow upward from the bottom edge
    edge, tip = (y0, y1) if up else (y1, y0)
    t = np.clip(np.linspace(0, 1, 512) / TIP, 0, 1)[:, None]  # rows from the edge to the tip
    gradient = cmap(np.repeat(t, 2, axis=1))
    image = ax.imshow(gradient, extent=(x0, x1, edge, tip), origin="lower", aspect="auto", interpolation="bilinear", zorder=Z_GLOBE)
    outline = MplPolygon(np.asarray(region.exterior.coords), closed=True, fc="none", ec=BORDER, lw=BORDER_WIDTH, zorder=Z_GLOBE + 0.1)
    ax.add_patch(outline)
    image.set_clip_path(outline)

    unit = DPI / frame.px_per_unit
    tick, pad = 0.08 * unit, 0.12 * unit
    left = corner in ("br", "tr")  # labels left of the bar (the globe is to the left)
    for v in np.arange(0, top + 1, TICK_STEP):
        y = edge + (tip - edge) * TIP * v / top
        cut = LineString([(x0 - 1, y), (x1 + 1, y)]).intersection(region)
        if cut.is_empty:
            continue
        cx0, _, cx1, _ = cut.bounds
        x = cx0 if left else cx1
        sign = -1 if left else 1
        ax.plot([x, x + sign * tick], [y, y], color=BORDER, lw=BORDER_WIDTH, zorder=Z_GLOBE + 0.1, solid_capstyle="butt",
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])
        text = "sea level" if v == 0 else f"{int(v):,} m" + ("+" if v == top else "")
        ax.text(x + sign * pad, y, text, fontsize=TICK_SIZE, color=TEXT_COLOR, ha="right" if left else "left", va="center",
                path_effects=HALO, zorder=Z_GLOBE + 0.2)
    title_y = tip + (pad if up else -pad)
    title_x = x1 if left else x0
    ax.text(title_x, title_y, title, fontsize=TITLE_SIZE, fontweight="bold", color=TEXT_COLOR, ha="right" if left else "left",
            va="bottom" if up else "top", path_effects=HALO, zorder=Z_GLOBE + 0.2)
