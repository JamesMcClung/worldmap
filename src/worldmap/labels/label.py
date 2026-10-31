"""Country labels: which pieces get one, where it is anchored, its size, and whether it sits on its country ('in')
or beside it with a leader line ('out'). All positions and geometry are in poster pixels."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import geopandas as gpd
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.text import Text
from shapely import maximum_inscribed_circle
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from worldmap.colorspace import RGB, darken
from worldmap.data import is_labeled
from worldmap.earth import Fill
from worldmap.frame import DPI, MapFrame
from worldmap.labels.geometry import rect, two_line

if TYPE_CHECKING:  # regions.py imports RENAME from here
    from worldmap.labels.regions import ChildRegion

RENAME = {"People's Republic of China": "China", "United States of America": "United States", "Democratic Republic of the Congo": "DR Congo", "Republic of the Congo": "Congo", "Central African Republic": "C.A.R.", "Bosnia and Herzegovina": "Bosnia & Herz.", "United Arab Emirates": "UAE", "Dominican Republic": "Dominican Rep."}
MIN_PT, MAX_PT = 6, 30
FONT_SCALE = 1.6  # font size (pt) = FONT_SCALE / 10 x sqrt(area), with the area measured in points^2
POINTS_PER_INCH = 72
LABEL_ZORDER = 6
HALO = [pe.withStroke(linewidth=2.2, foreground=(1, 1, 1, 0.55))]
BOX_PAD_X, BOX_PAD_Y = 1.06, 1.15  # label box = rendered text extent scaled by these

# Island-heavy countries: several polygons of at least SIGNIFICANT_SHARE of the largest, the largest being under
# ISLAND_LARGEST_SHARE of their total, and a convex hull narrower than ISLAND_MAX_HULL_WIDTH of the map.
SIGNIFICANT_SHARE = 0.02
ISLAND_LARGEST_SHARE = 0.65
ISLAND_MAX_HULL_WIDTH = 0.25
FORCE_ISLAND = {"Greece"}
INSCRIBED_CIRCLE_DIVISOR = 5000  # inscribed-circle tolerance = map width / this

# 'out' labels: covering at least OUT_COVER of the country, or wider than OUT_WIDTH_RATIO x a country that is less
# than OUT_SHORT_HEIGHTS label heights tall.
OUT_COVER = 0.6
OUT_WIDTH_RATIO = 1.6
OUT_SHORT_HEIGHTS = 6
# Wrap an 'in' label onto two lines when wider than WRAP_WIDTH_RATIO x its country and the country is at least
# WRAP_MIN_HEIGHTS label heights tall.
WRAP_WIDTH_RATIO = 0.9
WRAP_MIN_HEIGHTS = 2.5

@dataclass
class Label:
    name: str
    area: float  # of the piece it labels, map units^2; bigger labels are placed first
    color: RGB
    text: Text
    width: float  # padded text box
    height: float
    anchor_x: float  # where the label belongs (and its leader's dot, if 'out')
    anchor_y: float
    geom: BaseGeometry  # the piece it labels
    out: bool = False  # True: placed beside the country with a leader line


def measure(label: Label, renderer: RendererBase) -> None:
    """Set the label's box size from its rendered text, with a little padding."""
    ext = label.text.get_window_extent(renderer)
    label.width, label.height = ext.width * BOX_PAD_X, ext.height * BOX_PAD_Y


def anchor_piece(polys: list[Polygon], frame: MapFrame, tolerance: float, force_island: bool) -> tuple[tuple[float, float], BaseGeometry, float, bool]:
    """Where a label for these polygons goes: (anchor, the geometry it labels, its area, island-heavy?).

    The anchor is the centre of the biggest polygon's largest inscribed circle. Island-heavy pieces use their
    convex hull's centroid and a hull-based size instead."""
    largest = max(polys, key=lambda p: p.area)
    significant = [p for p in polys if p.area >= SIGNIFICANT_SHARE * largest.area]
    significant_area = sum(p.area for p in significant)
    hull = unary_union(significant).convex_hull
    hull_x0, _, hull_x1, _ = hull.bounds
    island = len(significant) >= 2 and largest.area / significant_area < ISLAND_LARGEST_SHARE and (hull_x1 - hull_x0) < ISLAND_MAX_HULL_WIDTH * frame.width
    if force_island and len(significant) >= 2:
        island = True
    if island:
        return hull.centroid.coords[0], unary_union(significant), max(significant_area, hull.area), True
    return maximum_inscribed_circle(largest, tolerance=tolerance).coords[0], largest, largest.area, False


def new_label(ax: Axes, renderer: RendererBase, frame: MapFrame, name: str, anchor: tuple[float, float], geom: BaseGeometry, area: float, color: RGB, italic: bool = False) -> Label:
    """A measured label for `name` at `anchor` (map units), sized by `area`. Child regions are set in italics."""
    ppu = frame.px_per_unit
    x, y = anchor
    size = float(np.clip(FONT_SCALE * np.sqrt(area) * ppu / DPI * POINTS_PER_INCH / 10, MIN_PT, MAX_PT))
    text = ax.text(x, y, name, fontsize=size, ha="center", va="center", color=color, zorder=LABEL_ZORDER, path_effects=HALO, fontstyle="italic" if italic else "normal")
    label = Label(name, area, color, text, 0.0, 0.0, (x - frame.xmin) * ppu, (y - frame.ymin) * ppu, frame.to_px(geom))
    measure(label, renderer)
    return label


def make_labels(countries_projected: gpd.GeoDataFrame, frame: MapFrame, ax: Axes, renderer: RendererBase, fill_of: Callable[[int, Point | None], Fill], children: Sequence["ChildRegion"] = ()) -> list[Label]:
    """One label per labelled country, plus an italic one per child region (--regions), largest first (see
    anchor_piece for where each goes)."""
    tolerance = frame.width / INSCRIBED_CIRCLE_DIVISOR
    labels, island_names = [], []
    for i, row in countries_projected.iterrows():
        if not is_labeled(row):
            continue
        polys = [p for p in (row.geometry.geoms if isinstance(row.geometry, MultiPolygon) else [row.geometry]) if not p.is_empty]
        if not polys:
            continue
        anchor, geom, area, island = anchor_piece(polys, frame, tolerance, row.NAME_EN in FORCE_ISLAND)
        if island:
            island_names.append(row.NAME_EN)
        color = darken(fill_of(i, geom.representative_point() if geom.geom_type == "Polygon" else None))
        labels.append(new_label(ax, renderer, frame, RENAME.get(row.NAME_EN, row.NAME_EN), anchor, geom, area, color))
    for child in children:
        # Anchored on and sized by the region's largest polygon: a hull would blow up archipelagos spread over an
        # ocean (French Polynesia, Saint Helena with Ascension and Tristan da Cunha).
        largest = max(child.polys, key=lambda p: p.area)
        anchor = maximum_inscribed_circle(largest, tolerance=tolerance).coords[0]
        color = darken(fill_of(child.feature, largest.representative_point()))
        labels.append(new_label(ax, renderer, frame, child.text, anchor, largest, largest.area, color, italic=True))
    labels.sort(key=lambda label: -label.area)
    print("island-heavy (hull-centred):", island_names)
    return labels


def decide_out_and_wrap(labels: list[Label], renderer: RendererBase) -> None:
    """Mark labels that would hide their country (or are far too wide for it) as 'out', and wrap long names onto two
    lines when an 'in' label is wider than its country."""
    for label in labels:
        x0, y0, x1, y1 = label.geom.bounds
        box = rect(label.anchor_x, label.anchor_y, label.width, label.height)
        covers = box.intersection(label.geom).area / label.geom.area if label.geom.area > 0 else 1.0
        too_wide = label.width > OUT_WIDTH_RATIO * (x1 - x0) and (y1 - y0) < OUT_SHORT_HEIGHTS * label.height
        label.out = covers >= OUT_COVER or too_wide
    for label in labels:
        x0, y0, x1, y1 = label.geom.bounds
        wrapped = two_line(label.text.get_text())
        if not label.out and wrapped and label.width > WRAP_WIDTH_RATIO * (x1 - x0) and (y1 - y0) > WRAP_MIN_HEIGHTS * label.height:
            label.text.set_multialignment("center")
            label.text.set_linespacing(1.0)
            label.text.set_text(wrapped)
            measure(label, renderer)
