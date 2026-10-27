"""Country labels: which pieces get one, where it is anchored, its size, and whether it sits on its country ('in')
or beside it with a leader line ('out'). All positions and geometry are in poster pixels."""

from collections.abc import Callable
from dataclasses import dataclass

import geopandas as gpd
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.text import Text
from shapely import maximum_inscribed_circle
from shapely.geometry import MultiPolygon, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from worldmap.colorspace import RGB, darken
from worldmap.data import is_labeled
from worldmap.earth import Fill
from worldmap.frame import DPI, MapFrame
from worldmap.labels.geometry import rect, two_line

RENAME = {"People's Republic of China": "China", "United States of America": "United States", "Democratic Republic of the Congo": "DR Congo", "Republic of the Congo": "Congo", "Central African Republic": "C.A.R.", "Bosnia and Herzegovina": "Bosnia & Herz.", "United Arab Emirates": "UAE", "Dominican Republic": "Dominican Rep."}
MIN_PT, MAX_PT = 6, 30
FORCE_ISLAND = {"Greece"}
HALO = [pe.withStroke(linewidth=2.2, foreground=(1, 1, 1, 0.55))]

# Separately labelled exclaves: (country, label, a lon/lat point inside it).
EXCLAVES = [("France", "French Guiana", (-53, 4)), ("Russia", "Kaliningrad (Russia)", (21, 54.7))]


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
    label.width, label.height = ext.width * 1.06, ext.height * 1.15


def make_labels(countries_projected: gpd.GeoDataFrame, frame: MapFrame, ax: Axes, renderer: RendererBase, fill_of: Callable[[int, Point | None], Fill]) -> list[Label]:
    """One label per labelled country (plus the EXCLAVES), largest first.

    The anchor is the centre of the biggest polygon's largest inscribed circle. Island-heavy countries (several
    significant polygons, the largest under 65% of their total, or FORCE_ISLAND) use their convex hull's centroid and
    a hull-based size instead."""
    ppu = frame.px_per_unit
    exclave_pts = {country: (name, Point(frame.to_map.transform(*lonlat))) for country, name, lonlat in EXCLAVES}
    labels, island_names = [], []
    for i, row in countries_projected.iterrows():
        if not is_labeled(row):
            continue
        polys = [p for p in (row.geometry.geoms if isinstance(row.geometry, MultiPolygon) else [row.geometry]) if not p.is_empty]
        if not polys:
            continue
        big = max(polys, key=lambda p: p.area)
        pieces = []
        sig = [p for p in polys if p.area >= 0.02 * big.area]
        hull = unary_union(sig).convex_hull
        hb = hull.bounds
        island = len(sig) >= 2 and big.area / sum(p.area for p in sig) < 0.65 and (hb[2] - hb[0]) < 0.25 * frame.width
        if row.NAME_EN in FORCE_ISLAND and len(sig) >= 2:
            island = True
        if island:
            island_names.append(row.NAME_EN)
            pieces.append((row.NAME_EN, hull.centroid.coords[0], unary_union(sig), max(sum(p.area for p in sig), hull.area)))
        else:
            c = maximum_inscribed_circle(big, tolerance=frame.width / 5000).coords[0]
            pieces.append((row.NAME_EN, c, big, big.area))
        if row.NAME_EN in exclave_pts:
            name, pt = exclave_pts[row.NAME_EN]
            for p in polys:
                if p.contains(pt):
                    c = maximum_inscribed_circle(p, tolerance=frame.width / 5000).coords[0]
                    pieces.append((name, c, p, p.area))
        for nm, c, geom, area in pieces:
            nm = RENAME.get(nm, nm)
            size = float(np.clip(1.6 * np.sqrt(area) * ppu / DPI * 72 / 10, MIN_PT, MAX_PT))
            color = darken(fill_of(i, geom.representative_point() if geom.geom_type == "Polygon" else None))
            text = ax.text(c[0], c[1], nm.upper() if size > 14 else nm, fontsize=size, ha="center", va="center", color=color, zorder=6, path_effects=HALO)
            label = Label(nm, area, color, text, 0.0, 0.0, (c[0] - frame.xmin) * ppu, (c[1] - frame.ymin) * ppu, frame.to_px(geom))
            measure(label, renderer)
            labels.append(label)
    labels.sort(key=lambda l: -l.area)
    print("island-heavy (hull-centred):", island_names)
    return labels


def decide_out_and_wrap(labels: list[Label], renderer: RendererBase) -> None:
    """Mark labels that would hide their country (or are far too wide for it) as 'out', and wrap long names onto two
    lines when an 'in' label is wider than its country."""
    for label in labels:
        gb = label.geom.bounds
        box = rect(label.anchor_x, label.anchor_y, label.width, label.height)
        covers = box.intersection(label.geom).area / label.geom.area if label.geom.area > 0 else 1.0
        label.out = covers >= 0.6 or (label.width > 1.6 * (gb[2] - gb[0]) and (gb[3] - gb[1]) < 6 * label.height)
    for label in labels:
        gb = label.geom.bounds
        if not label.out and " " in label.text.get_text() and label.width > 0.9 * (gb[2] - gb[0]) and (gb[3] - gb[1]) > 2.5 * label.height:
            label.text.set_multialignment("center")
            label.text.set_linespacing(1.0)
            label.text.set_text(two_line(label.text.get_text()))
            measure(label, renderer)
