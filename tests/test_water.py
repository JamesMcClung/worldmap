import geopandas as gpd
import numpy as np
import pytest
import shapely
import shapely.affinity
from shapely.geometry import Point

from worldmap.curves import arc_lengths, bend
from worldmap.data import load_lakes, project
from worldmap.frame import MapFrame, make_frame
from worldmap.labels.water import OCEAN_PT, SEA_MAX_PT, WATER_MIN_INSIDE, WATER_MIN_PT, WATER_TEXT, classify, fit_curve, kinds_from_labels, load_water_bodies, make_water_labels


@pytest.fixture(scope="module")
def lakes_projected() -> gpd.GeoDataFrame:
    return project(load_lakes())


def test_classify() -> None:
    assert classify("ocean") == "oceans"
    assert classify("gulf") == "seas" and classify("fjord") == "seas"
    assert classify("channel") == "straits"
    assert classify("reef") is None and classify("river") is None and classify("generic") is None


def test_kinds_from_labels() -> None:
    assert kinds_from_labels({"countries"}) == set()
    assert kinds_from_labels({"seas", "countries"}) == {"seas"}
    assert kinds_from_labels({"water"}) == {"oceans", "seas", "lakes", "straits"}


def test_oceans(lakes_projected: gpd.GeoDataFrame) -> None:
    oceans = load_water_bodies({"oceans"}, lakes_projected)
    texts = [b.text for b in oceans]
    assert sorted(texts) == sorted(["ATLANTIC OCEAN", "PACIFIC OCEAN", "PACIFIC OCEAN", "INDIAN OCEAN", "ARCTIC OCEAN", "SOUTHERN OCEAN"])
    pacific = [b.geom.centroid.x for b in oceans if b.text == "PACIFIC OCEAN"]
    assert min(pacific) < 0 < max(pacific)


def test_seas_lakes_and_straits(lakes_projected: gpd.GeoDataFrame) -> None:
    seas = {b.text for b in load_water_bodies({"seas"}, lakes_projected)}
    assert {"Mediterranean Sea", "Caspian Sea", "Bay of Bengal", "Skagerrak"} <= seas
    assert not seas & {"Great Barrier Reef", "Amazon River", "Strait of Gibraltar"}
    everything = load_water_bodies({"oceans", "seas", "lakes", "straits"}, lakes_projected)
    names = [b.text for b in everything]
    assert "Strait of Gibraltar" in names and "Lake Victoria" in names
    assert names.count("Bering Sea") == 1 and names.count("Caspian Sea") == 1
    assert everything[0].kind == "oceans"
    ranks = [b.rank for b in everything if b.kind != "oceans"]
    assert ranks == sorted(ranks)


def test_bend_straight_and_circle() -> None:
    line = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert np.allclose(bend(line, np.array([2.0, 5.0]), np.array([1.0, -1.0]), 0.01), [[2, 1], [5, -1]])
    theta = np.linspace(np.pi, 0, 500)  # upper half circle, radius 10, left to right
    arc = np.stack([10 * np.cos(theta), 10 * np.sin(theta)], -1)
    pts = bend(arc, np.array([arc_lengths(arc)[-1] / 2]), np.array([2.0]), 0.01)
    assert np.allclose(pts, [[0, 12]], atol=1e-2)  # left of rightward travel over the top is up, away from the centre


def test_fit_curve_horizontal_and_reading_direction() -> None:
    ellipse = shapely.affinity.scale(Point(0, 0).buffer(1), 10, 2)
    c = fit_curve(ellipse, 90)
    assert c[0, 0] < c[-1, 0] and np.ptp(c[:, 1]) < 0.5


def test_fit_curve_steep_never_upside_down() -> None:
    red_sea_like = shapely.affinity.rotate(shapely.affinity.scale(Point(0, 0).buffer(1), 10, 1), -60)
    c = fit_curve(red_sea_like, 90)
    assert c[0, 0] < c[-1, 0]  # never right to left (upside down)
    assert c[0, 1] > c[-1, 1]  # this one runs down to the right


def test_fit_curve_vertical_reads_bottom_to_top() -> None:
    tall = shapely.affinity.scale(Point(0, 0).buffer(1), 1, 10)
    c = fit_curve(tall, 90)
    assert c[0, 1] < c[-1, 1]


def test_fit_curve_tilt_clamped() -> None:
    tall = shapely.affinity.scale(Point(0, 0).buffer(1), 1, 10)
    c = fit_curve(tall, 20)
    angle = np.degrees(np.arctan2(c[-1, 1] - c[0, 1], c[-1, 0] - c[0, 0]))
    assert abs(angle) <= 20 + 1e-6


def test_fit_curve_follows_a_bend() -> None:
    ring = Point(0, 0).buffer(10).difference(Point(0, 0).buffer(8))
    upper = ring.intersection(shapely.box(-10, 2, 10, 10))  # an upward arch
    c = fit_curve(upper, 90)
    assert c[len(c) // 2, 1] > c[0, 1] and c[len(c) // 2, 1] > c[-1, 1]


@pytest.fixture(scope="module")
def frame() -> MapFrame:
    return make_frame()


def test_water_label_layout(frame: MapFrame, lakes_projected: gpd.GeoDataFrame) -> None:
    bodies = load_water_bodies({"oceans", "seas"}, lakes_projected)
    labels = make_water_labels(bodies, frame)
    med = next(l for l in labels if l.text == "Mediterranean Sea")
    atlantic = next(l for l in labels if l.text == "ATLANTIC OCEAN")
    assert atlantic.size == OCEAN_PT and WATER_MIN_PT <= med.size <= SEA_MAX_PT
    assert len(med.parts) == len("MediterraneanSea")
    assert med.inside_share(med.anchor_x, med.anchor_y) >= WATER_MIN_INSIDE
    width = med.parts[:, 2].max() - med.parts[:, 0].min()
    med.scale(0.5)
    assert med.parts[:, 2].max() - med.parts[:, 0].min() == pytest.approx(width / 2)


def test_hopeless_bodies_left_out(frame: MapFrame, lakes_projected: gpd.GeoDataFrame) -> None:
    bodies = load_water_bodies({"oceans", "seas", "lakes", "straits"}, lakes_projected)
    labels = make_water_labels(bodies, frame)
    assert 0 < len(labels) < len(bodies)
    assert all(l.inside_share(l.anchor_x, l.anchor_y) >= WATER_MIN_INSIDE for l in labels)


def test_draw_water_labels(frame: MapFrame, lakes_projected: gpd.GeoDataFrame) -> None:
    from matplotlib.patches import PathPatch

    from worldmap.labels.placement import place_labels
    from worldmap.render import draw_water_labels, new_figure

    med = [l for l in make_water_labels(load_water_bodies({"seas"}, lakes_projected), frame) if l.text == "Mediterranean Sea"]
    p = place_labels([], frame, water=med)
    p.x[0] += 100  # somewhere other than its anchor
    fig, ax = new_figure(frame)
    before = len(ax.patches)
    draw_water_labels(ax, med, p, frame)
    added = ax.patches[before:]
    assert len(added) == 1 and isinstance(added[0], PathPatch)
    assert np.allclose(added[0].get_facecolor()[:3], WATER_TEXT)
    ext = added[0].get_path().get_extents()
    gx0, gy0, gx1, gy1 = med[0].glyphs.get_extents().extents
    px, py = frame.px_to_map(np.array([p.x[0] + (gx0 + gx1) / 2]), np.array([p.y[0] + (gy0 + gy1) / 2]))
    assert abs((ext.x0 + ext.x1) / 2 - px[0]) * frame.px_per_unit < 1 and abs((ext.y0 + ext.y1) / 2 - py[0]) * frame.px_per_unit < 1
