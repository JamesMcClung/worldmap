import geopandas as gpd
import numpy as np
import pytest
import shapely
import shapely.affinity
from shapely.geometry import LineString, Point

from worldmap.curves import arc_lengths, bend
from worldmap.data import load_lakes
from worldmap.frame import MapFrame, make_frame
from worldmap.labels.text import PX_PER_PT
from worldmap.labels.water import (OCEAN_MIN_PT, OCEAN_PT, RIVER_MAX_PT, RIVER_MIN_PT, RIVER_SCALE_MAX, SEA_MAX_PT, WATER_MIN_INSIDE, WATER_MIN_PT,
                                   WATER_TEXT, best_stretch, classify, fit_curve, kinds_from_labels, load_water_bodies,
                                   longest_paths, make_water_labels, river_label, river_name, river_size, WaterBody)


@pytest.fixture(scope="module")
def lakes_projected() -> gpd.GeoDataFrame:
    return make_frame().project(load_lakes())


def test_classify() -> None:
    assert classify("ocean") == "oceans"
    assert classify("gulf") == "seas" and classify("fjord") == "seas"
    assert classify("channel") == "straits"
    assert classify("reef") is None and classify("river") is None and classify("generic") is None


def test_kinds_from_labels() -> None:
    assert kinds_from_labels({"countries"}) == set()
    assert kinds_from_labels({"seas", "countries"}) == {"seas"}
    assert kinds_from_labels({"straits"}) == {"straits"}


def test_oceans(lakes_projected: gpd.GeoDataFrame) -> None:
    oceans = load_water_bodies({"oceans"}, lakes_projected, make_frame())
    texts = [b.text for b in oceans]
    assert sorted(texts) == sorted(["ATLANTIC OCEAN", "PACIFIC OCEAN", "PACIFIC OCEAN", "INDIAN OCEAN", "ARCTIC OCEAN", "SOUTHERN OCEAN"])
    pacific = [b.geom.centroid.x for b in oceans if b.text == "PACIFIC OCEAN"]
    assert min(pacific) < 0 < max(pacific)


def test_seas_lakes_and_straits(lakes_projected: gpd.GeoDataFrame) -> None:
    seas = {b.text for b in load_water_bodies({"seas"}, lakes_projected, make_frame())}
    assert {"Mediterranean Sea", "Caspian Sea", "Bay of Bengal", "Skagerrak"} <= seas
    assert not seas & {"Great Barrier Reef", "Amazon River", "Strait of Gibraltar"}
    everything = load_water_bodies({"oceans", "seas", "lakes", "straits"}, lakes_projected, make_frame())
    names = [b.text for b in everything]
    assert "Strait of Gibraltar" in names and "Lake Victoria" in names
    assert "Saint Lawrence River" not in names  # a 'channel' in Natural Earth, but a river: not labelled yet
    assert names.count("Bering Sea") == 1 and names.count("Caspian Sea") == 1
    assert everything[0].kind == "oceans"
    sizes = [b.full_size for b in everything]
    assert sizes == sorted(sizes, reverse=True)


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



def test_fit_curve_round_a_ring_follows_it_below_its_middle() -> None:
    ring = Point(0, 0).buffer(10).difference(Point(0, 0).buffer(8))  # the Southern Ocean round the pole
    c = fit_curve(ring, 90)
    radius = np.hypot(c[:, 0], c[:, 1])
    assert np.all((radius > 8.5) & (radius < 9.5))  # along the ring, never across its hole
    assert c[0, 0] < c[-1, 0] and c[len(c) // 2, 1] < -8  # reads left to right, under the hole



def test_fit_curve_island_at_the_middle_is_not_a_ring() -> None:
    lake = shapely.affinity.scale(Point(0, 0).buffer(1), 10, 2).difference(Point(0, 0).buffer(0.5))  # Lake Kivu's island
    c = fit_curve(lake, 90)
    assert np.ptp(c[:, 0]) > 10 and np.ptp(c[:, 1]) < 0.5  # straight along it, as without the island


@pytest.fixture(scope="module")
def frame() -> MapFrame:
    return make_frame()


def test_water_label_layout(frame: MapFrame, lakes_projected: gpd.GeoDataFrame) -> None:
    bodies = load_water_bodies({"oceans", "seas"}, lakes_projected, make_frame())
    labels = make_water_labels(bodies, frame)
    med = next(l for l in labels if l.text == "Mediterranean Sea")
    atlantic = next(l for l in labels if l.text == "ATLANTIC OCEAN")
    assert OCEAN_MIN_PT <= atlantic.size <= OCEAN_PT and WATER_MIN_PT <= med.size <= SEA_MAX_PT
    assert len(med.parts) == len("MediterraneanSea")
    assert med.inside_share(med.anchor_x, med.anchor_y) >= WATER_MIN_INSIDE


def test_hopeless_bodies_left_out(frame: MapFrame, lakes_projected: gpd.GeoDataFrame) -> None:
    bodies = load_water_bodies({"oceans", "seas", "lakes", "straits"}, lakes_projected, make_frame())
    labels = make_water_labels(bodies, frame)
    assert 0 < len(labels) < len(bodies)
    assert all(l.inside_share(l.anchor_x, l.anchor_y) >= WATER_MIN_INSIDE for l in labels)


def test_draw_water_labels(frame: MapFrame, lakes_projected: gpd.GeoDataFrame) -> None:
    from matplotlib.patches import PathPatch

    from test_placement import placed
    from worldmap.render import draw_water_labels, new_figure

    med = [l for l in make_water_labels(load_water_bodies({"seas"}, lakes_projected, make_frame()), frame) if l.text == "Mediterranean Sea"]
    p = placed(med, frame)
    p.x[0] += 100  # somewhere other than its anchor
    fig, ax = new_figure(frame)
    before = len(ax.patches)
    draw_water_labels(ax, p, frame)
    added = ax.patches[before:]
    assert len(added) == 1 and isinstance(added[0], PathPatch)
    assert np.allclose(added[0].get_facecolor()[:3], WATER_TEXT)
    ext = added[0].get_path().get_extents()
    gx0, gy0, gx1, gy1 = med[0].glyphs.get_extents().extents
    px, py = frame.px_to_map(np.array([p.x[0] + (gx0 + gx1) / 2]), np.array([p.y[0] + (gy0 + gy1) / 2]))
    assert abs((ext.x0 + ext.x1) / 2 - px[0]) * frame.px_per_unit < 1 and abs((ext.y0 + ext.y1) / 2 - py[0]) * frame.px_per_unit < 1


def test_fit_curve_steep_and_bent_never_tips_past_vertical() -> None:
    ring = Point(0, 0).buffer(10).difference(Point(0, 0).buffer(8))
    arch = shapely.affinity.rotate(ring.intersection(shapely.box(-10, 2, 10, 10)), 80, origin=(0, 0))  # chord at ~80 degrees
    c = fit_curve(arch, 90)
    assert (np.diff(c[:, 0]) >= -1e-9).all()  # every step still goes right: no glyph leans upside down


def test_river_name() -> None:
    assert river_name("Amazon") == "Amazon"
    assert river_name("Paraná River") == "Paraná"
    assert river_name("Grande") == "Rio Grande"
    assert river_name("Sulina branch") is None and river_name("Suez Canal") is None
    assert river_name(None) is None and river_name(float("nan")) is None


def test_river_size() -> None:
    assert river_size(0) == river_size(1) == RIVER_MAX_PT
    assert river_size(6) == RIVER_MIN_PT == river_size(9)
    assert river_size(2) > river_size(4)


def test_river_names_stay_small_in_a_crop() -> None:
    from worldmap.crop import box_region, crop_frame
    from worldmap.labels.water import load_river_bodies

    f = crop_frame(box_region((5, 45, 15, 52)), 0)  # the Alps and Germany: label_scale near the cap
    assert f.label_scale > RIVER_SCALE_MAX
    rivers = make_water_labels(load_river_bodies(f), f)
    assert rivers and max(r.size for r in rivers) <= RIVER_MAX_PT * RIVER_SCALE_MAX
    assert max(r.min_size for r in rivers) <= WATER_MIN_PT * RIVER_SCALE_MAX


def test_every_drawn_river_is_a_candidate() -> None:
    """Rivers of every drawn scalerank get a body (and so may be named), not only the bigger ones."""
    from worldmap.crop import box_region, crop_frame
    from worldmap.labels.water import load_river_bodies

    bodies = load_river_bodies(crop_frame(box_region((5, 45, 15, 52)), 0))
    assert max(b.rank for b in bodies) > 6


def test_river_name_must_stay_within_an_em_of_its_river() -> None:
    """Its glyph centres count as on the river (its label's geom) within 1 em of the line, not beyond: a name the
    solver slides off its stretch (where it was bent to fit) doesn't float beside the river."""
    frame = make_frame()
    em = river_size(1) * PX_PER_PT
    x0, y0 = 3000.0, 1500.0
    mx, my = frame.px_to_map(np.array([x0, x0 + 40 * em]), np.array([y0, y0]))
    label = river_label(WaterBody("Straight", "rivers", 1, LineString(np.column_stack([mx, my])), 100.0), frame)
    assert label is not None
    assert label.geom.contains(Point(x0 + 20 * em, y0 + 0.9 * em))
    assert not label.geom.contains(Point(x0 + 20 * em, y0 + 1.1 * em))


def test_longest_paths_joins_gaps_and_takes_the_trunk() -> None:
    trunk = [LineString([(0, 0), (10, 0)]), LineString([(10.5, 0), (20, 0)])]  # a 0.5 gap (a lake)
    tributary = LineString([(10, 0), (10, 3)])
    far = LineString([(100, 0), (104, 0)])
    paths = longest_paths([*trunk, tributary, far], join=1.0)
    assert len(paths) == 2
    assert paths[0].length == pytest.approx(20)  # 0..10, bridged to 10.5..20
    assert paths[1].length == pytest.approx(4)


def test_longest_paths_survives_loops() -> None:
    braid = [LineString([(0, 0), (5, 1), (10, 0)]), LineString([(0, 0), (5, -1), (10, 0)]), LineString([(10, 0), (30, 0)])]
    ring = LineString([(50, 0), (51, 1), (52, 0), (50, 0)])
    paths = longest_paths([*braid, ring], join=0.1)
    assert paths[0].length > 30 and len(paths) == 2


def test_best_stretch_prefers_straight_and_level() -> None:
    zigzag = [(x, 5 * (x % 2)) for x in range(0, 50)]  # wiggly first half...
    straight = [(x, 0.0) for x in range(50, 100)]  # ...level second half
    stretch = best_stretch(np.array(zigzag + straight, float), 30)
    assert stretch[:, 0].min() >= 49 and np.ptp(stretch[:, 1]) < 1e-9


def test_crop_without_marine_areas() -> None:
    from worldmap.crop import box_region, crop_frame

    f = crop_frame(box_region((10, 10, 20, 20)), 0)  # Chad and Nigeria: no sea in sight
    bodies = load_water_bodies({"oceans", "seas", "straits", "lakes"}, f.project(load_lakes()), f)
    assert all(b.kind == "lakes" for b in bodies)


def bodies_by_text(frame: MapFrame) -> dict[str, list]:
    out: dict[str, list] = {}
    for b in load_water_bodies({"oceans", "seas", "lakes", "straits", "rivers"}, frame.project(load_lakes()), frame):
        out.setdefault(b.text, []).append(b)
    return out


@pytest.fixture(scope="module")
def world_bodies() -> dict[str, list]:
    return bodies_by_text(make_frame())


def test_full_size_of_water(world_bodies: dict[str, list]) -> None:
    assert 6e4 <= world_bodies["Lake Victoria"][0].full_size <= 7.5e4
    a, b = world_bodies["PACIFIC OCEAN"]
    assert a.full_size == b.full_size > 1e8
    assert world_bodies["Mediterranean Sea"][0].full_size > 1e6
    assert 2.4e3 <= world_bodies["Danube"][0].full_size <= 3.2e3
    colorados = sorted(world_bodies["Colorado"], key=lambda b: b.geom.centroid.y)  # Argentina's, then the US's
    assert len(colorados) == 2 and colorados[0].full_size != colorados[1].full_size
    assert 1.8e3 <= colorados[1].full_size <= 2.6e3


def test_water_full_size_is_frame_independent(world_bodies: dict[str, list]) -> None:
    from worldmap.crop import box_region, crop_frame

    crop = bodies_by_text(crop_frame(box_region((5, 40, 20, 49)), 0))
    for name in ("Danube", "Adriatic Sea"):
        assert crop[name][0].full_size == pytest.approx(world_bodies[name][0].full_size, rel=1e-9)
    label = make_water_labels(world_bodies["Lake Victoria"], make_frame())[0]
    assert label.full_size == world_bodies["Lake Victoria"][0].full_size


def test_inside_share_counts_glyphs_by_their_centres() -> None:
    """As the solver has it: a glyph box is on the water if its centre is, weighted by its area. A lake 5 px high
    under glyph boxes 14 px high is all on; a glyph whose centre is off is off, however much of its box is on."""
    from test_placement import water_label

    parts = np.array([[-50.0 + 10 * k, -7, -40 + 10 * k, 7] for k in range(10)])
    lake = water_label("Lake", (0, 0), shapely.box(-52, -2.5, 52, 2.5), parts, WATER_MIN_PT, kind="lakes")
    assert lake.inside_share(0, 0) == pytest.approx(1.0)
    assert lake.inside_share(30, 0) == pytest.approx(0.7)  # the right 3 glyphs' centres are past x = 52
