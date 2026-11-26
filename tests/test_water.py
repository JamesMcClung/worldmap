import geopandas as gpd
import numpy as np
import pytest
import shapely
import shapely.affinity
from shapely.geometry import LineString, Point

from worldmap.curves import arc_lengths, bend
from worldmap.frame import MapFrame, make_frame
from worldmap.labels.text import PX_PER_PT
from worldmap.labels.curved import CURVED_MIN_INSIDE, Body, CurvedLabel
from worldmap.labels.water import (OCEAN_MIN_PT, OCEAN_PT, RIVER_MAX_PT, RIVER_MIN_PT, SEA_MAX_PT, STYLES,
                                   WATER_MIN_PT, WATER_TEXT, best_stretch, classify,
                                   load_water_bodies, longest_paths, make_water_labels, river_label, river_name, river_size)
from worldmap.labels.priority import Unplaced
from worldmap.render import drawable_lakes


@pytest.fixture(scope="module")
def lakes_projected() -> gpd.GeoDataFrame:
    return drawable_lakes(make_frame()).shapes


def test_classify() -> None:
    assert classify("ocean") == "oceans"
    assert classify("gulf") == "seas" and classify("fjord") == "seas"
    assert classify("channel") == "straits"
    assert classify("reef") is None and classify("river") is None and classify("generic") is None


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


@pytest.fixture(scope="module")
def frame() -> MapFrame:
    return make_frame()


def test_water_label_layout(frame: MapFrame, lakes_projected: gpd.GeoDataFrame) -> None:
    bodies = load_water_bodies({"oceans", "seas"}, lakes_projected, make_frame())
    labels = [l for l in make_water_labels(bodies, frame) if isinstance(l, CurvedLabel)]
    med = next(l for l in labels if l.text == "Mediterranean Sea")
    atlantic = next(l for l in labels if l.text == "ATLANTIC OCEAN")
    assert OCEAN_MIN_PT <= atlantic.size <= OCEAN_PT and WATER_MIN_PT <= med.size <= SEA_MAX_PT
    assert len(med.parts) == len("MediterraneanSea")
    assert med.inside_share(med.anchor_x, med.anchor_y) >= CURVED_MIN_INSIDE


def test_hopeless_bodies_have_no_label(frame: MapFrame, lakes_projected: gpd.GeoDataFrame) -> None:
    bodies = load_water_bodies({"oceans", "seas", "lakes", "straits"}, lakes_projected, make_frame())
    fits = make_water_labels(bodies, frame)
    labels = [l for l in fits if isinstance(l, CurvedLabel)]
    assert len(fits) == len(bodies) and 0 < len(labels) < len(bodies)
    assert all(l.inside_share(l.anchor_x, l.anchor_y) >= CURVED_MIN_INSIDE for l in labels)


def test_make_water_labels_one_per_body(frame: MapFrame, lakes_projected: gpd.GeoDataFrame) -> None:
    bodies = [b for b in load_water_bodies({"rivers"}, lakes_projected, frame) if b.text in ("Nile", "Suez Canal")]
    fits = make_water_labels(bodies, frame)
    assert {b.text: isinstance(w, Unplaced) for b, w in zip(bodies, fits)} == {"Nile": False, "Suez Canal": True}
    suez = next(b for b in bodies if b.text == "Suez Canal")
    assert Unplaced("Suez Canal", "rivers", suez.full_size) in fits


def test_draw_curved_labels(frame: MapFrame, lakes_projected: gpd.GeoDataFrame) -> None:
    from matplotlib.patches import PathPatch

    from test_placement import placed
    from worldmap.render import draw_curved_labels, new_figure

    med = [l for l in make_water_labels(load_water_bodies({"seas"}, lakes_projected, make_frame()), frame) if isinstance(l, CurvedLabel) and l.text == "Mediterranean Sea"]
    p = placed(med, frame)
    p.x[0] += 100  # somewhere other than its anchor
    fig, ax = new_figure(frame)
    before = len(ax.patches)
    draw_curved_labels(ax, p, frame)
    added = ax.patches[before:]
    assert len(added) == 1 and isinstance(added[0], PathPatch)
    assert np.allclose(added[0].get_facecolor()[:3], WATER_TEXT)
    ext = added[0].get_path().get_extents()
    gx0, gy0, gx1, gy1 = med[0].glyphs.get_extents().extents
    px, py = frame.px_to_map(np.array([p.x[0] + (gx0 + gx1) / 2]), np.array([p.y[0] + (gy0 + gy1) / 2]))
    assert abs((ext.x0 + ext.x1) / 2 - px[0]) * frame.px_per_unit < 1 and abs((ext.y0 + ext.y1) / 2 - py[0]) * frame.px_per_unit < 1


def test_river_name() -> None:
    assert river_name("Amazon") == "Amazon"
    assert river_name("Paraná River") == "Paraná"
    assert river_name("Grande") == "Rio Grande"
    assert river_name("Suez Canal") == "Suez Canal" and river_name("Sulina branch") == "Sulina branch"
    assert river_name(None) is None and river_name(float("nan")) is None


def test_river_size() -> None:
    assert river_size(0) == RIVER_MAX_PT
    assert river_size(8) == RIVER_MIN_PT == river_size(9)
    assert river_size(2) - river_size(4) == pytest.approx(river_size(4) - river_size(6))  # linear in the line width


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
    label = river_label(Body("Straight", "rivers", 1, LineString(np.column_stack([mx, my])), 100.0, STYLES["rivers"]), frame)
    assert label is not None
    assert label.geom.contains(Point(x0 + 20 * em, y0 + 0.9 * em))
    assert not label.geom.contains(Point(x0 + 20 * em, y0 + 1.1 * em))


def test_river_name_stays_off_drawn_lakes() -> None:
    """Natural Earth's lines run on through lakes (Lake Centerline: the Jordan through the Dead Sea), hidden under
    the lake fill: the name goes on the river where it shows."""
    frame = make_frame()
    em = river_size(1) * PX_PER_PT
    x0, y0 = 3000.0, 1500.0
    mx, my = frame.px_to_map(np.array([x0, x0 + 100 * em]), np.array([y0, y0]))
    line = LineString(np.column_stack([mx, my]))
    lx, ly = frame.px_to_map(np.array([x0 + 20 * em, x0 + 60 * em]), np.array([y0 - 5 * em, y0 + 5 * em]))
    lake = shapely.box(lx[0], ly[0], lx[1], ly[1])  # the middle, where the name would go
    label = river_label(Body("Straight", "rivers", 1, line, 100.0, STYLES["rivers"]), frame, lake)
    assert label is not None and label.anchor_x > x0 + 60 * em


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
    bodies = load_water_bodies({"oceans", "seas", "straits", "lakes"}, drawable_lakes(f).shapes, f)
    assert all(b.kind == "lakes" for b in bodies)


def bodies_by_text(frame: MapFrame) -> dict[str, list]:
    out: dict[str, list] = {}
    for b in load_water_bodies({"oceans", "seas", "lakes", "straits", "rivers"}, drawable_lakes(frame).shapes, frame):
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
    assert isinstance(label, CurvedLabel)
    assert label.full_size == world_bodies["Lake Victoria"][0].full_size


def test_inside_share_counts_glyphs_by_their_centres() -> None:
    """As the solver has it: a glyph box is on the water if its centre is, weighted by its area. A lake 5 px high
    under glyph boxes 14 px high is all on; a glyph whose centre is off is off, however much of its box is on."""
    from test_placement import water_label

    parts = np.array([[-50.0 + 10 * k, -7, -40 + 10 * k, 7] for k in range(10)])
    lake = water_label("Lake", (0, 0), shapely.box(-52, -2.5, 52, 2.5), parts, WATER_MIN_PT, kind="lakes")
    assert lake.inside_share(0, 0) == pytest.approx(1.0)
    assert lake.inside_share(30, 0) == pytest.approx(0.7)  # the right 3 glyphs' centres are past x = 52


def test_river_ranks_by_name_and_nearest_body() -> None:
    from worldmap.data import load_rivers
    from worldmap.labels.priority import UNLABELED
    from worldmap.labels.water import load_river_bodies, river_ranks

    f = make_frame()
    rivers = f.project(load_rivers())
    bodies = [b for b in load_river_bodies(f) if b.text in ("Colorado", "Nile")]
    assert len([b for b in bodies if b.text == "Colorado"]) == 2  # the US's and Argentina's
    ranks = river_ranks(rivers, bodies, np.array([2 if b.geom.centroid.y < 0 else 3 for b in bodies]))
    named = rivers.name_en.map(river_name)
    colorado = (named == "Colorado").values
    south = rivers.geometry.centroid.y.values < 0
    assert (ranks[colorado & south] == 2).all() and (ranks[colorado & ~south] == 3).all()
    assert (ranks[(named == "Nile").values] == 3).all()
    assert (ranks[named.isna().values] == UNLABELED).all() and (ranks[(named == "Danube").values] == UNLABELED).all()
