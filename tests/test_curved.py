import numpy as np
import shapely
import shapely.affinity
from shapely.geometry import Point

from worldmap.labels.curved import fit_curve


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


def test_fit_curve_steep_and_bent_never_tips_past_vertical() -> None:
    ring = Point(0, 0).buffer(10).difference(Point(0, 0).buffer(8))
    arch = shapely.affinity.rotate(ring.intersection(shapely.box(-10, 2, 10, 10)), 80, origin=(0, 0))  # chord at ~80 degrees
    c = fit_curve(arch, 90)
    assert (np.diff(c[:, 0]) >= -1e-9).all()  # every step still goes right: no glyph leans upside down


def test_styles_by_kind() -> None:
    from worldmap.labels.curved import OCEAN_TRACKING_EM
    from worldmap.labels.water import STYLES, WATER_KINDS

    assert STYLES.keys() == set(WATER_KINDS)
    assert STYLES["oceans"].tracking_em == OCEAN_TRACKING_EM and STYLES["oceans"].cap_centred
    assert STYLES["seas"].tracking_em == 0 and not STYLES["seas"].cap_centred
    assert STYLES["rivers"].italic and not STYLES["seas"].italic
