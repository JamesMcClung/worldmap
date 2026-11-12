import numpy as np
import pytest
import shapely

from worldmap.labels.geometry import nearest_point_on_rect, segment_hits_rects, segments_cross, two_line


def hits(p0, p1, rect) -> bool:
    return bool(segment_hits_rects(np.array(p0, float), np.array(p1, float), np.array(rect, float)))


def test_segment_hits_rect() -> None:
    assert hits((0, 0), (10, 0), (4, -1, 6, 1))
    assert not hits((0, 0), (10, 0), (4, 2, 6, 3))


def test_segment_parallel_inside_edge() -> None:
    assert hits((5, -5), (5, 5), (4, -1, 6, 1))


def cross(a0, a1, b0, b1) -> bool:
    return bool(segments_cross(*(np.array(p, float) for p in (a0, a1, b0, b1))))


def test_segments_cross() -> None:
    assert cross((0, 0), (2, 2), (0, 2), (2, 0))  # X
    assert not cross((0, 0), (2, 0), (1, 0), (1, 2))  # T: touching only
    assert not cross((0, 0), (2, 0), (0, 1), (2, 1))  # parallel


def test_nearest_point_on_rect() -> None:
    nx, ny = nearest_point_on_rect(np.array(10.0), np.array(0.0), 0.0, 0.0, 4.0, 2.0)
    assert (float(nx), float(ny)) == (2.0, 0.0)


def test_two_line() -> None:
    assert two_line("Central African Republic") == "Central African\nRepublic"
    assert two_line("Chad") is None


def test_coverage_matches_the_exact_area() -> None:
    """A summed-area table of a country-like shape gives each box's area over it to within its edges' pixels."""
    from worldmap.labels.geometry import Coverage

    shape = shapely.Polygon([(0, 0), (300, 40), (260, 200), (120, 150), (30, 210)]).difference(shapely.Point(150, 90).buffer(30))
    rng = np.random.default_rng(1)
    x0, y0 = rng.uniform(-40, 300, 300), rng.uniform(-40, 220, 300)
    boxes = np.column_stack([x0, y0, x0 + rng.uniform(5, 60, 300), y0 + rng.uniform(5, 30, 300)])
    exact = shapely.area(shapely.intersection(shapely.box(*boxes.T), shape))
    rim = 2 * (boxes[:, 2] - boxes[:, 0] + boxes[:, 3] - boxes[:, 1])  # a pixel's error along every edge, at worst
    assert (np.abs(Coverage.of(shape).area(boxes) - exact) <= rim).all()


def test_coverage_of_boxes_off_the_shape_and_of_nothing() -> None:
    from worldmap.labels.geometry import Coverage

    assert Coverage.of(shapely.box(0, 0, 10, 10)).area(np.array([[20.0, 20, 30, 30], [-5, -5, 15, 15]])) == pytest.approx([0, 100])
    assert Coverage.of(shapely.Polygon()).area(np.array([[0.0, 0, 1, 1]])) == pytest.approx([0])


def test_coverage_contains_matches_shapely_away_from_the_edge() -> None:
    from worldmap.labels.geometry import Coverage

    shape = shapely.Polygon([(0, 0), (300, 40), (260, 200), (120, 150), (30, 210)]).difference(shapely.Point(150, 90).buffer(30))
    rng = np.random.default_rng(2)
    x, y = rng.uniform(-40, 340, 2000), rng.uniform(-40, 250, 2000)
    clear = shapely.distance(shapely.points(x, y), shape.boundary) > 1.5  # a cell's diagonal from the edge
    assert (Coverage.of(shape).contains(x, y) == shapely.contains_xy(shape, x, y))[clear].all()
    assert Coverage.of(shapely.Polygon()).contains(np.array([0.0]), np.array([0.0])).tolist() == [False]

