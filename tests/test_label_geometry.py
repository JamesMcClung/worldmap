import numpy as np

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
