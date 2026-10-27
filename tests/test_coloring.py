import pytest

from worldmap.coloring import color_graph


def test_color_graph_is_proper() -> None:
    edges = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (0, 2)}  # 5-cycle plus a chord
    k, color_class = color_graph(5, edges, [1.0] * 5)
    assert k == 3
    assert all(color_class[i] != color_class[j] for i, j in edges)


def test_color_graph_respects_kmin() -> None:
    k, _ = color_graph(2, {(0, 1)}, [1.0, 1.0], kmin=5)
    assert k == 5


def test_color_graph_unsolvable_raises() -> None:
    k4 = {(i, j) for i in range(4) for j in range(i + 1, 4)}
    with pytest.raises(RuntimeError):
        color_graph(4, k4, [1.0] * 4, kmax=3)
