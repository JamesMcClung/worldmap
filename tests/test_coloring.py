import geopandas as gpd
import pytest
from shapely.geometry import box

from worldmap.coloring import cache_key, color_graph, load_or_compute_coloring, merge_into_parents, spot_check
from worldmap.data import is_labeled, load_countries, load_disputed, parent_index
from worldmap.disputed import dispute_edges, load_disputes


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


def test_merge_into_parents() -> None:
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6), box(1, 0, 2, 1)], crs="EPSG:4326")
    merged, major, node_of = merge_into_parents(gdf, [True, False, True], {1: 0})
    assert len(merged) == 2 and node_of == [0, 0, 1] and major == [True, True]
    assert merged.geometry.iloc[0].area == pytest.approx(2.0)


def test_cache_key_covers_major_and_parents() -> None:
    c = load_countries()
    major = [is_labeled(r) for _, r in c.iterrows()]
    parents = parent_index(c)
    key = cache_key(c, major, parents, set())
    flipped = list(major)
    flipped[0] = not flipped[0]
    assert cache_key(c, flipped, parents, set()) != key
    assert cache_key(c, major, {}, set()) != key
    assert cache_key(c, major, parents, {(0, 1)}) != key


def test_coloring_keeps_dependencies_with_parents_and_dispute_pairs_apart() -> None:
    """With the same inputs as cli.main, so the cached coloring.json stays valid for real runs."""
    c = load_countries()
    parents = parent_index(c)
    edges = dispute_edges(load_disputes(c, load_disputed()))
    col = load_or_compute_coloring(c, [is_labeled(r) for _, r in c.iterrows()], False, parents, edges).color_class
    assert all(col[d] == col[p] for d, p in parents.items())
    assert all(col[a] != col[b] for a, b in edges)
    assert spot_check(c, col) == []
