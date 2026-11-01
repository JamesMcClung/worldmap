import geopandas as gpd
import numpy as np
import pytest

from worldmap.curves import arc_lengths, bend
from worldmap.data import load_lakes, project
from worldmap.labels.water import classify, kinds_from_labels, load_water_bodies


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
