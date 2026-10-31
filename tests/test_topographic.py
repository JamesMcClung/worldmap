import numpy as np
import pytest

from worldmap.topographic import Grid, elevation_grid, sample


@pytest.fixture(scope="module")
def grid() -> Grid:
    return elevation_grid()


def test_grid_shape(grid: Grid) -> None:
    assert grid.z.shape == (3600, 7200) and grid.z.dtype == np.int16


@pytest.mark.parametrize("lon, lat, lo, hi", [
    (86.925, 27.988, 7000, 9000),  # Everest (3' means lower the peak)
    (142.2, 11.35, -11100, -9000),  # Mariana Trench
    (10.0, 25.0, 100, 2000),  # Sahara
    (51.0, 42.0, -900, -20),  # Caspian Sea floor: below 0, so water
    (35.5, 31.5, -800, -100),  # Dead Sea
    (-30.0, 0.0, -6000, -2000),  # mid-Atlantic
])
def test_known_places(grid: Grid, lon: float, lat: float, lo: float, hi: float) -> None:
    assert lo <= sample(grid, np.array([lon]), np.array([lat]))[0] <= hi


def test_sample_wraps_antimeridian(grid: Grid) -> None:
    a, b = sample(grid, np.array([179.99, -179.99]), np.array([65.0, 65.0]))  # Chukotka
    assert abs(a - b) < 150


def test_netherlands_value_recorded(grid: Grid) -> None:
    """What the data says about the Dutch polders (Flevoland): near sea level, so they may read as sea."""
    v = sample(grid, np.array([5.5]), np.array([52.5]))[0]
    assert -10 < v < 10
