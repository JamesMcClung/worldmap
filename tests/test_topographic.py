import numpy as np
import pytest
from matplotlib.colors import to_rgb

from worldmap.coloring import OCEAN
from worldmap.topographic import Grid, colorize, elevation_grid, sample, shade


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


def test_sea_is_flat_ocean_without_ocean_flag() -> None:
    rgb = colorize(np.array([[-5000.0, -1.0]]), np.zeros((1, 2), bool), ocean=False)
    assert np.allclose(rgb, to_rgb(OCEAN), atol=1e-3)


def test_sea_darkens_with_depth() -> None:
    rgb = colorize(np.array([[-1.0, -4000.0, -20000.0]]), np.zeros((1, 3), bool), ocean=True)
    assert np.allclose(rgb[0, 0], to_rgb(OCEAN), atol=0.01) and rgb[0, 1].sum() > rgb[0, 2].sum()
    assert np.allclose(rgb[0, 2], to_rgb("#2f5f8c"), atol=1e-3)  # below 8,000 m: the bottom colour


def test_land_ramp_ends() -> None:
    rgb = colorize(np.array([[1.0, 9000.0]]), np.zeros((1, 2), bool), ocean=False)
    assert np.allclose(rgb[0, 0], to_rgb("#5f9e5a"), atol=0.01) and np.allclose(rgb[0, 1], 1.0, atol=1e-3)


def test_ice_wins_over_sea_and_land() -> None:
    rgb = colorize(np.array([[-300.0, 3000.0]]), np.ones((1, 2), bool), ocean=True)
    assert (rgb > 0.75).all()  # white to pale blue, never sea blue or brown


def test_flat_sea_unshaded_without_ocean() -> None:
    z = np.random.default_rng(0).normal(-3000, 500, (20, 20))
    rgb = colorize(z, np.zeros(z.shape, bool), False)
    assert np.array_equal(shade(rgb, z, 5000.0, z <= 0, False), rgb)
