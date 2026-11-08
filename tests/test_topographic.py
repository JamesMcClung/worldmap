import numpy as np
import pytest
import shapely
from matplotlib.colors import to_rgb

from worldmap.coloring import OCEAN
from worldmap.data import load_ice, load_lakes, load_land
from worldmap.frame import make_frame
from worldmap.topographic import (LAND_BELOW_SEA, Grid, PosterGrid, coastline, colorize, draw_topography, elevation_grid,
                                  polygon_mask, poster_grid, relief, sample, shade)


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


def no(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, bool)


def test_sea_is_flat_ocean_without_ocean_flag() -> None:
    rgb = colorize(np.array([[-5000.0, -1.0, 300.0]]), no((1, 3)), no((1, 3)), ocean=False)
    assert np.allclose(rgb, to_rgb(OCEAN), atol=1e-3)  # water above 0 m (coastal averages) too


def test_sea_darkens_with_depth() -> None:
    rgb = colorize(np.array([[-1.0, -4000.0, -20000.0]]), no((1, 3)), no((1, 3)), ocean=True)
    assert np.allclose(rgb[0, 0], to_rgb(OCEAN), atol=0.01) and rgb[0, 1].sum() > rgb[0, 2].sum()
    assert np.allclose(rgb[0, 2], to_rgb("#2f5f8c"), atol=1e-3)  # below -8,000 m: the bottom colour


def test_water_above_sea_level_is_lighter_than_ocean() -> None:
    rgb = colorize(np.array([[0.0, 250.0, 900.0]]), no((1, 3)), no((1, 3)), ocean=True)
    assert rgb[0, 0].sum() < rgb[0, 1].sum() < rgb[0, 2].sum()


def test_land_scale_ends() -> None:
    land = np.ones((1, 3), bool)
    rgb = colorize(np.array([[1.0, 9000.0, -2000.0]]), land, no((1, 3)), ocean=False)
    assert np.allclose(rgb[0, 0], to_rgb("#5f9e5a"), atol=0.02) and np.allclose(rgb[0, 1], 1.0, atol=1e-3)
    assert np.allclose(rgb[0, 2], to_rgb(LAND_BELOW_SEA), atol=1e-3)  # below -500 m: the bottom colour


def test_land_below_sea_level_is_land() -> None:
    """The Dead Sea shore, Qattara, Death Valley: the land mask decides, not the sign of the height."""
    land = np.ones((1, 2), bool)
    rgb = colorize(np.array([[-400.0, -100.0]]), land, no((1, 2)), ocean=True)
    sea = colorize(np.array([[-400.0, -100.0]]), no((1, 2)), no((1, 2)), ocean=True)
    assert not np.allclose(rgb, sea, atol=0.05)
    assert (rgb[..., 1] > rgb[..., 2]).all()  # green, not blue


def test_ice_wins_over_sea_and_land() -> None:
    rgb = colorize(np.array([[-300.0, 3000.0]]), np.array([[False, True]]), np.ones((1, 2), bool), ocean=True)
    assert (rgb > 0.75).all()  # white to pale blue, never sea blue or brown


def test_flat_sea_unshaded_without_ocean() -> None:
    z = np.random.default_rng(0).normal(-3000, 500, (20, 20))
    rgb = colorize(z, no(z.shape), no(z.shape), False)
    assert np.array_equal(shade(rgb.copy(), z, 5000.0, np.ones(z.shape, bool), False), rgb)


@pytest.fixture(scope="module")
def pg() -> PosterGrid:
    return poster_grid(make_frame(), 50)


def px(pg: PosterGrid, lon: float, lat: float) -> tuple[int, int]:
    """Index of the poster pixel nearest (lon, lat)."""
    x, y = make_frame().to_map.transform(lon, lat)
    return int(np.abs(pg.y - y).argmin()), int(np.abs(pg.x - x).argmin())


def test_inside_mask(pg: PosterGrid) -> None:
    assert pg.inside[px(pg, 0, 0)] and pg.inside[px(pg, 0, 89)] and pg.inside[px(pg, 179.5, 0)]
    assert not pg.inside[0, 0] and not pg.inside[-1, -1] and not pg.inside[0, -1]


def test_ice_mask(pg: PosterGrid) -> None:
    ice = polygon_mask(make_frame().project(load_ice()), make_frame(), pg)
    assert ice[px(pg, -40, 75)] and ice[px(pg, 0, -85)]  # Greenland, Antarctica
    assert ice[px(pg, -170, -80)]  # Ross Ice Shelf: ice over sea
    assert not ice[px(pg, 10, 25)] and not ice[px(pg, -30, 0)]  # Sahara, Atlantic


def test_islands_in_ice_shelves_are_ice(pg: PosterGrid) -> None:
    """Natural Earth cuts the islands out of the ice shelves, but they are ice-covered too (Roosevelt, Berkner)."""
    ice = polygon_mask(make_frame().project(load_ice()), make_frame(), pg)
    assert ice[px(pg, -162.22, -79.28)] and ice[px(pg, -60.55, -80.31)]


def test_land_mask(pg: PosterGrid) -> None:
    land = polygon_mask(make_frame().project(load_land()), make_frame(), pg)
    assert land[px(pg, 27.5, 29.8)] and land[px(pg, -116.85, 36.25)] and land[px(pg, 5.5, 52.5)]  # Qattara, Death Valley, polders
    assert not land[px(pg, 51, 42)] and not land[px(pg, -30, 0)]  # Caspian, Atlantic


def test_polygon_mask_covers_exactly_its_pixels(pg: PosterGrid) -> None:
    """Pixel j's centre is at j + 0.5 pixel widths: a polygon over pixels 10..19 fills those, not 10..20."""
    import geopandas as gpd
    from shapely.geometry import box

    f = make_frame()
    pw, ph = f.width / len(pg.x), f.height / len(pg.y)
    poly = box(f.xmin + 10 * pw, f.ymax - 20 * ph, f.xmin + 20 * pw, f.ymax - 10 * ph)
    mask = polygon_mask(gpd.GeoDataFrame(geometry=[poly]), f, pg)
    rows, cols = np.nonzero(mask)
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (10, 19, 10, 19)


@pytest.fixture(scope="module")
def coast():
    """The coastline, as one geometry in map units."""
    from shapely.geometry import MultiLineString

    return MultiLineString(coastline(make_frame().project(load_land()), make_frame().project(load_ice())))


def test_coastline_follows_land_and_ice(coast) -> None:
    from shapely.geometry import Point

    f = make_frame()
    at = lambda lon, lat: Point(f.to_map.transform(lon, lat))
    assert coast.distance(at(27.5, 29.8)) > 100_000  # Qattara is inland
    assert coast.distance(at(-175, -81)) > 100_000  # mid Ross Ice Shelf: no grounding line drawn through the ice
    assert coast.distance(at(-175, -78.2)) < 60_000  # the Ross Ice Shelf front is coast
    assert coast.distance(at(12.25, 41.73)) < 2_000  # the Italian coast at Ostia
    assert coast.distance(at(51.0, 42.0)) < 150_000  # the Caspian, a hole in the land, keeps its shore


def test_coastline_drops_slivers(coast) -> None:
    rings = [shapely.Polygon(line) for line in shapely.get_parts(coast)]
    assert min(r.area for r in rings) > 2_000  # the smallest islet is ~2,300 m^2; slivers are under 1 m^2
    assert len(rings) < 7_000  # ~3,700 slivers between ice shelves and land are dropped


def test_relief_lights_north_west_slopes() -> None:
    z = np.zeros((5, 5))  # flat
    assert np.allclose(relief(z, 1000.0), 0, atol=1e-6)
    rising_south_east = np.add.outer(np.arange(5.0), np.arange(5.0)) * 300  # faces the north-west light
    assert (relief(rising_south_east, 1000.0) > 0).all() and (relief(-rising_south_east, 1000.0) < 0).all()


def peak_bytes(fn) -> int:
    import tracemalloc

    tracemalloc.start()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def test_colorize_memory_stays_near_its_output() -> None:
    """At 200 dpi the grid is 7200 x 3693: full float64 RGBA copies per ramp cost GBs (review finding)."""
    z = np.random.default_rng(0).normal(0, 3000, (1000, 1000)).astype(np.float32)
    ice, land = z > 4000, z > 0
    out_bytes = z.size * 3 * 4  # float32 RGB
    assert peak_bytes(lambda: colorize(z, land, ice, True)) < 2.5 * out_bytes


def test_sample_memory_stays_near_its_output(grid: Grid) -> None:
    lon = np.random.default_rng(1).uniform(-180, 180, (1000, 1000)).astype(np.float32)
    lat = np.random.default_rng(2).uniform(-90, 90, (1000, 1000)).astype(np.float32)
    assert peak_bytes(lambda: sample(grid, lon, lat)) < 8 * lon.nbytes


def test_shade_memory_stays_near_its_input() -> None:
    z = np.random.default_rng(0).normal(0, 3000, (1000, 1000)).astype(np.float32)
    rgb = colorize(z, z > 0, np.zeros(z.shape, bool), True)
    assert peak_bytes(lambda: shade(rgb, z, 5000.0, z <= 0, True)) < 2.5 * rgb.nbytes


def test_draw_topography_holds_little_memory() -> None:
    """matplotlib's ContourSet keeps a float64 copy of the grid alive per call (~665 MB each at 200 dpi)."""
    import tracemalloc

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from worldmap.render import new_figure

    frame = make_frame()
    elevation_grid(), load_ice(), load_lakes()  # warm the caches outside the measurement
    fig, ax = new_figure(frame)
    tracemalloc.start()
    draw_topography(ax, frame, 50, True)
    held = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    plt.close(fig)
    pixels = 1800 * 923
    assert held < 40 * pixels  # the RGBA image is 16 bytes a pixel; the rest is lines and patches
