# Topographic map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `make_map --type topographic [--ocean]` renders the globe as a hill-shaded height map with contours, ice on its own scale, lakes, a coastline and right-triangle colour bars in the canvas corners; the political map is unchanged.

**Architecture:** ETOPO 2022 (60") is downloaded once and reduced to a cached 3-arc-minute lon/lat grid. Each run samples it at every poster pixel inside the globe (inverse Robinson), colours and hill-shades it into one RGBA image, then draws contours, coastline, lakes and colour bars as vector layers on top. New modules: `topographic.py` (grid, colours, raster, contours) and `colorbars.py` (corner bar geometry and drawing); `cli.main` branches on `--type` right after the frame is made.

**Tech Stack:** Python 3.14, numpy, matplotlib (Agg), shapely 2, pyproj, Pillow, **rasterio** (new: reads the Deflate/floating-point-predictor tiled GeoTIFF with decimated reads).

**Spec:** `docs/superpowers/specs/2026-10-31-topographic-design.md`

## Global Constraints

- `--type {political,topographic}`, default `political`; `--ocean` ignored with `--type political`.
- With `--type topographic`: `--color`, `--label`, `--disputed`, `--recolor` are ignored; no borders, state lines, labels or stripes.
- Default output file stays `worldmap.png` (`worldmap_pins.png` with `--pins`) for both types.
- Political output must not change: the `default`, `earth`, `pins`, `regions`, `disputed` fingerprints stay as they are.
- ETOPO: `https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif`, fetched into `.cache/` only for `--type topographic`.
- Ice outlines: Natural Earth `physical/ne_10m_glaciated_areas.zip` + `physical/ne_10m_antarctic_ice_shelves_polys.zip`. Ice wins over sea.
- Lakes: `load_lakes()`, flat `OCEAN` with a coastline, on both plain and `--ocean` maps.
- Ocean without `--ocean`: flat `OCEAN` (`#e3f0f7`). With `--ocean`: `OCEAN` at sea level to `#2f5f8c` at 8,000 m, lightly shaded.
- Hill shading lit from azimuth 315°. Heights above a bar's top take the top colour; the tip label gets a `+`.
- Colour bars: right angle in the canvas corner, curved side following the globe at a fixed gap, haloed (white-outlined) tick labels in the gap, ticks every 1,000 m. Height bottom-right (sea level at the bottom edge), depth top-right only with `--ocean` (sea level at the top edge), ice bottom-left (mirrored). Titles `Height`, `Depth`, `Ice` near the tips, haloed.
- Commits follow the user's `X: desc` / `X; *: desc` / `+X` scheme with the `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` trailer. Never amend.

## Review Focus

1. **Land below sea level** (Dead Sea, Qattara, Caspian shores, Dutch polders, Death Valley): expected to read as land where it is land. Pinned only as far as the data allows (Task 1 test records the Netherlands/Dead Sea values); check crops of each and raise anything wrong with the user rather than inventing a fix.
2. **The antimeridian and the poles**: no seam, gap or smear at lon ±180 (Chukotka, Fiji) or along Robinson's flat pole lines. Task 1 tests `sample` wrap-around; Task 3 tests the inside mask at the poles and edges.
3. **Speed and memory at 200 dpi**: a warm-cache `--type topographic --ocean` run should stay under ~40 s and not hold float64 copies of the 7200 x 3693 grid. Task 5 prints the run time; Task 3 uses float32.
4. **Transparency outside the globe**: nothing but the colour bars and their labels is painted outside the outline. Task 5 test checks a corner pixel outside the bars is transparent.
5. **`--dpi 50`, `--pins`, `--pick` with `--type topographic`**: work as on the political map (layout identical, pins drawn above everything). Task 5 tests `--dpi 50 --pins` renders.

---

### Task 1: Elevation data and the cached grid

**Files:**
- Modify: `pyproject.toml` (add `"rasterio>=1.4"` to `dependencies`), then `.venv/bin/pip install -e '.[dev]'`
- Modify: `src/worldmap/data.py`
- Create: `src/worldmap/topographic.py`
- Test: `tests/test_topographic.py`

**Interfaces:**
- Produces (`data.py`): `ETOPO = "ETOPO_2022_v1_60s_N90W180_surface.tif"`, `GLACIERS = "ne_10m_glaciated_areas.zip"`, `ICE_SHELVES = "ne_10m_antarctic_ice_shelves_polys.zip"`; `fetch()` knows ETOPO's URL (`ETOPO_URL`) and that the two ice files are under `physical/`; `load_ice() -> gpd.GeoDataFrame` (both files concatenated, geometry only).
- Produces (`topographic.py`): `GRID_STEP_DEG = 0.05`; `ELEVATION_RULE = "etopo2022-60s-mean-3min/v1"`; `@dataclass Grid(z: np.ndarray /* int16 metres, shape (3600, 7200), row 0 = northmost */, step: float)`; `elevation_grid() -> Grid` (reads ETOPO with rasterio at shape (3600, 7200), `Resampling.average`, caches `.cache/etopo_3min.npy` plus a key file with ETOPO's size and `ELEVATION_RULE`; rebuilds when either changes); `sample(grid: Grid, lon: np.ndarray, lat: np.ndarray) -> np.ndarray` (float32, bilinear between cell centres, lon wraps across ±180, lat clamped).

- [ ] **Step 1: Write the failing tests** in `tests/test_topographic.py`:

```python
@pytest.fixture(scope="module")
def grid() -> Grid:
    return elevation_grid()

def test_grid_shape(grid): assert grid.z.shape == (3600, 7200) and grid.z.dtype == np.int16

@pytest.mark.parametrize("lon, lat, lo, hi", [
    (86.925, 27.988, 7000, 9000),     # Everest (3' means lower the peak)
    (142.2, 11.35, -11100, -9000),    # Mariana Trench
    (10.0, 25.0, 100, 2000),          # Sahara
    (51.0, 42.0, -900, -20),          # Caspian Sea floor: below 0, so water
    (35.5, 31.5, -800, -100),         # Dead Sea
    (-30.0, 0.0, -6000, -2000),       # mid-Atlantic
])
def test_known_places(grid, lon, lat, lo, hi):
    assert lo <= sample(grid, np.array([lon]), np.array([lat]))[0] <= hi

def test_sample_wraps_antimeridian(grid):
    a, b = sample(grid, np.array([179.99, -179.99]), np.array([65.0, 65.0]))  # Chukotka
    assert abs(a - b) < 150

def test_netherlands_value_recorded(grid):  # Review Focus 1: documents what the data says about the polders
    v = sample(grid, np.array([5.5]), np.array([52.5]))[0]  # Flevoland
    assert -10 < v < 10
```

- [ ] **Step 2:** `.venv/bin/pytest tests/test_topographic.py -q` — Expected: FAIL (`ImportError: cannot import name 'elevation_grid'`). The first real run downloads 466 MB.
- [ ] **Step 3:** Implement the `data.py` and `topographic.py` pieces above. The Mariana and Everest bounds are guesses for 3' means; if the data falls outside them, look at the value and ledger a ruling rather than bending the code.
- [ ] **Step 4:** `.venv/bin/pytest tests/test_topographic.py -q && .venv/bin/mypy src` — Expected: all pass, mypy clean. Second run is fast (cache hit; < 2 s for the fixture).
- [ ] **Step 5:** Commit `topographic; *: +elevation_grid` (pyproject, data.py, topographic.py, tests).

### Task 2: Colours and shading

**Files:** Modify `src/worldmap/topographic.py`; Test `tests/test_topographic.py`

**Interfaces:**
- Produces: `LAND_TOP, ICE_TOP, SEA_BOTTOM = 6000, 4000, 8000` (metres); `LAND_RAMP` (stops from mockup C: `(0,"#5f9e5a"), (0.08,"#a8c97f"), (0.2,"#e8dc95"), (0.4,"#c9a26b"), (0.65,"#9a6b4a"), (0.85,"#d8cfc8"), (1,"#ffffff")`), `ICE_RAMP` (`#ffffff` -> pale blue `#cfe3f2`), `SEA_RAMP` (`OCEAN` -> `#2f5f8c`), each a `LinearSegmentedColormap`; `colorize(z: np.ndarray, ice: np.ndarray, ocean: bool) -> np.ndarray` (float32 RGB, shape z.shape + (3,); land where `z > 0 & ~ice`, ice where `ice`, sea elsewhere; values clipped at each top); `shade(rgb, z, pixel_m: float, sea: np.ndarray, ocean: bool) -> np.ndarray` (`LightSource(azdeg=315, altdeg=45)` hillshade on `z`; land and ice `rgb * (0.65 + 0.45 * hs)`, sea `rgb * (0.85 + 0.2 * hs)` with `--ocean` and untouched without; result clipped to [0, 1]). `VERT_EXAG` is a module constant, starting at the mockups' 0.05-per-20-km look, tuned on crops in Task 3.

- [ ] **Step 1: Write the failing tests:**

```python
def test_sea_is_flat_ocean_without_ocean_flag():
    rgb = colorize(np.array([[-5000.0, -1.0]]), np.zeros((1, 2), bool), ocean=False)
    assert np.allclose(rgb, to_rgb(OCEAN), atol=1e-3)

def test_sea_darkens_with_depth():
    rgb = colorize(np.array([[-1.0, -4000.0, -20000.0]]), np.zeros((1, 3), bool), ocean=True)
    assert np.allclose(rgb[0, 0], to_rgb(OCEAN), atol=0.01) and rgb[0, 1].sum() > rgb[0, 2].sum()
    assert np.allclose(rgb[0, 2], to_rgb("#2f5f8c"), atol=1e-3)  # below 8,000 m: the bottom colour

def test_land_ramp_ends():
    rgb = colorize(np.array([[1.0, 9000.0]]), np.zeros((1, 2), bool), ocean=False)
    assert np.allclose(rgb[0, 0], to_rgb("#5f9e5a"), atol=0.01) and np.allclose(rgb[0, 1], 1.0, atol=1e-3)

def test_ice_wins_over_sea_and_land():
    rgb = colorize(np.array([[-300.0, 3000.0]]), np.ones((1, 2), bool), ocean=True)
    assert (rgb > 0.75).all()  # white to pale blue, never sea blue or brown

def test_flat_sea_unshaded_without_ocean():
    z = np.random.default_rng(0).normal(-3000, 500, (20, 20))
    rgb = colorize(z, np.zeros(z.shape, bool), False)
    assert np.array_equal(shade(rgb, z, 5000.0, z <= 0, False), rgb)
```

- [ ] **Step 2:** Run them — Expected: FAIL (`colorize` not defined).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `.venv/bin/pytest tests/test_topographic.py -q` — Expected: PASS.
- [ ] **Step 5:** Commit `topographic: +colorize, +shade`.

### Task 3: The poster raster, contours, coastline, ice and lakes

**Files:** Modify `src/worldmap/topographic.py`, `src/worldmap/data.py` (nothing else); Test `tests/test_topographic.py`

**Interfaces:**
- Consumes: Task 1 `Grid`, `elevation_grid`, `sample`, `load_ice`; Task 2 `colorize`, `shade`; `frame.MapFrame`, `render.holed_patches`, `render.Z_*`, `data.load_lakes`, `data.project`.
- Produces: `@dataclass PosterGrid(x: np.ndarray, y: np.ndarray /* 1-D pixel-centre map coords, y descending */, lon, lat: np.ndarray /* float32 2-D */, inside: np.ndarray /* bool 2-D */)`; `poster_grid(frame: MapFrame, dpi: float) -> PosterGrid` (one cell per output pixel at `min(dpi, DPI)`; `inside` = the inverse-projected point is finite, within ±180/±90, and re-projects to itself within one pixel); `ice_mask(ice_projected: gpd.GeoDataFrame, frame: MapFrame, pg: PosterGrid) -> np.ndarray` (rasterised with Pillow `ImageDraw.polygon`, holes cleared, as `earth.py` does); `draw_topography(ax: Axes, frame: MapFrame, dpi: float, ocean: bool) -> None` (everything on the globe: RGBA image clipped to the outline at `Z_COUNTRY`, alpha 0 outside `inside`; contours; coastline; lakes via `holed_patches` in `OCEAN` with a `BORDER` edge at `Z_LAKES`).
- Contours (starting values, tuned on full-res crops; ledger any change): land every 1,000 m, `#5a4a3a`, 0.3 pt, alpha 0.6, on `z` masked to land-not-ice; ice every 1,000 m in a grey-blue `#7a8fa3`, same width, masked to ice; with `--ocean`, every 2,000 m, `#1f4466`, 0.3 pt, alpha 0.5, masked to sea-not-ice. Coastline: contour at 0 of `where(ice, max(z, 1), z)`, `BORDER`, `BORDER_WIDTH`. All contour inputs NaN outside `inside`.

- [ ] **Step 1: Write the failing tests:**

```python
@pytest.fixture(scope="module")
def pg() -> PosterGrid: return poster_grid(make_frame(), 50)

def px(pg, lon, lat):  # index of the pixel nearest (lon, lat)
    x, y = make_frame().to_map.transform(lon, lat)
    return np.abs(pg.y - y).argmin(), np.abs(pg.x - x).argmin()

def test_inside_mask(pg):
    assert pg.inside[px(pg, 0, 0)] and pg.inside[px(pg, 0, 89.9)] and pg.inside[px(pg, 179.5, 0)]
    assert not pg.inside[0, 0] and not pg.inside[-1, -1] and not pg.inside[0, -1]

def test_ice_mask(pg):
    ice = ice_mask(project(load_ice()), make_frame(), pg)
    assert ice[px(pg, -40, 75)] and ice[px(pg, 0, -85)]          # Greenland, Antarctica
    assert ice[px(pg, -170, -80)]                                # Ross Ice Shelf: ice over sea
    assert not ice[px(pg, 10, 25)] and not ice[px(pg, -30, 0)]   # Sahara, Atlantic
```

- [ ] **Step 2:** Run them — Expected: FAIL (`poster_grid` not defined).
- [ ] **Step 3:** Implement `poster_grid`, `ice_mask`, `draw_topography`.
- [ ] **Step 4:** `.venv/bin/pytest tests/test_topographic.py -q && .venv/bin/mypy src` — Expected: PASS, clean.
- [ ] **Step 5: Visual check.** A scratchpad script draws `new_figure(frame)` + `draw_topography(ax, frame, 200, True)` and saves; crop at full resolution: Alps, Himalaya, Greenland edge, Ross Ice Shelf, Netherlands, Dead Sea/Qattara, Great Lakes, Lake Victoria, Chukotka/antimeridian, both poles. Tune `VERT_EXAG`, contour spacing and widths until it matches mockups C/E at poster scale; ledger every tuned value. Record the run time.
- [ ] **Step 6:** Commit `topographic: +draw_topography` (body: the tuned values).

### Task 4: Corner colour bars

**Files:** Create `src/worldmap/colorbars.py`; Test `tests/test_colorbars.py`

**Interfaces:**
- Consumes: `MapFrame`; Task 2 ramps and tops.
- Produces: `BAR_GAP_IN = 0.7`, `BAR_MARGIN_IN = 0.2` (poster inches; starting values, tuned on the full render); `Corner = Literal["br", "tr", "bl"]`; `bar_region(frame: MapFrame, corner: Corner) -> Polygon` (the canvas corner's quadrant minus the globe outline buffered by the gap, minus a margin from the canvas edges; the largest part); `draw_colorbar(ax: Axes, frame: MapFrame, corner: Corner, cmap: Colormap, top: float, title: str) -> None` (gradient image clipped to the region: value 0 at the canvas edge — bottom for `br`/`bl`, top for `tr` — rising to `top` at 97% of the way to the tip; outline in `BORDER`; a tick every 1,000 m on the curved side with a label on the globe side: `sea level`, `1,000 m`, …, the last one `f"{top:,} m+"`; title at the tip; every text has `patheffects.withStroke(linewidth=2.5, foreground="white")`).
- Labels for `br`/`tr` are right-aligned left of the bar, for `bl` left-aligned right of it.

- [ ] **Step 1: Write the failing tests:**

```python
@pytest.mark.parametrize("corner", ["br", "tr", "bl"])
def test_bar_region_clears_globe_and_canvas(corner):
    f = make_frame(); r = bar_region(f, corner)
    gap = BAR_GAP_IN * DPI / f.px_per_unit
    assert r.area > 0 and r.distance(Polygon(f.outline)) >= gap * 0.99
    assert box(f.xmin, f.ymin, f.xmax, f.ymax).contains(r)

def test_bar_region_corners():
    f = make_frame()
    assert bar_region(f, "br").centroid.x > 0 and bar_region(f, "br").centroid.y < 0
    assert bar_region(f, "tr").centroid.y > 0 and bar_region(f, "bl").centroid.x < 0

def test_labels_have_halo_and_plus():
    fig, ax = plt.subplots(); f = make_frame()
    draw_colorbar(ax, f, "br", LAND_RAMP, 6000, "Height")
    texts = [t.get_text() for t in ax.texts]
    assert texts[0] == "sea level" and "6,000 m+" in texts and "Height" in texts
    assert all(t.get_path_effects() for t in ax.texts)
```

- [ ] **Step 2:** Run — Expected: FAIL (module missing).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `.venv/bin/pytest tests/test_colorbars.py -q && .venv/bin/mypy src` — PASS, clean.
- [ ] **Step 5:** Commit `+colorbars`.

### Task 5: `--type` / `--ocean`, wiring, fingerprints, docs

**Files:** Modify `src/worldmap/cli.py`, `tests/test_cli.py`, `tests/test_regression.py`, `tests/conftest.py`, `tests/fingerprints.json`, `README.md`, `CLAUDE.md`; Create `examples/topographic.png`

**Interfaces:**
- Consumes: Task 3 `draw_topography`; Task 4 `draw_colorbar`.
- Produces: `parse_args` gains `--type` (`choices=["political", "topographic"]`, default `political`) and `--ocean` (`store_true`); `cli.main` after `make_frame()` and the `--pick` early return: `if args.type == "topographic": draw_topographic(...)` with its own figure (`new_figure`), `draw_topography`, bars (height `br` always; depth `tr` only with `--ocean`; ice `bl` always), pins as today, `savefig`. The political path is not touched apart from being skipped, and the political-only loads (countries colouring, disputes, states) do not run in topographic mode. Print the run's wall time.
- `conftest.environment(mode)`: for modes starting with `topographic`, add `etopo`, `glaciers`, `ice_shelves` file sizes and `rasterio`/GDAL versions.

- [ ] **Step 1: Write the failing tests:**

```python
# test_cli.py
def test_type_default_political(): assert parse_args([]).type == "political" and not parse_args([]).ocean
def test_type_topographic(): a = parse_args(["--type", "topographic", "--ocean"]); assert (a.type, a.ocean) == ("topographic", True)
def test_type_unknown_errors(capsys):
    with pytest.raises(SystemExit): parse_args(["--type", "sky"])

# test_regression.py
def test_render_topographic(tmp_path): check_fingerprint("topographic", render(tmp_path, "--type", "topographic"))
def test_render_topographic_ocean(tmp_path): check_fingerprint("topographic_ocean", render(tmp_path, "--type", "topographic", "--ocean"))
def test_topographic_outside_is_transparent(tmp_path):
    im = np.asarray(Image.open(render(tmp_path, "--type", "topographic", "--dpi", "50", "--pins", str(TESTS / "pins.txt"))))
    h, w = im.shape[:2]
    assert im[h // 2, 2, 3] == 0 and im[2, w // 10, 3] == 0   # left edge at the equator; top-left corner (no bar there)
```

- [ ] **Step 2:** `.venv/bin/pytest tests/test_cli.py tests/test_regression.py -q` — Expected: the new CLI tests fail (no `type`), the topographic renders fail (unrecognized arguments).
- [ ] **Step 3:** Implement the CLI and `conftest` changes.
- [ ] **Step 4:** `UPDATE_REFERENCE=1 .venv/bin/pytest tests/test_regression.py -q -k topographic`, then `.venv/bin/pytest -q` — Expected: all pass; the five political fingerprints unchanged (`git diff tests/fingerprints.json` shows only the two new entries). Check the run time printed (Review Focus 3).
- [ ] **Step 5:** Commit `cli; *: --type topographic, --ocean` (cli, tests, conftest, fingerprints).
- [ ] **Step 6:** Render `examples/topographic.png` with `--type topographic --ocean --dpi 50`; update `README.md` (flags, data download size) and `CLAUDE.md` (setup line, module table rows for `topographic.py` and `colorbars.py`, a `--type topographic` section: data, cache, drawing order, colour bars, known risk). Commit `CLAUDE; *: --type topographic` (README, CLAUDE, example).
