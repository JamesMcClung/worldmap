# Readability Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the single 1,070-line `make_map.py` into a typed, tested package of focused modules. The rendered output stays pixel-identical, and `--ocean` is removed.

**Architecture:** First, a pixel-fingerprint regression test runs the CLI in a subprocess. Next, `--ocean` is deleted. Then pieces move out of `make_map.py` one module at a time. Each move keeps the program running and the fingerprints green, until `make_map.py` becomes `cli.py`. Naming and magic-number cleanup happens last, module by module.

**Tech Stack:** Python ≥3.10, geopandas, shapely 2, matplotlib, numpy, pyproj, Pillow; pytest and mypy as dev extras.

**Spec:** `docs/superpowers/specs/2026-10-27-readability-cleanup-design.md`

## Global Constraints

- **Pixel stability.** After Task 1, every commit keeps `pytest` green. A pure move, rename or typing commit must leave the fingerprints unchanged.
- **Deliberate output changes.** If a simplification changes the output, re-run with `UPDATE_REFERENCE=1` and commit `tests/fingerprints.json` in the same commit. The message body says what changed visually and why. In this plan, no task is expected to change the output.
- **Commit format.** Follow the owner's global CLAUDE.md:
  - Message forms: `X: desc`, `X; *: desc`, `+X`, `-X` or `->X`, where `X` is the filename without its extension.
  - Use only `git add` and `git commit`. No amend, squash, branch switching, push, `git mv` or `git rm`. Delete files with `rm`, then `git add` the path.
  - Every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Import side effects.** Importing any `worldmap` module has no side effects. Only `cli.main` may import `matplotlib.pyplot`, and only after deciding the backend, because `--pick` needs the interactive backend.
- **CLI.** `make_map [out] --recolor --earth --pins FILE --pin-size INCHES --pick FILE`. Only `--ocean` is removed. The `.cache/` layout is unchanged.
- **Data.** Tests use the real Natural Earth data in `.cache/`, with no stand-in data. The rng seed stays fixed at 7 as a module constant.
- **Typing.** Annotate every function. The code should pass `mypy src` in its default mode (no `--strict`), but no hook enforces it.
- **Style.** No class hierarchies, registries or config frameworks. Use plain functions and small dataclasses.

## Review Focus

- **`--pick` backend.** A module that imports `matplotlib.pyplot` at import time silently forces a backend, and the picker window then never opens. Covered by `test_import_has_no_pyplot` (Task 1, re-run every task).
- **Console script.** After `make_map.py` becomes `cli.py`, the `make_map` command must still work once `pip install -e .` is re-run. Covered by `test_console_script_help` (Task 9).
- **Removed flag.** Passing `--ocean` must give a clean argparse error, not be silently accepted. Covered by `test_ocean_flag_rejected` (Task 2).
- **Different library versions.** On another machine (different matplotlib or freetype), the regression test must skip with a message naming the version that differs, not fail. Covered by `test_version_mismatch_skips` (Task 1).
- **Missing fingerprint.** A missing fingerprint must fail the test, never quietly record whatever the code draws now. Covered by `test_missing_fingerprint_fails` (Task 1).

---

### Task 1: Regression test harness

**Files:**
- Create: `src/worldmap/__main__.py`, `tests/conftest.py`, `tests/test_regression.py`, `tests/pins.txt`, `tests/fingerprints.json`, `tests/test_imports.py`
- Modify: `pyproject.toml`, `.gitignore`

**Interfaces:**
- Produces:
  - `python -m worldmap ARGS…`: a stable entry point the tests run in a subprocess. It survives the later `make_map` → `cli` rename by changing one import line.
  - `tests/conftest.py`:
    - `render(tmp_path: Path, *args: str) -> Path`: runs `[sys.executable, "-m", "worldmap", "out.png", *args]` with `cwd=tmp_path`, `check=True`, and returns `tmp_path / "out.png"`.
    - `fingerprint(png: Path) -> str`: SHA-256 of `f"{arr.shape}|{arr.dtype}".encode() + arr.tobytes()`, where `arr = np.asarray(Image.open(png))`. Only decoded pixels are hashed, so PNG metadata doesn't matter.
    - `environment() -> dict[str, str]`: the versions of `python` (`platform.python_version()`), `matplotlib`, `freetype` (`matplotlib.ft2font.__freetype_version__`), `numpy`, `shapely`, `geos` (`shapely.geos_version_string`), `geopandas`, `pyproj`, `pillow`.
    - `check_fingerprint(mode: str, png: Path, store: Path = FINGERPRINTS) -> None`: does the compare, update and skip logic.
  - `tests/fingerprints.json`: `{mode: {"sha256": str, "env": {…}}}`, with modes `"default"`, `"earth"` and `"pins"`.

- [ ] **Step 1: Packaging and ignore rules.**
  - `pyproject.toml`:
    - add `[project.optional-dependencies] dev = ["pytest>=8", "mypy>=1.10"]`
    - add `[tool.pytest.ini_options] testpaths = ["tests"]`
  - `.gitignore`: add `!tests/*.json` and `!tests/*.txt`.
  - Run `.venv/bin/pip install -e '.[dev]'`. Expected: it installs pytest and mypy.
  - Commit `pyproject: +dev extras, pytest config`, then commit `gitignore: allow tests/*.json, tests/*.txt`.
- [ ] **Step 2: `src/worldmap/__main__.py`.** It contains `from worldmap.make_map import main` and `main()`. `make_map.main` already parses `sys.argv` through `parse_args()`. Check that `cd "$(mktemp -d)" && /path/.venv/bin/python -m worldmap -h` lists the flags. Commit `+__main__`.
- [ ] **Step 3: Write the harness tests (no fingerprints yet).**
  - `test_regression.py`:
    - `test_render_default`: `check_fingerprint("default", render(tmp_path))`
    - `test_render_earth`: `check_fingerprint("earth", render(tmp_path, "--earth"))`
    - `test_render_pins`: `check_fingerprint("pins", render(tmp_path, "--pins", str(TESTS / "pins.txt")))`
    - `test_missing_fingerprint_fails(tmp_path)`: writes a 2×2 PNG, calls `check_fingerprint("nope", png, store=tmp_path/"fp.json")` with `"{}"` in the store, and expects `pytest.fail` (catch it as `pytest.fail.Exception`). The message must contain `UPDATE_REFERENCE=1`.
    - `test_version_mismatch_skips(tmp_path)`: a store whose entry has the right hash but `env["matplotlib"] = "0.0"`. It expects `pytest.skip.Exception`, with a message containing `matplotlib`.
  - `tests/pins.txt`: five pins, including one near the antimeridian (`179.5,-16.5`), plus a `#` comment line.
- [ ] **Step 4: Run the tests.** `.venv/bin/pytest -v`. Expected: the two unit tests pass, and the three render tests fail with "no fingerprint for 'default' … run with UPDATE_REFERENCE=1".
- [ ] **Step 5: Implement `check_fingerprint`.** It does three things in order:
  - If `UPDATE_REFERENCE=1` is set, it writes the hash and env, sorted and indented, then copies the PNG to `.cache/reference/<mode>.png`.
  - Otherwise, if the entry is missing, it calls `pytest.fail` with the message above.
  - Otherwise, if the env differs, it calls `pytest.skip`, listing each differing key as recorded → current. If the env matches and the hash differs, it calls `pytest.fail`, naming the saved reference PNG path and the new PNG path so the two can be compared by eye.
- [ ] **Step 6: Record the fingerprints and confirm they're stable.**
  - Run `UPDATE_REFERENCE=1 .venv/bin/pytest -v`, then `.venv/bin/pytest -v` twice.
  - Expected: all tests pass on both plain runs. If a fingerprint differs between runs, stop. The render isn't deterministic, so find and fix the cause first, with a separate commit for that fix.
- [ ] **Step 7: Import side-effect test.** `test_imports.py::test_import_has_no_pyplot` runs a subprocess that imports every `worldmap` submodule (via `pkgutil.walk_packages`) and asserts that `"matplotlib.pyplot" not in sys.modules`. Expected: PASS.
- [ ] **Step 8: Commit.** First `+conftest`. Then `+test_regression`, including `pins.txt` and `fingerprints.json` (they are only used by this test). Then `+test_imports`.

### Task 2: Remove `--ocean`

**Files:**
- Modify: `src/worldmap/make_map.py`, `README.md:36`, `CLAUDE.md:10,91-92`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `make_map.main(argv: list[str] | None = None) -> None`. `argv` is passed through to `parse_args`, so tests can call `main` in-process for argument errors.

- [ ] **Step 1: Write a failing test.** `test_cli.py::test_ocean_flag_rejected`: `with pytest.raises(SystemExit) as e: main(["--ocean"])`, then `assert e.value.code == 2`.
- [ ] **Step 2: Run it.** Expected: FAIL, because `main` takes no argv.
- [ ] **Step 3: Remove the flag and every `OCEAN_AWARE` branch.** Keep only the code the default path runs:
  - **Delete:** `fits`, `max_growth`, `GROW_STEP`, `GROW_MAX`, `local_land`, `LAND_PX`, the `land_local` loop, the shrink-before-out loop and its restore, and the growth loop.
  - **Collapse to the non-ocean side:**
    - the `area` choice in the label pieces (`big.area`)
    - the wrap threshold (`0.9`)
    - the cost branch that checks the label stays inside its country
    - the candidate-grid `linspace`s
  - **Wrap with `two_line`:** replace the inline word-splitting wrap with a call to `two_line()`.
  - **`main(argv)`:** add the `argv` parameter.
- [ ] **Step 4: Run the tests.** `.venv/bin/pytest -v`. Expected: all pass, with the fingerprints unchanged.
- [ ] **Step 5: Commit.** First `make_map: -ocean mode`. Then `test_cli: +test_ocean_flag_rejected`; `test_cli.py` is a new file, so the message is `+test_cli`. Then `README: -ocean` and `CLAUDE: -ocean`.

### Task 3: `colorspace.py`

**Files:**
- Create: `src/worldmap/colorspace.py`, `tests/test_colorspace.py`
- Modify: `src/worldmap/make_map.py`

**Interfaces:**
- Produces:
  - `RGB = tuple[float, float, float]`
  - `srgb_to_lin(c: ArrayLike) -> np.ndarray` and `lin_to_srgb(c: ArrayLike) -> np.ndarray`
  - `oklab_to_linear_rgb(L: float, a: float, b: float) -> np.ndarray` (renamed from `lab_to_lin`)
  - `oklch_to_srgb(L: float, C: float, h: float) -> RGB | None` (renamed from `lch_to_rgb`)
  - `srgb_to_oklch(rgb: ArrayLike) -> tuple[float, float, float]` (renamed from `srgb_to_lch`)
  - `fit_gamut(L: float, C: float, h: float, steps: int = 40) -> tuple[RGB, float] | None`: the first in-gamut colour over `np.linspace(C, 0, steps)`, together with the chroma it used.
  - `darken(rgb: RGB, L: float = 0.26) -> RGB`

- [ ] **Step 1: Tests.**
  - `test_round_trip`: for `(0.2, 0.5, 0.8)`, `oklch_to_srgb(*srgb_to_oklch(rgb))` is approximately `rgb` (`abs=1e-6`).
  - `test_out_of_gamut_is_none`: `oklch_to_srgb(0.5, 0.5, 30)` is `None`.
  - `test_fit_gamut_reduces_chroma`: `rgb, c = fit_gamut(0.5, 0.5, 30)` gives `c < 0.5` and `rgb` within `[0, 1]`.
  - `test_darken_keeps_hue`: the hue of `darken((0.9, 0.6, 0.3))` is within 2° of the input's hue, and its L is approximately 0.26.
- [ ] **Step 2: Run them.** Expected: FAIL with `ModuleNotFoundError`.
- [ ] **Step 3: Move the colour helpers into `colorspace.py`.** Rename them and add `fit_gamut`. Replace the three chroma-reduction loops with `fit_gamut`: `make_palette` (which uses the returned chroma for `labs`), `earth_fill` and `darken`. `darken` keeps its `(0.1, 0.1, 0.1)` fallback. The colour helpers live where they're used until later tasks move those users.
- [ ] **Step 4: Run the tests.** `.venv/bin/pytest -v`. Expected: all pass, with the fingerprints unchanged.
- [ ] **Step 5: Commit.** First `+colorspace` (moving the helpers counts as the secondary change), then `+test_colorspace`.

### Task 4: `data.py`

**Files:**
- Create: `src/worldmap/data.py`
- Modify: `src/worldmap/make_map.py`

**Interfaces:**
- Produces:
  - Constants: `CACHE: Path` (still `<repo>/.cache`; same depth as now), `NE_URL`, `COUNTRIES`, `STATES`, `LAKES`, `RASTER`, `LAKE_MIN_KM2`, `DRY_LAKES`, `STATE_COUNTRIES`, `CRS`, `NOT_LABELED`, `LABELED_EXTRA`.
  - `fetch(name: str) -> Path`
  - `is_labeled(row: pd.Series) -> bool` (renamed from `labeled`)
  - `load_countries() -> gpd.GeoDataFrame`, which keeps the index reset
  - `load_states() -> gpd.GeoDataFrame`, which is already filtered to `STATE_COUNTRIES`
  - `load_lakes() -> gpd.GeoDataFrame`, which already applies the size and dry-lake filters
  - `project(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame` (renamed from `prep`)

- [ ] **Step 1: Move the data code and add the three loaders.** `main` calls the loaders.
- [ ] **Step 2: Run the tests.** `.venv/bin/pytest -v`. Expected: all pass, with the fingerprints unchanged.
- [ ] **Step 3: Commit.** `+data`

### Task 5: `coloring.py`

**Files:**
- Create: `src/worldmap/coloring.py`, `tests/test_coloring.py`
- Modify: `src/worldmap/make_map.py`

**Interfaces:**
- Consumes: `fetch`, `CACHE`, `COUNTRIES` from `data`; `srgb_to_oklch`, `fit_gamut`, `RGB` from `colorspace`.
- Produces:
  - `Edge = tuple[int, int]`
  - `adjacency(gdf) -> set[Edge]`
  - `add_proximity_edges(gdf, edges: set[Edge], major: list[bool], k: int, max_gap: float, min_deg: int) -> set[Edge]`
  - `color_graph(n: int, edges: Iterable[Edge], weights: Sequence[float], kmin: int = 1, kmax: int = 8, limit: int = 2_000_000) -> tuple[int, list[int]]`
  - `make_palette(k: int, class_pairs: set[Edge]) -> list[RGB]`
  - `@dataclass Coloring: k: int; color_class: list[int]; class_pairs: set[Edge]`
  - `load_or_compute_coloring(countries, major: list[bool], recolor: bool) -> Coloring`: includes the `coloring.json` cache, with the same key, including `rule`
  - `spot_check(countries, color_class: Sequence[int]) -> list[tuple[str, str]]`: returns the same-coloured pairs; `main` prints them as now. Baikonur is assigned before the check, as now.

- [ ] **Step 1: Tests.**
  - `test_color_graph_is_proper`: on a 5-cycle plus a chord `(0, 2)`, the returned `k == 3` and no edge joins two nodes of the same class.
  - `test_color_graph_respects_kmin`: with `kmin=5` on a single edge, `k == 5`.
  - `test_color_graph_unsolvable_raises`: K4 with `kmax=3` raises `RuntimeError`.
- [ ] **Step 2: Run them.** Expected: FAIL (import error).
- [ ] **Step 3: Move the colouring code.** Hoist `N_COLORS`, `MAX_GAP` and `MIN_DEG` into the module. Move `import time` and `STRtree` to the top.
- [ ] **Step 4: Run the tests.** `.venv/bin/pytest -v`. Expected: all pass, with the fingerprints unchanged. Also run once with `--recolor` (`.venv/bin/python -m worldmap /tmp/x.png --recolor` in the scratchpad), then confirm `pytest` still passes. That shows the recomputed colouring matches the cache.
- [ ] **Step 5: Commit.** First `+coloring`, then `+test_coloring`.

### Task 6: `frame.py` and `earth.py`

**Files:**
- Create: `src/worldmap/frame.py`, `src/worldmap/earth.py`
- Modify: `src/worldmap/make_map.py`

**Interfaces:**
- Produces in `frame.py`:
  - `POSTER_WIDTH_IN = 36` and `DPI = 200`
  - `@dataclass(frozen=True) MapFrame`, with these fields:
    - `to_map: Transformer` and `to_lonlat: Transformer`
    - `outline: list[tuple[float, float]]`
    - `xmin, ymin, xmax, ymax: float`
  - On `MapFrame`, these members:
    - `width`, `height` and `px_per_unit` (a property, `POSTER_WIDTH_IN * DPI / width`)
    - `to_px(geom, simplify: float = 2.0)`
    - `px_to_map(px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
  - `make_frame() -> MapFrame`: builds the outline and the 0.6% pad exactly as now.
- Produces in `earth.py`:
  - `earth_values(countries) -> list[list[list[float]]]`
  - `earth_fill(green_minus_red: float, ice_fraction: float) -> RGB`
  - `Regions = dict[int, list[tuple[Polygon, RGB | str]]]`
  - `build_regions(countries_projected, countries) -> Regions`
  - `fill_at(regions: Regions, i: int, pt: Point | None) -> RGB | str`

- [ ] **Step 1: Move the code.** `main` builds a `MapFrame` and uses it everywhere it used `t`, `edge`, `minx`/`maxx`/`miny`/`maxy`, `PPU` and `TO_PX`. Keep the conversion to pixels as the same `affine_transform` call.
- [ ] **Step 2: Run the tests.** `.venv/bin/pytest -v`. Expected: all pass, with the fingerprints unchanged.
- [ ] **Step 3: Commit.** First `+frame`, then `+earth`, as separate commits. `frame` goes first because `earth` doesn't depend on it.

### Task 7: `pins.py`

**Files:**
- Create: `src/worldmap/pins.py`, `tests/test_pins.py`
- Modify: `src/worldmap/make_map.py`

**Interfaces:**
- Consumes: `MapFrame`, `DPI` and `POSTER_WIDTH_IN` from `frame`.
- Produces:
  - `LonLat = tuple[float, float]`
  - `load_pins(path: Path) -> list[LonLat]`
  - `@dataclass(frozen=True) PinStyle`: one field per shading constant now inline in `pin_sprite`, with the current values as defaults. The fields cover the canvas pad, key-light direction, ground/sky levels and edges, dark band, window strips, colour ramp stops and colours, rim darkening, specular glare and core, and shadow offset, radius and strength. Names describe the effect, e.g. `ground_level`, `sky_level`, `strip_width`, `specular_core_power`.
  - `pin_sprite(diameter_px: float, style: PinStyle = PinStyle()) -> np.ndarray`, returning an RGBA float array
  - `draw_pins(ax: Axes, frame: MapFrame, pins: Sequence[LonLat], sprite: np.ndarray, diameter_in: float, pad: float) -> list[AxesImage]`
  - `pick_pins(path: Path, rendered_png: Path, frame: MapFrame, diameter_in: float) -> None`, which imports `matplotlib.pyplot` inside the function

- [ ] **Step 1: Tests.**
  - `test_load_pins_parses_comments_and_spaces`: the file `"# hi\n10,20\n 30 40 # x\n\n"` loads as `[(10.0, 20.0), (30.0, 40.0)]`.
  - `test_load_pins_rejects_garbage`: the line `"abc\n"` raises `SystemExit`, with `":1:"` in the message.
  - `test_load_pins_rejects_out_of_range`: `"200,0\n"` raises `SystemExit`.
  - `test_pin_sprite_shape_and_alpha`: `pin_sprite(48)` has shape `(n, n, 4)`, where `n == round(2 * style.pad * 24)`. All values are in `[0, 1]`, the centre alpha is 1, and the corner alpha is 0.
  - `test_pick_without_render_exits`: in an empty `tmp_path`, `main(["--pick", "p.txt"])` (with `monkeypatch.chdir(tmp_path)`) raises `SystemExit`, with `"run make_map first"` in the message.
- [ ] **Step 2: Run them.** Expected: FAIL (import error).
- [ ] **Step 3: Move the pin code.** Replace `pin_sprite`'s literals with `style.<field>`, with the values unchanged. Rename the internals descriptively: `normal`, `reflect_x`, `reflect_y`, `smoothstep`, `sphere_alpha`, `shadow_alpha`. `main` checks for the rendered PNG before calling `pick_pins`.
- [ ] **Step 4: Run the tests.** `.venv/bin/pytest -v`. Expected: all pass, with the `pins` fingerprint unchanged.
- [ ] **Step 5: Commit.** First `+pins`, then `+test_pins`.

### Task 8: `labels/` package

**Files:**
- Create:
  - `src/worldmap/labels/__init__.py`, which is empty
  - `src/worldmap/labels/geometry.py`
  - `src/worldmap/labels/label.py`
  - `src/worldmap/labels/placement.py`
  - `tests/test_label_geometry.py`
- Modify: `src/worldmap/make_map.py`

**Interfaces:**
- Consumes: `MapFrame`, `darken`, `fill_at`, `is_labeled`.
- Produces in `labels/geometry.py`:
  - `rect(x, y, w, h) -> Polygon`
  - `nearest_point_on_rect(px, py, x, y, w, h) -> tuple[np.ndarray, np.ndarray]` (renamed from `nearest_pt`)
  - `segment_hits_rects(p0: np.ndarray, p1: np.ndarray, rects: np.ndarray) -> np.ndarray`
  - `segments_cross(a0, a1, b0, b1) -> np.ndarray`
  - `two_line(text: str) -> str | None`
- Produces in `labels/label.py`:
  - `@dataclass Label`, with fields:
    - `name: str`, `area: float`, `color: RGB`
    - `text: Text` (the matplotlib artist)
    - `width, height: float` (px)
    - `anchor_x, anchor_y: float` (px)
    - `geom: Polygon | MultiPolygon` (px)
    - `home` (px)
    - `country: BaseGeometry` (px)
    - `out: bool = False`
  - `MIN_PT = 6` and `MAX_PT = 30`
  - `make_labels(countries_projected, frame: MapFrame, ax: Axes, renderer, fill_of: Callable[[int, Point | None], RGB | str]) -> list[Label]`: covers anchors, size, the French Guiana and Kaliningrad extras, the island rule and `FORCE_ISLAND`; sorted largest first
  - `measure(label: Label, renderer) -> None`: the 1.06 / 1.15 padding
  - `decide_out_and_wrap(labels: list[Label], renderer) -> None`
- Produces in `labels/placement.py`:
  - `@dataclass(frozen=True) CostWeights`: one field per number in `cost_batch` and `candidates`, defaulting to today's values. Names include `overlap`, `dot_cover`, `leader_hits_label`, `leader_length`, `leader_too_short`, `leader_crosses_leader`, `outside_country`, `drift`, `outside_globe`, `leader_target_ratio` (1.3), and the step and ring scales.
  - `@dataclass Placement`: holds `x, y, width, height, anchor_x, anchor_y: np.ndarray` and `out: np.ndarray[bool]`; the old `X/Y/W_/H_/AX/AY/OUT_`.
  - `place_labels(labels: list[Label], frame: MapFrame, seed: int = SEED, weights: CostWeights = CostWeights()) -> Placement`: covers seeding the out labels, `optimize(40)`, the 12 shrink-and-retry rounds and the unresolved-overlap print, with `SEED = 7`.
  - `leader_ends(p: Placement) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Geometry tests.**
  - `test_segment_hits_rect`: a horizontal segment `(0,0)→(10,0)` hits `[4,-1,6,1]` and misses `[4,2,6,3]`.
  - `test_segment_parallel_inside_edge`: the vertical segment `(5,-5)→(5,5)` hits `[4,-1,6,1]`.
  - `test_segments_cross`: the X shape is True; T-touching and parallel segments are False.
  - `test_nearest_point_on_rect`: the point `(10, 0)` against the rect centred at `(0,0)`, size 4×2, gives `(2, 0)`.
  - `test_two_line`: `two_line("Central African Republic") == "Central African\nRepublic"` and `two_line("Chad") is None`.
- [ ] **Step 2: Run them.** Expected: FAIL (import error).
- [ ] **Step 3: Move the geometry helpers, then the label building, then placement.** Make `cost_batch` and `candidates` module functions that take `(p: Placement, labels, i, …, weights, outline_px)`. Drop the `if False else` in the call to `segments_cross`. `seed` is used to make `np.random.default_rng(seed)` once, as now.
- [ ] **Step 4: Run the tests.** `.venv/bin/pytest -v`. Expected: all pass, with the fingerprints unchanged.
- [ ] **Step 5: Commit.**
  - `+labels/geometry`, with the empty `labels/__init__.py` in the same commit
  - `+test_label_geometry`
  - `+labels/label`
  - `+labels/placement`
  - Each commit leaves the program running.

### Task 9: `render.py`, `make_map` → `cli`

**Files:**
- Create: `src/worldmap/render.py`, `src/worldmap/cli.py`
- Delete: `src/worldmap/make_map.py`
- Modify: `src/worldmap/__main__.py`, `pyproject.toml`, `tests/test_cli.py`

**Interfaces:**
- Produces in `render.py`:
  - Colour constants: `OCEAN`, `BORDER`, `STATE_LINE`.
  - `new_figure(frame: MapFrame) -> tuple[Figure, Axes]`: covers the size, the globe fill and stroke, and the transparent background.
  - `draw_countries(ax, countries_projected, fill_of, regions: Regions | None) -> None`: covers Antarctica white and Baikonur hatching. `regions is None` means graph colours.
  - `draw_states(ax, states_projected) -> None`
  - `draw_lakes(ax, lakes_projected) -> None`
  - `subtract_lakes(countries_projected, lakes_projected) -> gpd.GeoDataFrame`
  - `draw_labels(ax, labels: list[Label], placement: Placement, frame: MapFrame) -> None`: places the text, leaders and dots.
  - `patches(geom)` and `holed_patches(geom)`, with the dead `for r in interiors: pass` removed.
- Produces in `cli.py`: `parse_args(argv)` and `main(argv: list[str] | None = None) -> None`, which read as about 40 lines of stage calls.
- `pyproject.toml`: `make_map = "worldmap.cli:main"`

- [ ] **Step 1: Test.** `test_cli.py::test_console_script_help`: `subprocess.run([Path(sys.executable).parent / "make_map", "-h"], check=True, capture_output=True)` succeeds, and its stdout contains `--pin-size` but not `--ocean`.
- [ ] **Step 2: Move the drawing code into `render.py`.** Do this in `make_map.py` first. Run `pytest` and commit `+render`.
- [ ] **Step 3: Rename `make_map.py` to `cli.py`.** Create `cli.py` with `make_map.py`'s remaining contents and delete `make_map.py`. Update `__main__.py`, the `pyproject` entry point and the `test_cli` imports. Run `.venv/bin/pip install -e .`.
- [ ] **Step 4: Run the tests.** `.venv/bin/pytest -v`. Expected: all pass, with the fingerprints unchanged. `.venv/bin/make_map -h` works.
- [ ] **Step 5: Commit.** `->cli`. Git detects the rename from the content.

### Task 10: Naming and magic numbers, module by module

**Files:**
- Modify: each module under `src/worldmap/`.

Rules:
- **Variable names.** No variable names under 3 letters except loop indices `i`, `j`, `k`, and the coordinates `x`, `y`, `w`, `h`. Colour maths keeps the standard `L`, `C`, `h`, `a`, `b`, but nothing else uses capitals.
- **Constants.** Every unexplained numeric literal becomes a named module constant, or a dataclass field where the module already has a config dataclass. The exceptions are 0, 1, 2, 0.5 and array indices.
- **Dead code.** Remove dead code and unused imports.

- [ ] **Step 1: Clean up one module per commit.**
  - Order: `colorspace`, `data`, `coloring`, `frame`, `earth`, `pins`, `labels/geometry`, `labels/label`, `labels/placement`, `render`, `cli`.
  - After each module, run `.venv/bin/pytest -v`. Expected: all pass, with the fingerprints unchanged.
  - Commit each as `<module>: descriptive names, named constants`.
- [ ] **Step 2: Type check.** Run `.venv/bin/mypy src`. Expected: no errors (stubs that are missing may be ignored via `[tool.mypy] ignore_missing_imports = true` in `pyproject.toml`, committed as `pyproject: mypy config`).

### Task 11: Docs

**Files:**
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Update `CLAUDE.md`.** "How make_map.py is organised" becomes a module map that matches the table in the spec. Add a "Tests" section covering:
  - `pip install -e '.[dev]'`
  - `pytest`, including its runtime
  - `UPDATE_REFERENCE=1`
  - the version-mismatch skip
  - the rule that a commit which deliberately changes the output also commits `fingerprints.json`

  Replace "there are no automated tests" under Gotchas. Update every function name that changed.
- [ ] **Step 2: Update `README.md`.** Mention tests and the dev install.
- [ ] **Step 3: Commit.** `CLAUDE: module layout, tests`, then `README: tests`.
