# worldmap

The `worldmap` package (console script `make_map`, also `python -m worldmap`) renders a poster-sized political world map PNG (country borders, state/province borders for
large countries, non-overlapping country labels), or with `--type topographic` a hill-shaded height map. Output default: `worldmap.png`, an RGBA PNG 7200 x 3693 px (36 in wide at 200 dpi), transparent outside the globe outline.

## Setup and run

```
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # deps in pyproject.toml; [dev] adds pytest, mypy
.venv/bin/make_map [out.png] [--type political|topographic] [--ocean] [--recolor] [--color colorblind|earth] [--label LIST] [--disputed] [--rivers] [--pins FILE] [--pin-size IN] [--dpi DPI] [--crop NAME|lon,lat:lon,lat] [--pick FILE]
```

`--dpi` only changes the `savefig` dpi (and the pin sprite's pixel size): layout is always measured at
`frame.DPI` = 200, so a low-res render is the poster scaled down. `examples/` is built at `--dpi 50`.
A run takes ~7-12 s. Output goes to the current directory. After changing the entry point in `pyproject.toml`, re-run `pip install -e .`.

Natural Earth data (public domain) is downloaded by `data.fetch(name)` on first use into `.cache/` at the repo root, located
relative to `data.py` (`CACHE`), so the package must be installed editable (`-e`). The land-cover raster (~75 MB zip,
394 MB unzipped into `.cache/ne1/`) is fetched only for `--color earth`; ETOPO 2022 (466 MB GeoTIFF) and the Natural Earth ice files only for
`--type topographic`; ETOPO's 15" tiles (`etopo15_tile`, 4-33 MB each, into `.cache/etopo15/`) only for zoomed
topographic crops. `fetch`/`fetch_all` ask y/N first with the total size (HEAD request) and the folder, once for all
missing files; without a tty they exit with that message. `fetch_all(optional=True)` returns None when declined (the
tiles: the crop falls back to 1'). `coloring.json`, `earth.json` and `etopo_3min.npy`/`.json` live in `.cache/` too.
`.cache/` is gitignored. User-facing docs are in `README.md`; MIT `LICENSE`.

## Tests

`.venv/bin/pytest` (~155 s). Unit tests cover the pure pieces (colour maths, `color_graph`, pin parsing and sprite, label geometry,
CLI errors), and `test_import_has_no_pyplot` guards that importing any module doesn't pick a matplotlib backend (`--pick` needs to).
`test_regression.py` renders the default, `--color earth`, `--pins`, `--label countries,territories`, `--label countries,water` (`water`), `--disputed`, `--rivers`, `--type topographic`, `--type topographic --ocean`, `--type topographic --label water` (`topographic_water`), `--crop France` (`crop`), a Bering Strait box crop with territories, water and `--disputed` (`crop_water`) an Alps box crop with `--type topographic --ocean` (`topographic_crop`; reads four 15" tiles, recorded in its
environment, so it skips where they aren't downloaded) and `--crop pacific --label countries,oceans,seas` (`crop_pacific_water`, Equal Earth) maps in a subprocess and compares a SHA-256 of the decoded pixels
with `tests/fingerprints.json`:
- A commit that changes the output on purpose re-records with `UPDATE_REFERENCE=1 .venv/bin/pytest` and commits `fingerprints.json`
  in the same commit, saying in the body what changed visually. Refactors must leave the fingerprints unchanged.
- Each fingerprint records the library versions it was made with (matplotlib, freetype, numpy, shapely, GEOS, ...). With other
  versions the render tests *skip* rather than fail; re-record after upgrading.
- A missing fingerprint fails. Rendered PNGs are kept in `.cache/reference/` (`<mode>.png`, and `<mode>.new.png` on a mismatch) to compare by eye.
- The test reads the `coloring.json`/`earth.json` caches, so changes to colouring or land-cover sampling need a `--recolor` run
  (or deleting `earth.json`) to be exercised.
- `conftest.private_caches` (session, autouse) points the in-process tests' `coloring.json`/`earth.json` at a temp copy, so a
  test with other inputs can't overwrite the real caches; the render subprocesses still read the real ones.
- `conftest.environment(mode)` keys data files per mode (`raster` for earth, ETOPO etc. for topographic*, `marine` for *water*, `rivers` for *river* and *water*):
  a new key on an existing mode would make it skip, so add keys only for the modes that use them.

`.venv/bin/mypy src` should report no issues (default mode, missing stubs ignored).

## Module layout (`src/worldmap/`)

Importing any module has no side effects; only `cli.main` (after choosing the backend) and functions that need it import `matplotlib.pyplot`.

| Module | Contents |
|---|---|
| `cli.py` | `parse_args` (`--label`: `label_kinds`, a list that replaces the default `countries`; `none` = no labels), `graph_fills` (palette + Baikonur), `main()`: the pipeline as a sequence of stage calls |
| `data.py` | `fetch`, `fetch_all`, `confirm_download`, `etopo15_tile`, file names (`MARINE`: the marine label areas; `RIVERS`), `CRS`, `STATE_COUNTRIES`, lakes filter, `is_dependency`, `parent_index`, `is_labeled`, `load_*` (`load_rivers`: scalerank <= `RIVER_MAX_RANK`, no canals) |
| `frame.py` | `MapFrame` (transformers, outline, extent, `crs`, `globe`, `clip`, `label_scale`, `width_in`/`height_in`, `px_per_unit`, `units_per_in`, `canvas`, `project`, `visible`, `to_px`, `px_to_map`), `make_frame`, `cut`, `join_at_antimeridian`, `lon_ranges`, `DPI`, `POSTER_WIDTH_IN` |
| `crop.py` | `--crop`: `parse_box`, `box_region`, `extent`, `sphere_centre`, `clip_boxes`, `reach`, `seam_boxes`, `globe_outline`, `crop_frame`, `Candidate`, `candidates`, `matches`, `describe`, `checklist`, `choose` |
| `colorspace.py` | OKLab/OKLCH conversions, `fit_gamut`, `darken` |
| `coloring.py` | `merge_into_parents`, `adjacency`, `add_proximity_edges`, `color_graph`, `make_palette`, `load_or_compute_coloring` (cache), `spot_check`, `OCEAN` |
| `disputed.py` | `--disputed`: `parse_note`, `ALIASES`, `HAND_CLAIMS`, `LABEL_TEXT`, `Dispute`, `load_disputes`, `dispute_edges`, `dispute_fills`, `stripe_parts`, `border_lines`, `disputed_label_args` |
| `topographic.py` | `--type topographic`: `elevation_grid` (ETOPO -> cached 3' grid), `Grid` (a window has an origin), `elevation_window`, `read_etopo`, `read_tiles`, `elevation_for`, `sample`, `Scale` (`LAND`, `SEA`, `ICE`), `Contours`, `colorize`, `relief`, `shade`, `soften`, `poster_grid`, `polygon_mask`, `coastline`, `draw_topography` |
| `colorbars.py` | `--type topographic`: `bar_region`, `globe_side`, `inner_side`, `curved_title`, `draw_colorbar` (right-triangle bars in the canvas corners; rectangles in a crop) |
| `earth.py` | `--color earth`: `earth_values` (raster sampling, cache), `earth_fill`, `build_regions`, `fill_at` |
| `pins.py` | `load_pins`, `PinStyle` (every sprite shading constant), `pin_sprite`, `draw_pins`, `pick_pins` |
| `labels/label.py` | `Label` dataclass, `make_labels` (anchors, sizes, exclaves), `decide_out_and_wrap` |
| `labels/regions.py` | `--label territories`: `ChildRegion`, `HAND_PICKED` (parent + lon/lat boxes), `child_regions` |
| `labels/water.py` | `--label oceans,seas,lakes,straits,rivers,water`: `kinds_from_labels`, `classify`, `WaterBody`, `load_water_bodies`, `river_name`, `longest_paths`, `load_river_bodies`, `best_stretch`, `river_label`, `fit_curve`, `body_curve`, `WaterLabel`, `make_water_labels`, style constants |
| `labels/placement.py` | `CostWeights`, `Placement` (per-label arrays, water glyph boxes), `cost_batch`/`water_cost`, `candidates`, `clashes`, `place_labels` (`fixed`), `place_with_water` (drop loop), `place_all` (rivers last), `leader_ends` |
| `labels/geometry.py` | `rect`, `nearest_point_on_rect`, `segment_hits_rects`, `segments_cross`, `two_line` |
| `curves.py` | `arc_lengths`, `along`, `bend` (points along a polyline + its left normal): text bending for `colorbars.curved_title` and water labels |
| `render.py` | `new_figure`, `draw_countries`, `draw_stripes`, `draw_borders`, `draw_states`, `draw_rivers`, `draw_lakes`, `subtract_lakes`, `draw_labels`, `draw_water_labels`, line widths and z-orders |

Pipeline (`cli.main`):
1. **Colouring** of all country features. Dependencies (`is_dependency`: `SOVEREIGNT != ADMIN`, not a lease or
   "Indeterminate"; 46 of them, e.g. Greenland, Puerto Rico, Hong Kong) are first dissolved into their parent
   (`merge_into_parents`), so they always share its colour. Then border adjacency (`adjacency`), plus across-water links
   (`add_proximity_edges`: major countries within `MAX_GAP` = 600 km, biggest first, kept only while the graph stays 5-colourable;
   then each feature topped up to >= 3 major-country neighbours), then exact DSATUR backtracking (`color_graph`).
   Usage is balanced by country *count*, with the largest countries given weight 0 so their colours aren't penalised
   (`UNWEIGHTED_LARGEST = 10`, but the test is `rank < 10`, so 9 are exempt; kept for output stability).
2. **Drawing**: full Robinson globe outline (poles included; the canvas is its bounding box plus a small pad; Antarctica is drawn in
   hardcoded white with no label), outline stroked, figure background transparent, country fills, state lines for `STATE_COUNTRIES`, Baikonur special case.
   The country polygons cover lakes, so every lake in `ne_10m_lakes.zip` (a `physical/` NE file) of >= `LAKE_MIN_KM2`,
   minus the usually-dry `DRY_LAKES`, is painted back in `OCEAN` at zorder 2.5 (above the state lines that cross them;
   `holed_patches` keeps islands). The lakes are then subtracted (`subtract_lakes`), so label placement treats them as water.
3. **Labels**: anchor points, font size, in/out mode, wrapping, then a cost-minimising placement
   (coordinate descent over candidate positions, rng seed `SEED = 7`) with leader lines for labels that would hide their country.

## Colour scheme

- 5 country hues, equispaced with the ocean hue pinned as the 6th (so land never looks like sea). Ocean colour is fixed.
- Lightness depends on hue: darkest at red, lightest at green (raised-cosine ramp) so red/green colour-blind viewers
  can separate them. Tune via `PALETTE_CHROMA`, `L_RED`, `L_GREEN` in `coloring.py`. Label text/lines/dots are the fill's own
  OKLab hue, darkened (`darken()`, L=0.26).
- Hues are assigned to colour classes to maximise the OKLab distance between classes that touch.
- Baikonur (Kazakh land leased to Russia) is hardcoded to Russia's fill, striped with Kazakhstan's colour by `draw_stripes`
  (its dispute alone, without `--disputed`), its border drawn again above the stripes.
- Verified approximately with a Machado deuteranopia/protanopia simulation, not with a real colour-blind viewer.

## Caching

`coloring.json` (in `.cache/`) stores each country's colour class and which classes touch. It is rebuilt when the
country file, country list, the major-country, dependency or dispute-edge lists (`cache_key`), `N_COLORS`, `MAX_GAP`, `MIN_DEG` or
`COLORING_RULE` change, or with `--recolor`.
Palette tweaks never change which countries share a class. If you change the colouring *logic*, bump `COLORING_RULE`.
The script prints a spot-check of same-coloured pairs (Korea/Japan, Malaysia/Philippines, Italy/Greece, France/UK,
UK/Norway, India/Sri Lanka, Russia/US) on every run; it should be empty.

## `--color earth` colouring mode

The default, `--color colorblind`, is the graph colouring above.

Replaces the graph colouring with a natural palette derived from real imagery: the Natural Earth I land-cover raster
(`.cache/ne1/NE1_LR_LC.tif`, from https://naciscdn.org/naturalearth/10m/raster/NE1_LR_LC.zip, fetched automatically).
`earth_values()` rasterises each *polygon* (not country) and takes the mean of (G - R) over its land pixels (ocean = white) plus
its fraction of bluish-white "ice" pixels. `earth_fill()` maps G-R linearly (`GR_DESERT`..`GR_LUSH`) onto an OKLCH sand -> light-green
ramp and blends toward ice. Every disjoint polygon is coloured independently (Alaska, French Guiana, Kaliningrad, Baikonur, islands);
polygons with < `MIN_PIX` pixels sample a 7x7 window at their centroid, then fall back to the whole country. Labels take the colour
of the polygon they sit on (`fill_at(regions, i, pt)`; `Regions` maps country -> [(projected polygon, fill)]). Results are cached in `earth.json`
(keyed on file sizes + `EARTH_RULE`; bump it if the sampling logic changes). The graph colouring still runs but is unused; Baikonur's stripes are off.

## `--disputed`

Natural Earth's `ne_10m_admin_0_disputed_areas.zip` (`data.DISPUTED`, 99 features, same release as the countries file) is read
on every run, because the colouring uses it too. `load_disputes` turns each feature's `NOTE_BRK` ("Admin. by India; Claimed
by Pakistan", "Self admin.; Claimed by Georgia", "Leased to Russia by Kazakhstan") into the countries involved, the one running
it first; `ALIASES` maps the note's short names (`U.K.`, `China`...) to `NAME_EN`, `HAND_CLAIMS` covers entries without a note
(Kosovo, Gaza...), and an unknown name is an error. Features without a claim (DMZ, Ceuta, Isla Aves "multiple") are skipped.
Today: 72 two-country areas, 11 with 3+ countries, 6 breakaways (self-governing, no countries-file feature: Abkhazia, Donetsk...).
- Colouring: `dispute_edges` adds an edge between the two countries of every 2-country dispute, with or without the flag,
  so their stripes always differ (only the US / Marshall Islands, Wake Island, needed it).
- Stripes (`stripe_parts`): 45-degree '/' bands `STRIPE_WIDTH_IN` wide, counted from the map origin. 3+ countries: plain
  `DISPUTED_GREY`; breakaways: the parent's colour and grey. Disputed polygons are drawn untrimmed (a few islets exist only in this file).
- Borders: with `--disputed`, fills have no edges and `draw_borders` draws `border_lines`: a disputed outline is dashed
  (`BORDER_DASH`) where it meets a country involved (the one running it counts, so Kashmir's edge inside India is a new
  dashed line) or another disputed area sharing a country, within `SNAP_M`; every other ring of the country polygons stays solid.
- Z-order: fills 1, stripes 1.5, Baikonur's edge 1.6, borders 1.7, states 2, lakes 2.5 (borders must stay below the lakes).
  With `--disputed`, state lines move to 1.65, under the borders, or they hide dashes that follow a state line (Kashmir/Himachal).
- Labels (only with `countries` in `--label`): self-governing areas get upright labels via `make_labels(relabel=, extra=)`;
  `LABEL_TEXT` shortens official names (Northern Cyprus, Donetsk, Luhansk). Other disputed areas get none.
- `--color earth`: no stripes, but dashed borders and labels. Natural Earth's data is used as it is (it predates 2022-23 changes).

## `--type topographic`

`cli.main` branches to `draw_topographic` right after the frame (and `--pick`), so none of the political data, colouring
or country labels run. `--color`, `--disputed`, `--recolor` and `--label countries,territories` are ignored; water kinds in
`--label` add water labels (`place_with_water` with no country labels), else no label stage runs. `--ocean` is ignored with `--type political`.
- Data: ETOPO 2022 60" ice-surface GeoTIFF (`data.ETOPO`, Deflate + float predictor, read with rasterio), reduced to `GRID_STEP_DEG` = 3' block means as int16 and cached in `etopo_3min.npy` (keyed on the file size
  and `ELEVATION_RULE`; bump it if the reduction changes). Ice: Natural Earth `glaciated_areas` + `antarctic_ice_shelves_polys`
  (`load_ice`; the shelves' holes are dropped, the islands in them are ice-covered). Land vs water: Natural Earth
  `ne_10m_land` (`load_land`), not the sign of the height. Lakes: the political `load_lakes()`.
- Raster (`draw_topography`): `poster_grid` inverse-projects every pixel centre at `min(dpi, DPI)` (one pixel per
  cell; `inside` = re-projects to itself), `sample` is bilinear and wraps at +-180. Masks (`polygon_mask`) are
  rasterised with rasterio by pixel centre. Ice wins over land and water (ice shelves). Colours are signed `Scale`s
  (`start` at the colour bar's canvas edge, `end` at its tip): `LAND` -500..6000 m (land below sea level is darker
  green), `SEA` +500..-8000 m with `--ocean` (water above 0 m is ETOPO's averaging along coasts) else flat `OCEAN`,
  `ICE` 0..4000 m. Shading is `relief()`, a Lambert term relative to flat ground (not matplotlib's
  `LightSource.hillshade`, which rescales to the steepest slope on the globe and flattened the Alps), scaled by
  `VERT_EXAG`, `LAND_SHADE`, `SEA_SHADE`.
- Layers: raster at `Z_COUNTRY` clipped to the outline, contours just above (`LAND_CONTOURS`, `ICE_CONTOURS`,
  `SEA_CONTOURS`, dashed), coastline at `Z_BORDERS`, lakes at `Z_LAKES`, colour bars at `Z_GLOBE`. The coastline
  (`coastline`) is the outline of the union of the land and ice polygons, as in the political map, minus the
  slivers the union leaves where ice shelves meet land (`SLIVER_M2`, `SLIVER_HOLE_M2`; they drew stray dashes).
  It takes ~3 s. Tracing it from the smoothed pixel mask instead rounded the coast and left river mouths inland.
- Memory: a 200 dpi run peaks around 3 GB. Work in blocks or in place on the whole-poster arrays; don't use
  `ax.contour` (its ContourSet keeps a float64 copy of the grid per call, ~665 MB each), use `contour_lines` (contourpy).
- Colour bars (`colorbars.py`): height bottom-right, depth top-right (`--ocean` only), ice bottom-left. Each is its
  corner quadrant minus the globe grown by `BAR_GAP_IN`, minus `BAR_MARGIN_IN`; signed labels in the gap with a white
  halo (ends marked `≥`/`≤`), the map's contour lines across the bar, and sea level in the coastline's style.
  The title (`curved_title`, `TITLE_SIZE` bold) is a `TextPath` warped onto the bar's globe-side curve
  (`globe_side`), centred on it, inside the bar `TITLE_INSET_IN` from it (descenders clear the curve in the top-right bar).

## `--crop`

A zoomed map of one region, redrawn in a Lambert azimuthal equal-area projection centred on it (`crop.crop_frame`);
a region reaching over `AZIMUTHAL_MAX_DEG` = 60 degrees from its centre (`reach`: the Pacific, Atlantic and Indian
oceans) gets Equal Earth centred on its longitude instead, clipped a hair either side of the far meridian
(`seam_boxes`), measured along the equator too (widest there), the canvas cut to the globe (`globe_outline`).
- Every stage projects through the frame: `frame.project(gdf)` (there is no `data.project`). The world frame
  (`globe=True`) clips to the +-180/90 box and projects to Robinson exactly as before. A crop frame (`globe=False`)
  first clips to lon/lat `clip` boxes (`clip_boxes`: the canvas's lon/lat extent + `CLIP_PAD_DEG`; LAEA is singular
  at the antipode), joins pieces Natural Earth splits at +-180 when the crop crosses it (`join_at_antimeridian`,
  snapped to `SEAM_GRID_DEG`, or the cut shows as a border through Chukotka), then `cut`s to the canvas grown by
  `PROJECT_PAD` (repairing invalid input: the marine areas). Layers whose rows split a feature at +-180 (the land:
  Wrangel Island) pass `dissolve=True`. `frame.visible` cuts to the canvas itself: labels use it,
  so a neighbour cut by the edge is labelled on its visible part.
- Size: the long side is `POSTER_WIDTH_IN` (`width_in`/`height_in`); the world map is unchanged.
- Labels: `frame.label_scale` = sqrt(zoom over the world map), 1..`LABEL_SCALE_MAX` = 2.5, multiplies every size
  limit (country `MIN_PT`/`MAX_PT`, water and river sizes, `SHRINK_MIN_PT`, `WaterLabel.min_size`).
- `--crop lon,lat:lon,lat`: corners in any order, the narrower longitude span wins (so a box can cross +-180); no
  margin. `--crop NAME`: substring of a label's text, ignoring case and accents (`fold`), among the `--label` kinds
  only (`candidates`, built in the world frame, then lon/lat). A country's extent leaves out what territories label
  separately (`HAND_PICKED`, which also has the far-flung islands for this). `Candidate.aliases` match too: a
  country's `NAME_EN` (UAE), a territory's parent (so `united` lists Alaska (US) beside the United States).
  Several matches: a `questionary` `checklist` (no select-all/invert keys; exact first, `describe` = text, kind,
  lon/lat `extent`); the crop covers all checked; without a tty, exit with the list. Margin `CROP_MARGIN` = 10%. Candidates are inverted from the Robinson world frame with x nudged
  by `EDGE_NUDGE` (points on +-180 don't invert).
- Drawing: ocean fills the canvas, no globe stroke; `child_regions` skips hand-picked regions out of frame (it
  raises only on the globe); out-of-frame disputes are dropped after the colouring; the Pacific is one body (one
  name candidate too).
  `--color earth`: `build_regions(countries, frame)` projects polygon by polygon so each keeps its value.
- Topographic: `elevation_for` steps up from the cached 3' grid to a window of the 60" file when 3' cells span >
  `FINE_ABOVE_PX` = 2 pixels, and to the 15" tiles when 1' cells do too (`elevation_window`, from the lon/lat of every
  8th shown pixel, + `FINE_PAD_DEG`; runs across +-180 are read in two pieces). `soften` still blurs heights over
  about a cell when a cell spans > 2 pixels (bilinear heights shade as flat facets). A 200 dpi square crop peaks
  ~4.3 GB (more pixels than the world poster). Colour bars are rectangles (`BAR_RECT_WIDTH_IN`, `BAR_RECT_LENGTH`) titled along `inner_side`,
  over the map: water labels are placed in `label_outline` (the canvas minus the bar corners), and a crop's water
  label with a glyph outside the outline is dropped (`placement.outside_outline`).
- `--pick` with the same `--crop` inverts the crop's projection.

## Pins (`--pick`, `--pins`)

Digitises the physical pin-the-hometown poster. A pin list is a text file, one `lon,lat` per line (`#` comments allowed).
- `make_map --pick pins.txt` opens an interactive window showing the *already rendered* output PNG (default `worldmap.png`,
  downsampled 4x; errors if it doesn't exist). Left-click adds a pin (inverse Robinson to lon/lat; clicks outside the globe are
  ignored), right-click or `u` undoes, closing the window appends the new pins to the file (existing lines untouched; undo only
  reaches this session's pins). The picker runs right after the globe extent is computed, then exits, so it skips the label work.
- `make_map [out.png] --pins pins.txt [--pin-size 0.24]` draws golden shaded spheres (`pin_sprite`, tuned through `PinStyle`: shaded by a reflected environment: cream sky, dark band above a domed horizon, gold ground, window strips at the sides; styled after `pin-example.png`
  environment for a chrome-like look, broad specular glare with a hot core, baked drop shadow) at zorder 10, above labels. Pins are drawn northernmost first so southern ones overlap them.
  Diameter is in inches on the 36 in poster. Default output with `--pins` is `worldmap_pins.png`, so the bare map isn't overwritten.
- `--pick` skips `matplotlib.use("Agg")` to get the interactive backend. Pins on the antimeridian are clipped by the canvas edge.

## Label placement notes

- Which features get labels: `is_labeled()` (sovereign countries and a few extras; dependencies, small territories and Antarctica excluded).
  Names shortened in `RENAME`.
- `--label territories` adds italic `<region> (<parent>)` labels (`labels/regions.py`): every dependency that doesn't touch its
  parent (so not Hong Kong/Macau), plus `HAND_PICKED` parts of a parent's polygons (Alaska, Kaliningrad, Tasmania...),
  selected by testing each polygon's representative point against lon/lat boxes. Child labels are anchored on and sized
  by their largest polygon (a hull-based size makes spread-out archipelagos huge).
- Anchor = centre of the largest inscribed circle of the biggest polygon; island-heavy countries (several significant
  polygons, largest < 65% of total, plus `FORCE_ISLAND`) use the convex-hull centroid and a hull-based size.
- Font size scales with sqrt(area), clamped to 6-30 pt. A label covering >= 60% of its country (or much wider than a
  short narrow one) becomes an "out" label with a leader line. Long names wrap onto 2 lines when much wider than their
  country; stuck labels shrink slightly (floor ~5 pt) rather than being dropped.
- Cost terms: label must lie inside the globe outline, label-label overlap, covering another leader's dot, leader/label and leader/leader crossings,
  drifting outside the country, leader length; weights in `CostWeights`. All labels are movable.

## Water labels (`labels/water.py`)

- Bodies: Natural Earth `ne_10m_geography_marine_polys` (`data.MARINE`, rough label areas, never drawn) + the drawn lakes.
  `classify`: ocean -> oceans; sea/gulf/bay/sound/fjord/lagoon/inlet -> seas; strait/channel -> straits;
  reefs, rivers, unnamed `generic` areas and names ending " River" (Saint Lawrence is a 'channel') never. North + South
  Atlantic are one ATLANTIC; the Pacific is split at x = 0 into two bodies. Each body is labelled once, on its largest
  polygon; order = oceans, then scalerank, then area (most important first; the drop loop removes from the end).
- Curve (`fit_curve`): principal axis of interior sample points, tilt clamped, quadratic fit with the bend clamped
  (`MAX_BEND_DEG`, `MAX_TILT_DEG`), read left to right (bottom to top within 1 degree of vertical). Oceans too, fitted on
  the whole ocean, so ATLANTIC and the left PACIFIC run pole to pole (a level label at the widest water put them in the
  south). Sizes fill `SEA_FILL` of the curve within `WATER_MIN_PT`..`SEA_MAX_PT` (oceans
  `OCEAN_MIN_PT`..`OCEAN_PT`, letter-spaced caps; all upright).
- `WaterLabel`: one bent `Path` per label plus a halo-padded box per glyph (`parts`), both relative to its position.
  Bodies whose name doesn't sit >= `WATER_MIN_INSIDE` over them at their anchor are left out up front (~160 of ~380 kept).
- Solver: water labels come after the country labels in `Placement`; x/y is their anchor position. Country terms skip
  water labels' boxes (so with no water every array is as before), plus overlap with water glyph boxes and their own
  leader crossing them. Water labels: glyph-box overlap with everything, other leaders/dots, `outside_water` (glyph
  centres over the body, approximate), `water_drift`, outside globe. `clashes` -> (country, water); country labels shrink
  only for country clashes, water labels for anything.
- `place_with_water`: re-solves (warm start, country sizes reset) dropping per group of failing labels the least
  important; a failing label (off its water, or clashing at its minimum size) is grouped with the water labels it
  touches where it is or at its anchor, so the solver pushing a random rival off its water can't drop the wrong one.
- Drawing: `PathPatch` in `WATER_TEXT` (OKLCH L 0.50 in the ocean hue) with an `OCEAN`-coloured halo, under the country
  labels. `--label countries,water` takes ~28 s (vs ~10 s), mostly the country labels' extra passes.

## `--rivers`

Natural Earth `ne_10m_rivers_lake_centerlines` (`data.RIVERS`), scalerank 0 (biggest) to `RIVER_MAX_RANK` = 8;
9+ are short tributaries, canals are dropped. `render.draw_rivers` draws them in `RIVER` (OKLCH 0.54/0.10/240, darker
than the ocean so it shows on every fill), `river_width` thinning linearly from `RIVER_WIDTH_MAX` to `RIVER_WIDTH_MIN`,
thin ones first. `Z_RIVERS` = 2.2: above the state lines, below the lakes, which hide the centrelines through them.
Labels ignore them.

`--type topographic` draws them too.

## River labels (`--label rivers`)

- `rivers` is a water kind; `water` = all kinds (oceans, seas, lakes, straits, rivers). `--label rivers` or `water`
  also draws the rivers (`cli.draws_rivers`).
- Bodies (`load_river_bodies`): the drawn rivers grouped by `river_name(name_en)` (English, merges Rhein/Rhin; drops
  " River", `NOT_RIVERS` delta channels and canals, `RIVER_NAMES` fixes), scalerank <= `RIVER_LABEL_MAX_RANK`. Natural
  Earth breaks rivers at lakes, reservoirs and junctions, so `longest_paths` joins ends within `RIVER_JOIN_M` and takes
  the longest path through each network's maximum spanning tree (an exhaustive search blew up on deltas). Same-named
  networks `RIVER_REPEAT_M` apart are labelled separately (Colorado US/Argentina). `WaterBody.geom` is the line.
- `river_label`: italic, `river_size` by rank; `best_stretch` picks the straightest, most level stretch near the middle
  for the name + an em; `fit_curve` on it; `lay_out(rise=)` puts the baseline `RIVER_GAP_EM` above the curve plus
  however far the river wiggles above it (`height_above`). Its `geom` is a `RIVER_CORRIDOR_EM` buffer of the whole river.
- Placement: `place_all` places countries + other water first, then the rivers with all of those `fixed` (so a river
  name can't push a country label off its country: it once moved Czech Republic to Denmark). Rivers come last, so the drop loop drops them first. Drawn in `RIVER_TEXT` with the country labels' white halo.
- `--label countries,rivers` takes ~13 s; ~180 river names kept.

## Gotchas

- macOS BSD `sed -i` needs `sed -i ''`; prefer python/Edit for file edits.
- Keep scratch files (previews, crops, simulations) in the session scratchpad, not the project dir.
- Leave `README.md` and `examples/` alone until the branch's work is done, and ask before updating them: every
  re-rendered PNG adds 0.5-2.5 MB to the repo history.
- Natural Earth `TYPE` includes odd features (Baikonur, Bir Tawil, Siachen, UN buffer zone...) that are drawn but
  deliberately not labelled. `NAME_EN` is used as the country key.
- The antimeridian: projection is centred on lon 0 and clipped to +-180; across-water adjacency uses a +-world-width
  shift so Russia/US (Bering Strait) still link.
- Output changes are checked visually by cropping regions from `worldmap.png`, and pinned by the fingerprint tests.
