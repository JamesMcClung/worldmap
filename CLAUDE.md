# worldmap

One module, `src/worldmap/make_map.py` (console script `make_map`), renders a poster-sized political world map PNG (country borders, state/province borders for
large countries, non-overlapping country labels). Output default: `worldmap.png`, an RGBA PNG 7200 x 3693 px (36 in wide at 200 dpi), transparent outside the globe outline.

## Setup and run

```
python3 -m venv .venv && .venv/bin/pip install -e .   # deps in pyproject.toml
.venv/bin/make_map [out.png] [--recolor] [--earth]
```

A run takes ~7-12 s. Output goes to the current directory.

Natural Earth data (public domain) is downloaded by `fetch(name)` on first use into `.cache/` at the repo root, located
relative to `make_map.py` (`CACHE`), so the package must be installed editable (`-e`). The land-cover raster (~75 MB zip,
394 MB unzipped into `.cache/ne1/`) is fetched only for `--earth`. `coloring.json` and `earth.json` live in `.cache/` too.
`.cache/` is gitignored. User-facing docs are in `README.md`; MIT `LICENSE`.

## How make_map.py is organised (top to bottom)

Module level holds constants and pure helpers; `main()` parses the command line (`parse_args()`, argparse, so `-h` lists every flag) and runs the whole pipeline, with the helpers
that need its state (`fill_of`, `draw_pins`, `pick_pins`, `cost_batch`, `candidates`, `optimize`...) nested inside it.
Importing the module has no side effects.

1. **Colour maths** (OKLab/OKLCH helpers) and `make_palette`.
2. **Colouring** of all country features: border adjacency (`adjacency`), plus across-water links
   (`add_proximity_edges`: real countries within 600 km, biggest first, kept only while the graph stays 5-colourable;
   then each feature topped up to >= 3 real-country neighbours), then exact DSATUR backtracking (`color_graph`).
   Usage is balanced by country *count*, with the 10 largest countries given weight 0 so their colours aren't penalised.
3. **Drawing**: full Robinson globe outline (poles included; the canvas is its bounding box plus a small pad; Antarctica is drawn in
   hardcoded white with no label), outline stroked, figure background transparent, country fills, state lines for `STATE_COUNTRIES`, Baikonur special case.
   The country polygons cover lakes, so every lake in `ne_10m_lakes.zip` (a `physical/` NE file) of >= `LAKE_MIN_KM2`,
   minus the usually-dry `DRY_LAKES`, is painted back in `OCEAN` at zorder 2.5 (above the state lines that cross them;
   `holed_patches` keeps islands). The lakes are then subtracted from `countries_p`, so label placement treats them as water.
4. **Labels**: anchor points, font size, in/out mode, wrapping, then a cost-minimising placement
   (coordinate descent over candidate positions) with leader lines for labels that would hide their country.

## Colour scheme

- 5 country hues, equispaced with the ocean hue pinned as the 6th (so land never looks like sea). Ocean colour is fixed.
- Lightness depends on hue: darkest at red, lightest at green (raised-cosine ramp) so red/green colour-blind viewers
  can separate them. Tune via `C`, `l_red`, `l_green` in `make_palette()`. Label text/lines/dots are the fill's own
  OKLab hue, darkened (`darken()`, L=0.26).
- Hues are assigned to colour classes to maximise the OKLab distance between classes that touch.
- Baikonur (Kazakh land leased to Russia) is hardcoded to Russia's fill with Kazakhstan-colour hatching.
- Verified approximately with a Machado deuteranopia/protanopia simulation, not with a real colour-blind viewer.

## Caching

`coloring.json` (in `.cache/`) stores each country's colour class and which classes touch. It is rebuilt when the
country file, country list, `N_COLORS`, `MAX_GAP`, `MIN_DEG` or the `rule` string in `_key` change, or with `--recolor`.
Palette tweaks never change which countries share a class. If you change the colouring *logic*, bump `rule`.
The script prints a spot-check of same-coloured pairs (Korea/Japan, Malaysia/Philippines, Italy/Greece, France/UK,
UK/Norway, India/Sri Lanka, Russia/US) on every run; it should be empty.

## `--earth` colouring mode

Replaces the graph colouring with a natural palette derived from real imagery: the Natural Earth I land-cover raster
(`.cache/ne1/NE1_LR_LC.tif`, from https://naciscdn.org/naturalearth/10m/raster/NE1_LR_LC.zip, fetched automatically).
`earth_values()` rasterises each *polygon* (not country) and takes the mean of (G - R) over its land pixels (ocean = white) plus
its fraction of bluish-white "ice" pixels. `earth_fill()` maps G-R linearly (`GR_DESERT`..`GR_LUSH`) onto an OKLCH sand -> light-green
ramp and blends toward ice. Every disjoint polygon is coloured independently (Alaska, French Guiana, Kaliningrad, Baikonur, islands);
polygons with < `MIN_PIX` pixels sample a 7x7 window at their centroid, then fall back to the whole country. Labels take the colour
of the polygon they sit on (`fill_of(i, pt)`; `REGIONS` maps country -> [(projected polygon, fill)]). Results are cached in `earth.json`
(keyed on file sizes + `rule`; bump `rule` if the sampling logic changes). The graph colouring still runs but is unused; Baikonur's hatching is off.

## Pins (`--pick`, `--pins`)

Digitises the physical pin-the-hometown poster. A pin list is a text file, one `lon,lat` per line (`#` comments allowed).
- `make_map --pick pins.txt` opens an interactive window showing the *already rendered* `OUT` (default `worldmap.png`,
  downsampled 4x; errors if it doesn't exist). Left-click adds a pin (inverse Robinson to lon/lat; clicks outside the globe are
  ignored), right-click or `u` undoes, closing the window appends the new pins to the file (existing lines untouched; undo only
  reaches this session's pins). The picker runs right after the globe extent is computed, then exits, so it skips the label work.
- `make_map [out.png] --pins pins.txt [--pin-size 0.24]` draws golden shaded spheres (`pin_sprite`: shaded by a reflected environment: cream sky, dark band above a domed horizon, gold ground, window strips at the sides; styled after `pin-example.png`
  environment for a chrome-like look, broad specular glare with a hot core, baked drop shadow) at zorder 10, above labels. Pins are drawn northernmost first so southern ones overlap them.
  Diameter is in inches on the 36 in poster. Default output with `--pins` is `worldmap_pins.png`, so the bare map isn't overwritten.
- `--pick` skips `matplotlib.use("Agg")` to get the interactive backend. Pins on the antimeridian are clipped by the canvas edge.

## Label placement notes

- Which features get labels: `labeled()` (sovereign countries and a few extras; small territories and Antarctica excluded).
  Names shortened in `RENAME`. France's French Guiana and Russia's Kaliningrad get separate labels.
- Anchor = centre of the largest inscribed circle of the biggest polygon; island-heavy countries (several significant
  polygons, largest < 65% of total, plus `FORCE_ISLAND`) use the convex-hull centroid and a hull-based size.
- Font size scales with sqrt(area), clamped to 6-30 pt. A label covering >= 60% of its country (or much wider than a
  short narrow one) becomes an "out" label with a leader line. Long names wrap onto 2 lines when much wider than their
  country; stuck labels shrink slightly (floor ~5 pt) rather than being dropped.
- Cost terms: label must lie inside the globe outline (`OUTLINE_PX`), label-label overlap, covering another leader's dot, leader/label and leader/leader crossings,
  drifting outside the country, leader length. All labels are movable.

## Gotchas

- macOS BSD `sed -i` needs `sed -i ''`; prefer python/Edit for file edits.
- Keep scratch files (previews, crops, simulations) in the session scratchpad, not the project dir.
- Natural Earth `TYPE` includes odd features (Baikonur, Bir Tawil, Siachen, UN buffer zone...) that are drawn but
  deliberately not labelled. `NAME_EN` is used as the country key.
- The antimeridian: projection is centred on lon 0 and clipped to +-180; across-water adjacency uses a +-world-width
  shift so Russia/US (Bering Strait) still link.
- Output is checked visually by cropping regions from `worldmap.png`; there are no automated tests.
