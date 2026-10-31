# Topographic map (`--type topographic`) — design

From the "Topographic" and "Map Type" items in `todo.md`.

## Intent

A second kind of poster: the same Robinson globe, but showing the shape of the land instead of countries. No
borders, no labels, just the coastline, height colours with hill shading, and contour lines. An option adds the sea
floor. Mountain ranges should jump out; the poster should look like a good physical atlas page.

This is the first non-political map type, so it also introduces `--type`, which later types (geological, sky...)
will join.

## Goals

- **`--type {political,topographic}`**, default `political`. Political is today's map, unchanged.
- **`--ocean`**: with `--type topographic`, the sea is coloured by depth. Ignored with `--type political` (may change
  later).
- **Nothing political** on a topographic map: no country borders, state lines, country or disputed labels, stripes.
  The only dark line is the coastline (land/water edge, including lake shores); the globe outline stays.
- **Land** (agreed from mockup C): classic atlas tints by height, green lowlands -> yellow-tan -> brown mountains
  -> near-white peaks, with hill shading lit from the north-west (azimuth 315°). Contour lines in a darker brown.
- **Ocean**:
  - without `--ocean`: flat `OCEAN` (`#e3f0f7`), as on the political map;
  - with `--ocean` (mockup E): from `OCEAN` at sea level to a deep blue (mockup used `#2f5f8c` at 8,000 m), lightly
    hill-shaded, with depth contours in a darker blue.
- **Ice** (Greenland, Antarctica, glaciers): coloured by the height of the ice surface on its own scale, white rising
  to a pale blue at the top (about 4,000 m covers Antarctica's dome), with contours like the land. Without this, the
  ice sheets read as 3,000 m brown mountains.
- **Lakes**: every lake the political map draws (`load_lakes`: >= `LAKE_MIN_KM2`, minus `DRY_LAKES`) is flat
  `OCEAN`, with a coastline, on both `--ocean` and plain maps. Lake surfaces in the elevation data are land
  heights, so without this the lakes vanish (they did in the mockups, except the Caspian).
- **Colour bars**, in the empty canvas corners outside the globe (agreed from the mockup):
  - each is a right triangle with its right angle in a canvas corner, its two straight sides along the canvas edges,
    and its third side curved, following the globe outline at a fixed gap;
  - tick labels sit in that gap, between the bar and the globe, with a white outline (halo) so they read on a
    transparent or dark background;
  - **height**: bottom-right; sea level along the bottom edge, highest value at the tip; ticks every 1,000 m;
  - **depth** (only with `--ocean`): top-right; sea level along the top edge, deepest at the tip;
  - **ice**: bottom-left, mirrored; white at the bottom to pale blue at the tip;
  - each bar has a short title (Height, Depth, Ice) near its tip, also haloed.
- **`--pins`** works on both types. `--dpi`, `--pick` work as today.

## Non-goals

- Labels of mountain ranges, peaks, or any other features (a later step: "labeled mountain pins" in `todo.md`).
- Rivers, climate, geology.
- Other `--color` palettes for topographic.

## Anti-goals

- Lakes or ice that disappear or turn into fake mountains.
- Colour bar labels hidden on a transparent/black background, or colour bars overlapping the globe.
- Any change to the political map's output (its fingerprints must not move).

## Data

**ETOPO 2022, 60 arc-second, ice surface** (NOAA NCEI, public domain):
`https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif`
(466 MB GeoTIFF; a NetCDF version exists too). Heights in metres, land and sea floor in one grid. The user chose
ETOPO for both land and sea (not Natural Earth's bathymetry polygons); if the coastline it gives looks wrong, fixes
are considered then.

- Downloaded once into `.cache/` like the other data, only for `--type topographic`.
- At 200 dpi the poster is ~3 arc-minutes per pixel, so 60 arc-second data is ~3x finer than needed. The builder
  should cache a reduced copy (e.g. resampled to the poster grid, or a coarser lon/lat grid) so later runs are fast,
  keyed like `earth.json`.
- Reading it may need a new dependency (no raster library is installed today; `earth.py` reads its TIFF with
  Pillow, which may not handle this one). Builder's choice; add it to `pyproject.toml`.

**Ice outlines**: Natural Earth `physical/ne_10m_glaciated_areas.zip`, plus `ne_10m_antarctic_ice_shelves_polys.zip`
for Antarctica's floating ice shelves. *(My call; the user asked "do we have a map of ice sheets?" and was told this
would be the source.)* Alternative if those prove poor: ETOPO's bedrock grid, ice = surface minus bedrock > 0
(another 478 MB).

**Lakes**: the existing `load_lakes()`.

**Land vs water**: decided by height > 0 m, except lakes (above) and *known risk*: land below sea level (Caspian
depression shores, Dead Sea, Qattara, Death Valley, Dutch polders) may come out as sea. Check crops of those places;
if it looks wrong, it's a fix for later, raised with the user.

## Drawing

Bottom to top: globe background / height+shade raster (land, ice, ocean) -> lakes -> contours -> coastline ->
globe outline -> colour bars -> pins.

- The raster is clipped to the globe outline; the canvas outside it stays transparent, as today.
- The look should match the mockups (`mock_world.py` in the session scratchpad used a `LightSource` hillshade
  blended as `colour * (0.65 + 0.45 * shade)` with ~0.05 vertical exaggeration); exact ramps, shading strength and
  line widths are tuned on full-resolution crops.
- **Contour spacing**: left to the builder, tuned on full-resolution crops (the 500 m used in the Alps mockup is
  probably too busy at poster scale; 1,000 m on land looked reasonable in the whole-poster mockup). Ice contours use
  the land spacing; depth contours may be coarser.
- Antarctica: no special white fill on the topographic map; it is ice and gets the ice scale.
- *My calls, from the builder check:* ice is hill-shaded like land; where the ice outlines and ETOPO disagree
  (ice shelves float over sea pixels), ice wins; heights above a bar's top value (Everest, Himalayan glaciers above
  the ice scale's top) take the top colour, and the tip label shows it with a `+` (e.g. `6,000 m+`).

## Command line

- `--type` uses `choices=["political", "topographic"]`, default `political`.
- With `--type topographic`, `--color`, `--label`, `--disputed` and `--recolor` are ignored (mirrors `--ocean`
  being ignored in political mode). *(My call.)*
- Default output file stays `worldmap.png` (`worldmap_pins.png` with `--pins`), whatever the type. *(My call.)*

## Tests

- Unit: `--type`/`--ocean` parsing; colour bar geometry (stays outside the globe by the gap, right angle at the
  corner, ticks inside the canvas); whatever pure helpers the builder adds (ramps, ice/lake masks).
- Regression: new fingerprints `topographic` and `topographic_ocean`; all existing fingerprints unchanged.
- `README.md`, `CLAUDE.md` and `examples/` (a `topographic.png` at `--dpi 50`) updated.

## Left to the builder

Colour ramp stops and shading strength (matching mockups C/E), contour spacing and colours, colour bar gap,
size and tick font, the reading library and the reduced cache's format, module layout (probably a new
`topographic.py`, documented in `CLAUDE.md`), and how the political pipeline in `cli.main` is split from the
topographic one.
