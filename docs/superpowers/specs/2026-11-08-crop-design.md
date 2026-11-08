# Crop (`--crop`)

## Intent

Make a map of one country or region instead of the whole world. The common case is "show me France" or
"show me the Colorado River"; a lon/lat box covers everything else. A crop is a **zoomed, redrawn** map, not
a cut-out of the poster: borders, labels and the topographic raster are laid out again at the new scale, in a
projection centred on the region, so small places become readable and shapes are true.

## What the user decided

**Syntax.** One flag, two forms:

- `--crop NAME`: NAME is matched as a substring (case-insensitive) against the label texts of the kinds
  selected by `--label` only. `--crop India` is ambiguous with `--label countries,oceans` (India, INDIAN OCEAN)
  but not with the default `--label countries`. Kinds not in `--label` are never candidates.
  - Exactly one match: crop to it, no prompt.
  - Several: the CLI **prompts interactively** with a multi-select list of every match, exact matches first, each
    showing its kind and coordinates so identical texts are distinguishable (e.g.
    `Colorado (river) 109.5°W 36.2°N` / `Colorado (river) 64.3°W 39.4°S`). The user checks one or more items;
    the crop's extent is the union of the checked items. So with `--label countries,territories`,
    `--crop France` lists France, Réunion (France), French Guiana (France)...; checking France alone gives
    mainland France, checking France and Martinique frames both.
  - Exact matches never silently win: `--crop Niger` lists Niger, Nigeria (and the Niger river if `rivers`
    is selected).
  - None: exit with an error.
- `--crop lon,lat:lon,lat`: two opposite corners of a box, **lon first**, the same order as pin files.

**Extent of a named feature** = what its own label stands for. Parts that `--label territories` labels
separately are *not* included: France is mainland France + Corsica (no French Guiana, Réunion...), the US is
the lower 48 (no Alaska, Hawaii). `--crop Alaska` shows Alaska. This split applies whether or not
`territories` is in `--label`. A river's extent is the whole labelled river; a water body's, its polygon.

**Far-flung pieces become territories.** Country polygons far from the mainland that nobody has split out yet
are added to `labels/regions.HAND_PICKED`, so they are excluded from their country's extent and appear in the
prompt list like the other territories: at least Mayotte (France), Bouvet and Jan Mayen (Norway), Macquarie
Island (Australia), Sala y Gómez (Chile), Trindade (Brazil), the Caribbean Netherlands, and New Zealand's and
Kiribati's outlying islands (the builder checks the countries file for others). This changes the
`--label countries,territories` render; re-record its fingerprint in that commit. A less hand-picked rule
(non-contiguous = territory) was considered and rejected for now, since it misfires on island countries.

**Zoomed and reprojected.** A crop is drawn in a projection centred on the region (see "Builder's calls").
Everything is redrawn at the new scale: fills, borders, states, rivers, lakes, labels, topography.

**Neighbours drawn as normal.** Everything inside the frame is drawn and labelled as on the poster; nothing is
greyed out or de-emphasised.

**Output name unchanged.** Same defaults as today (`worldmap.png`, `worldmap_pins.png` with `--pins`).

**Colour bars** (`--type topographic`) become rectangles, still in the canvas corners (height bottom-right,
depth top-right with `--ocean`, ice bottom-left), drawn over the map.

**`--pick` must work with crops**, but is the last step of the work. With `--crop` the picker inverts the crop's
projection to get lon/lat.

## Goals

- `--crop` works with every existing option: both `--type`s, `--color earth`, `--disputed`, `--rivers`,
  every `--label` kind, `--pins`, `--dpi`, and finally `--pick`.
- Without `--crop`, output is byte-for-byte unchanged: every existing fingerprint in `tests/fingerprints.json`
  still matches, except `territories` after the `HAND_PICKED` additions.

## Non-goals

- Higher-resolution source data. Natural Earth 10m is the finest vector data we have; tight crops (a small
  country) will show coarse coastlines at poster size. Accepted.
- New label kinds that only make sense zoomed in (cities, etc.).
- Changing the output file name per crop.

## Anti-goals

- A crop that looks like a sheared piece of the Robinson poster.
- A bare name silently picking one of several matches.
- Labels leaving the frame, or a neighbour cut by the frame losing its label because its anchor is outside.

## Constraints from the code

- The projection is global today: `data.CRS` (Robinson) is used by `data.project`, `frame.make_frame`, and
  indirectly everywhere a projected GeoDataFrame is used. The crop's projection must come from the frame,
  so `project()` and its callers take the frame's CRS. Without `--crop` the frame is the current Robinson one.
- `MapFrame.outline` is the globe outline, used for the ocean fill and stroke (`render.new_figure`), the
  "inside the globe" placement cost (`placement.py:276`), colour-bar regions (`colorbars.bar_region`), the
  raster clip (`topographic.py:257`) and the picker. In a crop the outline is the canvas rectangle: ocean fills
  the whole canvas, no globe stroke, no transparent corners, and placement keeps labels inside the frame.
- An azimuthal projection of the whole world blows up near the antipode, so data must be clipped (in lon/lat
  or after projecting) to a neighbourhood of the frame before projection and drawing.
- Colouring (`coloring.json`) is global and stays as it is, so a country keeps its poster colour.
- Layout is measured at `frame.DPI`; `--dpi` only rescales. Font sizes come from `px_per_unit`, so they grow
  with the zoom automatically, within the existing pt clamps (6-30 pt countries, water's own clamps).
- `--type topographic` samples the cached 3' ETOPO grid (~5.5 km). Crops much smaller than a continent will
  look soft. Using the 60" source for small crops is optional (builder's call, see below); a coarse look is
  acceptable for a first version.
- Antimeridian: a crop can straddle +-180 (Fiji, Bering Strait). A centred projection handles this naturally,
  but `data.project` clips at +-180 and `labels/water.py` splits the Pacific at x = 0 in Robinson; both
  must still behave sensibly in a crop.
- Pins: `pins.py:76` checks lon/lat ranges and draws via the frame; pins outside the crop are simply not visible.

## Builder's calls (Claude's suggestions, not user decisions; change if there's a reason)

- **Projection:** Lambert azimuthal equal-area (`+proj=laea`) centred on the region's centre. Equal-area keeps
  the area-based font sizing meaningful; shapes hold up from country to continent size. For a box crop the
  centre is the box's lon/lat midpoint (going east from the first corner's longitude if the box crosses the
  antimeridian; see next point).
- **Frame:** the smallest rectangle in the crop projection containing the region (the projected box, or the
  feature's geometry), plus a margin of 10% of its size on each side for a named crop; none for a box.
- **Size:** the long side of the image is 36 in (7200 px at 200 dpi); the other side follows the frame's aspect
  ratio. (Today width is always 36 in; a tall crop like Chile would otherwise be metres tall.)
- **What names are searched:** the label texts of the selected `--label` kinds: countries (after `RENAME`),
  territories (`child_regions`), water bodies, rivers, and with `--disputed` the disputed areas that get labels.
  Water bodies and rivers are searched from the full body lists, not only those whose name fits on the poster
  (`WATER_MIN_INSIDE` filtering is redone at the crop's scale). The listed location is the label's anchor
  point in lon/lat. `--label none` with a name crop is an error.
- **Prompt library:** `questionary` (checkbox prompt) as a new dependency. When stdin is not a terminal
  (tests, scripts) an ambiguous name exits with the list instead of prompting.
- **Box corners in any order:** of the two possible longitude spans between the corners, the narrower is
  used, so `170,50:-160,70` and `-160,70:170,50` both mean the 30° box across the antimeridian.
- **Resolving the crop before projecting:** names and extents must be found before the crop's frame exists.
  Doing this in the Robinson frame (cheap, and child-region selection already uses it) or in lon/lat is the
  builder's choice, but must not run the full label pipeline twice.
- **Cut-off neighbours:** a label's anchor is computed on the part of its feature inside the frame, so a
  neighbour cut by the edge is labelled on its visible part.
- **Colour bars:** rectangle in each corner, size proportional to the frame's short side, with the existing
  labels, title (straight instead of curved is fine), contour ticks and sea-level line; drawn with a background
  so they read over the map.
- **Tests:** at least `--crop France` (political) and one topographic crop added to `test_regression.py`
  (`fingerprints.json` updated in the same commit); unit tests for crop-argument parsing, name matching and
  ordering of the ambiguity list, and the antimeridian box rule.

## Left to the builder

- Module layout (a `crop.py` module is suggested), how the frame type changes (e.g. outline becomes the canvas
  rectangle vs a flag), and the order of commits (frame/projection refactor with unchanged fingerprints first).
- How water labels' Pacific split and other Robinson-specific assumptions are adapted.
- Whether small topographic crops sample the 60" ETOPO directly.
- How `--pick` learns the crop (it takes the same `--crop` argument as the render).
