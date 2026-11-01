# Water labels (`--label oceans,seas,lakes,water`)

## Intent

Name the oceans, seas and lakes on the poster the way a printed atlas does: quiet blue lettering that feels
printed into the water, curving along each water body, sitting behind the country names in importance.
They must look good on the political map (with or without country labels) and on the topographic map.

## What the user decided

**`--label` kinds.** Four new values in the existing comma-separated `--label` list, freely mixed with
`countries` / `territories` (still defaulting to `countries`, so the default map is unchanged):

| Kind | Labels |
|---|---|
| `oceans` | the oceans |
| `seas` | seas, gulfs, bays and similar open-water bodies |
| `lakes` | lakes |
| `water` | all of the above, plus straits and channels (only reachable through `water`) |

- Reefs are never labelled.
- Rivers are **out of scope**. A `rivers` kind is added later together with drawing rivers (the map doesn't
  draw them yet). The marine file's river-mouth polygons (Amazon, Yangtze, Columbia) are skipped.

**Oceans.** One ATLANTIC (Natural Earth's North and South Atlantic merged), one INDIAN, ARCTIC and SOUTHERN,
and PACIFIC **twice**: once near the left edge of the map and once near the right edge (the antimeridian
splits it).

**Style.**
- Oceans: ALL CAPS and big, the largest water names on the map, with letter spacing so the name spreads
  across the open ocean.
- Seas, gulfs, bays, lakes, straits: smaller, in their normal case, in italics (the atlas convention for
  water, and it sets them apart from the upright country names).
- Colour: entirely blue. Text is a deeper blue in the ocean's hue (`coloring.OCEAN`, `#e3f0f7`). The trim
  (halo) is also blue, a light one close to the water colour, *not* the white halo the country labels use.
  Overall similar shades to the water, quieter than the country names.
- One fixed style on both map types, even where `--type topographic --ocean` shades the sea darker by depth.

**Shape.** Every water label **curves along its water body's long direction**. It does not strictly have to
stay inside the water body.

**Placement and dropping.**
- Water labels join the existing placement solver as equals: they can push country labels around and they
  can be shrunk like them.
- Water labels **never** get leader lines (never 'out').
- When the solver can't resolve the layout, water labels are dropped one at a time and the solver runs
  again, until it succeeds. Country and territory labels are never dropped (today's behaviour stays: they
  shrink and may be left overlapping).
- "Try to fit them all": the full list is tried. Dropping is what thins it, not an up-front cut.
- **Country labels shrink only for country labels.** In a clash between a water label and a country (or
  territory) label, only the water label shrinks; country labels shrink only when they clash with each other.
  So dropping water labels never leaves a country name smaller than it would be without them.
- **A water label is also dropped when it ends up too small or mostly off its water body**, not only when it
  overlaps something: the goal is an atlas-like map, not every bay at the minimum font size. Thresholds
  (a minimum font size above the shrink floor, a minimum share of the text over its water body) are the
  builder's to tune by eye.
- **Country leader lines stay clear of water text** like any other label (the `leader_hits_label` and
  `dot_cover` terms apply to water labels), and a leader crossing a water label counts as a clash that can
  get the water label dropped. *(my call)*

## Data

- **Marine:** Natural Earth `ne_10m_geography_marine_polys.zip` (physical, 306 features; add to
  `data.PHYSICAL`, fetched like the rest). Fields used: `name`, `featurecla`, `scalerank` (0 = the 7 ocean
  pieces, 1 = major seas such as the Mediterranean, Caribbean, Bay of Bengal, ... up to 9).
  Class mapping:
  - `ocean` -> `oceans` (merging/duplication above)
  - `sea`, `gulf`, `bay`, `sound`, `fjord`, `lagoon`, `inlet` -> `seas`
  - `strait`, `channel` -> `water` only
  - `reef`, `river`, unnamed features (all the `generic` ones) -> never
- **Lakes:** the lakes the map already draws (`data.load_lakes()`, 90 named lakes of >= `LAKE_MIN_KM2`
  minus `DRY_LAKES`), with their `name` and `scalerank`.
- The Caspian Sea is in the marine file as a sea (the lake file doesn't have it), so it is a `seas` label.
  If a name ever appears in both files, it is labelled once. *(builder's call how)*
- Natural Earth's marine polygons are drawn roughly (they are label areas, not coastlines); they are used
  only to decide where a label goes and which way it runs, never drawn.

## Behaviour by map type

- **Political:** water labels are made alongside the country/territory labels and placed in the same
  solver run.
- **Topographic:** today `--label` is ignored and no label stage runs. With water kinds in `--label`, the
  topographic map runs the label stage with only water labels; `countries` and `territories` stay ignored
  there. Since the default `--label` is `countries`, the default topographic map still has no labels.
  *(my call)*

## Ordering and dropping *(my calls, open to the builder's judgement)*

- **Importance:** oceans first, then by `scalerank` (lower = more important), then by area. Oceans are the
  last water labels to be dropped.
- **"Solver fails"** = after descent and shrinking, some water label still clashes with another label
  (overlap or a leader crossing it), or is too small, or is mostly off its water body. Too small / off its
  water drops that label directly; for clashes, the least important water label *among those involved* is
  dropped. The solver then runs again, warm-started from the current positions. Repeat until no water label
  fails. Overlaps between country labels alone end the loop as today.
- Dropping a label that isn't in conflict wouldn't help, so unrelated labels elsewhere are kept. Several
  failing labels may be dropped per round if that keeps the run time reasonable *(builder's call)*.

## How (left to the builder, with guidance)

The solver (`labels/placement.py`) treats every label as an axis-aligned box moved by coordinate descent.
Curved labels don't fit that directly; the builder chooses how to represent them. A suggestion that fits
the existing vectorised cost: a curved label is a chain of small boxes (one per glyph or short run of
glyphs) that move together, and its candidate moves are slides along its curve and small sideways shifts
rather than free x/y steps.

- **The curve:** a smooth, gentle path along the water body's long direction (a low-order fit through its
  middle, not a wiggly medial axis), with the text centred on it and read left to right (flip if needed,
  so nothing is upside down). Ocean curves may span a large part of the ocean; seas follow their own shape.
- **Glyph bending:** `colorbars.curved_title` already warps a `TextPath` onto a curve; reuse or generalise
  it rather than writing a second one.
- **Sizes:** oceans get a fixed large size and tracking; other water labels scale with their body like the
  country labels, with their own (smaller) caps. Tune by eye on the poster.
- **Cost terms:** a soft pull to stay over its water body (like `outside_country`) and near its curve, the
  usual overlap and outside-globe terms, other labels' leader terms (see above), and none for leaders of
  their own. Weights are the builder's to tune.
- **Drawing:** a curved `PathPatch` (or equivalent) with a blue `withStroke` halo, at the label z-order.

## Constraints

- The default output, and every existing regression mode, must render identically (fingerprints unchanged)
  until a commit deliberately changes it. Water labels are opt-in, so this should come free.
- Add regression modes for at least `--label countries,water` on the political map and a water-labelled
  topographic map, and unit tests for the pure parts (class mapping, ocean merging, the curve, ordering).
- Run time: `--label countries,water` should stay within a few times today's ~10 s run; the drop-and-resolve
  loop should warm-start rather than restart.
- Same module conventions as the rest (no import side effects, `mypy src` clean). Likely a new
  `labels/water.py`; CLAUDE.md and README updated.

## Non-goals

- Rivers and river labels (later, with river rendering).
- Labels for undersea features, mountain ranges, or other physical regions.
- Adapting label colour to the topographic depth shading underneath.
- Labels in other languages.

## Anti-goals (would make it a failure even if it works)

- Water labels that compete visually with the country names (too dark, or with white halos).
- Upside-down or jagged, wiggly text.
- Water labels with leader lines, or country labels dropped to make room for water.
- Changing the default map's output.
