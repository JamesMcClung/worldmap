# `--size`: poster size, with labels in absolute units

## Why

Every map is 36 in on its long side today, crops too. On a small crop (the Dead Sea, say), that canvas looks empty and
wrong: the rivers and labels are small next to the area they cover. Maps don't need to be that big. The poster size
should be something you can choose, and a crop's default should shrink with how much ground it covers, though much less
than proportionally.

## Goals

- `--size IN` sets the poster's long side in inches, for the world map and for crops.
- Every text size is in absolute units (pt on the poster), whatever the size or zoom. A bigger poster gives labels more
  room, so more of them fit; a smaller poster fits fewer.
- Without `--size`, the world map is 36 in (as today), and a crop gets a default that grows with its region but is
  never below a floor.

## Non-goals and anti-goals

- No sizes in map km (text or river widths). That makes text grow as you zoom in, which breaks the absolute-units rule.
- No river width or discharge data. Natural Earth has only `scalerank`, and widths stay `river_width(scalerank)` in
  absolute units.
- `--dpi` is unchanged: layout is still measured at `DPI` = 200, and `--dpi` only rescales the saved image.
- Anti-goal: a default crop so small that its labels and rivers are cramped or unreadable (hence the floor).

## Approaches

Let *s* be the region's long side measured on the 36 in world poster (France ≈ 1.35 in, Dead Sea ≈ 0.14 in).

1. **Power law with a floor (chosen):** size = 36 × (*s*/36)^p, at least a floor. Bigger regions get bigger posters,
   and small regions still have room. Chosen by the user. It was the option they had in mind too.
2. World scale with a floor: max(*s*, floor). This is simpler, but the size stops tracking the region until it's
   continent-sized.
3. A fixed default for crops (e.g. 12 in). This is the most predictable, but a crop of Asia gets cramped.

Also rejected: requiring `--size` with every `--crop`, and a pure world-scale default (France at 1.5 in). Both make
the user pick a size every time.

## Design

- **`--size IN`** (`positive_float`; the default is None, meaning "the map's default"). It sets the long side in inches.
- **`MapFrame` gets a long-side field** (e.g. `size_in`, default `POSTER_WIDTH_IN` = 36). It replaces the constant in
  `width_in`/`height_in`, so `px_per_unit`, `units_per_in`, the figure size, the topographic raster and the pins all
  follow it. `cli` sets it after the frame is built, from `--size` or the default.
- **Crop default:** `CROP_SIZE_POWER` = 0.4 and `CROP_MIN_IN` = 6 in `crop.py`, with
  *s* = 36 × (crop long side in metres) / (world frame width in metres), the same comparison `with_label_scale` makes
  now. Rough results: US 16 in, Europe 14 in, France 10 in, Dead Sea 6 in (4 in without the floor). The floor applies
  only to the default: an explicit `--size` can be any positive value. Tune both constants by eye on a few real crops
  (France, the Dead Sea, Europe, the Bering box, the Pacific) and record the values chosen.
- **`label_scale` is removed:** the `MapFrame` field, `with_label_scale`, `LABEL_SCALE_MAX`, `RIVER_SCALE_MAX`, and every
  multiplication by it (country `MIN_PT`/`MAX_PT`, `SHRINK_MIN_PT`, water and river sizes, `WaterLabel.min_size`).
  Crop labels then use the world map's point limits, so many countries in a crop will hit `MAX_PT` (30 pt).
  That's expected.
- **Other inch-based constants stay physical:** `LAKE_MIN_IN2`, `STRIPE_WIDTH_IN`, the colour bars, `--pin-size`.
  So a bigger poster shows smaller lakes.
- **Crop recipe** (`cropping to France; the same as --crop ...`): this stays as it is. The centre crop it prints
  has the same extent, so it gets the same default size. An explicit `--size` is a separate flag the user keeps.

## Consequences

- The world fingerprints are unchanged (its `label_scale` is already 1, and its size is still 36). The crop
  fingerprints (`crop`, `crop_water`, `topographic_crop`, `crop_pacific_water`) change, because of both the smaller
  default size and the removal of `label_scale`. Re-record them, and describe the visual change in the commit body.
- Pixel count and topographic memory scale with size². The default crops get smaller and cheaper, but a large
  `--size` on a topographic crop can need many GB. There's no guard against that (left to the user).
- Update `CLAUDE.md` (the `--crop` "Size" and "Labels" notes, `label_scale`, the CLI usage line). Leave `README.md`
  and `examples/` until the branch is done (ask first).

## Left to the builder

- The field name and where the default is computed (in `crop` or `cli`).
- The commit split. Probably: removing `label_scale` (crop output changes), then the `MapFrame` size field (a
  refactor, fingerprints unchanged), then `--size` and the crop default (crop output changes).
- Tests: unit tests for the default-size formula (floor, world gives 36, monotonic in *s*) and for `--size`
  parsing/errors. No new render fingerprint unless one is cheap, since the suite is already slow.
