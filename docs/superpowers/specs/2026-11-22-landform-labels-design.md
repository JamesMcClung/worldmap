# Landform labels (`--label mountains,uplands,deserts,lowlands,land`)

## Intent

Name the mountain ranges, plateaus, deserts and plains on the poster, curving along each one like the sea names
do. This is the first step towards `--world` (maps of the Moon, Mars and other bodies): Mars' montes, planitiae
and valles and the Moon's maria will be labelled with exactly this machinery, so it is built and tuned on
Earth first, where the pipeline, data and fingerprint tests already exist. Peaks and craters are not part of it.

## What the user decided

- **Approach A**: the curved-label machinery in `labels/water.py` is pulled out into a general curved area
  label (`labels/curved.py`), styled per kind; water keeps its bodies and rivers, a new `labels/land.py` loads
  the landforms. (Rejected: B, adding landform kinds to `water.py`, where "water" becomes a misnomer and
  `kind == ...` checks multiply as Mars adds more; C, upright country-style labels, which can't follow the
  Andes or the Urals and would go 'out' with leader lines.)
- **Style is by kind, never by size.** Oceans keep their letter-spaced capitals because they are oceans; no
  landform kind gets capitals, however big (no SAHARA).
- **Peaks are deferred.** Natural Earth's elevation points (Everest...) need a marker kept clear and a name
  beside it; that builds on pins and comes later.
- Lander pins (Moon/Mars) are a later idea, already in `todo.md`.

## Data

Natural Earth `ne_10m_geography_regions_polys.zip` (`physical/`, ~2 MB, 1047 features, public domain):
cartographers' label areas, the same sort of thing as the marine areas — smooth blobs along a range's spine,
not outlines, overlapping freely (the Taklamakan inside the Tarim Basin). Columns are upper case (`FEATURECLA`,
`NAME`, `NAME_EN`, `SCALERANK`, ...). New `data.REGIONS` constant, fetched by `data.fetch` on first use, only
when a land kind is in `--label` (also for `--crop NAME` with a land kind). 885 Polygons, 162 MultiPolygons.

## Kinds

| `--label` kind | Natural Earth `FEATURECLA` | count |
|---|---|---|
| `mountains` | `Range/mtn` | 222 |
| `uplands` | `Plateau` | 72 |
| `deserts` | `Desert` | 58 |
| `lowlands` | `Plain`, `Basin`, `Lowland`, `Valley`, `Gorge`, `Depression` | 55 |
| `land` | all four above, in that order (as `water` expands in place) | |

Not labelled (my call; each could be a kind later): `Island`, `Island group`, `Pen/cape`, `Peninsula`, `Isthmus`,
`Coast`, `Continent` (todo.md has `continents` as its own item), `Geoarea` (Siberia, Patagonia, Wilkes Land...),
`Delta`, `Wetlands`, `Foothills`, `Tundra`, `Lake` (the lakes come from the lakes file), `Dragons-be-here`.

## Design

### Refactor first: `labels/curved.py`

Every step leaves every fingerprint unchanged.

- `WaterLabel` -> `CurvedLabel` (in `curved.py`), `lay_out`, `extend`, `fit_curve` and its helpers
  (`interior_points`, `turn`, `rotate`, `ring_arc`), `largest`, and the size fit from `make_water_labels` move
  there. `water.py` keeps the water bodies, `classify`, rivers, ocean handling.
- A `Style` per kind replaces the `kind == "oceans"` / `kind == "rivers"` checks in laying out and drawing:
  italic, capitals/tracking (`OCEAN_TRACKING_EM`), the x-height or cap-height centring, text colour, halo colour,
  size range (min, max) and fill share. `CurvedLabel.min_size` reads it. Rivers keep their own layout path
  (`river_label`), only their style moves.
- `Placeable.water` -> `curved` ("on its feature by its glyphs' centres, never 'out'"), and `Placement.water`
  with it. Names in `priority.py`, `placement.py`, `kernels.py` call sites, `render.py` and `cli.py` follow.
  `WATER_STEP_SCALES`, `WATER_MIN_INSIDE` and the halo padding are about curved labels and are renamed where they
  move. The print in `cli.main` (`water labels:`) keeps its wording for water kinds.
- `render.draw_water_labels` -> `draw_curved_labels`, taking fill and halo from each label's style.

### Landforms: `labels/land.py`

- `LAND_KINDS`, `CLASSES` (the table above), `kinds_from_labels`, and `load_land_bodies(kinds, frame)`: project
  (`frame.project`, then `frame.visible`, `make_valid` as the marine areas need), each feature labelled on its
  largest polygon, `full_size` = km2 of the whole feature however much the frame shows (the file has no `km2`
  column, unlike the marine one: computed from the unprojected geometry in `data.EQUAL_AREA_CRS`), sorted biggest first; of two features with one name (Transantarctic Mountains, Canadian Shield, Great
  Rift Valley's two branches) the bigger is kept, as for seas.
- Text: `NAME_EN`, falling back to `NAME`. Some names are all capitals in the data (from Natural Earth's own
  styling); an all-capitals name is put in title case (my call; the builder checks the result reads right,
  e.g. lower-case "of", "de").
- Body type: reuse `WaterBody` renamed to a shared `Body` in `curved.py` (text, kind, rank, geom, full_size),
  or a sibling dataclass — the builder's call, whichever keeps `crop.candidates` simplest.
- Labels: `make_curved_labels` (the moved `make_water_labels` polygon branch) lays each out on `fit_curve` of its
  polygon; a label with under the minimum inside share at its anchor is `Unplaced`, as for seas.

### Style (my call, to tune by eye)

- Italic, in the name's own case, no tracking, one dark warm brown for all four kinds (around OKLCH L 0.40,
  C 0.07, hue ~55, through `fit_gamut`) so they read as terrain and stay apart from the blue water and the
  country colours. White halo at the country labels' alpha (as `RIVER_HALO`), not the water-coloured one: they
  sit on land.
- Sizes as seas (fill `SEA_FILL` of the curve, `WATER_MIN_PT`..`SEA_MAX_PT`) to start with; separate constants
  so they can be tuned independently.
- Z-order: with the water labels (`LABEL_ZORDER - 0.1`), under the country labels. Same style on both map types.

### Pipeline

- `--label` accepts the four kinds and `land` (expanding in place, like `water`). Help text lists them. The
  default (`countries`) is unchanged, so the default map and every existing fingerprint stay as they are.
- Political map (`cli.main`): land labels join the list passed to `place` with the countries and water labels.
- Topographic map (`draw_topographic`): land kinds also start the label stage (today only water kinds do);
  a crop's land labels are placed in `label_outline` like water labels.
- Placement: as water labels (curved): on their own area by glyph centres, `overhang` per glyph, `check_step`
  skips one clashing or hanging off by over `OVERHANG_MAX` (never 'out'), ranked by `full_size` with every other
  label, cut by `CUT_SHARE` like every kind. Overlapping areas need nothing special: each label only has to sit
  on its own.
- `--crop NAME`: `crop.candidates` includes the land kinds asked for (`KINDS` gains `mountain`, `upland`,
  `desert`, `lowland`), so `--label land --crop Alps` works.

### Consequence to be aware of

Ranking is by area across every kind, so the Sahara (~9 M km2) ranks with Russia and Canada, ahead of the Sahel
countries, which then have to fit around it. That is how seas already behave ("the frame's biggest features
win, whatever their kind"); within a rank, `--label` order decides (`countries,land` puts countries first).
The todo's `+`/`-` rank modifiers are the general fix and are not part of this.

## Tests

- Unit: `land.classify`, the title-casing, `label_kinds` expanding `land` in place, a `Style` lookup per kind,
  `load_land_bodies` on a small frame (kinds filtered, duplicates dropped, biggest first).
- `test_import_has_no_pyplot` covers the new modules.
- Regression: a new fingerprint `land`: `--label countries,land` (world, political). `conftest.environment`
  gets a `regions` key (the file size) for that mode only. The refactor commits leave the existing fingerprints
  unchanged; the landform commits add the new one.
- `mypy src` clean.

## Goals, non-goals, anti-goals

- Goals: readable landform names on both map types and in crops; a curved-label module and per-kind style that
  `--world` can extend with new kinds without touching `water.py`.
- Non-goals: peaks and other point features; continents, geoareas, islands, coasts; `--world` itself;
  rank modifiers.
- Anti-goals: any change to existing renders during the refactor; size-based styling; landform labels that look
  like water or country labels.

## Left to the builder

Exact colour and sizes (checked by eye on world and crop renders: Himalayas, Andes, Sahara, Alps crop,
Tarim/Taklamakan overlap), the `Body` type question above, the curved module's exact split, commit order within
"refactor first, then land", and CLAUDE.md/README updates (README only after asking).
