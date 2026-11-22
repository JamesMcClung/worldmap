# `--size` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--size IN` option for the poster's long side, give crops a default size that shrinks with their region,
and make every text size absolute (no `label_scale`).

**Architecture:** `MapFrame` gets a `size_in` field, which replaces the `POSTER_WIDTH_IN` constant in
`width_in`/`height_in`. Everything else (`px_per_unit`, the figure, the raster) already derives from those. `crop.py`
sets a crop's default `size_in` where it set `label_scale` before, and `cli.make_map_frame` overrides it with `--size`.

**Tech Stack:** Python, matplotlib, shapely/pyproj, pytest (+xdist).

**Spec:** `docs/superpowers/specs/2026-11-22-size-option-design.md`

## Global Constraints

- The world map stays 36 in (`POSTER_WIDTH_IN`). The world fingerprints must not change in any task.
- Crop default: size = 36 × (*s*/36)^`CROP_SIZE_POWER`, at least `CROP_MIN_IN`, with starting values
  `CROP_SIZE_POWER = 0.4` and `CROP_MIN_IN = 6`. *s* is the crop's long side measured on the 36 in world poster. The floor
  applies only to the default.
- `--dpi` stays as it is: layout is measured at `DPI` = 200.
- Inch-based constants stay physical: `LAKE_MIN_IN2`, `STRIPE_WIDTH_IN`, colour bars, `--pin-size`.
- No sizes in map km, and no river-width data.
- Commit messages follow the user's scheme (`X; *: desc`, `+Y`/`-Y`), each with a `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` trailer.
  Only `git add`/`commit`/`mv`; never amend.
- A commit that changes output re-records with `UPDATE_REFERENCE=1 .venv/bin/pytest` and says in its body what changed visually.
  A refactor leaves `tests/fingerprints.json` unchanged.
- `CLAUDE.md` is updated in the same commit as the change it describes. `README.md` and `examples/` are left alone.
- `.venv/bin/mypy src` stays clean.

## Review Focus

1. **A crop wider than the world poster.** An Equal Earth crop (the Pacific) can measure about as wide as the Robinson globe, or a
   little wider in metres. Expected: the default never exceeds 36 in. Pinned by `test_default_size_capped_at_poster_width` (Task 4).
2. **`--size` with a crop.** An explicit `--size` must win over the crop default, including a value below `CROP_MIN_IN`.
   Pinned by `test_size_overrides_crop_default` (Task 3).
3. **A tall crop.** `size_in` is the *long* side, so a crop taller than wide gets `height_in == size_in`. Pinned by
   `test_tall_frame_long_side_is_size_in` (Task 2).
4. **The topographic Alps crop at its new default size** (~7 in). It must still read the four 15" tiles its environment records,
   or the fingerprint skips forever. Expected: it still reads them (1' cells are ~3 px there, above `FINE_ABOVE_PX` = 2). Check
   in Task 4 that the run prints no fallback note and that the test runs rather than skips.
5. **`--pick` after a `--size` render.** It shows the PNG stretched to the frame's extent, so pins still land in the right
   place. Only the pin preview's size depends on `size_in`. No test: check by hand once (Task 3).

---

### Task 1: Remove `label_scale`

**Files:**
- Modify: `src/worldmap/frame.py` (the `label_scale` field)
- Modify: `src/worldmap/crop.py` (`LABEL_SCALE_MAX`, `with_label_scale` and its five call sites: return the bare frame)
- Modify: `src/worldmap/labels/label.py` (`Label.label_scale`, `min_size`, `new_label`'s `s`, the `SHRINK_MIN_PT` comment)
- Modify: `src/worldmap/labels/water.py` (`RIVER_SCALE_MAX`, `WaterLabel.label_scale`, `lay_out`'s `label_scale` parameter, `river_label`, `make_water_labels`)
- Modify: `tests/test_crop.py`, `tests/test_water.py`, `tests/test_label.py`, `tests/test_placement.py`, `tests/fingerprints.json`, `CLAUDE.md`

**Interfaces:**
- Produces: `lay_out(body, curve, size, geom_px, rise=None) -> WaterLabel` (no `label_scale`).
  `Label.min_size == SHRINK_MIN_PT`, and `WaterLabel.min_size` is `OCEAN_MIN_PT` or `WATER_MIN_PT`, with no multiplier.
  `crop_frame`/`centre_frame` return frames with no size set (Task 4 adds one).

- [ ] **Step 1: Rewrite the tests that pin scaling into ones that pin absolute sizes**
  - `test_crop.py`: delete `test_label_scale_grows_with_zoom`. Rename `test_label_sizes_scale_in_a_crop` to
    `test_label_sizes_are_absolute_in_a_crop`, asserting `size == pytest.approx(MAX_PT)` and `pytest.approx(MIN_PT)`.
  - `test_water.py`: rename `test_river_names_stay_small_in_a_crop` to `test_river_names_absolute_in_a_crop`
    (same Alps frame): `max(r.size) <= RIVER_MAX_PT` and `max(r.min_size) == WATER_MIN_PT`. Drop the `RIVER_SCALE_MAX` import.
  - `test_label.py::test_label_min_size`: only `label.min_size == SHRINK_MIN_PT`.
  - `test_placement.py`: delete `test_water_floor_scales_with_the_frame`.
- [ ] **Step 2: Run** `.venv/bin/pytest -n0 tests/test_crop.py tests/test_water.py tests/test_label.py -k "absolute or min_size"`. Expected: FAIL (sizes are 2.5× in the crop).
- [ ] **Step 3: Remove the field, the constants and every multiplication** listed under Files. `grep -rn label_scale src tests` should come back empty.
- [ ] **Step 4: Run** the step 2 command (PASS), then `.venv/bin/mypy src` (clean), then `UPDATE_REFERENCE=1 .venv/bin/pytest`. Expected:
  only `crop`, `crop_water` and `crop_pacific_water` change. `topographic_crop` has no labels, so it stays the same.
  Confirm with `git diff tests/fingerprints.json`.
- [ ] **Step 5: Look at** `.cache/reference/crop.png` against what's committed (the labels should be smaller), and update `CLAUDE.md`:
  the `frame.py` row, the `--crop` "Labels" bullet, and the river-label "capped at … `RIVER_SCALE_MAX`" line.
- [ ] **Step 6: Commit** `frame; *: -label_scale`. The body says that crop labels now use the world map's point limits,
  so they're up to 2.5× smaller, and that many countries hit `MAX_PT`.

### Task 2: `MapFrame.size_in`

**Files:**
- Modify: `src/worldmap/frame.py` (`MapFrame`, `width_in`, `height_in`)
- Test: `tests/test_frame.py`

**Interfaces:**
- Produces: `MapFrame.size_in: float = POSTER_WIDTH_IN`, the long side in inches. `width_in` = `size_in` if
  `width >= height` else `size_in * width / height`, and `height_in` mirrors it.

- [ ] **Step 1: Write the tests** in `test_frame.py`:
  - `test_size_in_sets_long_side`: `f = replace(make_frame(), size_in=18)`, then `f.width_in == 18` and
    `f.px_per_unit == pytest.approx(make_frame().px_per_unit / 2)`.
  - Rename `test_tall_frame_long_side_is_poster_width` to `test_tall_frame_long_side_is_size_in`, adding a
    `size_in=10` case: `height_in == 10` and `width_in == 5`.
- [ ] **Step 2: Run** `.venv/bin/pytest -n0 tests/test_frame.py`. Expected: FAIL (`size_in` is an unexpected keyword).
- [ ] **Step 3: Add the field** after `clip`, and use it in `width_in`/`height_in`. Update the docstring.
- [ ] **Step 4: Run** `tests/test_frame.py` (PASS), then the full `.venv/bin/pytest`. Expected: all pass, fingerprints unchanged.
- [ ] **Step 5: Commit** `frame: +size_in`.

### Task 3: `--size`

**Files:**
- Modify: `src/worldmap/cli.py` (`parse_args`, `make_map_frame`)
- Test: `tests/test_cli.py`
- Modify: `CLAUDE.md` (the usage line, and `--dpi` "is the poster scaled down")

**Interfaces:**
- Consumes: `MapFrame.size_in` (Task 2).
- Produces: `args.size: float | None` (`--size IN`, `type=positive_float`, `default=None`). `make_map_frame(args)` returns
  `replace(frame, size_in=args.size)` when it's set, for the world map and for every crop form.

- [ ] **Step 1: Write the tests** in `test_cli.py`:
  - `test_size_parses`: `parse_args([]).size is None`, `parse_args(["--size", "12"]).size == 12`, and `"0"`, `"-3"` and `"x"`
    each raise `SystemExit` (the same pattern as the `--dpi` test).
  - `test_size_sets_the_world_frame`: `make_map_frame(parse_args(["--size", "18"])).width_in == 18`.
  - `test_size_overrides_crop_default`: `make_map_frame(parse_args(["--crop", "-5:9,42:51", "--size", "3"]))` has
    `max(width_in, height_in) == pytest.approx(3)`, which is below `CROP_MIN_IN`.
- [ ] **Step 2: Run** `.venv/bin/pytest -n0 tests/test_cli.py -k size`. Expected: FAIL (unrecognised argument).
- [ ] **Step 3: Add the flag.** Help text: "the poster's long side in inches (default 36 for the world map; a crop's grows
  with its region, at least 6); text sizes are fixed, so a bigger poster fits more labels". Change `--dpi`'s and
  `--pin-size`'s "on the 36 in poster" to "on the poster". In `make_map_frame`, apply `args.size` after the frame is built
  (move the existing body into a helper, or wrap the returns, whichever reads better).
- [ ] **Step 4: Run** `.venv/bin/pytest -n0 tests/test_cli.py` (PASS) and the full suite (fingerprints unchanged). Then by hand,
  in the scratchpad: `make_map w.png --size 12 --dpi 50`. Expected: 600 px wide, with fewer labels than the world map.
  Then `--pick` on it with the same `--size`: a pin clicked on Paris lands on Paris.
- [ ] **Step 5: Commit** `cli; *: +--size`.

### Task 4: The crop's default size

**Files:**
- Modify: `src/worldmap/crop.py` (`CROP_SIZE_POWER`, `CROP_MIN_IN`, `default_size_in`, and the five returns that Task 1 left bare)
- Test: `tests/test_crop.py`
- Modify: `tests/fingerprints.json`, `CLAUDE.md` (the `--crop` "Size" bullet)

**Interfaces:**
- Consumes: `MapFrame.size_in` (Task 2).
- Produces: `default_size_in(frame: MapFrame) -> float`. *s* = `POSTER_WIDTH_IN * max(frame.width, frame.height) / make_frame().width`
  (both in metres), and the result is `min(POSTER_WIDTH_IN, max(CROP_MIN_IN, POSTER_WIDTH_IN * (s / POSTER_WIDTH_IN) ** CROP_SIZE_POWER))`.
  `crop_frame` and `centre_frame` return `replace(frame, size_in=default_size_in(frame))`.

- [ ] **Step 1: Write the tests** in `test_crop.py`:
  - `test_default_size_floor`: a Dead Sea box crop (`box_region((35.3, 31.0, 35.7, 31.8))`, 0.1) has a long side `== CROP_MIN_IN`.
  - `test_default_size_grows_with_region`: France (`(-5, 42, 9, 51)`) < Europe (`(-25, 34, 45, 72)`) < US (`(-179, 18, -66, 72)`),
    with France between 8 and 12 in.
  - `test_default_size_capped_at_poster_width`: `w = make_frame()`, then `default_size_in(replace(w, xmax=w.xmin + 1.1 * w.width)) == POSTER_WIDTH_IN`
    (a frame wider than the globe; a whole-world box crop isn't usable here: `round_pole` turns it into a polar square).
  - `test_world_size_unchanged`: `make_frame().size_in == POSTER_WIDTH_IN`.
- [ ] **Step 2: Run** `.venv/bin/pytest -n0 tests/test_crop.py -k size`. Expected: FAIL (every crop is 36 in).
- [ ] **Step 3: Implement** `default_size_in` and the constants (`CROP_SIZE_POWER = 0.4`, `CROP_MIN_IN = 6`), and apply it in
  `crop_frame`/`centre_frame`. `centre_recipe` compares map-unit extents, so it needs no change.
- [ ] **Step 4: Run** the step 2 command (PASS) and `.venv/bin/mypy src` (clean).
- [ ] **Step 5: Tuning checkpoint (the user decides).** Render into the scratchpad at the default size, with
  `--label countries,water`: `--crop France`, `--crop "dead sea"`, `--crop Europe` (or the Europe box), `--crop 160:-150,48:72`,
  `--crop pacific`. Report each one's inch size, and show the PNGs to the user (using `Read`). Change `CROP_SIZE_POWER`/`CROP_MIN_IN`
  if they ask, and adjust step 1's France bounds to match.
- [ ] **Step 6: Re-record** with `UPDATE_REFERENCE=1 .venv/bin/pytest`. Expected: the four crop fingerprints change, `topographic_crop` runs
  (doesn't skip) and prints no tile fallback note, and the world fingerprints are unchanged.
- [ ] **Step 7: Update `CLAUDE.md`** (the "Size: the long side is `POSTER_WIDTH_IN`" bullet → the formula, its constants and the tuned values;
  the topographic memory figures for crops, which are now smaller) and **commit** `crop; *: default crop size grows with its region`.
  The body gives the formula, the tuned values and the sizes of the test crops.
