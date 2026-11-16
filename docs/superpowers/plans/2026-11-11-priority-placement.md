# Priority-ordered label placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Place labels tier by tier (5 size ranks per kind, `--label` order within a rank), each tier able to nudge the one before, skipping what doesn't fit, in place of the drop loop.

**Architecture:** Every label gets a frame-independent `full_size` (km² or km). A new module `labels/priority.py` ranks labels per kind, orders them into steps, and runs one `place_labels` solve per step with a per-label freedom (fixed / nudge / free). `place_labels` gains that freedom, a nudge cost and per-label shrink floors; `clashes` becomes symmetric across kinds. `place_with_water` and `place_all` are deleted.

**Tech Stack:** Python 3, numpy, shapely 2, geopandas, pyproj, matplotlib, pytest (+xdist), mypy.

**Spec:** `docs/superpowers/specs/2026-11-11-priority-placement-design.md`

## Global Constraints

- Commits follow `~/.claude/CLAUDE.md`: `X: desc` / `X; *: desc` / `+X`, `X` = filename without extension; trailer `Co-Authored-By: Claude <noreply@anthropic.com>`. Only `git add`, `git commit`, `git mv`. Never amend, push, or switch branch.
- Every commit compiles and runs. Refactor commits leave `tests/fingerprints.json` unchanged; a commit that changes output on purpose re-records with `UPDATE_REFERENCE=1 .venv/bin/pytest` in the same commit and says in the body what changed visually.
- `.venv/bin/pytest` and `.venv/bin/mypy src` pass at every commit.
- `N_RANKS = 5`, `RANK_STEP = 10` (in `labels/priority.py`). Rank 1 is >= biggest/`RANK_STEP`, rank k >= biggest/`RANK_STEP`^k, rank 5 everything smaller.
- Ranks are per kind, relative to the biggest *candidate* of that kind in this frame, using full (unclipped) sizes.
- Kinds: `countries` (disputed self-governing areas included), `territories`, `oceans`, `seas`, `lakes`, `straits`, `rivers`. `water` expands in place to `oceans, seas, lakes, straits, rivers`.
- No new CLI flags. Label sizing, wrapping and in/out decisions are unchanged.
- Don't touch `README.md` or `examples/`. Scratch renders go in the session scratchpad.

## Review Focus

- A kind with no candidates in the frame, or one candidate: steps for it are empty or one; no crash, no empty steps run.
- A label with `full_size` 0 or NaN (no area recorded): ranked 5, never crashes `log`.
- `--label` with repeats (`seas,water`, `countries,countries`): first occurrence wins, nothing placed twice.
- No labels at all (`--label none`; topographic without water kinds; a step where every new label fails): an empty or subset `Placement` draws fine.
- The world Pacific's two halves: both rank 1 with the same full size.
- A river cut by a crop edge, or broken by a lake inside it: its full size is the whole river's, not the visible piece's (Task 3's frame-independence test covers the crop; the Danube's range covers the joins).

---

### Task 1: `--label` keeps its order

**Files:**
- Modify: `src/worldmap/cli.py:51-59,75` (`label_kinds`, `--label` default)
- Modify: `src/worldmap/crop.py:333` (`candidates` hint), `src/worldmap/labels/water.py:64` (`kinds_from_labels` hint)
- Test: `tests/test_cli.py::test_label_flag`

**Interfaces:**
- Produces: `args.label: list[str]`, deduplicated, first occurrence kept, `water` *not* expanded (default `["countries"]`, `none` -> `[]`). `kinds_from_labels(label: Collection[str]) -> set[Kind]` and `crop.candidates(label: Collection[str], disputed: bool)` accept it.

- [ ] **Step 1: Update `test_label_flag`** to expect lists: `[]` -> `["countries"]`; `"countries,territories"` -> `["countries", "territories"]`; `"none"` -> `[]`; `"countries,oceans,seas,lakes,water"` -> that list; add `"seas,countries,seas"` -> `["seas", "countries"]`. Keep the error cases.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_cli.py::test_label_flag -n0` — FAIL (sets).
- [ ] **Step 3: Implement.** `label_kinds` returns `list(dict.fromkeys(kinds))`; default `["countries"]`; `crop.candidates` uses `set(label) & {...}` where it intersects; type hints to `Collection[str]`.
- [ ] **Step 4: Run** `.venv/bin/pytest` and `.venv/bin/mypy src` — all pass, fingerprints unchanged.
- [ ] **Step 5: Commit** `cli; *: --label is an ordered list`

### Task 2: Full sizes of country, territory and disputed labels

**Files:**
- Modify: `src/worldmap/data.py` (`with_km2`, used by `load_countries`, `load_disputed`, `load_lakes`, `load_marine`)
- Modify: `src/worldmap/labels/label.py` (`Label.kind`, `Label.full_size`, `new_label`, `make_labels`)
- Modify: `src/worldmap/labels/regions.py` (`ChildRegion.km2`, `child_regions`)
- Modify: `src/worldmap/disputed.py` (`Dispute.km2`, `load_disputes`, `disputed_label_args`)
- Modify: `src/worldmap/crop.py` (unpacks `extra`)
- Test: `tests/test_label.py` (create), `tests/test_regions.py`

**Interfaces:**
- Produces:
  - `data.with_km2(gdf) -> GeoDataFrame`: adds column `km2` = area in `EQUAL_AREA_CRS` / 1e6 of each lon/lat row. Columns survive `frame.project`.
  - `Label.kind: str = "countries"` (`"territories"` for child regions) and `Label.full_size: float = 0.0` (km²), new trailing dataclass fields; `new_label(..., kind="countries", full_size=0.0)`.
  - `ChildRegion.km2: float`: a dependency's `km2`; a hand-picked region's = area (km²) of the parent's *lon/lat* polygons whose representative point is in its boxes (same test as now, on the unprojected geometry).
  - `Dispute.km2: float` from the row's `km2`; `extra` items become `(text, polygon, fill, km2)`.
  - `make_labels` sets `full_size` from `countries_projected.km2[i]`, the extra's `km2`, or the child's `km2`.

- [ ] **Step 1: Write failing tests** (`tests/test_label.py`, real data, world frame and `crop_frame` for France via the existing crop test helpers):
  - `test_full_size_is_frame_independent`: France's `full_size` is the same (rel 1e-9) on the world map and in a France crop, and in 5.4e5..6.6e5 km².
  - `test_full_size_of_territories`: with `--label territories` children, Alaska (US) is in 1.4e6..1.8e6 km², Tasmania in 6e4..7.5e4.
  - `test_full_size_of_breakaways`: with disputes, every breakaway extra label has `full_size > 0`.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_label.py -n0` — FAIL.
- [ ] **Step 3: Implement** as in Interfaces.
- [ ] **Step 4: Run** `.venv/bin/pytest` and `.venv/bin/mypy src` — pass, fingerprints unchanged.
- [ ] **Step 5: Commit** `label; *: +Label.full_size, the whole country's area`

### Task 3: Full sizes of water labels

**Files:**
- Modify: `src/worldmap/labels/water.py` (`WaterBody.full_size`, `WaterLabel.full_size`, `load_water_bodies`, `load_river_bodies`, `lay_out`)
- Test: `tests/test_water.py`

**Interfaces:**
- Produces: `WaterBody.full_size: float = 0.0`, `WaterLabel.full_size: float = 0.0` (trailing fields; `lay_out` copies it).
  - marine and lake bodies: the row's `km2`;
  - oceans: sum of `km2` over *all* `load_marine()` ocean rows of that `ocean_word` (not only the visible ones); both Pacific halves get the whole Pacific's;
  - rivers: total geodesic length (km, `pyproj.Geod(ellps="WGS84").geometry_length`) of the *river* the body is on: the unclipped `load_rivers()` features of that `river_name` joined into networks the way the labels are (pieces whose ends are within `RIVER_JOIN_M`, and networks `RIVER_REPEAT_M` apart are different rivers). A body takes the length of the unclipped network it lies on. So Natural Earth's segments (split at lakes, reservoirs, junctions) add up to one river, but the two Colorados (US, Argentina) each get their own length.

- [ ] **Step 1: Write failing tests** in `tests/test_water.py`:
  - `test_full_size_of_water`: world frame, all kinds: Lake Victoria in 6e4..7.5e4 km²; the two PACIFIC OCEAN bodies have equal `full_size` > 1e8; Mediterranean Sea > 1e6; Danube length in 2.4e3..3.2e3 km; the two Colorado bodies have different `full_size`s, the US one in 1.8e3..2.6e3 km.
  - `test_water_full_size_is_frame_independent`: the Danube's and the Adriatic's `full_size` equal (rel 1e-9) on the world map and in an Alps box crop.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** as in Interfaces (compute the per-name river lengths and per-ocean areas once in each loader).
- [ ] **Step 4: Run** `.venv/bin/pytest` and `.venv/bin/mypy src` — pass, fingerprints unchanged.
- [ ] **Step 5: Commit** `water: +WaterLabel.full_size, the whole body's area or river's length`

### Task 4: Ranks and steps

**Files:**
- Create: `src/worldmap/labels/priority.py`
- Test: `tests/test_priority.py` (create)

**Interfaces:**
- Produces (pure, no data):
  - `N_RANKS = 5`, `RANK_STEP = 10.0`
  - `label_order(label: Sequence[str]) -> list[str]`: `water` expanded in place to `oceans, seas, lakes, straits, rivers`, duplicates dropped (first wins).
  - `ranks_per_kind(kinds: Sequence[str], sizes: Sequence[float]) -> np.ndarray` (int, 1..`N_RANKS`): `1 + floor(log(biggest_of_kind / size) / log(RANK_STEP))`, clipped; size <= 0 or NaN -> `N_RANKS`.
  - `Ranker = Callable[[Sequence[str], Sequence[float]], np.ndarray]`
  - `steps(kinds: Sequence[str], sizes: Sequence[float], order: Sequence[str], rank: Ranker = ranks_per_kind) -> list[tuple[int, list[int]]]`: `(rank, indices)` for rank 1..5, then kind in `order`, biggest first; empty steps left out; kinds not in `order` left out.

- [ ] **Step 1: Write failing tests**:
  - `test_label_order`: `["countries","water","seas"]` -> `["countries","oceans","seas","lakes","straits","rivers"]`; `["rivers","countries","rivers"]` -> `["rivers","countries"]`.
  - `test_ranks_relative_to_biggest_of_kind`: kinds `["seas"]*4 + ["lakes"]*2`, sizes `[1000, 100.1, 99.9, 1e-9, 5, 0.04]` -> `[1, 1, 2, 5, 1, 3]`.
  - `test_ranks_zero_and_nan`: sizes `[10, 0, nan]` one kind -> `[1, 5, 5]`.
  - `test_steps_order`: kinds `["countries","seas","countries","seas","rivers"]`, sizes `[1, 100, 1000, 1, 5]`, order `["countries","rivers","seas"]` -> `[(1, [2]), (1, [4]), (1, [1]), (3, [3]), (4, [0])]`.
  - `test_steps_empty`: `steps([], [], ["countries"]) == []`.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_priority.py -n0` — FAIL (no module).
- [ ] **Step 3: Implement** as in Interfaces.
- [ ] **Step 4: Run** — pass; mypy clean.
- [ ] **Step 5: Commit** `+priority`

### Task 5: Per-label freedom, nudge cost and floors in `place_labels`

**Files:**
- Modify: `src/worldmap/labels/placement.py` (`CostWeights.nudge`, `place_labels`, `cost_batch`/`water_cost`, `place_with_water`, `place_all`, `subset`)
- Test: `tests/test_placement.py`

**Interfaces:**
- Produces:
  - `FIXED, NUDGE, FREE = 0, 1, 2`
  - `CostWeights.nudge: float = 2.0`: per height (country) or em (water) of distance from `home`, for `NUDGE` labels only.
  - `place_labels(labels, frame, seed=SEED, weights=CostWeights(), water=(), start=None, freedom: np.ndarray | None = None, home: tuple[np.ndarray, np.ndarray] | None = None, floors: np.ndarray | None = None, seed_out: bool | None = None) -> Placement`. `fixed: int` is removed.
    - `freedom` (per label, default all `FREE`): `FIXED` never moves or shrinks; `NUDGE` moves with the nudge cost from `home` (default `start`) and shrinks no lower than its `floors` entry; `FREE` as today.
    - `floors` (pt per label; default `SHRINK_MIN_PT * label_scale` for country labels, `min_size` for water): no label shrinks below its entry.
    - `seed_out` (default `start is None`): seed the *free* 'out' labels in order before descending, as today.
    - The descent visits `movable[rng.permutation(len(movable))]`, movable = non-`FIXED` indices: with a fixed prefix this is the current `fixed + rng.permutation(n - fixed)`, so output is unchanged.
  - `subset(p: Placement, keep: np.ndarray) -> Placement`: the placement of the kept labels only, `parts`/`owner` reindexed.
- `place_with_water`/`place_all` pass `freedom` equivalent to their old `fixed` prefix (they are deleted in Task 7).

- [ ] **Step 1: Write failing tests** (reuse the file's fixtures):
  - `test_fixed_labels_stay_put`: rewritten with `freedom=np.array([FIXED, FREE])`; same assertions.
  - `test_nudged_label_moves_less`: the Atlantis + Sea-on-top setup; the country `NUDGE` (home = its anchor), the sea `FREE`: country's displacement from its anchor is less than with both `FREE`, and > 0 or the sea moved.
  - `test_floors_hold`: an always-clashing (monkeypatched `clashes`) sea of size 14 with `floors=[12.0]` ends at size >= 12.
  - `test_subset`: placement of one country + two water labels, `subset` keeping the country and the second water label: `n == 2`, `owner` only `1`, glyph boxes equal those of the original second label.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_placement.py -n0` — the new ones FAIL.
- [ ] **Step 3: Implement** as in Interfaces (the nudge term goes into both cost functions for `NUDGE` labels).
- [ ] **Step 4: Run** `.venv/bin/pytest` and `.venv/bin/mypy src` — pass, **fingerprints unchanged**.
- [ ] **Step 5: Commit** `placement: per-label freedom (fixed, nudged, free) and shrink floors`

### Task 6: Symmetric clashes

**Files:**
- Modify: `src/worldmap/labels/placement.py` (`clash`, `clashes`, `place_labels` shrink rule, `place_with_water` caller)
- Test: `tests/test_placement.py`; re-record `tests/fingerprints.json`

**Interfaces:**
- Produces:
  - `clash(p: Placement, i: int, among: np.ndarray) -> bool`: label i (in boolean mask `among`) overlaps another label of `among` (box or glyph boxes, any kind), or a leader of an `among` label crosses i's box/glyphs, or i's leader crosses another's box/glyphs. A box covering the leader's own dot doesn't count (as in `cost_batch`).
  - `clashes(p: Placement, among: np.ndarray | None = None) -> list[int]`: every i in `among` (default all) with `clash`.
  - Shrink rounds shrink every non-`FIXED` clashing label, country or water, down to its floor.

- [ ] **Step 1: Write failing tests**: `test_water_pushes_country_but_only_water_shrinks` becomes `test_country_can_shrink_for_water` (a sea fixed on top: the free country now shrinks, or moves, and `clashes(p) == []` or it is at its floor); `test_clash_is_symmetric`: two overlapping labels, one country one water, both in `clashes`; a leader through a country box makes that box clash.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement**; `place_with_water` uses `[i for i in clashes(p) if i >= n_country]` for water clashing.
- [ ] **Step 4: Run** `UPDATE_REFERENCE=1 .venv/bin/pytest`, then `.venv/bin/pytest` — pass; mypy clean. Compare `.cache/reference/*.png` with the previous ones (copy them to the scratchpad *before* re-recording) by cropping Europe, the Caribbean and the Pacific crop; note what changed.
- [ ] **Step 5: Commit** `placement; *: clashes are the same for every kind` (body: what changed visually; includes `fingerprints.json`).

### Task 7: Place by priority

**Files:**
- Modify: `src/worldmap/labels/priority.py` (`place_by_priority`, `NUDGE_MIN_SCALE`)
- Modify: `src/worldmap/labels/placement.py` (delete `place_with_water`, `place_all`, `water_touching`; keep `outside_outline`)
- Modify: `src/worldmap/cli.py:186-198,225-231` (both label stages)
- Test: `tests/test_priority.py`, `tests/test_placement.py` (delete the drop-loop tests); re-record `tests/fingerprints.json`

**Interfaces:**
- Consumes: `steps`, `label_order` (Task 4); `place_labels(freedom, home, floors, seed_out)`, `subset`, `FIXED/NUDGE/FREE` (Task 5); `clash` (Task 6); `full_size`, `Label.kind` (Tasks 2-3).
- Produces: `NUDGE_MIN_SCALE = 0.9`; `place_by_priority(labels: list[Label], water: list[WaterLabel], frame: MapFrame, order: Sequence[str]) -> tuple[Placement, list[Label], list[WaterLabel]]` (placement of the kept labels, country first then water, as the draw functions expect).

Algorithm (per step `(r, new)`):
1. Placement items = kept country labels + new country labels, then kept water + new water.
2. Freedom: new `FREE`; kept of rank `r` or `r - 1` `NUDGE` (home = where they settled after their own step, floor = max(normal floor, `NUDGE_MIN_SCALE` x their settled size)); other kept `FIXED`. Start: current positions; new labels at their anchors; `seed_out=True`.
3. After the solve, record every non-fixed label's position and size. Go through the new labels **smallest first**; one fails if it `clash`es among the kept + not-yet-skipped labels, or (water) its `inside_share` < `WATER_MIN_INSIDE`, or (crop) it is in `outside_outline`. Failures are skipped for good; no re-solve.
4. Kept new labels settle (home and size recorded).
Print the skipped names per kind once at the end.

cli: `place_by_priority(labels, water, frame, label_order(args.label))`, then `label.text.remove()` for every country label not kept; the topographic stage passes `[]` labels and `room`.

- [ ] **Step 1: Delete** `test_drop_least_important`, `test_off_water_is_dropped`, `test_country_labels_never_dropped`, `test_no_water_is_place_labels`, `test_drop_loop_ends_when_nothing_can_shrink`, `test_rivers_give_way_to_countries`, `test_crop_drops_water_labels_outside_the_outline` from `test_placement.py`; **write** in `test_priority.py` (fixtures moved to `tests/conftest.py` or imported from `test_placement`):
  - `test_smaller_of_a_clashing_pair_is_skipped`: Sea and Bay (crossing_water_labels), `full_size` 100 and 10, same kind: only Sea kept, `clashes == []`.
  - `test_off_water_is_skipped`: as the old test.
  - `test_country_can_lose_to_a_bigger_sea`: Atlantis (`full_size` 1) under a sea filling its whole country box with `full_size` 1e6, order `["seas","countries"]`, the country at its floor: no country label kept.
  - `test_label_order_breaks_ties_within_a_rank`: the same pair at rank 1 each: with `["countries","seas"]` the country is kept, with `["seas","countries"]` the sea is.
  - `test_two_ranks_down_is_fixed`: a country of rank 1 and a sea of rank 3 on top of it: the country's position equals where it was after its own step.
  - `test_crop_skips_water_labels_outside_the_outline`: as the old test.
  - `test_nothing_to_place`: `place_by_priority([], [], frame, ["countries"])` gives `n == 0`, two empty lists.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_priority.py -n0` — FAIL.
- [ ] **Step 3: Implement** `place_by_priority`, switch the cli, delete the old functions.
- [ ] **Step 4: Time it**: `time .venv/bin/make_map $SCRATCH/w.png --label countries,water` before (on Task 6's commit) and after; record both. Then `UPDATE_REFERENCE=1 .venv/bin/pytest`, `.venv/bin/pytest`, mypy. Compare renders (world `--label countries,water`, `crop_pacific_water`, `topographic_water`, `crop_water`) with the Task 6 ones by eye.
- [ ] **Step 5: Commit** `priority; *: +place_by_priority, in place of the drop loop` (body: visual changes, timing).

### Task 8: Every drawn river is a candidate

**Files:**
- Modify: `src/worldmap/labels/water.py` (`load_river_bodies` filters; `RIVER_LABEL_MAX_RANK` keeps only its role in `river_size`, renamed `RIVER_SIZE_MAX_RANK` with its comment updated)
- Test: `tests/test_water.py`; re-record fingerprints

- [ ] **Step 1: Write failing test** `test_every_drawn_river_is_a_candidate`: in an Alps box crop, some river body has `rank > 6` (Natural Earth scalerank).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement**: drop both `RIVER_LABEL_MAX_RANK` checks in `load_river_bodies`.
- [ ] **Step 4: Run** `UPDATE_REFERENCE=1 .venv/bin/pytest`, `.venv/bin/pytest`, mypy; look at `crop_water` and `topographic_crop`.
- [ ] **Step 5: Commit** `water: every drawn river may be named` (body: visual change).

### Task 9: Tune by eye, then the docs

**Files:**
- Modify: `src/worldmap/labels/priority.py` (`RANK_STEP`, `NUDGE_MIN_SCALE`), `src/worldmap/labels/placement.py` (`CostWeights.nudge`) only if the renders call for it
- Modify: `CLAUDE.md` (Pipeline step 3, Label placement notes, Water labels' solver/drop-loop paragraphs, River labels' placement paragraph, `labels/priority.py` row in the module table, `--label` being a list)

- [ ] **Step 1:** Render the world (`--label countries,water`), `--crop pacific --label countries,oceans,seas`, `--crop France --label countries,water`, an Alps topographic crop with `--label water`, at `--dpi 50`, into the scratchpad. Show them to the user with any constant changes proposed; change constants only with their OK, one commit per constant (`priority: RANK_STEP 10 -> N`, re-recording fingerprints).
- [ ] **Step 2:** Update `CLAUDE.md` to describe the step loop and remove the drop loop's description. Commit `CLAUDE: priority placement`.
