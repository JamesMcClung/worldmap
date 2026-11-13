# Label-driven features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop the fixed feature filters (river scalerank, lake km2). Lakes are drawn above a pixel-area threshold,
river lines are drawn according to which river labels place, and a saturated label kind is cut off at the first
rank where too many of its labels fail.

**Architecture:** `place_by_priority` learns about *unfit* features: features of a labelled kind whose label could
not even be laid out (a river shorter than its name, a bay too narrow for it). They are ranked with the rest and count
as failures. After each (rank, kind) group it checks the share that failed. At the first group with
>= `CUT_MIN_DROPS` failures making up >= `CUT_SHARE` of the group, it rolls the placement back to before that group,
and that kind is *cut* at that rank: no later rank of it is tried. A feature category whose drawing follows its labels
(rivers today) draws every feature whose rank is above its kind's cut. Features without a label rank `N_RANKS + 1`,
so they are drawn only when the kind was not cut. Lakes are not label-driven: they are drawn when their projected
area is >= `LAKE_MIN_PX` poster pixels.

**Tech Stack:** Python, numpy, shapely/geopandas, matplotlib; pytest with fingerprint regression tests.

**Spec:** this conversation's decisions, restated in Global Constraints below.

## Global Constraints

- Cut: a (rank, kind) group triggers when `failed >= CUT_MIN_DROPS = 5` **and** `failed / total >= CUT_SHARE = 0.3`,
  where failed = the group's labels not kept + its unfit features, and total = the group's labels + its unfit features.
- When a group triggers a cut, all of that group's labels are dropped (including the ones that placed), along with every later rank of that kind. The
  placement is restored to what it was before the group was solved: positions, sizes, `out` flags, and which labels are active or settled.
- `countries` and `territories` are never cut (`UNCUT`).
- A drawn feature is shown iff its rank < its kind's cut rank. With no cut every rank is shown. A feature with no label
  (an unnamed river line, or a kind that isn't in `--label`) ranks `N_RANKS + 1` = 6: shown iff its kind was not cut.
  A side branch of a named river takes the rank of the nearest body of that name.
- Unfit features get no special treatment. They are ranked by `full_size` together with every label, and each one
  counts as a failure in its group.
- Lakes: drawn iff projected area x `frame.px_per_unit ** 2` >= `LAKE_MIN_PX = 10` (poster pixels at layout `DPI`).
  `DRY_LAKES` stays. Lake labels can still be cut like any water kind, but that doesn't change which lakes are drawn.
- Rivers: no scalerank filter at all (Natural Earth goes to 12). Line widths still ramp from rank 0 to rank 8, then stay
  at `RIVER_WIDTH_MIN`.
- Every task ends with `.venv/bin/pytest` green and `.venv/bin/mypy src` clean.
- If a task changes the output, re-record the fingerprints (`UPDATE_REFERENCE=1 .venv/bin/pytest`) and commit
  `tests/fingerprints.json` in the same commit, saying in the body what changed visually.
- Commit messages follow the user's global CLAUDE.md scheme (`X; *: desc`, Co-Authored-By trailer). Never amend. Use
  `git add`/`commit` only.

## Review Focus

- Rolling back after a cut: the labels kept before the cut group must end up exactly where they would be if that group
  didn't exist (it may have nudged them). Pinned in Task 2.
- A group made only of unfit features (no placeable label of that kind in that rank) must still be able to trigger a cut. Pinned in Task 2.
- `--rivers` without `--label rivers`, and `--type topographic` with `--rivers`: there are no river labels, so nothing is
  cut and every line is drawn. Pinned in Task 5 (`draws_all_rivers_without_labels`).
- A crop where no lake reaches `LAKE_MIN_PX`: an empty lakes GeoDataFrame must flow through `draw_lakes`,
  `subtract_lakes` and `load_water_bodies`. Pinned in Task 1.
- 100% failure with fewer than 5 failures: no cut. Pinned in Task 2.

---

### Task 1: Lakes drawn by pixel area

**Files:**
- Modify: `src/worldmap/render.py` (add `LAKE_MIN_PX`, `drawable_lakes`)
- Modify: `src/worldmap/data.py` (remove `LAKE_MIN_KM2` and its filter from `load_lakes`; it now returns every lake except `DRY_LAKES`)
- Modify: `src/worldmap/cli.py`, `src/worldmap/topographic.py:409`, `src/worldmap/crop.py:365` (callers project with `drawable_lakes`)
- Modify: `tests/test_water.py:19,240,246`, `tests/test_regions.py:24`, `tests/test_topographic.py:214` as needed
- Test: `tests/test_render.py` (create it if it doesn't exist)

**Interfaces:**
- Produces: `render.drawable_lakes(frame: MapFrame) -> gpd.GeoDataFrame`, i.e. `frame.project(load_lakes())` keeping the rows with
  `area * frame.px_per_unit ** 2 >= LAKE_MIN_PX`. `LAKE_MIN_PX = 10`.

- [ ] **Step 1: Failing tests**

```python
def test_drawable_lakes_by_pixel_area() -> None:
    world = drawable_lakes(make_frame())
    px = world.area * make_frame().px_per_unit ** 2
    assert (px >= LAKE_MIN_PX).all() and "Lake Victoria" in set(world.name)
    assert len(world) > N  # N: well above the count at LAKE_MIN_KM2 = 2000; read both off the data
    alps = drawable_lakes(crop_frame(box_region(parse_box("5:11,44:48")), 0))
    assert "Lake Geneva" in set(alps.name)  # 580 km2: too small for the world map's old 2000 km2

def test_drawable_lakes_may_be_none() -> None:
    sea = centre_frame(parse_centre("-30,0/2,2"))  # mid-Atlantic
    lakes = drawable_lakes(sea)
    assert lakes.empty
    subtract_lakes(sea.project(load_countries()), lakes)  # no error on empty
```

Check `world`'s count and the lake names against the data before you fix the numbers in the asserts.

- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_render.py -n0`. Expected: ImportError on `drawable_lakes`.
- [ ] **Step 3:** Implement `drawable_lakes`, remove `LAKE_MIN_KM2`, and switch every caller of `frame.project(load_lakes())`
  to `drawable_lakes(frame)`.
- [ ] **Step 4:** Run `.venv/bin/pytest -n0 tests/test_render.py tests/test_water.py tests/test_regions.py`. Expected: PASS.
- [ ] **Step 5:** Re-record: `UPDATE_REFERENCE=1 .venv/bin/pytest`. Expect most fingerprints to change (many more lakes).
  Compare `.cache/reference/crop.png` by eye: the French lakes should be visible.
- [ ] **Step 6:** Commit `render; *: lakes drawn above LAKE_MIN_PX poster pixels, -LAKE_MIN_KM2` with the fingerprints.

### Task 2: Unfit features and the saturation cut in `place_by_priority`

**Files:**
- Modify: `src/worldmap/labels/priority.py`
- Modify: `src/worldmap/labels/placement.py` (add `Placement.copy`)
- Test: `tests/test_priority.py`

**Interfaces:**
- Produces, in `priority.py`:
  - `CUT_SHARE = 0.3`, `CUT_MIN_DROPS = 5`, `UNCUT = ("countries", "territories")`
  - `saturated(failed: int, total: int) -> bool`: the Global Constraints test
  - `class Unplaced(NamedTuple): name: str; kind: str; full_size: float`, a feature of a labelled kind with no label that fits
  - `place_by_priority(items, frame, order, rank=ranks_by_quantile, on_step=None, unfit: Sequence[Unplaced] = ()) -> Outcome`
  - `Outcome` gains `ranks: np.ndarray` (int, per entry of `[*items, *unfit]`) and `cut: dict[str, int]` (kind ->
    the rank it was cut at). `plan` and `kept` now index `[*items, *unfit]`; `kept` is False for every unfit entry.
    `placement` is unchanged (the kept items only).
  - `Outcome.shown(kind: str, ranks: np.ndarray) -> np.ndarray`: bool mask, `ranks < self.cut.get(kind, N_RANKS + 2)`.
  - `UNLABELED = N_RANKS + 1`
- Produces, in `placement.py`: `Placement.copy() -> Placement`. Every array field is copied; `items` is shared.

- [ ] **Step 1: Failing tests** (helpers `sized`, `atlantis`, `crossing_water_labels`, `FRAME` already in the file)

```python
def unfit(n: int, kind: str = "seas") -> list[Unplaced]:
    return [Unplaced(f"Bay {k}", kind, 1.0) for k in range(n)]

def test_cut_drops_the_whole_group_and_later_ranks() -> None:
    sea, bay = crossing_water_labels(FRAME)  # they don't clash once apart; both place alone
    one = lambda kinds, sizes: np.array([1, 2] + [1] * 5)
    out = place_by_priority([sized(sea, 100), sized(bay, 10)], FRAME, ["seas"], rank=one, unfit=unfit(5))
    assert out.cut == {"seas": 1} and not out.kept.any() and out.placement.n == 0
    assert list(out.shown("seas", np.array([1, 2, UNLABELED]))) == [False, False, False]

def test_no_cut_below_the_minimum_drops() -> None:
    sea, _ = crossing_water_labels(FRAME)
    out = place_by_priority([sized(sea, 100)], FRAME, ["seas"], rank=lambda k, s: np.ones(5, int), unfit=unfit(4))
    assert out.cut == {} and out.kept[0] and list(out.shown("seas", np.array([5, UNLABELED]))) == [True, True]

def test_saturated() -> None:
    assert saturated(5, 16) and not saturated(5, 17)  # 31% and 29%
    assert not saturated(4, 4) and saturated(5, 5)  # CUT_MIN_DROPS

def test_unfit_only_group_cuts() -> None:
    sea, _ = crossing_water_labels(FRAME)
    ranks = lambda kinds, sizes: np.array([2] + [1] * 5)
    out = place_by_priority([sized(sea, 1)], FRAME, ["seas"], rank=ranks, unfit=unfit(5))
    assert out.cut == {"seas": 1} and not out.kept[0]

def test_countries_are_never_cut() -> None:
    out = place_by_priority([atlantis(1)], FRAME, ["countries"], rank=lambda k, s: np.ones(6, int), unfit=unfit(5, "countries"))
    assert out.cut == {} and out.kept[0]

def test_cut_rolls_back_what_the_group_nudged() -> None:
    sea, bay = crossing_water_labels(FRAME)
    alone = place_by_priority([atlantis(1), sized(sea, 1)], FRAME, ["countries", "seas"], rank=lambda k, s: np.array([1, 2])).placement
    p = place_by_priority([atlantis(1), sized(sea, 1)], FRAME, ["countries", "seas"],
                          rank=lambda k, s: np.array([1, 2] + [2] * 5), unfit=unfit(5)).placement
    assert p.n == 1 and (p.x[0], p.y[0], p.size[0]) == (alone.x[0], alone.y[0], alone.size[0])
```

`test_cut_rolls_back_what_the_group_nudged` only proves something if the sea actually moves Atlantis when it's
placed. Pick positions where it does: check that `alone`'s Atlantis differs from Atlantis placed without the sea.
If it doesn't, build the sea on top of Atlantis's anchor with `water_label`.

- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_priority.py -n0`. Expected: ImportError on `Unplaced`.
- [ ] **Step 3: Implement.** `steps` is unchanged and runs over `[*items, *unfit]`'s kinds and sizes, so an
  unfit-only (rank, kind) group gets its own step. In the step loop, indices `>= len(items)` are unfit. Skip solving
  when a step has no items. Skip every step whose kind is already in `cut`. Take a snapshot (`p.copy()`, `active`,
  `settled`, `home_x`, `home_y`, `settled_size`) before solving the step. After the step, if its kind isn't in `UNCUT`
  and `saturated(failed, total)`, restore the snapshot, deactivate the step's items and record `cut[kind] = r`.
  `solve` closes over `p`, so restore by assigning the arrays back into `p` (or make `p` `nonlocal`).
- [ ] **Step 4:** Run `.venv/bin/pytest tests/test_priority.py tests/test_report.py -n0`. Expected: PASS. The report
  tests use `outcome.plan`/`kept`, and with no unfit those are unchanged.
- [ ] **Step 5:** Re-record the fingerprints. The water modes may change if a group of seas or rivers trips the cut. In
  the body, name the kinds and ranks cut (see the rank table printed by `make_map --label countries,water`).
- [ ] **Step 6:** Commit `priority; *: +unfit features and the saturation cut (CUT_SHARE, CUT_MIN_DROPS)`.

### Task 3: The report shows unfit features and cuts

**Files:**
- Modify: `src/worldmap/labels/report.py`, `src/worldmap/cli.py` (`place` passes `unfit` through and returns the `Outcome`)
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `Unplaced`, `Outcome.cut`, `Outcome.ranks` (Task 2)
- Produces: `summary(items, unfit, outcome, order, color, columns) -> str`. Unfit entries count in the table as
  dropped and appear by name in the dropped column. A line `cut: rivers from rank 4, seas from rank 5` sits between the
  skipped list and the table, left out when nothing was cut. `cli.place(items, frame, order, unfit=()) -> Outcome`.

- [ ] **Step 1: Failing test**

```python
def test_summary_shows_unfit_and_cuts() -> None:
    sea, bay = crossing_water_labels(FRAME)
    bays = [Unplaced(f"Bay {k}", "seas", 1.0) for k in range(5)]
    outcome = place_by_priority([sized(sea, 100)], FRAME, ["seas"], rank=lambda k, s: np.ones(6, int), unfit=bays)
    lines = summary([sea], bays, outcome, ["seas"], color=False, columns=None).split("\n")
    assert lines[1] == "cut: seas from rank 1"
    assert lines[4].startswith("seas") and "0/6" in lines[4]
```

- [ ] **Step 2:** Run it. Expected: TypeError (summary's arguments).
- [ ] **Step 3:** Implement. `rank_table` already works with any `kinds`/`names` sequences, so pass it those of
  `[*items, *unfit]`. Update `cli.place` and its two callers. They don't pass `unfit` yet.
- [ ] **Step 4:** Run `.venv/bin/pytest -n0 tests/test_report.py tests/test_cli.py`. Expected: PASS. The fingerprints don't change.
- [ ] **Step 5:** Commit `report; *: summary shows unfit features and cuts`.

### Task 4: `make_water_labels` reports the bodies that don't fit

**Files:**
- Modify: `src/worldmap/labels/water.py` (`make_water_labels`), `src/worldmap/cli.py` (both water label call sites)
- Test: `tests/test_water.py`

**Interfaces:**
- Produces: `make_water_labels(bodies, frame) -> list[WaterLabel | None]`, one per body in order, None where the name
  doesn't fit (too short a river, under `WATER_MIN_INSIDE`). `water.unplaced(body: WaterBody) -> Unplaced`.
- cli: `fits = make_water_labels(bodies, frame)`, `water = [w for w in fits if w]`,
  `unfit = [unplaced(b) for b, w in zip(bodies, fits) if w is None]`, passed to `place`.

- [ ] **Step 1: Failing test**

```python
def test_make_water_labels_one_per_body() -> None:
    f = make_frame()
    bodies = [b for b in load_water_bodies({"rivers"}, lakes(), f) if b.text in ("Nile", "Suez Canal")]
    fits = make_water_labels(bodies, f)
    assert len(fits) == 2 and {b.text: w is None for b, w in zip(bodies, fits)} == {"Nile": False, "Suez Canal": True}
```

- [ ] **Step 2:** Run it. Expected: FAIL (the list has 1 entry).
- [ ] **Step 3:** Implement and update the other callers of `make_water_labels` (grep tests too: they now filter Nones).
- [ ] **Step 4:** `.venv/bin/pytest`. Expect fingerprint mismatches in the water modes, because small bays and short rivers now
  count as failures and may cut. Re-record, and say which cuts appeared in the body.
- [ ] **Step 5:** Commit `water; *: make_water_labels one per body, the unfit ones counted by place_by_priority`.

### Task 5: River lines drawn by their labels' ranks

**Files:**
- Modify: `src/worldmap/labels/water.py` (add `river_ranks`), `src/worldmap/cli.py` (draw rivers after placement, in
  both the political and topographic paths)
- Test: `tests/test_water.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `Outcome.ranks`, `Outcome.shown`, `UNLABELED` (Task 2); `fits`/`unfit` (Task 4)
- Produces: `river_ranks(rivers_projected: gpd.GeoDataFrame, bodies: list[WaterBody], ranks: Sequence[int]) -> np.ndarray`.
  It gives each line the rank of the nearest body whose text equals `river_name(line.name_en)`, or `UNLABELED` when there is
  none (unnamed, or rivers not labelled). `cli.body_ranks(fits, outcome, first: int) -> list[int]` maps each body to its
  entry in `outcome.ranks` (labels from index `first`, the unfit ones after all items).
- cli: `draw_rivers(ax, rivers_p[outcome.shown("rivers", river_ranks(rivers_p, river_bodies, ranks))])`. Without
  river labels it draws every line.

- [ ] **Step 1: Failing tests**

```python
def test_river_ranks_by_name_and_nearest_body() -> None:
    f = make_frame()
    rivers = f.project(load_rivers())
    bodies = [b for b in load_river_bodies(f) if b.text in ("Colorado", "Nile")]
    ranks = river_ranks(rivers, bodies, [3] * len(bodies))
    named = rivers.name_en.map(river_name)
    assert (ranks[named.isin(["Colorado", "Nile"]).values] == 3).all()
    assert (ranks[named.isna().values] == UNLABELED).all()

def test_cut_rivers_are_not_drawn() -> None:  # a unit test of the selection, not a render
    cut = Outcome(placement=..., plan=[], kept=np.zeros(0, bool), ranks=np.zeros(0, int), cut={"rivers": 2})
    assert list(cut.shown("rivers", np.array([1, 2, UNLABELED]))) == [True, False, False]

def test_draws_all_rivers_without_labels() -> None:
    uncut = Outcome(placement=..., plan=[], kept=np.zeros(0, bool), ranks=np.zeros(0, int), cut={})
    assert uncut.shown("rivers", np.array([1, 5, UNLABELED])).all()
```

(`placement=...` stands for any `Placement`, e.g. `Placement.of([])`.)

- [ ] **Step 2:** Run them. Expected: ImportError.
- [ ] **Step 3:** Implement. In `cli.main` and `draw_topographic`, move `draw_rivers` after `place` (zorder already
  keeps it below the labels and lakes). Keep the river bodies from `load_water_bodies` (`kind == "rivers"`).
- [ ] **Step 4:** `.venv/bin/pytest`. Re-record. The water modes change if rivers were cut in Task 4. Look at
  `.cache/reference/water.png` to check that the lines left match the names.
- [ ] **Step 5:** Commit `cli; *: river lines drawn iff their label's rank isn't cut`.

### Task 6: No river rank filter

**Files:**
- Modify: `src/worldmap/data.py` (`load_rivers` loses its filter; `RIVER_MAX_RANK` moves to render as
  `RIVER_THIN_RANK = 8`, the end of the width ramp), `src/worldmap/render.py`, `tests/test_data.py`

- [ ] **Step 1:** Change `test_load_rivers` to `assert rivers.scalerank.max() == 12`. Run it: FAIL.
- [ ] **Step 2:** Implement. Grep for `RIVER_MAX_RANK` everywhere, CLAUDE.md included.
- [ ] **Step 3:** `.venv/bin/pytest`. Re-record. Expect more river names in crops and more lines unless cut. Note the run time of
  `make_map --label countries,water` (was ~17 s) in the commit body.
- [ ] **Step 4:** Commit `data; *: load_rivers keeps every rank, -RIVER_MAX_RANK`.

### Task 7: CLAUDE.md

- [ ] Update: the `data.py`/`render.py`/`priority.py`/`report.py` rows, "Drawing" (lakes by `LAKE_MIN_PX`), `--rivers`
  (no rank cap, label-driven), "Priority placement" (unfit, cut, rollback, `UNCUT`), and the river-label counts. Commit `CLAUDE: label-driven features`.
