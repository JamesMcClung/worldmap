# Child Regions (`--regions`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dependencies take their parent's political colour and lose their default labels, and a new `--regions` flag
labels 44 dependencies plus 22 hand-picked outlying regions in italics as `<region> (<parent>)`.

**Architecture:**
- `data.py` defines what a dependency is and maps each to its parent.
- `coloring.py` colours a graph in which each dependency is dissolved into its parent's node.
- A new `labels/regions.py` builds the list of child regions: dependencies by rule, plus a hand-picked table of parent
  name and lon/lat boxes.
- `labels/label.py` turns child regions into italic `Label`s, which flow through placement unchanged.

**Tech Stack:** Python 3, geopandas/shapely/pyproj, matplotlib, pytest, mypy.

**Spec:** `docs/superpowers/specs/2026-10-27-child-regions-design.md`

## Global Constraints

- **Dependency:** a feature with `SOVEREIGNT != ADMIN` and `TYPE` not in `{"Lease", "Indeterminate"}`. Its parent is the
  feature whose `ADMIN == SOVEREIGNT`. Today's data gives 46 of them.
- **Labels:** a dependency is labelled with `--regions` unless it touches its parent (today only Hong Kong and Macau
  touch). That gives 44 labelled dependencies plus 22 hand-picked regions, so 66 child labels.
- **Label text:** `f"{region} ({parent label text})"`. The parent's text is after `RENAME` ("United States",
  "East Timor"). Labels are italic and otherwise follow the normal size, uppercase, halo and out/wrap rules.
- **No flag:** no child region is labelled.
- **Colouring:** in political mode a dependency always has its parent's fill, and the colouring must not depend on
  `--regions`.
- **Unchanged:** Baikonur and Guantánamo (leases), Palestine and Siachen (indeterminate), and the disputed entries
  (Kosovo, Northern Cyprus, Somaliland, Western Sahara...).
- **Commits:** follow the user's global CLAUDE.md format (`X: desc`, `X; *: desc`, `+X`), with a
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` trailer. Use only `git add` and `git commit`. Never amend.
- **Fingerprints:** a commit that changes the output re-records with `UPDATE_REFERENCE=1 .venv/bin/pytest
  tests/test_regression.py` and commits `tests/fingerprints.json` in the same commit, with the visual change in the body.
  A refactor must leave the fingerprints unchanged.
- **Checks before every commit:** `.venv/bin/pytest` and `.venv/bin/mypy src` are clean.

## Review Focus

1. **Antimeridian:** Alaska's Aleutians reach east of 180°, so its boxes must cover both sides and must not pick up
   Russia's Commander Islands, which are Russian features anyway. Pinned in Task 4 by
   `test_alaska_includes_aleutians_east_of_180`.
2. **Overlaps:** a hand-picked box must not claim the parent's main body, and no polygon may be claimed by two regions
   (Madeira/Canaries, Guadeloupe/Martinique). Pinned in Task 4.
3. **Stale colouring cache:** a change to which features count as "major" or as dependencies must invalidate
   `coloring.json`. Pinned in Task 2 by the cache key test.
4. **Earth mode:** an island-group label's colour must come from the group's own largest polygon, not the parent's
   largest. Hawaii must not take the contiguous US land-cover colour. Pinned in Task 5.
5. **Flag off:** the default render must contain no italic labels and no Kaliningrad label. Pinned in Task 5, and by
   the default fingerprint.

---

### Task 1: Dependency rule in `data.py`

**Files:**
- Modify: `src/worldmap/data.py`
- Test: `tests/test_data.py` (new)

**Interfaces:**
- Produces:
  - `is_dependency(row: pd.Series) -> bool`
  - `parent_index(countries: gpd.GeoDataFrame) -> dict[int, int]`, mapping each dependency's row index to its parent's
    row index (unprojected or projected frames share the index).

- [ ] **Step 1: Write the failing tests** in `tests/test_data.py`. They use the real `load_countries()`, since `.cache/` is present.

```python
def test_dependencies_and_parents() -> None:
    c = load_countries()
    parents = parent_index(c)
    names = {c.NAME_EN[d]: c.NAME_EN[p] for d, p in parents.items()}
    assert len(parents) == 46
    assert names["Greenland"] == "Denmark" and names["Hong Kong"] == "People's Republic of China"
    assert names["Falkland Islands"] == "United Kingdom" and names["Aruba"] == "Netherlands"
    assert not {"Baikonur", "Guantanamo Bay Naval Base", "Palestine", "Siachen Glacier"} & names.keys()
    assert all(is_labeled(c.loc[p]) for p in parents.values())
```

- [ ] **Step 2: Run it.** `.venv/bin/pytest tests/test_data.py -v` should FAIL with an ImportError.

- [ ] **Step 3: Implement both functions in `data.py`.** Add a constant
  `NOT_DEPENDENCY_TYPES = ("Lease", "Indeterminate")`. If a dependency has no parent row, raise `ValueError` naming it.

- [ ] **Step 4: Run the tests and mypy.** The new tests pass, and the full suite (fingerprints included) is unchanged.

- [ ] **Step 5: Commit** with message `data: +parent_index` and the trailer. Files: `src/worldmap/data.py`,
  `tests/test_data.py`.

### Task 2: Dissolve dependencies into their parent for the colouring

**Files:**
- Modify: `src/worldmap/coloring.py` (`load_or_compute_coloring`, `COLORING_RULE`)
- Modify: `src/worldmap/cli.py` (pass `parent_index(countries)`)
- Modify: `tests/fingerprints.json` (re-recorded)
- Test: `tests/test_coloring.py`

**Interfaces:**
- Consumes: `parent_index` from Task 1.
- Produces:
  - `merge_into_parents(gdf: gpd.GeoDataFrame, major: list[bool], parent_of: dict[int, int]) -> tuple[gpd.GeoDataFrame, list[bool], list[int]]`
    returns the merged frame (one row per non-dependency, with its dependencies' geometry unioned in), the merged
    `major` flags (the parent's), and `node_of`, which maps each original row position to its merged row position.
  - `load_or_compute_coloring(countries, major, recolor, parent_of: dict[int, int]) -> Coloring`. `color_class` stays
    per original feature, expanded through `node_of`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_merge_into_parents() -> None:
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6), box(1, 0, 2, 1)], crs="EPSG:4326")
    merged, major, node_of = merge_into_parents(gdf, [True, False, True], {1: 0})
    assert len(merged) == 2 and node_of == [0, 0, 1] and major == [True, True]
    assert merged.geometry[0].area == pytest.approx(2.0)

def test_dependencies_share_parent_class() -> None:
    c = load_countries()
    parents = parent_index(c)
    col = load_or_compute_coloring(c, [is_labeled(r) for _, r in c.iterrows()], False, parents).color_class
    assert all(col[d] == col[p] for d, p in parents.items())
    assert spot_check(c, col) == []

def test_cache_key_covers_major_and_parents(tmp_path, monkeypatch) -> None:
    # Point CACHE at tmp_path. Compute once, then flip one `major` flag. The second call must print "computed",
    # not "loaded".
```

  Use the `capsys` fixture for the last test. If running the colouring twice makes it too slow (more than about 30 s),
  call a small `cache_key(countries, major, parent_of) -> dict` helper instead and assert that the keys differ.

- [ ] **Step 2: Run them.** All three should FAIL.

- [ ] **Step 3: Implement.**
  - `merge_into_parents` uses `dissolve` on a group column (each row's own index, or its parent's for a dependency).
    Keep a deterministic order of non-dependency rows in their original order.
  - In `load_or_compute_coloring`, run `adjacency`, `add_proximity_edges`, the area weights and `color_graph` on the
    merged frame, then expand the result.
  - `class_pairs` comes from the merged edges.
  - Add `"major": [names of major rows]` and `"parents": sorted name pairs` to the cache key.
  - Bump `COLORING_RULE` to `"top10-exempt-count-balance/v3-dependencies-merged"`.
  - In `cli.main`, pass `parent_index(countries)`.

- [ ] **Step 4: Run the tests and check the result.** The unit tests pass. Run `.venv/bin/make_map` and check that the
  spot-check line prints `[]`.

- [ ] **Step 5: Re-record and look.** Run `UPDATE_REFERENCE=1 .venv/bin/pytest tests/test_regression.py`, then crop
  `.cache/reference/default.png` around the Caribbean, Greenland and the Channel to confirm that dependencies match
  their parent's fill.

- [ ] **Step 6: Commit** with message `coloring; *: dependencies take their parent's colour class`. In the body, say
  which fills changed: the 46 dependencies, plus any reshuffled countries you noticed. Files: `coloring.py`, `cli.py`,
  `tests/test_coloring.py`, `tests/fingerprints.json`.

### Task 3: Dependencies and exclaves are not labelled by default

**Files:**
- Modify: `src/worldmap/data.py` (`is_labeled`, `LABELED_EXTRA`, `NOT_LABELED`)
- Modify: `src/worldmap/labels/label.py` (remove `EXCLAVES` and its loop in `make_labels`)
- Modify: `tests/fingerprints.json`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `is_labeled(row)` is False for every dependency.

- [ ] **Step 1: Write the failing test.**

```python
def test_dependencies_not_labeled() -> None:
    c = load_countries()
    labeled = {r.NAME_EN for _, r in c.iterrows() if is_labeled(r)}
    assert not {"Greenland", "Faroe Islands", "Puerto Rico", "Falkland Islands"} & labeled
    assert {"Palestine", "Kosovo", "Western Sahara", "Israel", "Denmark"} <= labeled
    assert not {"Northern Cyprus", "Somaliland", "Baikonur"} & labeled
```

- [ ] **Step 2: Run it.** It should FAIL.

- [ ] **Step 3: Implement.**
  - `is_labeled` returns False when `is_dependency(row)`.
  - Drop the dependencies from `LABELED_EXTRA` (Puerto Rico, Falklands, Faroes) and from `NOT_LABELED` (Jersey,
    Guernsey, Isle of Man, Åland, Macau, Hong Kong, Sint Maarten, Aruba, Curaçao).
  - Delete `EXCLAVES` and the exclave branch in `make_labels`.
  - The "major" flags change, and Task 2's cache key handles that.

- [ ] **Step 4: Run the tests and mypy.** The new test passes, and the regression tests fail only on fingerprints.

- [ ] **Step 5: Re-record and look.** Re-record with `UPDATE_REFERENCE=1`. In a crop, check that Greenland, the
  Falklands, Kaliningrad and French Guiana have no labels and that no other labels vanished. Compare the label count
  printed by `make_map` before and after: it should drop by exactly 6.

- [ ] **Step 6: Commit** with message `data; *: dependencies and exclaves are unlabelled by default`. In the body, list
  the 6 removed labels and any colour reshuffle caused by the "major" change.

### Task 4: `labels/regions.py`, the list of child regions

**Files:**
- Create: `src/worldmap/labels/regions.py`
- Test: `tests/test_regions.py` (new)

**Interfaces:**
- Consumes: `parent_index`, `is_dependency` (Task 1), `RENAME` (`labels/label.py`), and `MapFrame.to_lonlat`.
- Produces:

```python
Box = tuple[float, float, float, float]  # lon0, lat0, lon1, lat1
HAND_PICKED: list[tuple[str, str, list[Box]]]  # (label name, parent NAME_EN, boxes)

@dataclass
class ChildRegion:
    name: str            # e.g. "Tasmania", "Falkland Islands"
    parent_name: str     # parent's label text after RENAME, e.g. "United States"
    feature: int         # row index whose fill colours it (the dependency itself, or the parent for hand-picked)
    polys: list[Polygon] # projected polygons it labels

    @property
    def text(self) -> str: ...  # f"{name} ({parent_name})"

def child_regions(countries: gpd.GeoDataFrame, countries_projected: gpd.GeoDataFrame, frame: MapFrame) -> list[ChildRegion]
```

- **Dependencies:** take every dependency whose unprojected geometry does not intersect its parent's
  `buffer(0.01)` (degrees). Use its `NAME_EN` and all its non-empty projected polygons.
- **Hand-picked:** take each projected polygon of the parent whose `representative_point()`, converted with
  `frame.to_lonlat`, lies in any of the region's boxes.
- `RENAME` lives in `label.py`. To avoid a cycle, `regions.py` imports it from there, and `label.py` must not import
  `regions.py` at module level (or move `RENAME` into `regions.py`, your choice).
- **Starting boxes** (lon0, lat0, lon1, lat1). Only polygon representative points are tested, so the boxes may overlap
  the parent's mainland:

| Region | Parent NAME_EN | Boxes |
|---|---|---|
| Kaliningrad | Russia | (19.5, 54.2, 23.0, 55.4) |
| French Guiana | France | (-55, 2, -51, 6) |
| Alaska | United States of America | (-180, 51, -129, 72), (170, 50, 180, 56) |
| Hawaii | United States of America | (-179, 18, -154, 29) |
| Svalbard | Norway | (10, 74, 35, 81) |
| Tasmania | Australia | (143.5, -44, 149, -39.5) |
| Galápagos | Ecuador | (-92.5, -1.6, -89, 1.7) |
| Easter Island | Chile | (-109.6, -27.3, -109.1, -26.9) |
| Cabinda | Angola | (11.5, -6, 13.2, -4.3) |
| Nakhchivan | Azerbaijan | (44.7, 38.8, 46.7, 39.8) |
| Oecusse | East Timor | (124, -9.5, 124.6, -9.1) |
| Musandam | Oman | (55.9, 25.6, 56.6, 26.5) |
| Canary Islands | Spain | (-18.5, 27.5, -13.3, 29.5) |
| Azores | Portugal | (-31.5, 36.8, -24.9, 40) |
| Madeira | Portugal | (-17.4, 32.3, -16.2, 33.2) |
| Zanzibar | Tanzania | (39, -6.6, 40, -4.8) |
| Socotra | Yemen | (52, 11.9, 54.6, 12.8) |
| Andaman & Nicobar | India | (92, 6.5, 94, 14) |
| Bioko | Equatorial Guinea | (8.4, 3.1, 9.0, 3.9) |
| Réunion | France | (55.1, -21.5, 55.9, -20.8) |
| Guadeloupe | France | (-61.9, 15.8, -61.0, 16.6) |
| Martinique | France | (-61.3, 14.3, -60.8, 14.9) |

  Adjust the boxes if a test shows one is wrong.

- [ ] **Step 1: Write the failing tests** in `tests/test_regions.py`, with a module-scoped fixture that runs
  `child_regions(c, project(c), make_frame())`.

```python
def test_counts(regions) -> None:
    assert len(regions) == 66
    names = {r.name for r in regions}
    assert not {"Hong Kong", "Macau"} & names
    assert {"Gibraltar", "Greenland", "Alaska", "Easter Island"} <= names

def test_text(regions) -> None:
    by = {r.name: r for r in regions}
    assert by["Tasmania"].text == "Tasmania (Australia)"
    assert by["Hawaii"].text == "Hawaii (United States)"
    assert by["Oecusse"].text == "Oecusse (East Timor)"

def test_hand_picked_select_polygons_but_not_mainland(regions, countries_projected) -> None:
    # Each hand-picked region has >= 1 polygon, none of which is its parent's largest polygon.
    # No polygon (compare by WKB) belongs to two regions.

def test_alaska_includes_aleutians_east_of_180(regions, frame) -> None:
    # Some Alaska polygon's representative point has lon > 170.

def test_parents_are_labeled_countries(regions) -> None:
    # Every parent_name is the RENAME'd NAME_EN of a row where is_labeled is True.
```

- [ ] **Step 2: Run them.** They should FAIL with an ImportError.

- [ ] **Step 3: Implement `regions.py`** as specified above.

- [ ] **Step 4: Run the tests and mypy.** All pass, and `test_import_has_no_pyplot` still passes.

- [ ] **Step 5: Commit** with message `+regions` (the body may mention the test file). Files: `regions.py`,
  `tests/test_regions.py`.

### Task 5: `--regions` flag: italic child labels

**Files:**
- Modify: `src/worldmap/labels/label.py` (`make_labels`)
- Modify: `src/worldmap/cli.py` (`parse_args`, `main`)
- Modify: `tests/test_regression.py`, `tests/fingerprints.json`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `child_regions`, `ChildRegion` (Task 4).
- Produces: `make_labels(countries_projected, frame, ax, renderer, fill_of, children: list[ChildRegion] = [])`
  (or `Sequence`, default empty). Each child gets a `Label` whose `name` is `child.text`, whose `Text` has
  `fontstyle="italic"`, and that otherwise goes through the same size, uppercase and halo code.

**Anchoring and colour:**
- Factor the per-country anchor logic (the island test with hull centroid or inscribed circle) into a helper
  `anchor_piece(polys, frame, tolerance, force_island: bool) -> tuple[tuple[float, float], BaseGeometry, float, bool]`
  that returns the anchor, geometry, area and island flag.
- Use the helper for both countries and children. The country output must stay byte-identical: run the fingerprint
  tests before adding children.
- A child's colour is `darken(fill_of(child.feature, max(child.polys, key=area).representative_point()))`.

- [ ] **Step 1: Do the refactor first.** Extract `anchor_piece` and run `.venv/bin/pytest tests/test_regression.py`.
  The fingerprints should be unchanged.

- [ ] **Step 2: Commit the refactor** with message `label: extract anchor_piece`.

- [ ] **Step 3: Write the failing tests.**
  - In `test_cli.py`: `assert parse_args([]).regions is False and parse_args(["--regions"]).regions is True`.
  - In `test_regression.py`: `test_render_regions`, which calls
    `check_fingerprint("regions", render(tmp_path, "--regions"))`.
  - A test in `tests/test_regions.py` that builds labels on an Agg figure through `make_labels` with the real data and
    `children=child_regions(...)`. Check that it yields 66 labels with `get_fontstyle() == "italic"`, and that with
    `children=[]` there are none.
  - **Earth colour:** with `fill_of` from `build_regions` (earth mode), the Hawaii label colour differs from the US
    label colour. Skip this check if the raster isn't cached.

- [ ] **Step 4: Run them.** They should FAIL. The regions fingerprint is missing.

- [ ] **Step 5: Implement.**
  - Add `--regions` (store_true) with the help text "label dependencies and outlying regions (Alaska, Greenland,
    Kaliningrad...) with their parent country".
  - In `main`, build the children only when the flag is set, after `subtract_lakes`, from `countries_p`, and pass them
    to `make_labels`.

- [ ] **Step 6: Check the output.**
  - The default, earth and pins fingerprints are unchanged.
  - Record `regions` with `UPDATE_REFERENCE=1`.
  - Render `make_map --regions` and `make_map --regions --earth` into the scratchpad and crop the spec's check areas:
    the Caribbean, the Channel and Isle of Man, the North Atlantic, Cyprus, Gibraltar, the Gulf of Oman, the Caucasus,
    Timor, the Pacific, the South Atlantic and the southern Indian Ocean.
  - If placement is visibly worse (overlaps, long tangled leaders), tune `CostWeights` only when it leaves the default
    fingerprints unchanged. Otherwise note it for the user and don't drop regions.

- [ ] **Step 7: Commit** with message `label; *: --regions labels child regions in italics`. In the body, describe the
  new `regions` fingerprint and any crowding seen.

### Task 6: Docs

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `todo.md`

- [ ] **Step 1: README.** Document `--regions` and the colour rule (dependencies take their parent's colour).
- [ ] **Step 2: CLAUDE.md.** Update the module table (`labels/regions.py`) and pipeline step 1 (dependencies are
  dissolved into their parent before colouring; the cache key includes the major and parent lists). In the label notes,
  replace the `EXCLAVES` sentence. Add the 4th regression mode to the Tests section.
- [ ] **Step 3: todo.md.** Remove the "Disconnected Regions" item and add the future notes: a `--regions` threshold, and
  disputed regions as children of several states. `todo.md` is untracked, so ask the user before committing it.
- [ ] **Step 4: Commit** README and CLAUDE.md separately: `README: --regions` and `CLAUDE: --regions`.
