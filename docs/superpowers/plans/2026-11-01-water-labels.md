# Water Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--label oceans,seas,lakes,water` puts curved, blue, atlas-style water names on the political and topographic maps, placed by the existing label solver.

**Architecture:** A new `labels/water.py` loads the water bodies (Natural Earth marine polygons + the drawn lakes), fits a gentle curve along each, and lays the name's glyphs along it as one rigid `Path` plus a chain of per-glyph boxes. The solver (`labels/placement.py`) gains these "water labels" next to the country boxes: they move rigidly, overlap through their glyph boxes, shrink only themselves, and a drop loop removes failing water labels and re-solves. A shared `curves.py` holds the glyph-bending maths, also used by the colour-bar titles.

**Tech Stack:** Python, matplotlib (`TextPath`, `PathPatch`, `patheffects`), shapely 2, numpy, geopandas.

**Spec:** `docs/superpowers/specs/2026-11-01-water-labels-design.md`

## Global Constraints

- Default `--label` stays `countries`; every existing regression fingerprint must stay unchanged by every task (run `.venv/bin/pytest tests/test_regression.py` in each task that touches placement, rendering or `colorbars`).
- Water labels never get leader lines; country/territory labels are never dropped.
- Country labels shrink only for clashes with other country labels; in a country-water clash only the water label shrinks.
- A water label is dropped when, after solving, it still clashes (overlap, or a leader crossing it) at its minimum size, or is mostly off its water body.
- Rivers, reefs and unnamed features are never labelled; straits and channels only with `water`.
- Oceans: one ATLANTIC (North + South merged), one each INDIAN, ARCTIC, SOUTHERN, and PACIFIC twice (left and right map edges).
- Style: oceans upright ALL CAPS, large, letter-spaced; everything else smaller, normal case, italic. Text a deeper blue in `coloring.OCEAN`'s hue, halo a light blue near `OCEAN` (no white halo). Same style on both map types.
- Commit messages follow the user's global scheme (`X: desc`, `X; *: desc`, `+X`), always with the `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` trailer; only `git add`/`commit`/`mv`.
- `.venv/bin/mypy src` clean; importing any module has no side effects (no pyplot import).
- `--label countries,water` should take at most a few times today's ~10 s.

## Review Focus

- `--label water` alone (no country labels) on the political map: the solver must work with zero box labels. Test in Task 6.
- Both PACIFIC labels survive near the globe edge (outside-globe cost is 1000): the Pacific pieces' anchors must leave room inside the outline. Asserted in Task 8's render check and pinned by the `water` fingerprint.
- Steep bodies (Red Sea, Gulf of California, Mozambique Channel) are never upside down: left to right, or bottom to top when within 1° of vertical. Test in Task 3.
- Water bodies split by the antimeridian or by islands (Bering Sea, Chukchi Sea): labelled once, on their largest piece. Test in Task 2.
- `--type topographic` without water kinds (including the default `countries`) renders exactly as today. The existing topographic fingerprints cover this; Task 9 must not run the label stage unless a water kind is asked for.

---

## File Structure

| File | Change |
|---|---|
| `src/worldmap/data.py` | `+MARINE` (in `PHYSICAL`), `+load_marine` |
| `src/worldmap/curves.py` | new: `arc_lengths`, `bend` (shared glyph bending) |
| `src/worldmap/colorbars.py` | `curved_title` uses `curves.bend` (no output change) |
| `src/worldmap/labels/water.py` | new: kinds, `WaterBody`, `load_water_bodies`, `fit_curve`, `WaterLabel`, `make_water_labels`, style constants |
| `src/worldmap/labels/placement.py` | water labels in `Placement`, `cost_batch`, shrink rule, `clashes`, `place_with_water` |
| `src/worldmap/render.py` | `+draw_water_labels` |
| `src/worldmap/cli.py` | `LABEL_KINDS`, help text, political + topographic wiring |
| `tests/test_water.py` | new: unit tests for `water.py` and `curves.py` |
| `tests/test_placement.py` | new: unit tests for water labels in the solver |
| `tests/test_regression.py`, `tests/conftest.py`, `tests/fingerprints.json` | `water`, `topographic-water` modes; `marine` in `environment` for them |
| `README.md`, `CLAUDE.md` | docs |

---

### Task 1: Load the marine polygons

**Files:**
- Modify: `src/worldmap/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `data.MARINE = "ne_10m_geography_marine_polys.zip"` (added to `PHYSICAL`); `load_marine() -> gpd.GeoDataFrame` (lon/lat, columns `name`, `featurecla`, `scalerank`, `geometry`, all 306 features, unfiltered).

- [ ] **Step 1: Write the failing test**

```python
def test_load_marine() -> None:
    marine = load_marine()
    assert {"name", "featurecla", "scalerank"} <= set(marine.columns)
    assert (marine.featurecla == "ocean").sum() == 7
    assert "Mediterranean Sea" in set(marine.name)
```

- [ ] **Step 2: Run it:** `.venv/bin/pytest tests/test_data.py::test_load_marine -v` → FAIL (ImportError).
- [ ] **Step 3: Implement** `MARINE`, add it to `PHYSICAL`, and `load_marine()` (`gpd.read_file(fetch(MARINE))[["name", "featurecla", "scalerank", "geometry"]]`).
- [ ] **Step 4: Run it again** → PASS (downloads ~1 MB on first run).
- [ ] **Step 5: Commit** `data; *: +MARINE, +load_marine` (`tests/test_data.py` is the secondary file).

---

### Task 2: Water bodies

**Files:**
- Create: `src/worldmap/labels/water.py`
- Test: `tests/test_water.py`

**Interfaces:**
- Consumes: `data.load_marine`, `data.project`, `data.load_lakes`.
- Produces:
  - `WATER_KINDS = ("oceans", "seas", "lakes", "water")`
  - `Kind = Literal["oceans", "seas", "lakes", "straits"]`
  - `kinds_from_labels(label: set[str]) -> set[Kind]`: `water` expands to all four; other entries (`countries`, `territories`) ignored.
  - `classify(featurecla: str) -> Kind | None`: `ocean` → oceans; `sea gulf bay sound fjord lagoon inlet` → seas; `strait channel` → straits; anything else (`reef`, `river`, `generic`) → None.
  - `@dataclass WaterBody: text: str; kind: Kind; rank: int; geom: Polygon` (projected map units; the piece it is labelled on).
  - `OCEAN_TEXT = {"Atlantic": "ATLANTIC OCEAN", "Pacific": "PACIFIC OCEAN", "Indian": "INDIAN OCEAN", "Arctic": "ARCTIC OCEAN", "Southern": "SOUTHERN OCEAN"}` (my call: the full name, in caps; change here if only ATLANTIC/PACIFIC is wanted).
  - `load_water_bodies(kinds: set[Kind], lakes_projected: gpd.GeoDataFrame) -> list[WaterBody]`, sorted most important first: oceans, then `rank` ascending, then area descending.

Rules for `load_water_bodies`:
- Marine features with no `name`, or `classify` None, or kind not in `kinds`, are skipped.
- Oceans: project, group by the ocean word in the name (`North Atlantic Ocean` + `South Atlantic Ocean` → Atlantic, case-insensitive), union each group. The Pacific union is split at x = 0 (map units) into a left and a right body, each labelled on its largest polygon. Every other ocean: its largest polygon.
- Seas / straits / lakes: the largest polygon of the projected geometry (a body cut by the antimeridian is labelled once). Lakes come from `lakes_projected` (`name`, `scalerank`).
- A name appearing in both the marine and lake lists is kept once (the marine entry).

- [ ] **Step 1: Write the failing tests**

```python
def test_classify() -> None:
    assert classify("ocean") == "oceans"
    assert classify("gulf") == "seas" and classify("fjord") == "seas"
    assert classify("channel") == "straits"
    assert classify("reef") is None and classify("river") is None and classify("generic") is None


def test_kinds_from_labels() -> None:
    assert kinds_from_labels({"countries"}) == set()
    assert kinds_from_labels({"seas", "countries"}) == {"seas"}
    assert kinds_from_labels({"water"}) == {"oceans", "seas", "lakes", "straits"}


def test_oceans(lakes_projected) -> None:
    oceans = load_water_bodies({"oceans"}, lakes_projected)
    texts = [b.text for b in oceans]
    assert sorted(texts) == sorted(["ATLANTIC OCEAN", "PACIFIC OCEAN", "PACIFIC OCEAN", "INDIAN OCEAN", "ARCTIC OCEAN", "SOUTHERN OCEAN"])
    pacific = [b.geom.centroid.x for b in oceans if b.text == "PACIFIC OCEAN"]
    assert min(pacific) < 0 < max(pacific)


def test_seas_lakes_and_straits(lakes_projected) -> None:
    seas = {b.text for b in load_water_bodies({"seas"}, lakes_projected)}
    assert {"Mediterranean Sea", "Caspian Sea", "Bay of Bengal", "Skagerrak"} <= seas
    assert not seas & {"Great Barrier Reef", "Amazon River", "Strait of Gibraltar"}
    everything = load_water_bodies({"oceans", "seas", "lakes", "straits"}, lakes_projected)
    names = [b.text for b in everything]
    assert "Strait of Gibraltar" in names and "Lake Victoria" in names
    assert names.count("Bering Sea") == 1 and names.count("Caspian Sea") == 1
    assert everything[0].kind == "oceans"
    ranks = [b.rank for b in everything if b.kind != "oceans"]
    assert ranks == sorted(ranks)
```

(`lakes_projected` is a module fixture: `project(load_lakes())`.)

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_water.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** the interfaces above in `labels/water.py` (module docstring in the style of `label.py`).
- [ ] **Step 4: Run** → PASS; `.venv/bin/mypy src` clean.
- [ ] **Step 5: Commit** `+water` (the new module with `tests/test_water.py`).

---

### Task 3: Curves and glyph bending

**Files:**
- Create: `src/worldmap/curves.py`
- Modify: `src/worldmap/colorbars.py` (`curved_title`), `src/worldmap/labels/water.py`
- Test: `tests/test_water.py`

**Interfaces:**
- Produces:
  - `curves.arc_lengths(curve: np.ndarray) -> np.ndarray`: cumulative length along an (n, 2) polyline, starting at 0.
  - `curves.bend(curve: np.ndarray, u: np.ndarray, v: np.ndarray, h: float) -> np.ndarray`: the points at distance `u` along `curve` plus `v` along its left normal (tangent from `u ± h`), shape (len(u), 2). This is exactly the `at` / tangent / normal maths now inside `curved_title`.
  - `water.fit_curve(geom: Polygon, max_tilt_deg: float, n: int = 64) -> np.ndarray`: an (n, 2) smooth polyline in the same units as `geom`, along the body's long direction through its middle, reading left to right (first x < last x), or bottom to top when it is within 1° of vertical.

`fit_curve` algorithm (the tests don't pin it, so here it is):
1. Sample interior points of `geom` (a regular grid, ~2000 points, kept if inside).
2. Principal axis from their covariance; clamp its angle from horizontal to ±`max_tilt_deg`.
3. In the rotated frame (s along the axis, t across), fit t = a·s² + b·s + c by least squares; clamp |a| so the bend over the s-range (5th..95th percentile) turns the tangent by at most `MAX_BEND_DEG` = 40.
4. Sample n points over that s-range, rotate back, orient as above.

Constants in `water.py`: `MAX_BEND_DEG = 40`, `OCEAN_MAX_TILT_DEG = 20`, `SEA_MAX_TILT_DEG = 90` (tuned in Task 10).

- [ ] **Step 1: Write the failing tests**

```python
def test_bend_straight_and_circle() -> None:
    line = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert np.allclose(bend(line, np.array([2.0, 5.0]), np.array([1.0, -1.0]), 0.01), [[2, 1], [5, -1]])
    theta = np.linspace(np.pi, 0, 500)  # upper half circle, radius 10, left to right
    arc = np.stack([10 * np.cos(theta), 10 * np.sin(theta)], -1)
    pts = bend(arc, np.array([arc_lengths(arc)[-1] / 2]), np.array([2.0]), 0.01)
    assert np.allclose(pts, [[0, 12]], atol=1e-2)  # left of rightward travel over the top is up, away from the centre


def test_fit_curve_horizontal_and_reading_direction() -> None:
    ellipse = shapely.affinity.scale(Point(0, 0).buffer(1), 10, 2)
    c = fit_curve(ellipse, 90)
    assert c[0, 0] < c[-1, 0] and np.ptp(c[:, 1]) < 0.5


def test_fit_curve_steep_never_upside_down() -> None:
    red_sea_like = shapely.affinity.rotate(shapely.affinity.scale(Point(0, 0).buffer(1), 10, 1), -60)
    c = fit_curve(red_sea_like, 90)
    assert c[0, 0] < c[-1, 0]  # never right to left (upside down)
    assert c[0, 1] > c[-1, 1]  # this one runs down to the right


def test_fit_curve_tilt_clamped() -> None:
    tall = shapely.affinity.scale(Point(0, 0).buffer(1), 1, 10)
    c = fit_curve(tall, 20)
    angle = np.degrees(np.arctan2(c[-1, 1] - c[0, 1], c[-1, 0] - c[0, 0]))
    assert abs(angle) <= 20 + 1e-6


def test_fit_curve_follows_a_bend() -> None:
    ring = Point(0, 0).buffer(10).difference(Point(0, 0).buffer(8))
    upper = ring.intersection(shapely.box(-10, 2, 10, 10))  # an upward arch
    c = fit_curve(upper, 90)
    assert c[len(c) // 2, 1] > c[0, 1] and c[len(c) // 2, 1] > c[-1, 1]
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_water.py -v` → the new tests FAIL.
- [ ] **Step 3: Implement** `curves.py`; rewrite `curved_title` to call `arc_lengths` and `bend` with the same numbers (same `h = 0.05 * unit`, same normals), then `fit_curve`.
- [ ] **Step 4: Run** the water tests → PASS. Then `.venv/bin/pytest tests/test_colorbars.py tests/test_regression.py -k "topographic or colorbar"` → PASS with fingerprints unchanged (a mismatch means the refactor changed float order: match the old expression order exactly).
- [ ] **Step 5: Commit** twice: `+curves` (`curves.py` with `colorbars.py` and the bend test riding along; body: the colour bars render identically), then `water; *: +fit_curve`.

---

### Task 4: Water labels (glyphs, size, boxes)

**Files:**
- Modify: `src/worldmap/labels/water.py`
- Test: `tests/test_water.py`

**Interfaces:**
- Consumes: `WaterBody`, `fit_curve`, `curves.bend`, `frame.MapFrame` (`to_px`, `px_per_unit`, `xmin`, `ymin`), `colorspace.srgb_to_oklch`, `oklch_to_srgb`.
- Produces:
  - `@dataclass WaterLabel: text: str; kind: Kind; rank: int; geom: BaseGeometry` (the body in poster px) `; anchor_x: float; anchor_y: float` (px, the curve's middle) `; size: float` (pt) `; glyphs: Path` (px, relative to the anchor) `; parts: np.ndarray` (shape (k, 4): x0, y0, x1, y1 offsets in px from the anchor, one per non-space glyph, padded by the halo).
    - `scale(self, factor: float) -> None`: shrink glyphs, parts and `size` about the anchor.
    - `inside_share(self, x: float, y: float) -> float`: area of `parts` boxes (placed at x, y) over `geom`, divided by their total area.
  - `make_water_labels(bodies: list[WaterBody], frame: MapFrame) -> list[WaterLabel]`: one label per body, same order, minus the ones that already fail `WATER_MIN_INSIDE` at their starting size and position (this is the drop rule applied early, which keeps hundreds of hopeless bays out of the solver).
  - Constants: `OCEAN_PT = 30`, `OCEAN_TRACKING_EM = 0.6`, `SEA_MAX_PT = 14`, `WATER_MIN_PT = 6.5`, `SEA_FILL = 0.8` (text length / curve length), `WATER_MIN_INSIDE = 0.35`, `WATER_TEXT = oklch(L=0.50, C=0.08, h=OCEAN's hue)` , `WATER_HALO = OCEAN`, `WATER_HALO_PX = 2.4`. All starting values, tuned in Task 10.

Layout: glyph by glyph, `TextPath` per character at its prefix's advance (`TextToPath.get_text_width_height_descent` of `text[:k]` with the same `FontProperties`) plus `k × tracking`, italic except oceans; the whole run centred on the curve's middle (curve extended linearly at both ends if the text is longer); `v` offset so the text's x-height centre lies on the curve. Size: oceans `OCEAN_PT`; others `clip(SEA_FILL × curve length / text width at 1 pt, WATER_MIN_PT, SEA_MAX_PT)`. Curves fitted with `OCEAN_MAX_TILT_DEG` for oceans, else `SEA_MAX_TILT_DEG`.

- [ ] **Step 1: Write the failing tests**

```python
def test_water_label_layout(frame, lakes_projected) -> None:
    bodies = load_water_bodies({"oceans", "seas"}, lakes_projected)
    labels = make_water_labels(bodies, frame)
    med = next(l for l in labels if l.text == "Mediterranean Sea")
    atlantic = next(l for l in labels if l.text == "ATLANTIC OCEAN")
    assert atlantic.size == OCEAN_PT and WATER_MIN_PT <= med.size <= SEA_MAX_PT
    assert len(med.parts) == len("MediterraneanSea")
    assert med.inside_share(med.anchor_x, med.anchor_y) >= WATER_MIN_INSIDE
    width = med.parts[:, 2].max() - med.parts[:, 0].min()
    med.scale(0.5)
    assert med.parts[:, 2].max() - med.parts[:, 0].min() == pytest.approx(width / 2)


def test_hopeless_bodies_left_out(frame, lakes_projected) -> None:
    bodies = load_water_bodies({"oceans", "seas", "lakes", "straits"}, lakes_projected)
    labels = make_water_labels(bodies, frame)
    assert 0 < len(labels) < len(bodies)
    assert all(l.inside_share(l.anchor_x, l.anchor_y) >= WATER_MIN_INSIDE for l in labels)
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run** → PASS; mypy clean.
- [ ] **Step 5: Commit** `water: +WaterLabel, +make_water_labels`.

---

### Task 5: Water labels in the solver (no dropping yet)

**Files:**
- Modify: `src/worldmap/labels/placement.py`
- Test: `tests/test_placement.py` (new)

**Interfaces:**
- Consumes: `WaterLabel` (`parts`, `geom`, `scale`, `inside_share`, `size`, `anchor_x/y`).
- Produces:
  - `Placement` gains `water: np.ndarray` (bool, per label), `parts: np.ndarray` (all water glyph boxes, offsets, shape (m, 4)) and `owner: np.ndarray` (int, label index of each part). Labels are ordered countries first, then water. For a water label, `x`/`y` is where its *anchor* (the curve middle) is placed, not a box centre, and `width`/`height` are its parts' bounding box, used only to scale candidate steps; every water term goes through `parts`.
  - `place_labels(labels: list[Label], frame: MapFrame, seed: int = SEED, weights: CostWeights = CostWeights(), water: Sequence[WaterLabel] = (), start: tuple[np.ndarray, np.ndarray] | None = None) -> Placement` (`start`: warm-start centres for all labels).
  - `CostWeights` gains `outside_water: float = 20` (1 − `inside_share`) and `water_drift: float = 0.15` (distance from anchor, in heights).
  - `clashes(p: Placement) -> tuple[list[int], list[int]]`: (country labels overlapping another country label, water labels overlapping any label's box or glyph box, or crossed by an 'out' leader).

Rules:
- Country label i: existing terms unchanged, computed over the *non-water* others (so with no water labels every array and sum is identical to today), **plus** overlap with water glyph boxes (same `overlap` weight) and none of the leader terms against them except `leader_hits_label` for its own leader crossing water glyph boxes.
- Water label i: candidates are the existing `STEPS × STEP_SCALES` around its centre plus its anchor (no rings, no grid); cost = overlap of its translated glyph boxes with all country boxes and other water glyph boxes, `dot_cover` and `leader_hits_label` for other labels' leaders against its glyph boxes, `outside_water`, `water_drift`, and `outside_globe` if any glyph box leaves the outline.
- Pre-filter other boxes/parts by the candidates' joint envelope before the pairwise arrays (this is what keeps ~200 water labels × ~12 glyphs fast).
- Shrink rounds use `clashes`: country labels shrink as today when in the first list; water labels in the second list shrink with `WaterLabel.scale(SHRINK_FACTOR)` while `size > WATER_MIN_PT`.

- [ ] **Step 1: Write the failing tests.** Fixtures in `tests/test_placement.py`: `frame` (`make_frame()`); `two_labels` / `two_labels_copy()` (two far-apart `Label`s over the Sahara, built with `new_label` on an Agg figure, fresh each call); `one_label` (one `Label` in the mid-Atlantic, a 1000 px square `geom` around it); `water_on_top_of_it` (a `WaterLabel` with the same anchor and `geom`, three hand-made glyph boxes covering the country label, `size = 10`).

```python
def test_no_water_is_unchanged(frame, two_labels) -> None:
    a = place_labels(two_labels, frame)
    b = place_labels(two_labels_copy(), frame, water=[])
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)


def test_water_pushes_country_but_only_water_shrinks(frame, one_label, water_on_top_of_it) -> None:
    size = one_label[0].text.get_fontsize()
    p = place_labels(one_label, frame, water=[water_on_top_of_it])
    assert one_label[0].text.get_fontsize() == size
    assert clashes(p) == ([], [])


def test_water_only(frame, water_on_top_of_it) -> None:
    p = place_labels([], frame, water=[water_on_top_of_it])
    assert p.n == 1 and p.water.all()
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_placement.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new tests → PASS; then `.venv/bin/pytest tests/test_regression.py` → every fingerprint unchanged; mypy clean.
- [ ] **Step 5: Commit** `placement; *: water labels in the solver` (`tests/test_placement.py` secondary).

---

### Task 6: The drop loop

**Files:**
- Modify: `src/worldmap/labels/placement.py`
- Test: `tests/test_placement.py`

**Interfaces:**
- Produces: `place_with_water(labels: list[Label], water: list[WaterLabel], frame: MapFrame) -> tuple[Placement, list[WaterLabel]]` — the final placement and the water labels kept (same order as in it). With `water == []` it is exactly `place_labels(labels, frame)`.

Loop:
1. `p = place_labels(labels, frame, water=kept, start=previous centres of the kept labels or None)`.
2. Failing: water labels with `inside_share(p.x, p.y) < WATER_MIN_INSIDE` (dropped directly), and water labels in `clashes(p)[1]` at `size <= WATER_MIN_PT`. Group the clashing ones into connected clusters (labels whose boxes/glyph boxes touch); in each cluster drop the least important water label (largest index in the most-important-first order).
3. Stop when nothing fails; print the dropped names (`print("water labels dropped:", [...])`), like the existing "unresolved overlaps" line.

- [ ] **Step 1: Write the failing tests**

```python
def test_drop_least_important(frame) -> None:
    sea, bay = crossing_water_labels()  # helper: same anchor and tiny shared geom, both at WATER_MIN_PT, "Sea" first
    p, kept = place_with_water([], [sea, bay], frame)
    assert kept[0].text == "Sea"
    assert clashes(p) == ([], [])


def test_off_water_is_dropped(frame) -> None:
    lonely = water_label_far_from_its_geom()  # helper: glyph boxes 5000 px from a 10 px geom
    _, kept = place_with_water([], [lonely], frame)
    assert kept == []


def test_country_labels_never_dropped(frame, one_label, water_on_top_of_it) -> None:
    p, _ = place_with_water(one_label, [water_on_top_of_it], frame)
    assert (~p.water).sum() == 1
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS; regression unchanged.
- [ ] **Step 5: Commit** `placement: +place_with_water`.

---

### Task 7: Drawing

**Files:**
- Modify: `src/worldmap/render.py`
- Test: `tests/test_water.py`

**Interfaces:**
- Produces: `draw_water_labels(ax: Axes, water: list[WaterLabel], placement: Placement, frame: MapFrame) -> None`: each kept label's `glyphs` translated to its placed centre (`placement.x/y` at its index), converted to map units (`frame.px_to_map`), drawn as one `PathPatch` (`fc=WATER_TEXT`, `ec="none"`, `path_effects=[withStroke(linewidth=WATER_HALO_PX × 72 / DPI, foreground=WATER_HALO)]`, `zorder=LABEL_ZORDER - 0.1`, under the country labels).

- [ ] **Step 1: Write the failing test** `test_draw_water_labels`: draw one placed label on an Agg figure and assert one `PathPatch` was added with `facecolor` ≈ `WATER_TEXT` and its path's extent centred on the placed position (in map units, to 1 px).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `render: +draw_water_labels`.

---

### Task 8: Political map wiring

**Files:**
- Modify: `src/worldmap/cli.py`, `tests/test_cli.py`, `tests/test_regression.py`, `tests/conftest.py`, `tests/fingerprints.json`

**Interfaces:**
- Consumes: `kinds_from_labels`, `load_water_bodies`, `make_water_labels`, `place_with_water`, `draw_water_labels`.
- Produces: `LABEL_KINDS = ("countries", "territories", "oceans", "seas", "lakes", "water")`; `--label` help lists the new kinds ("water = oceans, seas, lakes, straits and channels").

In `main`, after `decide_out_and_wrap`: `water = make_water_labels(load_water_bodies(kinds_from_labels(args.label), lakes_p), frame)`, `placement, water = place_with_water(labels, water, frame)`, `draw_labels(...)` then `draw_water_labels(...)`. With no water kinds this must be the same calls as today. The final print adds `"water labels:", len(water)`.

- [ ] **Step 1: Write the failing tests:** `test_label_flag` accepts `oceans,seas,lakes,water` and still rejects `rivers`; `test_render_water` (`check_fingerprint("water", render(tmp_path, "--label", "countries,water"))`). In `conftest.environment`, add `"marine": str(os.path.getsize(fetch(MARINE)))` only when `"water" in mode` (so existing entries keep their env).
- [ ] **Step 2: Run** → CLI test FAILS; the render test fails with "no fingerprint".
- [ ] **Step 3: Implement** the wiring.
- [ ] **Step 4: Render and look**: `cd` to the scratchpad, `.venv/bin/make_map water.png --label countries,water`; crop the Mediterranean, Caribbean, both Pacific edges, Indonesia and the North Atlantic; check: both PACIFIC labels present, no upside-down text, no white halos, no country label smaller than in the default map. Then `.venv/bin/pytest tests/test_cli.py tests/test_regression.py` → all old fingerprints PASS.
- [ ] **Step 5: Commit** `cli; *: --label oceans, seas, lakes, water` (fingerprints not yet recorded: recorded after tuning in Task 10).

---

### Task 9: Topographic map wiring

**Files:**
- Modify: `src/worldmap/cli.py` (`draw_topographic`), `tests/test_regression.py`, `tests/fingerprints.json`

`draw_topographic` gets water labels only when `kinds_from_labels(args.label)` is non-empty: after the colour bars, `place_with_water([], make_water_labels(load_water_bodies(kinds, project(load_lakes())), frame), frame)` and `draw_water_labels`, before the pins. Update the `--type` help: `--label` water kinds apply to topographic; countries/territories are ignored.

- [ ] **Step 1: Test** `test_render_topographic_water` (`--type topographic --label water`), fingerprint mode `topographic-water`.
- [ ] **Step 2: Run** existing topographic tests → still PASS (no labels without water kinds).
- [ ] **Step 3: Implement**; render once and look at the Southern Ocean, Mediterranean and Arctic (label on hill shading, ice and the ice colour bar region).
- [ ] **Step 4: Commit** `cli; *: water labels on the topographic map`.

---

### Task 10: Tune by eye, record, document

**Files:**
- Modify: `src/worldmap/labels/water.py` and `placement.py` constants only; `tests/fingerprints.json`; `README.md`; `CLAUDE.md`; `examples/` (a `water.png` at `--dpi 50`)

- [ ] **Step 1:** Render `--label countries,water` and `--type topographic --ocean --label water` at 200 dpi; crop the regions from Task 8 plus the Great Lakes and East Africa. Tune `OCEAN_PT`, tracking, `SEA_MAX_PT`, `WATER_MIN_PT`, `WATER_MIN_INSIDE`, `WATER_TEXT` L/C, halo, tilt/bend limits, `outside_water`, `water_drift` until: oceans clearly biggest; Mediterranean, Caribbean, Adriatic, Aegean, Black, Caspian, Red Sea, Hudson Bay and at least the largest Great Lakes named; labels visibly quieter than country names; no white halos; nothing upside down; run time within a few × 10 s (report the numbers). Show crops to the user before recording.
- [ ] **Step 2:** `UPDATE_REFERENCE=1 .venv/bin/pytest tests/test_regression.py -k water`; confirm the other fingerprints are untouched (`git diff tests/fingerprints.json` shows only the two new entries).
- [ ] **Step 3:** README: the new `--label` kinds and one example image; CLAUDE.md: `labels/water.py` and `curves.py` in the module table, the water pipeline (bodies → curve → glyph boxes → solver → drop loop), the shrink rule, `topographic-water`/`water` regression modes, the marine file.
- [ ] **Step 4:** Full `.venv/bin/pytest` and `.venv/bin/mypy src` → PASS / clean.
- [ ] **Step 5: Commits:** `water: tuned sizes and colours` with `fingerprints.json` (body: what the new modes look like), then `examples: +water`, `README: --label water kinds`, `CLAUDE: water labels`.
