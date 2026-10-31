# Disputed Regions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `make_map --disputed` stripes every disputed area in the colours of the countries involved (plain grey for
3+ countries, parent + grey for breakaway areas), dashes its borders with those countries, and labels self-governing areas.

**Architecture:** A new `disputed.py` turns Natural Earth's disputed-areas file into `Dispute` records (which
countries are involved, breakaway or not, label text). The colouring always adds an edge for each 2-country dispute.
With `--disputed`, `render.py` draws fills without edges, then stripes, then borders as a separate line layer split into
solid and dashed parts; labels get the self-governing areas.

**Tech Stack:** Python 3, geopandas, shapely 2, matplotlib (Agg), pytest, mypy.

**Spec:** `docs/superpowers/specs/2026-10-31-disputed-regions-design.md`

## Global Constraints

- Commit messages follow the user's scheme (`~/.claude/CLAUDE.md`): `X: desc`, `X; *: desc`, `+X`; `X` = file name without
  extension. Never amend/squash; only `git add`, `git commit`, `git mv`. End each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Commits that change the output on purpose re-record with `UPDATE_REFERENCE=1 .venv/bin/pytest tests/test_regression.py`,
  commit `tests/fingerprints.json` in the same commit, and say in the body what changed visually. All other commits leave
  every fingerprint unchanged.
- `.venv/bin/mypy src` stays clean; importing any module stays side-effect free (no pyplot at import).
- Without `--disputed`, the only visible changes are Baikonur's border (Task 2) and colour shifts from the dispute
  edges (Task 3).
- Natural Earth data is used as it is (v5.1.1); no hand-drawn outlines.
- Scratch images (crops for review) go in the session scratchpad, not the repo.

## Review Focus

1. **Lakes inside or beside a disputed area** (Kashmir's lakes, Lake Victoria next to Ilemi): lakes must stay ocean-blue
   over stripes, and border lines must not cross lakes. Task 6 pins the z-order: borders below states and lakes.
2. **Borders between two uninvolved countries that run near a disputed area** must stay solid. Task 6 asserts that a
   point on the India–Bangladesh border is in the solid set, not the dashed one.
3. **`--color earth --disputed`**: no stripes, but dashes and labels; breakaway labels coloured from the land under them.
   Task 7 renders it once (smoke test, no fingerprint) and Task 8 covers the label colour.
4. **`--label none --disputed` and `--label territories --disputed`**: no self-governing labels unless `countries` is
   in the list. Task 8 asserts this.
5. **Antimeridian**: disputed polygons are clipped to ±180 by `project` like the countries. `cli.main` passes
   `project(load_disputed())` (Task 3), so nothing new is needed; Task 7's review crop includes the Kurils.

---

## File structure

- Create `src/worldmap/disputed.py`: `parse_note`, `ALIASES`, `HAND_CLAIMS`, `LABEL_TEXT`, `Dispute`, `load_disputes`,
  `dispute_edges`, `dispute_fills`, `DISPUTED_GREY`, `stripe_parts`, `border_lines`.
- Modify `src/worldmap/data.py`: `DISPUTED` file name, `load_disputed()`.
- Modify `src/worldmap/coloring.py`: `extra_edges` in `load_or_compute_coloring` and `cache_key`, bump `COLORING_RULE`.
- Modify `src/worldmap/render.py`: Baikonur edge fix, `draw_countries(..., disputed=...)`, `draw_stripes`, `draw_borders`.
- Modify `src/worldmap/labels/label.py`: `make_labels` gains `relabel` and `extra`.
- Modify `src/worldmap/cli.py`: `--disputed`, wiring.
- Tests: create `tests/test_disputed.py`; modify `tests/test_coloring.py`, `tests/test_cli.py`, `tests/test_regression.py`,
  `tests/conftest.py`.
- Docs: `README.md`, `CLAUDE.md`.

---

### Task 1: Parse the disputed-areas file

**Files:**
- Modify: `src/worldmap/data.py` (next to `LAKES`, and after `load_lakes`)
- Create: `src/worldmap/disputed.py`
- Test: `tests/test_disputed.py`

**Interfaces:**
- Produces:
  - `data.DISPUTED = "ne_10m_admin_0_disputed_areas.zip"` (`fetch` already maps it to `cultural/`).
  - `data.load_disputed() -> gpd.GeoDataFrame` (unprojected, like `load_countries`).
  - `disputed.parse_note(note: str | None) -> Note | None` with `Note = NamedTuple(admin: str | None, claimants: list[str], self_admin: bool)`.
  - `@dataclass class Dispute: name: str; admin: str; countries: list[int]; breakaway: bool; label: str | None; geometry: BaseGeometry`
    (`name`/`admin` = the file's `NAME_EN`/`ADMIN`; `countries` = row indices into the countries frame, the running
    country first, de-duplicated; for a breakaway only the parent).
  - `disputed.load_disputes(countries: gpd.GeoDataFrame, disputed: gpd.GeoDataFrame) -> list[Dispute]` (geometry copied
    from `disputed` as is, so pass a projected frame to get projected geometry).

- [ ] **Step 1: Write the failing tests** in `tests/test_disputed.py` (module fixtures `countries = load_countries()`,
  `disputes = load_disputes(countries, load_disputed())`):

```python
@pytest.mark.parametrize("note, expected", [
    ("Admin. by India; Claimed by Pakistan", Note("India", ["Pakistan"], False)),
    ("Admin. By Israel; Claimed by Syria", Note("Israel", ["Syria"], False)),
    ("Admin by U.K.; Claimed by Ireland, Denmark, and Iceland", Note("U.K.", ["Ireland", "Denmark", "Iceland"], False)),
    ("Admin. by American Samoa for U.S.A.; Claimed by Tokelau", Note("U.S.A.", ["Tokelau"], False)),
    ("Admin. by China; Ceded to China by Pakistan; Claimed by India", Note("China", ["India"], False)),
    ("Self admin.; Claimed by Azer.", Note(None, ["Azer."], True)),
    ("Claimed by Chile and Argentina; under survey", Note(None, ["Chile", "Argentina"], False)),
    ("Claimed by Colombia, Jamaica, Nicaragua and the United States", Note(None, ["Colombia", "Jamaica", "Nicaragua", "the United States"], False)),
    ("Leased to Russia by Kazakhstan", Note("Russia", ["Kazakhstan"], False)),
    ("Leased to U.S.A. by Cuba; Claimed by Cuba", Note("U.S.A.", ["Cuba", "Cuba"], False)),
])
def test_parse_note(note, expected): assert parse_note(note) == expected

@pytest.mark.parametrize("note", [None, "Admin. by Venezuela; Claimed by multiple", "UN Patrolled Zone in Syria", "Between Egypt and Sudan"])
def test_parse_note_without_claim(note): assert parse_note(note) is None

def test_counts(disputes):
    two = [d for d in disputes if not d.breakaway and len(d.countries) == 2]
    grey = [d for d in disputes if not d.breakaway and len(d.countries) > 2]
    assert (len(two), len(grey), sum(d.breakaway for d in disputes)) == (72, 11, 6)

def test_examples(disputes, countries):
    by = {(d.name, d.admin): [countries.NAME_EN[i] for i in d.countries] for d in disputes}
    assert by[("Kashmir", "India")] == ["India", "Pakistan"]
    assert by[("Taiwan", "Taiwan")] == ["Taiwan", "People's Republic of China"]
    assert by[("Abkhazia", "Georgia")] == ["Georgia"]
    assert by[("Kosovo", "Kosovo")] == ["Kosovo", "Serbia"]               # hand-written
    assert by[("Baikonur", "Baykonur Cosmodrome")] == ["Russia", "Kazakhstan"]
    assert by[("Guantanamo Bay Naval Base", "US Naval Base Guantanamo Bay")] == ["United States of America", "Cuba"]
    assert by[("Swains Island", "American Samoa")] == ["United States of America", "New Zealand"]
    assert ("Israel", "Israel") not in by and ("Ceuta", "Spain") not in by and ("Isla Aves", "Venezuela") not in by

def test_labels(disputes):
    assert {d.label for d in disputes if d.label} == {"Taiwan", "Somaliland", "Northern Cyprus", "Western Sahara", "Abkhazia",
        "South Ossetia", "Transnistria", "Nagorno-Karabakh", "Donetsk", "Luhansk"}

def test_unknown_claimant_is_an_error(countries):
    bad = gpd.GeoDataFrame({"NAME_EN": ["X"], "ADMIN": ["India"], "NOTE_BRK": ["Admin. by India; Claimed by Atlantis"]}, geometry=[Point(0, 0)])
    with pytest.raises(ValueError, match="Atlantis"):
        load_disputes(countries, bad)
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_disputed.py -q`. Expected: import error (`worldmap.disputed` missing).

- [ ] **Step 3: Implement** `load_disputed` in `data.py` and `disputed.py` with these exact tables:

```python
ALIASES = {"China": "People's Republic of China", "U.S.A.": "United States of America", "U.S.A": "United States of America",
           "the United States": "United States of America", "U.K.": "United Kingdom", "Azer.": "Azerbaijan",
           "UAE": "United Arab Emirates", "the Philippines": "Philippines", "the Seychelles": "Seychelles",
           "Tokelau": "New Zealand"}  # Tokelau is a New Zealand territory with no feature of its own
HAND_CLAIMS = {  # (NAME_EN, ADMIN) -> countries (NAME_EN), for entries Natural Earth leaves without a note
    ("Kosovo", "Kosovo"): ["Kosovo", "Serbia"], ("Gaza Strip", "Palestine"): ["Palestine", "Israel"],
    ("West Bank", "Palestine"): ["Palestine", "Israel"], ("Belize", "Belize"): ["Belize", "Guatemala"],
    ("Dragonja", "Croatia"): ["Croatia", "Slovenia"], ("Vukovar Island", "Croatia"): ["Croatia", "Serbia"],
    ("Šarengrad Island", "Croatia"): ["Croatia", "Serbia"], ("Ilemi Triangle", "South Sudan"): ["Kenya", "South Sudan"],
    ("North Borneo", "Malaysia"): ["Malaysia", "Philippines"],
    ("Scarborough Shoal", "Scarborough Reef"): ["People's Republic of China", "Philippines", "Taiwan"]}
LABEL_TEXT = {"Turkish Republic of Northern Cyprus": "Northern Cyprus", "Donetsk People's Republic": "Donetsk",
              "Luhansk People's Republic": "Luhansk"}
```

`parse_note`: split on `;`; case-insensitive `admin.? by A( for B)?` (admin = B if present, else A), `self admin`,
`leased to A by B` (admin A, claimant B), `claimed by LIST` (split on commas and `and`, `, and`); other parts ignored.
Return `None` when there are no claimants or they are exactly `["multiple"]`.
`load_disputes`: hand table first, else parse; names go through `ALIASES` and must match a countries `NAME_EN`
(else `ValueError` naming it). Self-governing: if the feature's `NAME_EN` is a countries `NAME_EN` it is the first
country (Taiwan, Somaliland, Western Sahara, Turkish Republic of Northern Cyprus), otherwise `breakaway=True`.
`label` = `LABEL_TEXT.get(name, name)` for self-governing entries, else `None`.

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_disputed.py -q && .venv/bin/mypy src`. Expected: all pass, no issues.

- [ ] **Step 5: Commit** `disputed.py`, `data.py`, `test_disputed.py`: `+disputed` (body: what it parses; `data.py` adds
  `DISPUTED`/`load_disputed` as the secondary change).

---

### Task 2: Baikonur's border above its hatching (default map)

**Files:** Modify `src/worldmap/render.py:69-84`; `tests/fingerprints.json`.

**Interfaces:** none new.

- [ ] **Step 1:** In `draw_countries`, after the hatch collection, draw Baikonur's outline again as an edge-only
  `PatchCollection` (`fc="none", ec=BORDER, lw=BORDER_WIDTH`) at a new `Z_BAIKONUR_EDGE = 1.6` (above the hatch at 1.5, below
  states at 2).
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_regression.py -q`. Expected: `default`, `pins`, `regions` fail as
  changed (`earth` unchanged: no hatch there).
- [ ] **Step 3:** Crop Baikonur (around lon 63.3, lat 45.9) from `.cache/reference/default.png` and `default.new.png`
  into the scratchpad and check the border is now continuous over the stripes.
- [ ] **Step 4:** Re-record (`UPDATE_REFERENCE=1`) and commit `render.py` + `fingerprints.json`:
  `render; *: Baikonur's border above its hatching` (body: only pixels around Baikonur change).

---

### Task 3: Two-country disputes always get different colours

**Files:** Modify `src/worldmap/disputed.py`, `src/worldmap/coloring.py:244-277`, `src/worldmap/cli.py:main`,
`tests/conftest.py:environment`; Test `tests/test_coloring.py`, `tests/test_disputed.py`; `tests/fingerprints.json`.

**Interfaces:**
- Consumes: `load_disputes`, `load_disputed` (Task 1).
- Produces:
  - `disputed.dispute_edges(disputes: list[Dispute]) -> set[tuple[int, int]]`: `(min, max)` country-index pairs of every
    non-breakaway dispute with exactly 2 countries.
  - `coloring.load_or_compute_coloring(countries, major, recolor, parent_of, extra_edges: set[tuple[int, int]] = set()) -> Coloring`
  - `coloring.cache_key(countries, major, parent_of, extra_edges)` includes `sorted` `[NAME_EN, NAME_EN]` pairs.

- [ ] **Step 1: Write the failing tests:**

```python
# test_disputed.py
def test_dispute_edges(disputes, countries):
    names = {tuple(sorted((countries.NAME_EN[a], countries.NAME_EN[b]))) for a, b in dispute_edges(disputes)}
    assert ("Marshall Islands", "United States of America") in names and ("India", "Pakistan") in names
    assert not any("Georgia" in p for p in names)  # breakaways add no edge

# test_coloring.py
def test_dispute_pairs_differ():
    countries = load_countries()
    edges = dispute_edges(load_disputes(countries, load_disputed()))
    major = [is_labeled(r) for _, r in countries.iterrows()]
    col = load_or_compute_coloring(countries, major, False, parent_index(countries), edges).color_class
    assert all(col[a] != col[b] for a, b in edges)
    # extend test_cache_key_covers_major_and_parents: a different extra_edges gives a different key
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_coloring.py tests/test_disputed.py -q`. Expected: fail (no `dispute_edges`, extra argument).
- [ ] **Step 3: Implement.** In `load_or_compute_coloring` map each pair through `node_of` (dependencies share their
  parent's node) and add it to the edge set after `add_proximity_edges`, skipping pairs on the same node. Bump
  `COLORING_RULE` to `"top10-exempt-count-balance/v4-dispute-edges"`. In `cli.main`, compute
  `disputes = load_disputes(countries, project(load_disputed()))` before the colouring and pass `dispute_edges(disputes)`
  (the file is fetched on every run, ~215 KB; this deviates from the spec's "only with `--disputed`" because the colouring
  needs it). In `conftest.environment`, add `"disputed": size of fetch(DISPUTED)` for every mode.
- [ ] **Step 4: Run** `.venv/bin/make_map --recolor` (spot-check line must print `[]`), then the full suite.
  Expected: unit tests pass; render tests fail if colours shifted, or skip because the environment gained `disputed`.
- [ ] **Step 5:** Look at the new default render, re-record, commit `coloring.py`, `disputed.py`, `cli.py`, tests,
  `fingerprints.json`: `coloring; *: two-country disputes always get different colours` (body: Wake Island US/Marshall
  Islands was the only clash; list which countries changed colour, from comparing `coloring.json` before and after).

---

### Task 4: Stripe geometry and fills

**Files:** Modify `src/worldmap/disputed.py`; Test `tests/test_disputed.py`.

**Interfaces:**
- Produces:
  - `DISPUTED_GREY = "#cfcfcf"` (my call; tune while reviewing Task 7's crops).
  - `STRIPE_WIDTH_IN = 0.05` (10 px at 200 dpi; tune in Task 7).
  - `stripe_parts(geom: BaseGeometry, n: int, width: float) -> list[BaseGeometry]`: `n` disjoint pieces of `geom`; piece
    `i` is `geom` ∩ the 45° bands `k` with `k % n == i`, band `k` = `k*width <= (x + y)/sqrt(2) < (k+1)*width`, measured from the
    map origin, so neighbouring areas' stripes line up.
  - `dispute_fills(d: Dispute, fill: Callable[[int], Fill]) -> list[Fill]`: breakaway → `[fill(parent), DISPUTED_GREY]`;
    more than 2 countries → `[DISPUTED_GREY]`; else `[fill(c) for c in d.countries]`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_stripe_parts_cover_and_balance():
    sq = box(0, 0, 100, 100)
    parts = stripe_parts(sq, 3, 2.0)
    assert len(parts) == 3
    assert abs(sum(p.area for p in parts) - sq.area) < 1e-6
    assert all(abs(p.area - sq.area / 3) < sq.area * 0.02 for p in parts)
    assert parts[0].intersection(parts[1]).area < 1e-9

def test_stripe_parts_line_up_across_shapes():
    a, b = box(0, 0, 10, 10), box(10, 0, 20, 10)
    assert stripe_parts(a, 2, 1.0)[0].union(stripe_parts(b, 2, 1.0)[0]).equals(stripe_parts(a.union(b), 2, 1.0)[0])

def test_dispute_fills():
    fill = {0: "#ff0000", 1: "#00ff00", 2: "#0000ff"}.__getitem__
    assert dispute_fills(Dispute("A", "A", [0, 1], False, None, Point(0, 0)), fill) == ["#ff0000", "#00ff00"]
    assert dispute_fills(Dispute("A", "A", [0, 1, 2], False, None, Point(0, 0)), fill) == [DISPUTED_GREY]
    assert dispute_fills(Dispute("A", "A", [2], True, "A", Point(0, 0)), fill) == ["#0000ff", DISPUTED_GREY]
```

(The `equals` check may need `.normalize()` or a symmetric-difference area `< 1e-9`; use the area form if `equals` is flaky.)

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_disputed.py -q`. Expected: fail on missing names.
- [ ] **Step 3: Implement.** Build the bands as rotated rectangles covering `geom.bounds`, union them per residue, intersect.
- [ ] **Step 4: Run** tests and mypy. Expected: pass.
- [ ] **Step 5: Commit** `disputed: stripe_parts, dispute_fills` (+ test file as secondary: `disputed; *: ...`).

---

### Task 5: Solid and dashed border lines

**Files:** Modify `src/worldmap/disputed.py`; Test `tests/test_disputed.py`.

**Interfaces:**
- Consumes: `Dispute` (Task 1), `parent_index` from `data.py`.
- Produces: `border_lines(countries_projected: gpd.GeoDataFrame, disputes: list[Dispute], parent_of: dict[int, int]) -> tuple[BaseGeometry, BaseGeometry]`
  returning `(solid, dashed)` lines in map units, with `SNAP_M = 500` (metres; < 1/9 px on the poster).

Algorithm (the part the signature doesn't determine):
- For each dispute `d`: `involved` = union of country polygons whose index, or whose `parent_of` index, is in
  `d.countries`, plus every other dispute's geometry that shares a country with `d`. `dashed_d = d.geometry.boundary ∩
  (involved − d.geometry).buffer(SNAP_M)`. `dashed` = union of all `dashed_d`.
- `solid` = union of the exterior and interior rings of every country polygon (Antarctica included), minus
  `dashed.buffer(SNAP_M)`.

- [ ] **Step 1: Write the failing tests** (module fixtures: projected countries via `project(load_countries())`,
  projected disputes, `parent_index`; `to_map(lon, lat)` helper via `pyproj.Transformer` to `CRS`):

```python
def test_kashmir_line_of_control_is_dashed(lines):
    solid, dashed = lines
    p = Point(to_map(74.0, 34.6))  # nearest point on the India/Pakistan line of control
    assert dashed.distance(p) < solid.distance(p) or solid.distance(p) > 2_000

def test_uninvolved_border_stays_solid(lines):
    solid, dashed = lines
    p = Point(to_map(88.6, 26.4))  # India/Bangladesh border, far from any dispute
    assert solid.distance(p) < 5_000 and dashed.distance(p) > 50_000

def test_taiwan_coast_is_not_dashed(lines, disputes_projected):
    taiwan = next(d for d in disputes_projected if d.name == "Taiwan")
    assert lines[1].intersection(taiwan.geometry.buffer(1_000)).length == 0

def test_kashmir_meets_rest_of_india_dashed(lines):
    # the internal edge between Kashmir and Himachal Pradesh is a new dashed line, not in the country borders
    p = Point(to_map(76.8, 32.9))
    assert lines[1].distance(p) < 20_000
```

Pick the exact coordinates from the data while writing the tests (print the nearest boundary vertex); the
assertions above are the intent.

- [ ] **Step 2: Run.** Expected: fail on missing `border_lines`.
- [ ] **Step 3: Implement** as above.
- [ ] **Step 4: Run** tests and mypy; check `border_lines` runs in a few seconds (`time`); if it's slow, simplify the
  ring geometry first, not the snap distance.
- [ ] **Step 5: Commit** `disputed; *: border_lines`.

---

### Task 6: Drawing: fills without edges, stripes, borders layer

**Files:** Modify `src/worldmap/render.py`.

**Interfaces:**
- Consumes: `stripe_parts`, `dispute_fills`, `border_lines`, `STRIPE_WIDTH_IN` (Tasks 4-5).
- Produces:
  - `draw_countries(ax, countries_projected, fill_of, regions, disputed: bool = False) -> None`: with `disputed`, fills
    get `ec="none"` and Baikonur's hatch is skipped.
  - `draw_stripes(ax: Axes, disputes: list[Dispute], fill: Callable[[int], Fill], width: float) -> None` at `Z_STRIPES = 1.5`.
  - `draw_borders(ax: Axes, solid: BaseGeometry, dashed: BaseGeometry) -> None`: `LineCollection`s, `BORDER`,
    `BORDER_WIDTH`, solid at `Z_BORDERS = 1.7`, dashed with `BORDER_DASH = (0, (2.5, 1.5))` at the same z, `capstyle="butt"`.
  - Final order: fills 1 → stripes / Baikonur hatch 1.5 → Baikonur edge 1.6 → borders 1.7 → states 2 → lakes 2.5 → globe 3.
    (The spec listed borders above lakes; that would draw borders across lakes, so they go below. Lakes keep their own
    `BORDER` shore outline.)

- [ ] **Step 1:** Implement. No unit test: Task 7's render test covers it; the existing fingerprints must not change.
- [ ] **Step 2: Run** `.venv/bin/pytest -q && .venv/bin/mypy src`. Expected: all pass, fingerprints unchanged.
- [ ] **Step 3: Commit** `render: draw_stripes, draw_borders` (no output change: nothing calls them yet).

---

### Task 7: `--disputed` flag

**Files:** Modify `src/worldmap/cli.py`; Test `tests/test_cli.py`, `tests/test_regression.py`; `tests/fingerprints.json`.

**Interfaces:**
- Consumes: Tasks 3-6.
- Produces: `parse_args(...).disputed: bool`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_disputed_flag():
    assert parse_args([]).disputed is False and parse_args(["--disputed"]).disputed is True

def test_render_disputed(tmp_path):  # test_regression.py
    check_fingerprint("disputed", render(tmp_path, "--disputed"))
```

- [ ] **Step 2: Run.** Expected: `test_disputed_flag` fails; `test_render_disputed` fails (unrecognised argument).
- [ ] **Step 3: Implement** in `main`: with `args.disputed`, `draw_countries(..., disputed=True)`; in political mode
  `draw_stripes(ax, disputes, lambda i: fill_of(i, None), STRIPE_WIDTH_IN * DPI / frame.px_per_unit)`; in both modes
  `draw_borders(ax, *border_lines(countries_p, disputes, parent_index(countries)))`. Use the projected countries
  *before* `subtract_lakes`. Help text: `"stripe disputed areas in the colours of the countries involved and dash their borders"`.
- [ ] **Step 4:** Render `--disputed --dpi 200` and `--color earth --disputed` (smoke: exits 0) into the scratchpad. Crop
  Kashmir/Aksai Chin, Crimea, Taiwan, Cyprus, Western Sahara, the Caucasus, the Kurils, the Falklands, Baikonur; check
  stripes are equal, under the borders, dashes on the right segments, lakes blue. Show the crops to the user and tune
  `STRIPE_WIDTH_IN`, `DISPUTED_GREY`, `BORDER_DASH` with them before recording.
- [ ] **Step 5:** Record the `disputed` fingerprint (others unchanged) and commit `cli; *: --disputed`.

---

### Task 8: Labels for self-governing areas

**Files:** Modify `src/worldmap/labels/label.py:103-128`, `src/worldmap/cli.py`; Test `tests/test_disputed.py`;
`tests/fingerprints.json`.

**Interfaces:**
- Consumes: `Dispute.label`, `Dispute.breakaway`, `Dispute.countries`, `DISPUTED_GREY` (Tasks 1, 4).
- Produces: `make_labels(..., countries: bool = True, relabel: dict[int, str] | None = None, extra: Sequence[tuple[str, Polygon, Fill]] = ())`:
  - `relabel`: countries-file rows labelled with this text even if `is_labeled` is false (Somaliland, Turkish Republic
    of Northern Cyprus → "Northern Cyprus"); also overrides `RENAME`. Only used when `countries` is true.
  - `extra`: upright labels for breakaways: `(text, largest polygon, fill)`; anchored and sized like child regions;
    text colour `darken(fill)`.
  - cli, with `--disputed` and `countries` in `--label`: `relabel` = `{d.countries[0]: d.label}` for non-breakaway
    labelled disputes; `extra` = one per breakaway, fill `DISPUTED_GREY` in political mode, or
    `fill_at(regions, d.countries[0], largest.representative_point())` with `--color earth`.

- [ ] **Step 1: Write the failing tests** (reuse `test_regions.py`'s `build_labels` approach: a figure from `new_figure`, Agg renderer):

```python
def test_disputed_labels(...):
    names = {l.name for l in labels_with_disputes}
    assert {"Somaliland", "Northern Cyprus", "Abkhazia", "Donetsk", "Luhansk", "Transnistria", "Taiwan"} <= names
    assert "Kashmir" not in names and "Crimean Peninsula" not in names
    assert not any(l.text.get_fontstyle() == "italic" for l in labels_with_disputes if l.name in {"Abkhazia", "Donetsk"})

def test_disputed_labels_need_countries(...):
    # make_labels(..., countries=False, relabel=..., extra=...) -> no Somaliland, no Abkhazia
```

- [ ] **Step 2: Run.** Expected: fail (unexpected keyword arguments).
- [ ] **Step 3: Implement** in `label.py` and the cli wiring.
- [ ] **Step 4:** Full suite: only `disputed` changes. Crop the Caucasus, Cyprus and Somaliland; check labels sit inside,
  readable, and placement didn't push others badly.
- [ ] **Step 5:** Re-record `disputed`, commit `label; *: --disputed labels self-governing areas`.

---

### Task 9: Docs and example

**Files:** `README.md`, `CLAUDE.md`, `examples/disputed.png` (new).

- [ ] **Step 1:** `make_map examples/disputed.png --disputed --dpi 50`.
- [ ] **Step 2:** README: `--disputed` in the usage block, the option table (stripes, grey for 3+ countries and for
  breakaways, dashed borders, labels for self-governing areas), and an example section with the image.
  Commit `README; *: --disputed` (with the example PNG).
- [ ] **Step 3:** CLAUDE.md: module table row for `disputed.py`; a `--disputed` section (data file, `HAND_CLAIMS`,
  `ALIASES`, the colouring's dispute edges, z-order, `SNAP_M`); the regression list gains `--disputed`; `Z_*` order.
  Commit `CLAUDE: --disputed`.
- [ ] **Step 4:** `.venv/bin/pytest -q && .venv/bin/mypy src`: all pass.
