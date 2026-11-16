# Placeable Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Country labels (`Label`) and water labels (`WaterLabel`) answer the same questions the placement code asks, so `place_labels` and `place_by_priority` stop branching on `isinstance` to get a label's name, size, size floor, move unit or to shrink it.

**Architecture:** A `Placeable` Protocol (`labels/placeable.py`) names the shared members; `Label` and `WaterLabel` gain the ones they lack (`Label.size`/`min_size`/`unit_px`/`shrink_toward`, `WaterLabel.name`/`unit_px`/`shrink_toward`). The two solvers then call them. The algorithms that really differ (the two cost kernels, `overhang_of`'s per-glyph vs per-slice test, water glyph boxes in `Placement.parts`) keep dispatching, on `p.water[i]`. Pure refactor: every fingerprint stays as it is.

**Tech Stack:** Python 3.14, `typing.Protocol`, pytest, mypy.

**Spec:** none: from the end-of-branch report of `priority-placement` (architecture idea 1). The motivation: `priority.place_by_priority` has local `size()`, `name()`, `floor()` helpers that each branch on the label's class, and `placement.place_labels` has two shrink loops (country / water) and a class test for the nudge unit.

## Global Constraints

- Refactor only: `.venv/bin/pytest` passes with `tests/fingerprints.json` unchanged (no `UPDATE_REFERENCE`).
- `.venv/bin/mypy src` reports no issues.
- Country labels keep their quirk: a country label above its floor shrinks by a whole `SHRINK_FACTOR` step, even if that takes it under the floor (water labels stop at the floor). It keeps the fingerprints; unifying it is a separate, output-changing decision for the user.
- Commits follow the user's convention (`X: desc`, `X; *: desc`, `+X`), one logical change each, with the `Co-Authored-By` trailer from the session.

## Review Focus

1. A country label shrunk inside `place_labels` must start the next priority step at its shrunk box: today `place_by_priority.solve` copies `p.width`/`p.height` back to the label; after Task 3 `Label.shrink_toward` keeps `label.width`/`height` in step itself and the copy-back goes. Test in Task 1.
2. The country quirk above (whole step under the floor): pinned by a test in Task 1.
3. `WaterLabel.text` is the string drawn and matched by `--crop` names; `name` must be an alias, not a rename. Test in Task 2.
4. A nudged country label's unit is its *current* height (after shrinking), as `p.height[i]` is today. `Label.unit_px` reads `self.height`, which Task 1 keeps current. Covered by the fingerprint renders.
5. Floors default: `SHRINK_MIN_PT * frame.label_scale` for country labels. `Label.min_size` needs the frame's scale, so `new_label` must set `Label.label_scale`; a test label built without it (tests construct `Label` directly) gets 1.0, as `make_frame()` (the world) has. Test in Task 1.

---

### Task 1: `Label` sizes and shrinks itself

**Files:**
- Modify: `src/worldmap/labels/label.py` (`Label`, `new_label`; `SHRINK_MIN_PT`, `SHRINK_FACTOR` move here from `placement.py`)
- Modify: `src/worldmap/labels/placement.py`, `src/worldmap/labels/priority.py` (import the two constants from `label`)
- Test: `tests/test_label.py`

**Interfaces:**
- Produces, on `Label`:
  - field `label_scale: float = 1.0` (after `full_size`, before `cover`), set by `new_label` to `frame.label_scale`
  - property `size -> float`: `float(self.text.get_fontsize())`
  - property `min_size -> float`: `SHRINK_MIN_PT * self.label_scale`
  - property `unit_px -> float`: `self.height` (a nudge is measured in heights)
  - method `shrink_toward(self, floor: float) -> float`: if `self.size > floor`, scale the font size, `width` and `height` by `SHRINK_FACTOR` and return it; else return 1.0
- `label.SHRINK_MIN_PT = 5`, `label.SHRINK_FACTOR = 0.94` (same values; `placement` re-imports them for its other users)

- [ ] **Step 1: Write the failing tests** in `tests/test_label.py`, building a label with `test_placement.country_label(make_frame(), "Atlantis", -35, 30, 500)`:

```python
def test_label_shrinks_a_whole_step_while_above_its_floor() -> None:
    label = country_label(make_frame(), "Atlantis", -35, 30, 500)
    size, w, h = label.size, label.width, label.height
    assert label.shrink_toward(size - 0.01) == SHRINK_FACTOR  # under the floor afterwards: the country quirk
    assert label.size == pytest.approx(size * SHRINK_FACTOR)
    assert (label.width, label.height) == pytest.approx((w * SHRINK_FACTOR, h * SHRINK_FACTOR))
    assert label.shrink_toward(label.size) == 1.0 and label.size == pytest.approx(size * SHRINK_FACTOR)


def test_label_min_size_and_unit() -> None:
    label = country_label(make_frame(), "Atlantis", -35, 30, 500)
    assert label.min_size == SHRINK_MIN_PT  # the world frame: label_scale 1
    label.label_scale = 2
    assert label.min_size == 2 * SHRINK_MIN_PT and label.unit_px == label.height
```

- [ ] **Step 2: Run** `.venv/bin/pytest -n0 tests/test_label.py -k "shrinks or min_size"`: FAIL (`AttributeError: 'Label' object has no attribute 'shrink_toward'`).
- [ ] **Step 3: Implement** the Interfaces above; move the two constants, fixing imports.
- [ ] **Step 4: Run** `.venv/bin/pytest -n0 tests/test_label.py tests/test_placement.py tests/test_priority.py` and `.venv/bin/mypy src`: pass, no issues.
- [ ] **Step 5: Commit** `label; *: +Label.size, min_size, unit_px, shrink_toward`.

### Task 2: `WaterLabel` answers the same

**Files:**
- Modify: `src/worldmap/labels/water.py` (`WaterLabel`)
- Test: `tests/test_water.py`

**Interfaces:**
- Produces, on `WaterLabel` (`size`, `min_size` already exist):
  - property `name -> str`: `self.text`
  - property `unit_px -> float`: `self.size * PX_PER_PT` (a nudge is measured in ems)
  - method `shrink_toward(self, floor: float) -> float`: if `self.size > floor`, `factor = max(SHRINK_FACTOR, floor / self.size)`, `self.scale(factor)`, return it; else 1.0. (`SHRINK_FACTOR` from `labels.label`.)

- [ ] **Step 1: Write the failing test**, with `test_placement.water_label("Sea", (0, 0), shapely.box(0, 0, 10, 10), np.array([[-30.0, -7, 30, 7]]), 10)`:

```python
def test_water_label_shrinks_to_its_floor_and_no_further() -> None:
    sea = water_label("Sea", (0, 0), shapely.box(0, 0, 10, 10), np.array([[-30.0, -7, 30, 7]]), 10)
    assert sea.name == sea.text == "Sea"
    assert sea.shrink_toward(9.9) == pytest.approx(0.99) and sea.size == pytest.approx(9.9)
    assert sea.shrink_toward(9.9) == 1.0
    assert sea.unit_px == pytest.approx(9.9 * PX_PER_PT)
```

- [ ] **Step 2: Run** `.venv/bin/pytest -n0 tests/test_water.py -k shrinks_to_its_floor`: FAIL (no attribute `name`).
- [ ] **Step 3: Implement** the Interfaces above.
- [ ] **Step 4: Run** `.venv/bin/pytest -n0 tests/test_water.py`: pass.
- [ ] **Step 5: Commit** `water: +WaterLabel.name, unit_px, shrink_toward`.

### Task 3: `Placeable`, and the solvers use it

**Files:**
- Create: `src/worldmap/labels/placeable.py`
- Modify: `src/worldmap/labels/placement.py` (`place_labels`), `src/worldmap/labels/priority.py` (`place_by_priority`)
- Modify: `CLAUDE.md` (module table: `labels/placeable.py` row)

**Interfaces:**
- Consumes: Task 1's and Task 2's members.
- Produces: `class Placeable(Covered, Protocol)` (`Covered` from `labels/geometry.py`) with read-only property `name -> str` (a property in the Protocol, or mypy rejects `WaterLabel`'s read-only one), `kind: str`, `full_size: float`, `anchor_x: float`, `anchor_y: float`, read-only properties `size`, `min_size`, `unit_px` (float) and `shrink_toward(self, floor: float) -> float`. Both label classes satisfy it structurally (no inheritance).

- [ ] **Step 1: Create** `placeable.py` with the Protocol and a one-line module docstring.
- [ ] **Step 2: `place_labels`**: `items: list[Placeable]`; the default `floors` is `[item.min_size for item in items]`; the nudge unit is `items[i].unit_px`; the two shrink loops become one over `clashing`: `f = items[i].shrink_toward(floors[i])`, then `p.width[i] *= f`, `p.height[i] *= f`, and for `p.water[i]` also `p.parts[p.owner == i] = items[i].parts`; the closing print uses `items[i].name`. `cost_batch` dispatches on `p.water[i]` (casting for mypy). No `isinstance` left in `placement.py` but that cast.
- [ ] **Step 3: `place_by_priority`**: drop the local `size()` and `name()` helpers (use `items[i].size`, `items[i].name`); `floor(i)` becomes `max(items[i].min_size, NUDGE_MIN_SCALE * settled[i][2]) if i in settled else items[i].min_size`; drop the loop in `solve` that copies `p.width`/`p.height` back to country labels (Task 1 keeps them current). `overhang_of` and the out/off checks dispatch on `p.water[k]`.
- [ ] **Step 4: Verify** `grep -n "isinstance" src/worldmap/labels/placement.py src/worldmap/labels/priority.py` shows only the casts in `cost_batch`/`overhang_of` (or nothing); `.venv/bin/pytest` passes with `git diff tests/fingerprints.json` empty; `.venv/bin/mypy src` clean.
- [ ] **Step 5: Commit** three commits, each passing the suite: `+placeable; *: the labels' shared members` (with CLAUDE.md's row), `placement: place_labels asks its labels (Placeable)`, `priority: place_by_priority asks its labels`.
