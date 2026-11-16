# Faster Placement Cost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the label solver's time, the biggest stage of a `--label countries,water` run, without changing a pixel.

**Architecture:** The end-of-branch report guessed that rebuilding the `Placement` every priority step was the main cost. A profile says otherwise (world map, `--label countries,water`, 21 s in all): `place_labels` minus its descent is ~0.1 s; the descent is 9.6 s, of which the country cost kernel is 3.9 s (15 272 calls, ~256 us each), the water kernel 3.5 s (40 155 calls, ~87 us), and `placement.glyph_boxes` 0.8 s (recomputed from scratch on every cost and clash call). The water kernel already gathers only the obstacles near its candidates' envelope; the country kernel loops over *every* country label for every candidate. So: (1) give `country_cost` the same envelope pre-filter, exactly (a skipped label adds exactly 0.0 to every term); (2) keep the glyph boxes in the `Placement`, moved with their label, instead of rebuilding them. Then measure again before anything bigger (a grid index).

**Tech Stack:** numba 0.68 (`@njit(cache=True)`), numpy, pytest.

**Spec:** none: from the end-of-branch report of `priority-placement` (architecture idea 2), corrected by the profile above.

## Global Constraints

- Output identical: `.venv/bin/pytest` passes with `tests/fingerprints.json` unchanged. Every skipped term must be exactly 0.0 (adding +0.0 leaves a float sum as it is), so the cost arrays are bit-identical, not merely close.
- `.venv/bin/mypy src` reports no issues.
- Timing is measured with `split.py` (Appendix; keep it in the scratchpad, not the repo: wraps `kernels.country_cost`, `kernels.water_cost`, `placement.glyph_boxes` with timers and prints `{name: (seconds, calls)}` after `cli.main`), on `--label countries,water --dpi 50`; quote before and after in each commit body.
- Commits follow the user's convention, with the session's `Co-Authored-By` trailer.

## Review Focus

1. An 'out' country label j far from our candidates whose *leader* still crosses the envelope: its leader-hits-our-box and leader-crosses-our-leader terms are not zero. The pre-filter must take j's whole extent: its box, its dot +- `DOT_RADIUS * height[j]`, and the segment dot -> leader end. Test in Task 1.
2. Our own leader (label i 'out') reaching from its dot to far candidates: the envelope already grows to the dot (`country_cost` does this for glyphs); the country filter must use that grown envelope. Test in Task 1.
3. A label whose box covers our dot: the "unless that box covers our dot" exemption must survive the filter (it is a per-term test inside the loop; the filter only drops labels whose every term is 0). Covered by the fingerprint renders.
4. `subset()` and the shrink loop change glyph boxes: after Task 2 both must keep `Placement.glyphs` in step (a shrunk water label's parts change size). Test in Task 2.
5. Water labels moved by `place_by_priority.solve`'s `start` (a new `Placement` per step) must have glyph boxes at their start positions, not their anchors. Test in Task 2.

---

### Task 1: `country_cost` skips country labels that can't reach its candidates

**Files:**
- Modify: `src/worldmap/labels/kernels.py` (`country_cost`)
- Test: `tests/test_placement.py`

**Interfaces:**
- Unchanged signatures. Inside `country_cost`, before the candidate loop, gather `near_labels` (int64 indices, like `near` for glyphs): every `j != i`, `not water[j]`, whose extent meets the (dot-grown, if `out[i]`) envelope. Extent = its box; if `out[j]`, also its dot +- `wt[DOT_RADIUS] * height[j]` and its leader end (`nearest point of its box to its dot`). The candidate loop then runs over `near_labels` only.

- [ ] **Step 1: Write the failing test** `test_near_labels_holds_every_label_that_costs`: a `Placement` of ~40 country labels (`country_label` at spread lon/lats, `place_labels(..., freedom=all FIXED)`), a few set `out` with dots far from their boxes, one of them with its leader running past label 0. For label 0 and for one 'out' label i, with its candidates `cx, cy = candidates(p, labels, i)`: each other label j's contribution is `cost_batch` on `subset(p, {i, j})` minus `cost_batch` on `subset(p, {i})`; assert every j with a nonzero contribution is in `kernels.near_labels(...)` for i's envelope, and that `near_labels` holds fewer than all of them (the filter does something).
- [ ] **Step 2: Run** `.venv/bin/pytest -n0 tests/test_placement.py -k holds_every_label`: FAIL (`kernels` has no `near_labels`).
- [ ] **Step 3: Implement** `near_labels(i: int, ex0: float, ey0: float, ex1: float, ey1: float, x, y, width, height, anchor_x, anchor_y, out, water, dot_radius: float) -> np.ndarray` (int64, njit) in `kernels.py`, and an njit `envelope(...)` both it and `country_cost` use so the test takes the same envelope; `country_cost` loops over `near_labels` only.
- [ ] **Step 4: Run** `.venv/bin/pytest` (fingerprints unchanged) and `split.py`; expect the country kernel well under its 3.9 s.
- [ ] **Step 5: Commit** `kernels: country_cost looks only at country labels within reach of its candidates` (body: before/after timings).

### Task 2: glyph boxes kept in the `Placement`

**Files:**
- Modify: `src/worldmap/labels/placement.py` (`Placement`, `glyph_boxes`, `place_labels`, `subset`, `move_to_best`, the shrink loop)
- Test: `tests/test_placement.py`

**Interfaces:**
- `Placement` gains `glyphs: np.ndarray` (parts, 4): every water glyph box where its label is now, and `first: np.ndarray` (int, n + 1): label k's glyphs are rows `first[k]:first[k + 1]` (`owner` is ascending by construction; country labels have empty ranges). Both built in `place_labels` and `subset`.
- `placement.move(p: Placement, i: int, x: float, y: float) -> None`: sets `p.x[i]`, `p.y[i]` and, for a water label, `p.glyphs[first[i]:first[i+1]] = p.parts[...] + (x, y, x, y)` (the same arithmetic as `glyph_boxes`, so bit-identical).
- `glyph_boxes(p)` returns `p.glyphs` (kept for its callers; delete it only if every caller is changed, grep `src` and `tests`).
- Every write to `p.x`/`p.y` or `p.parts` in `placement.py` goes through `move` or refreshes the label's glyph rows (start positions, `subset`, the shrink loop). `priority.py` writes none (verify by grep).

- [ ] **Step 1: Write the failing tests** `test_glyphs_follow_their_label` (after `place_labels([], frame, water=[sea, bay])`, `np.array_equal(p.glyphs, p.parts + np.stack([p.x[p.owner], p.y[p.owner]] * 2, -1)[:, [0, 1, 0, 1]])`, i.e. exactly what `glyph_boxes` computed before), the same after a shrink (`monkeypatch` `clashes` to always report the water labels, as `test_floors_hold` does), after `subset`, and with a `start` away from the anchors (Review Focus 4, 5).
- [ ] **Step 2: Run** them: FAIL (`Placement` has no `glyphs`).
- [ ] **Step 3: Implement** the Interfaces above.
- [ ] **Step 4: Run** `.venv/bin/pytest` (fingerprints unchanged), `.venv/bin/mypy src`, `split.py`: `glyph_boxes` near 0 s.
- [ ] **Step 5: Commit** `placement: +Placement.glyphs, moved with their label, in place of glyph_boxes rebuilding them` (body: timings). Then update CLAUDE.md's `--label countries,water` timing if it moved by a second or more (`CLAUDE: timings`).

### Task 3: Measure, and stop or go on

- [ ] **Step 1:** Run `split.py` and the cProfile line (`python -m cProfile -o prof.out -m worldmap ... --label countries,water --dpi 50`, sorted by cumulative). Report the stage times to the user.
- [ ] **Step 2:** If the water kernel still dominates the solver, propose (don't build) a uniform grid of glyph and box rects per `Placement`, rebuilt per `move`, for its obstacle gathering, with its expected saving from the numbers; otherwise stop: the rest of the run (water label layout ~6 s, mostly matplotlib text measuring; drawing ~2 s) is the next target, a different plan.

## Appendix: `split.py`

```python
import collections, sys, time
from worldmap import cli
from worldmap.labels import kernels, placement

t, n = collections.Counter(), collections.Counter()
for mod, name in ((kernels, "country_cost"), (kernels, "water_cost"), (placement, "glyph_boxes")):
    def wrap(*a, _f=getattr(mod, name), _n=name):
        s = time.perf_counter(); r = _f(*a); t[_n] += time.perf_counter() - s; n[_n] += 1
        return r
    setattr(mod, name, wrap)
cli.main(sys.argv[1:])
print({k: (round(v, 2), n[k]) for k, v in t.items()}, file=sys.stderr)
```

Run from the scratchpad: `.venv/bin/python split.py out.png --label countries,water --dpi 50`.
