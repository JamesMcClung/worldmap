# Readability cleanup of make_map — design

## Intent

`src/worldmap/make_map.py` (one ~1,070-line file) has become hard to read and change. The trigger was tweaking the pin
colouring: the shading code is a run of unnamed constants and few-letter variable names. More generally, there are no type
annotations, and about 470 lines sit inside `main()`, where ~10 nested functions share loose variables (`minx`, `PPU`, `t`,
`edge`, `X`, `W_`, `OUT_`, `REGIONS`, ...). So it's hard to know what a function depends on, and nothing can be called or
tested on its own.

The goal is a codebase where you can open the file for one concern (pins, label placement, colouring), see its inputs and
outputs from type signatures, and change a named constant instead of hunting for a bare number.

## Goals

1. **Architecture first:** split into a package of focused modules, with typed functions and no nested-in-`main` helpers.
2. **Tests:** a pixel-exact regression test of the rendered output, plus unit tests for pure pieces.
3. **Then the smaller things:** descriptive variable names, named constants for magic numbers, removal of dead or
   duplicated code.

## Non-goals

- No new features, flags, or visual changes made for their own sake.
- No change to the command-line interface (`make_map [out] --recolor --ocean --earth --pins --pin-size --pick`) or to the
  `.cache/` data layout.
- No tuning of the label-placement algorithm or the palette.

## Anti-goals

- A refactor that silently changes the map. Every output change must be deliberate and visible (see the stability rule).
- Abstraction for its own sake: class hierarchies, plugin registries, config frameworks. Prefer plain functions and small
  dataclasses.

## Output-stability rule (decided by the owner)

- The regression test requires **pixel-identical** output for a fixed rng seed.
- Small rendering differences are **allowed** when they make the code simpler. In that case the commit that causes the
  change also regenerates the reference images, and its message body says what changed visually and why.
- Pure-structure commits (moves, renames, typing) must keep the output pixel-identical.

## Target structure (agreed in outline; exact boundaries are the builder's call)

Package `src/worldmap/`:

| Module | Contents |
|---|---|
| `cli.py` | `parse_args` and a short `main()` that calls each stage in order. The console script `make_map` points here. |
| `data.py` | `fetch`, file names and URLs, loading and filtering countries/states/lakes, `prep` (clip and project) |
| `frame.py` | `MapFrame` dataclass: projection transformer, globe outline, extent, map↔pixel conversion (replaces `minx`/`PPU`/`t`/`edge`) |
| `colorspace.py` | OKLab/OKLCH conversions, one `fit_gamut` (replaces the three copies of the "reduce chroma until in gamut" loop), `darken` |
| `coloring.py` | `adjacency`, `add_proximity_edges`, `color_graph`, `make_palette`, the `coloring.json` cache, the spot-check print |
| `earth.py` | `earth_values`, `earth_fill`, building per-polygon regions |
| `pins.py` | `load_pins`, `pin_sprite` (shading constants named and grouped, e.g. a `PinStyle` dataclass), `draw_pins`, the `--pick` window |
| `labels/` | `Label` dataclass (replacing the empty class with attributes added later), anchor/size/wrap, segment and rectangle geometry helpers, placement (cost weights named in one place; candidates; optimiser) |
| `render.py` | figure setup, fills, borders, state lines, lakes, Baikonur hatching, saving |

State currently shared through closures in `main()` becomes explicit parameters or small dataclasses (e.g. `MapFrame`, a
placement-state object holding the label position arrays). Importing any module has no side effects.

## Tests

- **pytest**, in `tests/`, added as an optional dev dependency in `pyproject.toml`.
- **Regression (pixel-exact):** render the map and compare a SHA-256 of its decoded pixel data to a committed
  **fingerprint**. Covers the default map, `--earth`, and `--pins` with a small committed pin list. `--ocean` (~50 s) is
  covered by a test marked slow, skipped by default.
  - **Fingerprints are committed** (decided by the owner) in a small JSON file in `tests/`, one entry per mode, each with the
    hash plus the Python, matplotlib, freetype (and other rendering-relevant) versions it was made with. A commit that
    deliberately changes the output updates this file in the same commit, so `git bisect` and rebasing see the right
    expectation at every commit. (`*.json` is gitignored, so `.gitignore` needs an exception for it.)
  - If the current library versions differ from the recorded ones, the test skips with a clear message rather than
    failing misleadingly.
  - A missing fingerprint is a **failure**, never silently created. An explicit opt-in (e.g. `UPDATE_REFERENCE=1`)
    rewrites the fingerprints.
  - The rendered PNGs are also saved to `.cache/reference/` (not committed) so a failing diff can be inspected visually.
- **Unit tests** for pure pieces: colour conversion round-trips and `fit_gamut`; `color_graph` gives a proper colouring;
  `load_pins` parsing and errors; segment/rectangle intersection helpers; `pin_sprite` shape and alpha range; `two_line`.
- These are the first commits, before any restructuring, so every later commit is checked against them.

## Order of work

1. Add the test setup and regression references (from the current code).
2. Structural split, in small commits that each keep the program running (per the owner's commit rules in the global
   CLAUDE.md: one logical change per commit, `X: desc` / `X; *: desc` messages).
3. Type annotations throughout (can be folded into step 2 as each module is extracted).
4. Naming and magic-number cleanup, module by module, and removal of dead or duplicated code.
5. Update the repo `CLAUDE.md` and `README.md` to describe the new layout.

## Builder's calls (marked as such)

- **Committed pin list for the `--pins` test:** `*.txt` is gitignored, so add a `.gitignore` exception for files under
  `tests/` (or use another extension) (*my call*).
- **Tests use the real Natural Earth data** from `.cache/` (*my call*); no fixtures or mocks of the data.
- **The rng seed** (currently `default_rng(7)`) stays a fixed module constant (*my call*).
- **Type checking:** annotations are written to pass `mypy` in non-strict mode, but no type checker is enforced in a
  hook (*my call*).
- **Exact module boundaries, names and dataclass fields** are left to the builder, within the table above.
