# Priority-ordered label placement

## Why

Which labels exist is decided today by fixed cut-offs before placement starts (`is_labeled`,
`RIVER_LABEL_MAX_RANK`, `WATER_MIN_INSIDE`), and what doesn't fit is removed afterwards by
`place_with_water`'s drop loop, which re-solves everything each round. The cut-offs are tuned for the world
poster. On the world map, skipping small water labels is right; on a crop there is room for far more, but those
labels were never candidates, or get dropped as if the map were the world.

Placing labels in order of importance until nothing more fits fills any map with a fitting number of labels on
its own: sparse on the world poster, dense on a crop, without per-zoom tuning.

## Goals

- Labels are placed in priority tiers; a label that can't fit around the higher-priority ones is skipped.
- The same rule works for the world map and every crop (political and topographic).
- The `--label` order matters: it orders kinds within a tier.
- Replace the drop loop (`place_with_water`) and the special rivers-last pass (`place_all`).

## Non-goals

- Drawing smaller lakes and rivers on crops (wanted, but a later project). Labels stay limited to what is drawn.
- Changing how labels are sized, wrapped, or made 'out' with a leader (`make_labels`, `decide_out_and_wrap`,
  `lay_out`), or the cost terms themselves.
- New CLI flags.
- The standalone compiled label engine.

## Anti-goals

- Visibly worse results than today on the world map, the Pacific crop or the Alps crop. The bar is a by-eye
  comparison against the current renders (`.cache/reference/`); there is no particular failure the user fears.

## Design (decided with the user)

### Kinds

The kinds are the `--label` kinds: `countries`, `territories`, `oceans`, `seas`, `lakes`, `straits`, `rivers`
(`water` expands in place to oceans, seas, lakes, straits, rivers). Disputed self-governing areas (`--disputed`)
belong to `countries`. `--label` becomes an ordered list (first occurrence wins on repeats) instead of a set.

### Ranks

- Each kind ranks its own candidates by **full size**: area (countries, territories, oceans, seas, lakes,
  straits) or length (rivers), measured on the whole feature, not the part visible in the frame. In a France
  crop, Germany is as important as France.
- Only features **present in the frame** take part, and ranks are **relative to the biggest present feature of
  that kind**, in fixed steps: rank 1 is >= 1/`RANK_STEP` of the biggest, rank 2 >= 1/`RANK_STEP`^2, ...,
  rank 5 catches everything smaller. 5 ranks per kind; sparse kinds leave ranks empty (all oceans are rank 1).
  `RANK_STEP` starts at 10 and is tuned by eye.
- Within a (rank, kind), bigger goes first.
- "Present" means the features that produce a label candidate (after `WATER_MIN_INSIDE`, `is_labeled`, rivers
  too short for their name, the out-of-frame `child_regions` filter), not any visible geometry: a huge sea
  barely reaching into a crop shouldn't set the scale. (Builder's call.)
- **This ranking may not work out.** The user's likely fallback is ranking across all kinds at once, comparing
  sizes on one scale (length^2 of rivers vs area). Keep ranking a single swappable function from candidates to
  (rank, order) so either rule plugs into the same placement loop.

### Order of placement

With `--label countries,rivers,seas`: rank-1 countries, rank-1 rivers, rank-1 seas, then rank-2 countries, rivers,
seas, and so on down to rank 5. Each (rank, kind) is a step.

### Nudging

- While rank N is being placed, its labels are free. Rank N-1 labels may still **move and shrink a little**: both
  cost something (a cost growing with distance from where they settled, and a smaller allowed shrink than in their
  own rank). Ranks N-2 and above are fixed in place and size.
- A nudged label is never dropped.
- Each label is checked right after its own step (not its whole rank): if it fails (a water label off its water,
  or anything clashing at its minimum size) it is skipped. Once kept, later steps can only nudge it. So the
  `--label` order affects placement, not survival. (Builder's call.)
- **No more lopsided clashes.** Today a country label clashes only with country labels and water labels give way
  to everything (`placement.clashes`). Now any label clashes with any label, box, glyph or leader regardless of
  kind, and the label being placed gives way: a country can shrink and lose its label to a higher-priority sea.
  This applies to country labels too: a microstate can lose its label.

### Candidates

- `RIVER_LABEL_MAX_RANK` goes away: every drawn river (`RIVER_MAX_RANK`) is a candidate. The other existing
  filters stay as quality rules: `WATER_MIN_INSIDE` up front, `is_labeled` for countries, the marine `classify`
  rules, only drawn lakes.

### Output

Every fingerprint changes. Re-record them in the commit that switches placement over, and say what changed
visually.

## Left to the builder

- How to measure full size: an equal-area CRS (`data.EQUAL_AREA_CRS`) on the unclipped source features; rivers
  by the length of their whole joined path. The world Pacific's two halves share the whole ocean's size.
- The form and weights of the nudge costs, and how far a nudged label may shrink, tuned by eye.
- Whether a later same-rank step may nudge an earlier same-rank step's labels. Builder's default: yes, like
  rank N-1 (the user described only the rank-to-rank rule).
- The country-only cost asymmetries that follow from the clash rule (today country terms skip water boxes
  except glyph overlap); make the cost terms match the symmetric clash rule.
- How `place_labels` takes per-label freedom (free / nudgeable / fixed) instead of a `fixed` prefix, and how the
  `Placement` arrays grow as steps add labels.
- Keeping run time no worse than today (`--label countries,water` ~28 s); each step solves a smaller set, so it
  should get faster.
- Test changes: the unit tests for the drop loop, new ones for ranking.
