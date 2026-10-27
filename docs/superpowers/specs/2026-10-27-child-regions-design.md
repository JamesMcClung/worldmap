# Child regions (`--regions`) — design

From the "Disconnected Regions" item in `todo.md`.

## Intent

The poster should read as a map of countries. Dependencies and outlying or detached parts of a country (Alaska,
Greenland, Kaliningrad, the Canaries, Bermuda...) belong to their parent: same political colour, no label of their own.
Someone who wants a little more detail can pass `--regions` to name these places, each followed by its parent in
parentheses and set in italics so it reads as secondary to the country labels.

## Goals

- **Colour (political mode), always, with or without the flag:** every *dependency* (defined below) takes its parent's
  fill. This includes ones that are never labelled (Hong Kong, Macau).
- **Default labels (no flag):** no child region is labelled. This changes today's default poster (see "Changes to the
  default output").
- **`--regions`:** adds an italic label per *labelled child region* (defined below): `<region> (<parent>)`, e.g.
  *Tasmania (Australia)*, *GREENLAND (DENMARK)* (labels over `UPPERCASE_ABOVE_PT` are capitalised as usual).
- Works in both political and `--earth` mode, and with `--pins`.
- `--earth`: every polygon is already coloured on its own (`earth.py`), so child regions keep their own land-cover
  colour. No change needed there.
- Label text colour is the darkened fill of the region it sits on (as for exclaves today, via `fill_of`).

## Non-goals / anti-goals

- Baikonur behaviour is unchanged (todo: explicit non-goal). The same goes for the other lease, Guantánamo Bay.
- Disputed regions (Kosovo, Taiwan, Western Sahara, Northern Cyprus, Somaliland, Palestine, Siachen...) are a separate
  todo item. Their treatment must not change here.
- No threshold argument on `--regions` yet. The user wants to see the fully labelled version first and may later add one
  (e.g. a minimum size) to thin it out.
- Water labels are a separate todo item.
- Anti-goal: the flag must not change political colours, so the colouring (and `coloring.json`) doesn't depend on
  `--regions`.

## Definitions

### Dependencies (separate Natural Earth features)

A feature of `ne_10m_admin_0_countries` is a **dependency** when `SOVEREIGNT != ADMIN` and `TYPE` is not `"Lease"` or
`"Indeterminate"`. Its **parent** is the feature whose `ADMIN` equals its `SOVEREIGNT`. Today's data has 46 of these
(50 rows with `SOVEREIGNT != ADMIN`, minus Baikonur and Guantánamo (leases), and Palestine and Siachen (indeterminate)).
Every parent is a UN member state, which satisfies the user's rule that only UN members and observers can be parents. A
test should pin that: every dependency's parent is a labelled country.

Their parents: Australia (5), China (2), Denmark (2), Finland (1), France (8), Netherlands (3), New Zealand (2),
United Kingdom (17, including the TYPE "Disputed" Falklands, Gibraltar and British Indian Ocean Territory), United
States (6).

A dependency is **labelled** with `--regions` unless it touches its parent. Today that excludes only Hong Kong and Macau.
So 44 dependencies get labels: Greenland, Faroes, Åland, Puerto Rico, Guam, US Virgin Islands, Northern Marianas, American
Samoa, US Minor Outlying Islands, the French ones (New Caledonia, French Polynesia, St Martin, St Barthélemy,
St Pierre & Miquelon, Wallis & Futuna, Clipperton, French Southern & Antarctic Lands), Aruba, Curaçao, Sint Maarten, Cook
Islands, Niue, Norfolk Island, the other Australian territories, and all the UK ones (Jersey, Guernsey, Isle of Man,
Gibraltar, Akrotiri, Dhekelia, Bermuda, Caymans, Falklands, South Georgia, ...). Note that Gibraltar, Akrotiri and Dhekelia
touch *other* countries but not the UK, so by this rule they are labelled. This is intended: the user wants to see the
fully labelled version.

The adjacency test (touches, or within a small tolerance, of the parent geometry) is for the builder to choose.

### Parts of a parent's polygon (hand-picked)

These are not separate features. They are polygons inside the parent's MultiPolygon (verified by point-in-polygon on the
current data). The list is fixed and labelled only with `--regions`. Their colour is already the parent's.

| Region | Parent | Kind |
|---|---|---|
| Kaliningrad | Russia | exclave (labelled today via `EXCLAVES`) |
| French Guiana | France | overseas, mainland (labelled today via `EXCLAVES`) |
| Alaska | United States | detached |
| Hawaii | United States | island group |
| Svalbard | Norway | island group |
| Tasmania | Australia | island |
| Galápagos | Ecuador | island group |
| Easter Island | Chile | speck |
| Cabinda | Angola | exclave |
| Nakhchivan | Azerbaijan | exclave |
| Oecusse | East Timor | exclave |
| Musandam | Oman | exclave |
| Canary Islands | Spain | island group |
| Azores | Portugal | island group |
| Madeira | Portugal | island group |
| Zanzibar | Tanzania | island group |
| Socotra | Yemen | island |
| Andaman & Nicobar | India | island group |
| Bioko | Equatorial Guinea | island |
| Réunion | France | island |
| Guadeloupe | France | island group (tiny) |
| Martinique | France | island (tiny) |

With `--regions`, that's 44 + 22 = 66 child labels.

## Decided

- **Flag name:** `--regions`, a plain switch for now (my call). Help text along the lines of "label dependencies and
  outlying regions (Alaska, Greenland, Kaliningrad...) with their parent country".
- **Label text:** `<region name> (<parent's label text>)`, using the parent's name as it appears on the map, i.e. after
  `RENAME` ("United States", not "United States of America"; "East Timor" as NE names it). Dependencies use `NAME_EN`.
  (My call.)
- **Style:** italic (`fontstyle="italic"`), otherwise the same as country labels: font size from the region's own area
  with the usual clamps, uppercase above 14 pt, same halo, and the same in/out and wrap rules. Specks become 'out' labels
  with leaders through the existing rules. Long names ("South Georgia and the South Sandwich Islands (United Kingdom)")
  go through the normal two-line wrap.
- **Default labels removed:** Greenland, Faroe Islands, Puerto Rico and Falkland Islands no longer get country labels.
  The two `EXCLAVES` labels (Kaliningrad, French Guiana) appear only with `--regions`. `EXCLAVES` and the dependency
  entries in `LABELED_EXTRA` and `NOT_LABELED` are subsumed by the new rules. Palestine stays in `LABELED_EXTRA`.
- **Political colouring:** each dependency is merged into its parent's node *in the colouring graph*, not just repainted
  afterwards. So the parent's class must also differ from the dependency's neighbours, e.g. the UK vs Spain via
  Gibraltar, or Denmark vs Canada via Greenland's across-water links. This changes the colouring logic, so bump
  `COLORING_RULE`. Dependencies stop counting as "major" countries for `add_proximity_edges` (today `major` =
  `is_labeled`). Check the `spot_check` list and the 5-colourability still hold. (My call on the merge approach.)
- **Parent labels unchanged:** a parent's own label is anchored and sized as today (e.g. the US label stays on the
  contiguous states). (My call.)
- **Earth mode:** no colouring changes. A child label's colour follows the polygon it labels (for an island group, its
  largest polygon), via the existing `fill_of(i, pt)`.

## Left to the builder

- **How hand-picked regions are defined.** Suggested: per region, a parent name plus a lon/lat bounding box. The region is
  the parent's polygons whose representative point lies in the box. This generalises the `EXCLAVES` point test, which
  only works for single polygons. Boxes must be loose enough to catch every island in a group (Hawaii, Andaman & Nicobar,
  Svalbard) and tight enough not to grab mainland pieces.
- **Anchoring multi-polygon regions and dependencies:** either like island-heavy countries (convex-hull centroid,
  hull-based size) or the largest polygon's inscribed circle. Pick whichever reads better for Hawaii, the Canaries,
  French Polynesia and the Caymans.
- Where the region definitions live (likely a new `labels/regions.py`) and how `make_labels` gets told about the flag.
- Whether `Label` needs a `child` or `italic` field, or the `Text` just carries the style.
- Crowding: 66 extra labels, mostly small and clustered in the Caribbean, the Pacific and the Channel. Tune within the
  existing `CostWeights` if placement gets visibly worse, rather than dropping regions. Report what was seen, since this
  feeds the later threshold decision.

## Changes to the default output

The default, `--earth` and `--pins` renders all change: four labels are removed, and in political mode all 46
dependencies take their parent's fill, which may also reshuffle other countries' colour classes. Per CLAUDE.md:
re-record with `UPDATE_REFERENCE=1`, commit `tests/fingerprints.json` in the same commit as the change that alters the
output, and say in the body what changed visually. The `COLORING_RULE` bump forces the recolour.

Add a fourth regression mode, `--regions` (political), so the new labels are pinned too.

## Future (not this work)

- A threshold argument for `--regions`, once the user has seen the full version.
- User's idea for the "Disputed Regions" item: a disputed region might be modelled as a child of *several* states,
  with no label of its own.

## Testing

- Unit:
  - the dependency rule yields the expected set and every dependency's parent exists and is a labelled country
  - Hong Kong and Macau are the only adjacent, unlabelled ones
  - every hand-picked region selects at least one polygon of its parent, and no polygon is claimed by two regions
  - label text format
  - child labels are italic
  - without the flag, no child region is labelled
- Colouring: every dependency has its parent's class, and the `spot_check` pairs are still empty.
- Visual check by cropping, in both modes:
  - Caribbean
  - Channel Islands and Isle of Man
  - North Atlantic (Greenland, Faroes, Svalbard, Azores, Madeira, Canaries)
  - Cyprus (Akrotiri and Dhekelia)
  - Gibraltar
  - Gulf of Oman (Musandam)
  - Caucasus (Nakhchivan)
  - Timor
  - Pacific (Hawaii, Guam and the Marianas, New Caledonia, French Polynesia, Cook Islands, Easter Island, Galápagos)
  - South Atlantic and southern Indian Ocean
- Update `README.md` and `CLAUDE.md` (pipeline, colouring, label notes, the `EXCLAVES` mention) and tick the todo item.
