# Disputed regions (`--disputed`) — design

From the "Disputed Regions" item in `todo.md`.

## Intent

Today the poster shows Natural Earth's de facto borders: Kashmir is split along the line of control, Crimea is
simply part of whichever country the data puts it in, and Taiwan is a plain country. With `--disputed`, every area
that more than one country claims is drawn in equal-width stripes of the colours of everyone involved, and the borders
between it and its claimants are dashed. The stripes and dashes should make it reasonably clear who claims what without
extra labels. It does not have to be 100% clear: small islands may be too small to read, and that's fine.

## Goals

- **`--disputed` flag**, off by default. Without it, the output is unchanged except for the Baikonur fix below.
- **Stripes:** each disputed area is filled with equal-width stripes, one colour per country involved (the one that
  runs it plus every claimant). The order of the colours doesn't matter.
- **More than 2 countries involved** (counting the one running it): the whole area is filled plain `DISPUTED_GREY`
  instead, no stripes (user's decision: many colours would be too ambiguous). That covers the Spratlys, Paracels,
  Senkakus, Rockall, British Indian Ocean Territory, Tromelin, Bajo Nuevo, Serranilla, Sapodilla Cayes, Scarborough
  Shoal... Dashed borders still apply.
- **Stripe colours** are the political fills of those countries (`graph_fills`), so they match the countries around them.
- **The two countries of every 2-country dispute always get different colours.** The graph colouring only separates
  neighbours, so today the US and the Marshall Islands (Wake Island) share a colour class. Add an edge between the two
  countries of each 2-country dispute to the colouring graph (alongside `adjacency` and the proximity edges), and bump
  `COLORING_RULE`. The edges are added whether or not `--disputed` is given, so colours never depend on the flag; this
  may shift some colours on the default map (user agreed), with the spot-check still empty. Re-record the
  fingerprints in that commit.
- **Stripes are drawn under the border lines.** Today Baikonur's hatching is drawn over its border. Fix that in the
  default map too (this changes the default, earth and other fingerprints: re-record them in that commit).
- **Dashed borders:** a border segment between a disputed area and any country involved in it (the one running it
  or a claimant) is dashed instead of solid. The same goes for a segment between two disputed areas that share a country
  (Aksai Chin and Kashmir). Where it meets an uninvolved country, the border stays solid. *My call:* the running
  country counts too, because every disputed polygon lies inside one country's polygon (Kashmir and Arunachal inside
  India, Crimea inside Russia, Abkhazia inside Georgia...). Without this, the stripes of Kashmir would end against the
  rest of India with no line at all. So the dashed segments are mostly *new* lines (each disputed polygon's outline),
  not restyled country edges.
- **Labels:** a disputed area that governs itself and claims independence (Natural Earth note starts with
  "Self admin.") has a label: Taiwan, Kosovo, Somaliland, Northern Cyprus, Abkhazia, South Ossetia, Transnistria,
  Nagorno-Karabakh, the Donetsk and Luhansk regions, Western Sahara. Taiwan's label reads `Taiwan`. *My call:* the
  labels use plain geographic names, not Natural Earth's official-style ones: `Northern Cyprus` (not "Turkish Republic
  of Northern Cyprus"), `Donetsk`, `Luhansk`. These labels show with `--label countries` (upright, like countries). Every other
  disputed area gets no label of its own (Kashmir, Crimea, the Kurils...). Existing labels are unaffected: the
  Falklands keep their `--label territories` label, Israel and Palestine keep theirs.
- **`--color earth`:** no stripes (natural colours don't stand for countries), but the dashed borders still apply.
- Works with `--pins` and every `--label` combination.

## Non-goals / anti-goals

- Not a statement of who is right. Use Natural Earth's notes as they are; the hand-written table (below) only fills in
  entries Natural Earth leaves blank.
- No labels for disputed areas that don't claim independence: the colours and dashes do that job.
- The flag must not change which colour a country gets (the dispute edges above are always in the graph).
- Anti-goal: stripes that cover border lines, or stripes of visibly unequal width.

## Data

Natural Earth `cultural/ne_10m_admin_0_disputed_areas.zip` (v5.1.1, the same release as the countries file), fetched
with `data.fetch` like the other files (only when `--disputed` is given). It has 99 features. The fields that matter:

- `ADMIN` / `SOVEREIGNT`: who runs it.
- `NOTE_BRK`: free text such as `Admin. by India; Claimed by Pakistan`, `Self admin.; Claimed by Georgia`,
  `Claimed by China, Taiwan, Malaysia, the Philippines, and Brunei`, `Admin by U.K.; Claimed by Ireland, Denmark, and Iceland`.
- `TYPE`: `Disputed`, `Breakaway`, `Indeterminate`, `Lease`, `Overlay`, `Geo unit`, `Geo subunit`.

### Which features are striped

Every feature whose `NOTE_BRK` names at least one claimant, plus the hand-written table. That includes leases with a
claim (Guantánamo: US/Cuba) and geo (sub)units with a claim (South Georgia, Juan de Nova...).

Excluded: overlays with no claim (Korean DMZ, UNDOF zone, Northern Limit Line), Ceuta and Melilla (no claim noted),
`Claimed by multiple` (Isla Aves; there is no list to draw).

**Baikonur** (`Leased to Russia by Kazakhstan`) becomes equal Russia/Kazakhstan stripes with `--disputed`; without the
flag it keeps today's Russia fill with Kazakh hatching, moved under the border.

### Claimant names

`NOTE_BRK` uses short or informal names (`U.K.`, `U.S.A.`, `Azer.`, `UAE`, `the Philippines`, `the Seychelles`,
`Admin. By` with a capital B, `Admin by` without the dot). Parse it into a list of country names and map each one onto
a feature of the countries file through a small alias table. Every name must resolve; an unknown name is an error, not
a silent skip, and a test pins that every striped feature resolves.

### Hand-written claimants

Entries with no `NOTE_BRK` get their claimants from a table in the code (my call on the contents; the user will review
it in the plan/PR):

| Feature | Countries |
|---|---|
| Kosovo | Kosovo, Serbia |
| Gaza Strip, West Bank | Palestine, Israel |
| Belize (Guatemala's claim) | Belize, Guatemala |
| Dragonja | Croatia, Slovenia |
| Vukovar Island, Šarengrad Island | Croatia, Serbia |
| Ilemi Triangle (South Sudan's polygon) | Kenya, South Sudan |
| North Borneo | Malaysia, Philippines |
| Scarborough Shoal | China, Philippines, Taiwan |

Israel's own entry (no note) is not striped. Scarborough Shoal (3 countries) is grey.

### Self-governing areas without a country of their own

Taiwan, Kosovo, Somaliland, Northern Cyprus and Western Sahara are already features of the countries file with their own
colour, so their stripes are e.g. China + Taiwan. Abkhazia, South Ossetia, Transnistria, Nagorno-Karabakh and the
Donetsk/Luhansk regions are not: see the open questions.

## Drawing

Order, bottom to top: country fills → stripes → state lines → lakes → borders (solid and dashed) → labels → pins.

- Borders are currently the edge of each country's fill patch (`ec=BORDER`). For stripes to sit under them and for some
  segments to be dashed, borders have to become their own line layer above the stripes, at least with `--disputed`.
  Without the flag the output should stay as close to today as possible (it changes only around Baikonur).
- Stripe angle, width (fixed in poster inches so every area matches), and dash pattern are the builder's choice;
  show crops for review. Areas narrower than one stripe set may show only some colours; acceptable.
- Disputed polygons sit on top of the country polygons (Kashmir lies inside India's and Pakistan's polygons, Crimea
  inside Russia's). Stripe each disputed polygon's own outline (all of them are land: coastlines or land between
  borders, never sea), with lakes still painted over it. *My call:* don't trim them to the countries file's land, because
  a few small islands (Sapodilla Cayes, Doumeira, islets of the Senkakus and Chafarinas; under 10 km² in all) exist only
  in the disputed-areas file and would vanish. (Natural Earth names the Spratlys row "Wake Island"; match on `ADMIN`/`SOVEREIGNT`, not `NAME_EN`.)
- Label placement treats a striped area like the country under it. Label text colour for Taiwan and friends: their own
  darkened fill, as today.

## Tests

- Unit: `NOTE_BRK` parsing on the awkward strings above; every striped feature resolves to known countries; the
  hand-written table's names exist; which features get labels.
- Regression: add a `disputed` fingerprint (`--disputed`), and re-record the existing ones for the Baikonur fix in its
  own commit, saying what changed.

## Decided after review

- **Breakaway areas with no country of their own** (Abkhazia, South Ossetia, Transnistria, Nagorno-Karabakh, Donetsk,
  Luhansk): stripes alternate the parent's colour with a neutral light grey (`DISPUTED_GREY`, builder picks the value).
  Their labels use a dark grey text colour.
- **Data as it is:** Natural Earth 5.1.1 is used unchanged, even where it is out of date (Nagorno-Karabakh, the extent
  of Russian-held Ukraine). No hand-drawn outlines.

## Left to the builder

Stripe angle/width, dash pattern, how borders become a line layer, the claimant alias table, how disputed polygons
are clipped, and module layout (probably a new `disputed.py`, documented in `CLAUDE.md`).
