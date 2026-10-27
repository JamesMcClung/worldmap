"""Graph colouring of the country features: which countries share one of the N_COLORS colour classes, and the
palette those classes get. Neighbours (by land border, or across up to MAX_GAP of water) never share a class."""

import json
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import permutations

import geopandas as gpd
import numpy as np
from matplotlib.colors import to_rgb
from shapely import STRtree, affinity

from worldmap.colorspace import RGB, fit_gamut, srgb_to_oklch
from worldmap.data import CACHE, COUNTRIES, fetch

Edge = tuple[int, int]  # (i, j) feature indices, i < j

N_COLORS = 5  # country colour classes
MAX_GAP = 600_000  # metres: countries closer than this across water count as neighbours
MIN_DEG = 3  # every feature gets at least this many real-country neighbours
OCEAN = "#e3f0f7"  # fixed ocean colour; its hue is pinned as the (N_COLORS + 1)th hue
EQUAL_AREA_CRS = "EPSG:6933"
UNWEIGHTED_LARGEST = 10  # the biggest countries don't count towards colour balance (rank < this, so 9 in practice)
COLORING_RULE = "top10-exempt-count-balance/v2"  # bump when the colouring logic changes, to invalidate the cache
SPOT_CHECK_PAIRS = [("South Korea", "Japan"), ("Russia", "United States of America"), ("Malaysia", "Philippines"), ("Italy", "Greece"), ("France", "United Kingdom"), ("United Kingdom", "Norway"), ("India", "Sri Lanka")]


def adjacency(gdf: gpd.GeoDataFrame) -> set[Edge]:
    """Edges between features that share a border (more than a single point)."""
    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    edges = set()
    for i, g in enumerate(geoms):
        for j in tree.query(g, predicate="intersects"):
            if j > i and g.intersection(geoms[j]).length > 0:
                edges.add((i, int(j)))
    return edges


def add_proximity_edges(gdf: gpd.GeoDataFrame, edges: set[Edge], major: list[bool], k: int, max_gap: float = 600_000, min_deg: int = 3) -> set[Edge]:
    """Link countries separated by water so they don't share a colour, without exceeding k colours.
    Candidate links (real countries within `max_gap` metres of each other) are tried largest-country-first and
    kept only if the graph stays k-colourable; then each feature is topped up to `min_deg` real-country
    neighbours via its nearest ones under the same feasibility rule."""
    t0 = time.time()
    WORLD = 2 * np.pi * 6378137.0  # width of EPSG:6933 in metres (wrap-around for the antimeridian)
    g6933 = gdf.to_crs("EPSG:6933").geometry
    sim = [g.simplify(10_000) for g in g6933]
    area = np.array([g.area for g in g6933])
    n = len(sim)
    tree = STRtree(sim)
    have = set(edges)
    # Keep one valid k-colouring and extend the graph edge by edge, repairing the colouring locally when needed.
    _, col = color_graph(n, have, [1.0] * n, kmin=k, kmax=k)
    nbr = [set() for _ in range(n)]
    for i, j in have:
        nbr[i].add(j)
        nbr[j].add(i)

    def try_add(e: Edge) -> bool:
        u, v = e
        if col[u] != col[v]:
            ok = True
        else:  # same colour: recolour one endpoint if some other colour is free
            ok = False
            for x in sorted((u, v), key=lambda q: len(nbr[q])):
                free = [c for c in range(k) if c != col[x] and all(col[y] != c for y in nbr[x])]
                if free:
                    col[x] = free[0]
                    ok = True
                    break
        if ok:
            have.add(e)
            nbr[u].add(v)
            nbr[v].add(u)
        return ok

    def near(i: int, maxd: float) -> dict[int, float]:
        out: dict[int, float] = {}
        for dx in (0, WORLD, -WORLD):
            g = affinity.translate(sim[i], dx) if dx else sim[i]
            for j in tree.query(g, predicate="dwithin", distance=maxd):
                j = int(j)
                if j != i and major[j]:
                    out[j] = min(out.get(j, np.inf), g.distance(sim[j]))
        return out

    cands: dict[Edge, float] = {}
    for i in range(n):
        if major[i]:
            for j, d in near(i, max_gap).items():
                e = (min(i, j), max(i, j))
                if e not in have:
                    cands[e] = d
    added = 0
    for e in sorted(cands, key=lambda e: -min(area[e[0]], area[e[1]])):  # biggest countries first
        if try_add(e):
            added += 1
    deg = [0] * n
    for i, j in have:
        deg[i] += int(major[j])
        deg[j] += int(major[i])
    topped = 0
    for i in sorted(range(n), key=lambda q: deg[q]):
        if deg[i] >= min_deg:
            continue
        d = near(i, 5_000_000)
        for j in sorted(d, key=d.get):
            if deg[i] >= min_deg:
                break
            e = (min(i, j), max(i, j))
            if e in have or not try_add(e):
                continue
            deg[i] += 1
            deg[j] += int(major[i])
            topped += 1
    print("proximity edges: %d across water (<= %d km) + %d top-ups to >=%d neighbours (%.1fs)" % (added, max_gap // 1000, topped, min_deg, time.time() - t0))
    return have


def color_graph(n: int, edges: Iterable[Edge], weights: Sequence[float], kmin: int = 1, kmax: int = 8, limit: int = 2_000_000) -> tuple[int, list[int]]:
    """Fewest-colour proper colouring (exact DSATUR backtracking); prefers the least-used colour by area."""
    nbr: list[set[int]] = [set() for _ in range(n)]
    for i, j in edges:
        nbr[i].add(j)
        nbr[j].add(i)
    for k in range(kmin, kmax + 1):
        col = [-1] * n
        used = [0.0] * k
        steps = [0]

        def pick() -> int:
            best, key = None, None
            for v in range(n):
                if col[v] >= 0:
                    continue
                sat = len({col[u] for u in nbr[v] if col[u] >= 0})
                kk = (sat, len(nbr[v]))
                if key is None or kk > key:
                    best, key = v, kk
            return best

        def solve(left: int) -> bool:
            if left == 0:
                return True
            steps[0] += 1
            if steps[0] > limit:
                return False
            v = pick()
            taken = {col[u] for u in nbr[v] if col[u] >= 0}
            for c in sorted((c for c in range(k) if c not in taken), key=lambda c: used[c]):
                col[v] = c
                used[c] += weights[v]
                if solve(left - 1):
                    return True
                col[v] = -1
                used[c] -= weights[v]
            return False

        if solve(n):
            return k, col
    raise RuntimeError("no colouring found")


RED_HUE, GREEN_HUE = 29.0, 142.0  # OKLCH hues of sRGB red and green


def lightness_for_hue(h: float, l_red: float, l_green: float) -> float:
    """Lightness ramps (raised cosine) from darkest at red to lightest at green, so a red/green colour-blind viewer
    can tell those hues apart by brightness. Other hues interpolate along the two arcs between them."""
    t = (h - RED_HUE) % 360
    arc = (GREEN_HUE - RED_HUE) % 360
    if t <= arc:  # red -> green, rising
        u = t / arc
        return l_red + (l_green - l_red) * (1 - np.cos(np.pi * u)) / 2
    u = (t - arc) / (360 - arc)  # green -> red the long way round, falling
    return l_green - (l_green - l_red) * (1 - np.cos(np.pi * u)) / 2


def make_palette(k: int, class_pairs: set[Edge], C: float = 0.115, l_red: float = 0.64, l_green: float = 0.77) -> list[RGB]:
    """k country hues: the ocean's hue is one of k+1 equispaced hues (pinned); countries get the other k.
    Each hue gets its own lightness (dark red .. light green). The hues are assigned to colour classes so that
    classes that end up adjacent are as far apart as possible in OKLab (hue *and* brightness)."""
    ocean_h = srgb_to_oklch(to_rgb(OCEAN))[2]
    hues = [(ocean_h + (i + 1) * 360 / (k + 1)) % 360 for i in range(k)]
    rgbs, labs = [], []
    for h in hues:
        L = lightness_for_hue(h, l_red, l_green)
        rgb, cc = fit_gamut(L, C, h)  # most chroma the sRGB gamut allows, up to C
        rgbs.append(rgb)
        labs.append((L, cc * np.cos(np.radians(h)), cc * np.sin(np.radians(h))))
    D = np.linalg.norm(np.array(labs)[:, None] - np.array(labs)[None], axis=-1)
    best = max(permutations(range(k)), key=lambda p: (min(D[p[a], p[b]] for a, b in class_pairs), sum(D[p[a], p[b]] for a, b in class_pairs)))
    print("ocean hue %.0f; country hues/L:" % ocean_h, [(round(hues[best[c]]), round(labs[best[c]][0], 2)) for c in range(k)], "min adjacent OKLab distance %.3f" % min(D[best[a], best[b]] for a, b in class_pairs))
    return [rgbs[best[c]] for c in range(k)]


@dataclass
class Coloring:
    k: int  # number of colour classes used
    color_class: list[int]  # per country feature
    class_pairs: set[Edge]  # colour classes that touch somewhere


def load_or_compute_coloring(countries: gpd.GeoDataFrame, major: list[bool], recolor: bool) -> Coloring:
    """The graph colouring, cached in coloring.json so palette tweaks never reshuffle it and runs skip the
    border/distance work. Rebuilt when the inputs change or when `recolor` is set."""
    cache = CACHE / "coloring.json"
    key = {"n_colors": N_COLORS, "max_gap": MAX_GAP, "min_deg": MIN_DEG, "rule": COLORING_RULE, "files": {COUNTRIES: os.path.getsize(fetch(COUNTRIES))}}
    names = list(countries.NAME_EN)
    if not recolor and cache.exists():
        try:
            cached = json.loads(cache.read_text())
            if cached["key"] == key and cached["names"] == names:
                print("colouring: loaded from coloring.json")
                return Coloring(cached["k"], cached["col"], {(a, b) for a, b in cached["pairs"]})
        except (ValueError, KeyError):
            pass
    edges = add_proximity_edges(countries, adjacency(countries), major, N_COLORS, max_gap=MAX_GAP, min_deg=MIN_DEG)
    # Balance colour usage by country count, but don't count the largest countries, so their colours aren't penalised.
    areas = countries.to_crs(EQUAL_AREA_CRS).geometry.area
    weights = [0.0 if rank < UNWEIGHTED_LARGEST else 1.0 for rank in areas.rank(ascending=False, method="first")]
    k, color_class = color_graph(len(countries), edges, weights, kmin=N_COLORS)
    class_pairs = {(min(color_class[i], color_class[j]), max(color_class[i], color_class[j])) for i, j in edges}
    cache.write_text(json.dumps({"key": key, "names": names, "k": k, "col": color_class, "pairs": sorted(class_pairs)}))
    print("colouring: computed and saved to coloring.json")
    return Coloring(k, color_class, class_pairs)


def spot_check(countries: gpd.GeoDataFrame, color_class: Sequence[int]) -> list[tuple[str, str]]:
    """Pairs from SPOT_CHECK_PAIRS that ended up the same colour (should be empty)."""
    by_name = dict(zip(countries.NAME_EN, color_class))
    return [(a, b) for a, b in SPOT_CHECK_PAIRS if by_name[a] == by_name[b]]
