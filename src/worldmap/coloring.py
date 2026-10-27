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
WORLD_WIDTH_M = 2 * np.pi * 6378137.0  # width of EPSG:6933 in metres (shift for wrap-around at the antimeridian)
SIMPLIFY_M = 10_000  # outline simplification for the distance queries
TOP_UP_SEARCH_M = 5_000_000  # how far to look for top-up neighbours
COLORING_RULE = "top10-exempt-count-balance/v2"  # bump when the colouring logic changes, to invalidate the cache
SPOT_CHECK_PAIRS = [("South Korea", "Japan"), ("Russia", "United States of America"), ("Malaysia", "Philippines"), ("Italy", "Greece"), ("France", "United Kingdom"), ("United Kingdom", "Norway"), ("India", "Sri Lanka")]


def adjacency(gdf: gpd.GeoDataFrame) -> set[Edge]:
    """Edges between features that share a border (more than a single point)."""
    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    edges = set()
    for i, geom in enumerate(geoms):
        for j in tree.query(geom, predicate="intersects"):
            if j > i and geom.intersection(geoms[j]).length > 0:
                edges.add((i, int(j)))
    return edges


def neighbour_sets(n: int, edges: Iterable[Edge]) -> list[set[int]]:
    neighbours: list[set[int]] = [set() for _ in range(n)]
    for i, j in edges:
        neighbours[i].add(j)
        neighbours[j].add(i)
    return neighbours


def add_proximity_edges(gdf: gpd.GeoDataFrame, edges: set[Edge], major: list[bool], k: int, max_gap: float = MAX_GAP, min_deg: int = MIN_DEG) -> set[Edge]:
    """Link countries separated by water so they don't share a colour, without exceeding k colours.
    Candidate links (major countries within `max_gap` metres of each other) are tried largest-country-first and
    kept only if the graph stays k-colourable; then each feature is topped up to `min_deg` major-country
    neighbours via its nearest ones under the same feasibility rule."""
    start_time = time.time()
    geoms = gdf.to_crs(EQUAL_AREA_CRS).geometry
    simplified = [g.simplify(SIMPLIFY_M) for g in geoms]
    area = np.array([g.area for g in geoms])
    n = len(simplified)
    tree = STRtree(simplified)
    linked = set(edges)
    # Keep one valid k-colouring and extend the graph edge by edge, repairing the colouring locally when needed.
    _, color_class = color_graph(n, linked, [1.0] * n, kmin=k, kmax=k)
    neighbours = neighbour_sets(n, linked)

    def try_add(edge: Edge) -> bool:
        """Add `edge` if the colouring can stay proper (recolouring one endpoint if needed)."""
        u, v = edge
        if color_class[u] != color_class[v]:
            ok = True
        else:  # same colour: recolour one endpoint if some other colour is free
            ok = False
            for node in sorted((u, v), key=lambda q: len(neighbours[q])):
                free = [c for c in range(k) if c != color_class[node] and all(color_class[nb] != c for nb in neighbours[node])]
                if free:
                    color_class[node] = free[0]
                    ok = True
                    break
        if ok:
            linked.add(edge)
            neighbours[u].add(v)
            neighbours[v].add(u)
        return ok

    def near(i: int, max_dist: float) -> dict[int, float]:
        """Major features within max_dist of feature i (across the antimeridian too) -> their distance."""
        out: dict[int, float] = {}
        for shift in (0, WORLD_WIDTH_M, -WORLD_WIDTH_M):
            geom = affinity.translate(simplified[i], shift) if shift else simplified[i]
            for j in tree.query(geom, predicate="dwithin", distance=max_dist):
                j = int(j)
                if j != i and major[j]:
                    out[j] = min(out.get(j, np.inf), geom.distance(simplified[j]))
        return out

    candidates: dict[Edge, float] = {}
    for i in range(n):
        if major[i]:
            for j, dist in near(i, max_gap).items():
                edge = (min(i, j), max(i, j))
                if edge not in linked:
                    candidates[edge] = dist
    added = 0
    for edge in sorted(candidates, key=lambda e: -min(area[e[0]], area[e[1]])):  # biggest countries first
        if try_add(edge):
            added += 1
    major_degree = [0] * n
    for i, j in linked:
        major_degree[i] += int(major[j])
        major_degree[j] += int(major[i])
    topped = 0
    for i in sorted(range(n), key=lambda q: major_degree[q]):
        if major_degree[i] >= min_deg:
            continue
        nearby = near(i, TOP_UP_SEARCH_M)
        for j in sorted(nearby, key=nearby.__getitem__):
            if major_degree[i] >= min_deg:
                break
            edge = (min(i, j), max(i, j))
            if edge in linked or not try_add(edge):
                continue
            major_degree[i] += 1
            major_degree[j] += int(major[i])
            topped += 1
    print("proximity edges: %d across water (<= %d km) + %d top-ups to >=%d neighbours (%.1fs)" % (added, max_gap // 1000, topped, min_deg, time.time() - start_time))
    return linked


def color_graph(n: int, edges: Iterable[Edge], weights: Sequence[float], kmin: int = 1, kmax: int = 8, limit: int = 2_000_000) -> tuple[int, list[int]]:
    """Fewest-colour proper colouring with k in [kmin, kmax] (exact DSATUR backtracking, giving up on a k after
    `limit` steps). Each node takes the free colour with the least total weight so far."""
    neighbours = neighbour_sets(n, edges)
    for k in range(kmin, kmax + 1):
        color_class = [-1] * n
        used = [0.0] * k
        steps = 0

        def most_constrained() -> int:
            """Uncoloured node with the most distinct neighbour colours, then the most neighbours."""
            best, best_key = -1, (-1, -1)
            for v in range(n):
                if color_class[v] >= 0:
                    continue
                saturation = len({color_class[u] for u in neighbours[v] if color_class[u] >= 0})
                key = (saturation, len(neighbours[v]))
                if key > best_key:
                    best, best_key = v, key
            return best

        def solve(left: int) -> bool:
            nonlocal steps
            if left == 0:
                return True
            steps += 1
            if steps > limit:
                return False
            v = most_constrained()
            taken = {color_class[u] for u in neighbours[v] if color_class[u] >= 0}
            for c in sorted((c for c in range(k) if c not in taken), key=lambda c: used[c]):
                color_class[v] = c
                used[c] += weights[v]
                if solve(left - 1):
                    return True
                color_class[v] = -1
                used[c] -= weights[v]
            return False

        if solve(n):
            return k, color_class
    raise RuntimeError("no colouring found")


RED_HUE, GREEN_HUE = 29.0, 142.0  # OKLCH hues of sRGB red and green
PALETTE_CHROMA = 0.115  # target chroma (reduced where the gamut needs it)
L_RED, L_GREEN = 0.64, 0.77  # lightness at the red and green hues


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


def make_palette(k: int, class_pairs: set[Edge], C: float = PALETTE_CHROMA, l_red: float = L_RED, l_green: float = L_GREEN) -> list[RGB]:
    """k country hues: the ocean's hue is one of k+1 equispaced hues (pinned); countries get the other k.
    Each hue gets its own lightness (dark red .. light green). The hues are assigned to colour classes so that
    classes that end up adjacent are as far apart as possible in OKLab (hue *and* brightness)."""
    ocean_hue = srgb_to_oklch(to_rgb(OCEAN))[2]
    hues = [(ocean_hue + (i + 1) * 360 / (k + 1)) % 360 for i in range(k)]
    rgbs, labs = [], []
    for h in hues:
        L = lightness_for_hue(h, l_red, l_green)
        fitted = fit_gamut(L, C, h)  # most chroma the sRGB gamut allows, up to C
        assert fitted is not None
        rgb, chroma = fitted
        rgbs.append(rgb)
        labs.append((L, chroma * np.cos(np.radians(h)), chroma * np.sin(np.radians(h))))
    dist = np.linalg.norm(np.array(labs)[:, None] - np.array(labs)[None], axis=-1)

    def separation(perm: tuple[int, ...]) -> tuple[float, float]:
        """(closest, total) OKLab distance between touching classes, if class c gets hue perm[c]."""
        touching = [dist[perm[a], perm[b]] for a, b in class_pairs]
        return min(touching), sum(touching)

    best = max(permutations(range(k)), key=separation)
    print("ocean hue %.0f; country hues/L:" % ocean_hue, [(round(hues[best[c]]), round(labs[best[c]][0], 2)) for c in range(k)], "min adjacent OKLab distance %.3f" % separation(best)[0])
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
