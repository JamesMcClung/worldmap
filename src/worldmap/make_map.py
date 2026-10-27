"""Render a poster-sized world map (country + large-country state borders, non-overlapping labels).

Data: Natural Earth 10m (public domain), downloaded into .cache/ on first use. Run: make_map [out.png]
Pins: --pick pins.txt (click the rendered map to add pins) and --pins pins.txt [--pin-size inches] (draw them).
"""

import json
import os
import sys
import urllib.request
import zipfile

import geopandas as gpd
import matplotlib
import matplotlib.patheffects as pe
import numpy as np
import shapely
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon as MplPolygon
from pyproj import Transformer
from shapely import affinity, maximum_inscribed_circle
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union


def flag_value(name):
    """Remove `name VALUE` from sys.argv and return VALUE (None if the flag is absent)."""
    if name not in sys.argv:
        return None
    i = sys.argv.index(name)
    if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("--"):
        sys.exit("%s needs a value" % name)
    sys.argv.pop(i)
    return sys.argv.pop(i)


# Data files and caches live in the repo's .cache/ (found from this file, so it needs an editable install).
CACHE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache"))
NE_URL = "https://naciscdn.org/naturalearth/10m/"
COUNTRIES, STATES = "ne_10m_admin_0_countries.zip", "ne_10m_admin_1_states_provinces_lines.zip"
RASTER = "ne1/NE1_LR_LC.tif"  # unzipped from NE1_LR_LC.zip


def fetch(name):
    """Path of Natural Earth file `name` in CACHE, downloading it first if it isn't there yet."""
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = NE_URL + ("raster/NE1_LR_LC.zip" if name == RASTER else "cultural/" + name)
    print("downloading", url)
    part = path + ".part"
    urllib.request.urlretrieve(url, part)
    if name == RASTER:
        with zipfile.ZipFile(part) as z:
            z.extractall(os.path.dirname(path))
        os.remove(part)
    else:
        os.replace(part, path)
    return path


CRS ="+proj=robin +lon_0=0 +datum=WGS84"
W_IN, DPI = 36, 200  # 7200 px wide
STATE_COUNTRIES = {
    "United States of America",
    "Canada",
    "Russia",
    "China",
    "Brazil",
    "Australia",
    "India",
    "Mexico",
    "Argentina",
    "Kazakhstan",
    "Indonesia",
}
OCEAN, BORDER, STATE = "#e3f0f7", "#3a3a3a", "#7d7d7d"


# --- OKLab palette: 9 fills with maximal minimum pairwise distance (light enough for dark text) ---
def srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def lab_to_lin(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    return np.array([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s])


def lch_to_rgb(L, C, h):
    """OKLCH -> sRGB in [0,1], or None if out of gamut."""
    lin = lab_to_lin(L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h)))
    if (lin < -1e-6).any() or (lin > 1 + 1e-6).any():
        return None
    return np.clip(lin_to_srgb(np.clip(lin, 0, 1)), 0, 1)


def srgb_to_lch(rgb):
    lin = srgb_to_lin(rgb)
    l = np.cbrt(0.4122214708 * lin[0] + 0.5363325363 * lin[1] + 0.0514459929 * lin[2])
    m = np.cbrt(0.2119034982 * lin[0] + 0.6806995451 * lin[1] + 0.1073969566 * lin[2])
    s_ = np.cbrt(0.0883024619 * lin[0] + 0.2817188376 * lin[1] + 0.6299787005 * lin[2])
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s_
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s_
    b = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s_
    return L, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360


NOT_LABELED = {"Jersey", "Guernsey", "Isle of Man", "Åland", "Macau", "Hong Kong", "Sint Maarten", "Aruba", "Curaçao", "Northern Cyprus", "Turkish Republic of Northern Cyprus", "Somaliland"}
LABELED_EXTRA = {"Palestine", "Kosovo", "Western Sahara", "Israel", "Puerto Rico", "Falkland Islands", "Faroe Islands"}


def labeled(row):
    return row.NAME_EN in LABELED_EXTRA or (row.TYPE in ("Sovereign country", "Country", "Sovereignty") and row.NAME_EN not in NOT_LABELED)


def adjacency(gdf):
    """Edges between features that share a border (more than a single point)."""
    from shapely import STRtree

    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    edges = set()
    for i, g in enumerate(geoms):
        for j in tree.query(g, predicate="intersects"):
            if j > i and g.intersection(geoms[j]).length > 0:
                edges.add((i, int(j)))
    return edges


def add_proximity_edges(gdf, edges, major, k, max_gap=600_000, min_deg=3):
    """Link countries separated by water so they don't share a colour, without exceeding k colours.
    Candidate links (real countries within `max_gap` metres of each other) are tried largest-country-first and
    kept only if the graph stays k-colourable; then each feature is topped up to `min_deg` real-country
    neighbours via its nearest ones under the same feasibility rule."""
    import time

    from shapely import STRtree

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

    def try_add(e):
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

    def near(i, maxd):
        out = {}
        for dx in (0, WORLD, -WORLD):
            g = affinity.translate(sim[i], dx) if dx else sim[i]
            for j in tree.query(g, predicate="dwithin", distance=maxd):
                j = int(j)
                if j != i and major[j]:
                    out[j] = min(out.get(j, np.inf), g.distance(sim[j]))
        return out

    cands = {}
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


def color_graph(n, edges, weights, kmin=1, kmax=8, limit=2_000_000):
    """Fewest-colour proper colouring (exact DSATUR backtracking); prefers the least-used colour by area."""
    nbr = [set() for _ in range(n)]
    for i, j in edges:
        nbr[i].add(j)
        nbr[j].add(i)
    for k in range(kmin, kmax + 1):
        col = [-1] * n
        used = [0.0] * k
        steps = [0]

        def pick():
            best, key = None, None
            for v in range(n):
                if col[v] >= 0:
                    continue
                sat = len({col[u] for u in nbr[v] if col[u] >= 0})
                kk = (sat, len(nbr[v]))
                if key is None or kk > key:
                    best, key = v, kk
            return best

        def solve(left):
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


def lightness_for_hue(h, l_red, l_green):
    """Lightness ramps (raised cosine) from darkest at red to lightest at green, so a red/green colour-blind viewer
    can tell those hues apart by brightness. Other hues interpolate along the two arcs between them."""
    t = (h - RED_HUE) % 360
    arc = (GREEN_HUE - RED_HUE) % 360
    if t <= arc:  # red -> green, rising
        u = t / arc
        return l_red + (l_green - l_red) * (1 - np.cos(np.pi * u)) / 2
    u = (t - arc) / (360 - arc)  # green -> red the long way round, falling
    return l_green - (l_green - l_red) * (1 - np.cos(np.pi * u)) / 2


def make_palette(k, pairs, C=0.115, l_red=0.64, l_green=0.77):
    """k country hues: the ocean's hue is one of k+1 equispaced hues (pinned); countries get the other k.
    Each hue gets its own lightness (dark red .. light green). The hues are assigned to colour classes so that
    classes that end up adjacent are as far apart as possible in OKLab (hue *and* brightness)."""
    from itertools import permutations

    ocean_h = srgb_to_lch(to_rgb(OCEAN))[2]
    hues = [(ocean_h + (i + 1) * 360 / (k + 1)) % 360 for i in range(k)]
    rgbs, labs = [], []
    for h in hues:
        L = lightness_for_hue(h, l_red, l_green)
        for cc in np.linspace(C, 0, 40):  # most chroma the sRGB gamut allows, up to C
            rgb = lch_to_rgb(L, cc, h)
            if rgb is not None:
                break
        rgbs.append(tuple(rgb))
        labs.append((L, cc * np.cos(np.radians(h)), cc * np.sin(np.radians(h))))
    D = np.linalg.norm(np.array(labs)[:, None] - np.array(labs)[None], axis=-1)
    best = max(permutations(range(k)), key=lambda p: (min(D[p[a], p[b]] for a, b in pairs), sum(D[p[a], p[b]] for a, b in pairs)))
    print("ocean hue %.0f; country hues/L:" % ocean_h, [(round(hues[best[c]]), round(labs[best[c]][0], 2)) for c in range(k)], "min adjacent OKLab distance %.3f" % min(D[best[a], best[b]] for a, b in pairs))
    return [rgbs[best[c]] for c in range(k)]

N_COLORS, MAX_GAP, MIN_DEG = 5, 600_000, 3


# --- --earth mode: per-region vegetation from the Natural Earth I land-cover raster (NE1_LR_LC, 16200x8100, public domain) ---
# Each polygon (mainland, Alaska, French Guiana, Kaliningrad, islands...) is coloured independently by the mean (G - R) of
# the raster's land pixels inside it (desert tan ~ -12, rainforest green ~ +20), and by its fraction of icy (bluish-white) pixels.
GR_DESERT, GR_LUSH = -12.0, 20.0
MIN_PIX = 6  # polygons with fewer raster pixels borrow from a window around their centroid, then from their country
SAND, GREEN, ICE = (0.85, 0.09, 78), (0.84, 0.095, 135), "#eef3f4"  # OKLCH (L, C, hue deg)


def earth_values(countries):
    """[[(mean G-R, icy fraction) per polygon] per country], in the polygon order of `prep(countries)`. Cached in earth.json."""
    cache = os.path.join(CACHE, "earth.json")
    raster = fetch(RASTER)
    key = {"raster": os.path.getsize(raster), "countries": os.path.getsize(fetch(COUNTRIES)), "rule": "v1", "min_pix": MIN_PIX}
    if os.path.exists(cache):
        d = json.load(open(cache))
        if d["key"] == key:
            return d["vals"]
    from PIL import Image, ImageDraw

    Image.MAX_IMAGE_PIXELS = None
    im = np.asarray(Image.open(raster).convert("RGB"))
    H, W = im.shape[:2]

    def stats(px):
        px = px[px.min(1) < 250].astype(int)  # white = ocean
        if len(px) == 0:
            return None
        R, G, B = px.T
        return float(np.mean(G - R)), float(((B >= G) & (B >= R)).mean()), len(px)

    def to_img(co):
        return [((x + 180) / 360 * W, (90 - y) / 180 * H) for x, y in co]

    vals = []
    for geom in countries.geometry.intersection(box(-180, -90, 180, 90)):
        polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        got, pool = [], []
        for p in polys:
            if p.is_empty or p.geom_type != "Polygon":
                got.append(None)
                continue
            x0, y0, x1, y1 = p.bounds
            X0, X1 = max(0, int((x0 + 180) / 360 * W)), min(W, int((x1 + 180) / 360 * W) + 2)
            Y0, Y1 = max(0, int((90 - y1) / 180 * H)), min(H, int((90 - y0) / 180 * H) + 2)
            mk = Image.new("L", (X1 - X0, Y1 - Y0), 0)
            dr = ImageDraw.Draw(mk)
            dr.polygon([(a - X0, b - Y0) for a, b in to_img(p.exterior.coords)], fill=1)
            for r in p.interiors:
                dr.polygon([(a - X0, b - Y0) for a, b in to_img(r.coords)], fill=0)
            px = im[Y0:Y1, X0:X1][np.asarray(mk, bool)]
            pool.append(px)
            st = stats(px)
            if st is None or st[2] < MIN_PIX:  # tiny: look at a window around the centroid instead
                cx, cy = to_img([p.representative_point().coords[0]])[0]
                cx, cy = int(min(max(cx, 3), W - 4)), int(min(max(cy, 3), H - 4))
                st = stats(im[cy - 3 : cy + 4, cx - 3 : cx + 4].reshape(-1, 3))
            got.append(st)
        whole = stats(np.concatenate(pool)) if pool else None
        vals.append([list(st[:2]) if st else (list(whole[:2]) if whole else [10.0, 0.0]) for st in got])
    json.dump({"key": key, "vals": vals}, open(cache, "w"))
    return vals


def earth_fill(gr, ice):
    g = float(np.clip((gr - GR_DESERT) / (GR_LUSH - GR_DESERT), 0, 1))
    f = float(np.clip((ice - 0.15) / 0.6, 0, 1))  # bluish-white land = ice cap
    L, C, h = (1 - g) * np.array(SAND) + g * np.array(GREEN)  # hue 78 -> 135 interpolates through yellow-olive, no wrap
    for c in np.linspace(C, 0, 40):  # gamut-reduce
        rgb = lch_to_rgb(L, c, h)
        if rgb is not None:
            break
    return tuple((1 - f) * np.array(rgb) + f * np.array(to_rgb(ICE)))


# --- pins: golden spheres at (lon, lat) points ---
PIN_PAD = 1.4  # the sprite canvas extends this many sphere radii from the centre (room for the drop shadow)


def load_pins(path):
    """(lon, lat) pairs from a text file with one 'lon,lat' per line; '#' starts a comment."""
    pins = []
    for n, line in enumerate(open(path), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        try:
            lon, lat = (float(s) for s in line.replace(",", " ").split()[:2])
        except ValueError:
            sys.exit("%s:%d: expected 'lon,lat', got %r" % (path, n, line))
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            sys.exit("%s:%d: coordinates out of range: %r" % (path, n, line))
        pins.append((lon, lat))
    return pins


def pin_sprite(diameter_px):
    """RGBA gold sphere lit from the upper left (dark rim, bounce light from below, white highlight) with a soft drop shadow."""
    R = diameter_px / 2
    n = int(round(2 * PIN_PAD * R))
    c = np.linspace(-PIN_PAD, PIN_PAD, n)
    u, v = np.meshgrid(c, -c)  # in sphere radii, v up
    r = np.hypot(u, v)
    nz = np.sqrt(np.clip(1 - r**2, 0, 1))
    N = np.stack([u, v, nz], -1)

    def lit(d):
        d = np.asarray(d, float)
        return np.clip(N @ (d / np.linalg.norm(d)), 0, None)

    key = np.array([-0.45, 0.6, 0.65])
    key /= np.linalg.norm(key)

    def sstep(e0, e1, x):
        q = np.clip((x - e0) / (e1 - e0), 0, 1)
        return q * q * (3 - 2 * q)

    # Metal: shade by what the surface reflects, not by diffuse light. From the bottom up: warm gold ground, a crisp horizon,
    # a dark band just above it, a bright cream sky; plus a window-like bright strip down each side (stronger on the right).
    ry = 2 * nz * v + 0.7 * u**2  # elevation of the reflected view ray; the u^2 term bends the horizon into a dome
    rx = 2 * nz * u
    ground = 0.68 - 0.2 * (1 - sstep(-1.0, -0.3, ry))
    sky = 0.8 + 0.2 * sstep(0.4, 1.0, ry)
    shade = ground + (sky - ground) * sstep(0.05, 0.5, ry)
    y_band = -0.2
    shade += 0.1 * np.exp(-(((ry - y_band) / np.where(ry < y_band, 0.07, 0.3)) ** 2))
    strip = np.where(rx > 0, 0.55, 0.3) * np.exp(-(((np.abs(rx) - 0.85) / 0.14) ** 2)) * sstep(0.0, 0.35, ry)
    shade = np.clip(shade + strip, 0, 1)
    stops = [0, 0.35, 0.65, 0.9, 1]
    ramp = np.array([(0.30, 0.17, 0.02), (0.65, 0.42, 0.05), (0.92, 0.70, 0.16), (0.98, 0.80, 0.28), (1.0, 0.93, 0.68)])
    rgb = np.stack([np.interp(shade, stops, ramp[:, k]) for k in range(3)], -1)
    rgb *= (0.7 + 0.3 * np.sqrt(nz))[..., None]
    h = lit(key + [0, 0, 1])
    spec = np.clip(0.4 * h**12 + 0.8 * h**70, 0, 1)[..., None]  # broad soft glare plus a hot core
    rgb = rgb + (1 - rgb) * spec
    a = np.clip((1 - r) * R + 0.5, 0, 1)
    s = np.clip((1.15 - np.hypot(u - 0.12, v + 0.16)) / 0.35, 0, 1)
    sa = 0.45 * s * s * (3 - 2 * s)  # black shadow, offset down and right
    A = a + sa * (1 - a)
    rgb = np.where(A[..., None] > 0, rgb * a[..., None] / np.maximum(A, 1e-9)[..., None], 0)
    return np.dstack([rgb, A])


def prep(gdf):
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.intersection(box(-180, -90, 180, 90))
    return gdf.to_crs(CRS)


def patches(geom):
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if p.geom_type == "Polygon" and not p.is_empty:
            yield MplPolygon(np.asarray(p.exterior.coords), closed=True)
            for r in p.interiors:
                pass


def biggest(geom):
    return max(geom.geoms, key=lambda g: g.area) if isinstance(geom, MultiPolygon) else geom


RENAME = {"People's Republic of China": "China", "United States of America": "United States", "Democratic Republic of the Congo": "DR Congo", "Republic of the Congo": "Congo", "Central African Republic": "C.A.R.", "Bosnia and Herzegovina": "Bosnia & Herz.", "United Arab Emirates": "UAE", "Dominican Republic": "Dominican Rep."}
MIN_PT, MAX_PT = 6, 30
FORCE_ISLAND = {"Greece"}

halo = [pe.withStroke(linewidth=2.2, foreground=(1, 1, 1, 0.55))]


class Label:
    pass


def darken(rgb, L=0.26):
    """Same OKLab hue/chroma as the fill, but dark."""
    _, C, h = srgb_to_lch(rgb)
    for c in np.linspace(C, 0, 40):
        out = lch_to_rgb(L, c, h)
        if out is not None:
            return tuple(out)
    return (0.1, 0.1, 0.1)


def rect(x, y, w, h):
    return shapely.box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)


GROW_STEP, GROW_MAX = 1.12, 1.8


def fits(L, f, land):
    """Is there a spot where the label scaled by f sits mostly on its country + ocean (not on other land)?"""
    w, h = L.w * f, L.h * f
    gx0, gy0, gx1, gy1 = L.own.bounds
    us = np.linspace(gx0 - 0.5 * w, gx1 + 0.5 * w, 11)
    vs = np.linspace(gy0 - 0.8 * h, gy1 + 0.8 * h, 8)
    cx = np.append(np.repeat(us, len(vs)), L.ax)
    cy = np.append(np.tile(vs, len(us)), L.ay)
    boxes = shapely.box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    a_own = shapely.area(shapely.intersection(boxes, L.own))
    other = np.clip(shapely.area(shapely.intersection(boxes, land)) - a_own, 0, None) / (w * h)
    cover = shapely.area(shapely.intersection(boxes, L.geom)) / max(L.geom.area, 1e-9)
    need = min(0.6 * w * h, 0.5 * L.own.area)
    return bool(((other <= 0.03) & (cover < 0.6) & (a_own >= need)).any())


def two_line(text):
    words = text.split(" ")
    if len(words) < 2:
        return None
    k = min(range(1, len(words)), key=lambda j: abs(len(" ".join(words[:j])) - len(" ".join(words[j:]))))
    return " ".join(words[:k]) + "\n" + " ".join(words[k:])


def max_growth(L, pt0, land):
    f = 1.0
    while f * GROW_STEP <= GROW_MAX and pt0 * f * GROW_STEP <= MAX_PT and fits(L, f * GROW_STEP, land):
        f *= GROW_STEP
    return f


TARGET = 1.3


def nearest_pt(px_, py_, x, y, w, h):
    return np.clip(px_, x - w / 2, x + w / 2), np.clip(py_, y - h / 2, y + h / 2)


def seg_hits_rects(p0, p1, R):
    """p0,p1: (...,2) broadcastable against R (...,4) [x0,y0,x1,y1]; True where the segment crosses the rect."""
    d = p1 - p0
    tmin = np.zeros(np.broadcast(d[..., 0], R[..., 0]).shape)
    tmax = np.ones_like(tmin)
    ok = np.ones(tmin.shape, bool)
    for k in (0, 1):
        dk = d[..., k] * np.ones_like(tmin)
        lo, hi = R[..., k], R[..., k + 2]
        p = p0[..., k] * np.ones_like(tmin)
        par = np.abs(dk) < 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - p) / dk, (hi - p) / dk
        tn, tx = np.minimum(t1, t2), np.maximum(t1, t2)
        ok &= np.where(par, (p >= lo) & (p <= hi), True)
        tmin = np.where(par, tmin, np.maximum(tmin, tn))
        tmax = np.where(par, tmax, np.minimum(tmax, tx))
    return ok & (tmax >= tmin)


def seg_cross(a0, a1, b0, b1):
    """Proper segment-segment intersection, broadcasting over leading dims."""

    def cr(o, p, q):
        return (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])

    return (cr(a0, a1, b0) * cr(a0, a1, b1) < 0) & (cr(b0, b1, a0) * cr(b0, b1, a1) < 0)


STEPS = [(a * 0.25, b) for a in (-2, -1, 0, 1, 2) for b in (-1, 0, 1) if (a, b) != (0, 0)]
ANG = np.radians(np.arange(0, 360, 15))


def main():
    PINS, PICK = flag_value("--pins"), flag_value("--pick")  # pin-list files: --pins draws them, --pick adds to one by clicking
    PIN_IN = float(flag_value("--pin-size") or 0.24)  # pin diameter in inches on the 36-inch-wide poster

    if not PICK:  # --pick needs an interactive backend
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --ocean: experimental ocean-aware label placement (open water counts as the country; labels grow). Slower.
    OCEAN_AWARE = "--ocean" in sys.argv
    _args = [a for a in sys.argv[1:] if not a.startswith("--")]
    OUT = _args[0] if _args else ("worldmap_pins.png" if PINS else "worldmap.png")
    # --earth: colour countries by how they look from space (sandy desert -> light green vegetation) instead of graph colours.
    EARTH = "--earth" in sys.argv
    PALETTE = None  # set below via make_palette()

    countries = gpd.read_file(fetch(COUNTRIES))
    countries = countries.reset_index(drop=True)
    _major = [labeled(r) for _, r in countries.iterrows()]
    # The graph colouring (which countries share a colour class) is cached in coloring.json, so palette tweaks never
    # reshuffle it and runs skip the border/distance work. It is rebuilt when the inputs change or with --recolor.

    _CACHE = os.path.join(CACHE, "coloring.json")
    _key = {"n_colors": N_COLORS, "max_gap": MAX_GAP, "min_deg": MIN_DEG, "rule": "top10-exempt-count-balance/v2", "files": {COUNTRIES: os.path.getsize(fetch(COUNTRIES))}}
    _names = list(countries.NAME_EN)
    _cached = None
    if "--recolor" not in sys.argv and os.path.exists(_CACHE):
        try:
            _d = json.load(open(_CACHE))
            if _d["key"] == _key and _d["names"] == _names:
                _cached = _d
        except (ValueError, KeyError):
            pass
    if _cached:
        _k, _col, _cls_pairs = _cached["k"], _cached["col"], {tuple(p) for p in _cached["pairs"]}
        print("colouring: loaded from coloring.json")
    else:
        _edges = add_proximity_edges(countries, adjacency(countries), _major, N_COLORS, max_gap=MAX_GAP, min_deg=MIN_DEG)
        # Balance colour usage by country count, but don't count the 10 largest, so their colours aren't penalised.
        _areas = countries.to_crs("EPSG:6933").geometry.area
        _weights = [0.0 if r < 10 else 1.0 for r in _areas.rank(ascending=False, method="first")]
        _k, _col = color_graph(len(countries), _edges, _weights, kmin=N_COLORS)
        _cls_pairs = {(min(_col[i], _col[j]), max(_col[i], _col[j])) for i, j in _edges}
        json.dump({"key": _key, "names": _names, "k": _k, "col": _col, "pairs": sorted(_cls_pairs)}, open(_CACHE, "w"))
        print("colouring: computed and saved to coloring.json")

    def draw_pins(ax, pins, sprite):
        """Draw pins given as (lon, lat), northern ones first so southern ones overlap them. Returns the image artists."""
        half = PIN_PAD * PIN_IN / 2 * (maxx - minx) / W_IN
        out = []
        for lon, lat in sorted(pins, key=lambda p: -p[1]):
            x, y = t.transform(lon, lat)
            out.append(ax.imshow(sprite, extent=(x - half, x + half, y - half, y + half), zorder=10, interpolation="bilinear"))
        return out

    def pick_pins(path):
        """Show the rendered map (OUT) in a window; left-click adds a pin, right-click or `u` undoes one. Closing the window
        appends the new pins to `path` (existing lines are left alone)."""
        from matplotlib.path import Path
        from PIL import Image
        from pyproj import Transformer

        if not os.path.exists(OUT):
            sys.exit("--pick shows the rendered map: run make_map first to create %s" % OUT)
        im = Image.open(OUT).convert("RGBA").reduce(4)
        existing = load_pins(path) if os.path.exists(path) else []
        new = []
        sprite = pin_sprite(PIN_IN * DPI)
        outline = Path(np.array(edge))
        t_inv = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
        fig_ = plt.figure(figsize=(14, 14 * (maxy - miny) / (maxx - minx)))
        ax_ = fig_.add_axes([0, 0, 1, 1])
        ax_.imshow(np.asarray(im), extent=(minx, maxx, miny, maxy))
        ax_.axis("off")
        arts = []

        def redraw():
            for a in arts:
                a.remove()
            arts[:] = draw_pins(ax_, existing + new, sprite)
            ax_.set_xlim(minx, maxx)
            ax_.set_ylim(miny, maxy)
            fig_.canvas.draw_idle()

        def undo():
            if new:
                print("removed %.4f,%.4f" % new.pop())

        def on_click(e):
            if e.inaxes is not ax_ or getattr(fig_.canvas.toolbar, "mode", ""):  # ignore clicks while zoom/pan is active
                return
            if e.button == 3:
                undo()
            elif e.button == 1 and outline.contains_point((e.xdata, e.ydata)):
                new.append(t_inv.transform(e.xdata, e.ydata))
                print("added   %.4f,%.4f" % new[-1])
            redraw()

        def on_key(e):
            if e.key == "u":
                undo()
                redraw()

        def save(_):
            if not new:
                return print("no new pins")
            text = open(path).read() if os.path.exists(path) else ""
            with open(path, "a") as f:
                if text and not text.endswith("\n"):
                    f.write("\n")
                f.writelines("%.4f,%.4f\n" % p for p in new)
            print("added %d pins to %s (%d total)" % (len(new), path, len(existing) + len(new)))

        fig_.canvas.mpl_connect("button_press_event", on_click)
        fig_.canvas.mpl_connect("key_press_event", on_key)
        fig_.canvas.mpl_connect("close_event", save)
        redraw()
        print("left-click: add pin; right-click or u: undo; close the window to save to", path)
        plt.show()

    PALETTE = make_palette(_k, _cls_pairs)
    COLOR_OF = dict(zip(countries.index, _col))
    # Baikonur Cosmodrome: Kazakh territory leased to Russia -> Russia's fill, striped with Kazakhstan's colour.
    _idx = dict(zip(countries.NAME_EN, countries.index))
    COLOR_OF[_idx["Baikonur"]] = COLOR_OF[_idx["Russia"]]
    _by_name = {n: COLOR_OF[i] for i, n in zip(countries.index, countries.NAME_EN)}
    _pairs = [("South Korea", "Japan"), ("Russia", "United States of America"), ("Malaysia", "Philippines"), ("Italy", "Greece"), ("France", "United Kingdom"), ("United Kingdom", "Norway"), ("India", "Sri Lanka")]
    print("same-colour pairs among spot checks:", [p for p in _pairs if _by_name[p[0]] == _by_name[p[1]]])
    _name_of = dict(zip(countries.index, countries.NAME_EN))

    def fill_of(i, pt=None):
        """Country fill; in --earth mode, the fill of the country's polygon containing (or nearest) `pt`, else its largest."""
        if not EARTH:
            return PALETTE[COLOR_OF[i]]
        regs = REGIONS[i]
        if pt is None:
            return max(regs, key=lambda r: r[0].area)[1]
        return min(regs, key=lambda r: r[0].distance(pt))[1]

    states = gpd.read_file(fetch(STATES))
    states = states[states.ADM0_NAME.isin(STATE_COUNTRIES)]

    countries_p, states_p = prep(countries), prep(states)
    REGIONS = {}  # --earth: country index -> [(projected polygon, fill)]
    if EARTH:
        for (_i, _row), _v in zip(countries_p.iterrows(), earth_values(countries)):
            _g = _row.geometry
            _polys = list(_g.geoms) if isinstance(_g, MultiPolygon) else [_g]
            REGIONS[_i] = [(p, earth_fill(*v)) for p, v in zip(_polys, _v) if not p.is_empty]
            if not REGIONS[_i]:
                REGIONS[_i] = [(Polygon(), "#ffffff")]

    # Robinson outline of the whole globe (poles included); the canvas is its bounding box plus a little room for the stroke.

    t = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    lat = np.linspace(-90, 90, 200)
    edge = [t.transform(180 - 1e-6, la) for la in lat] + [t.transform(-180 + 1e-6, la) for la in lat[::-1]]
    _ex, _ey = np.array(edge).T
    _pad = 0.006 * (_ex.max() - _ex.min())
    minx, miny, maxx, maxy = _ex.min() - _pad, _ey.min() - _pad, _ex.max() + _pad, _ey.max() + _pad
    if PICK:
        pick_pins(PICK)
        return

    fig_w = W_IN
    fig_h = fig_w * (maxy - miny) / (maxx - minx)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0)  # transparent outside the globe outline
    ax.add_patch(MplPolygon(edge, closed=True, fc=OCEAN, ec="none", zorder=0))
    ax.add_patch(MplPolygon(edge, closed=True, fc="none", ec=BORDER, lw=2.2, zorder=3))

    for _, row in countries_p.iterrows():
        col = "#ffffff" if row.NAME_EN == "Antarctica" else fill_of(_)  # Antarctica: hardcoded white, unlabelled
        if EARTH and row.NAME_EN != "Antarctica":
            for p, c in REGIONS[_]:
                ax.add_collection(PatchCollection(list(patches(p)), fc=c, ec=BORDER, lw=0.6, zorder=1))
        else:
            ax.add_collection(PatchCollection(list(patches(row.geometry)), fc=col, ec=BORDER, lw=0.6, zorder=1))
        if row.NAME_EN == "Baikonur" and not EARTH:
            stripe = fill_of(_idx["Kazakhstan"])
            plt.rcParams["hatch.linewidth"] = 1.6
            ax.add_collection(PatchCollection(list(patches(row.geometry)), fc="none", ec=stripe, lw=0, hatch="////", zorder=1.5))
    states_p.plot(ax=ax, color=STATE, linewidth=0.35, zorder=2)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # Graticule-free; labels ----------------------------------------------------
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px_per_unit = fig.get_size_inches()[0] * DPI / (maxx - minx)

    # Everything below works in window pixels: px = (x - minx) * PPU, py = (y - miny) * PPU.
    PPU = px_per_unit
    TO_PX = [PPU, 0, 0, PPU, -minx * PPU, -miny * PPU]

    def to_px(g, simplify=2.0):
        return affinity.affine_transform(g, TO_PX).simplify(simplify, preserve_topology=False)

    labels, island_names = [], []
    gx, gy = t.transform(-53, 4)  # French Guiana
    kx, ky = t.transform(21, 54.7)  # Kaliningrad
    for _, row in countries_p.iterrows():
        if not labeled(row):
            continue
        polys = [p for p in (row.geometry.geoms if isinstance(row.geometry, MultiPolygon) else [row.geometry]) if not p.is_empty]
        if not polys:
            continue
        big = max(polys, key=lambda p: p.area)
        pieces = []
        sig = [p for p in polys if p.area >= 0.02 * big.area]
        hull = unary_union(sig).convex_hull
        hb = hull.bounds
        island = len(sig) >= 2 and big.area / sum(p.area for p in sig) < 0.65 and (hb[2] - hb[0]) < 0.25 * (maxx - minx)
        if row.NAME_EN in FORCE_ISLAND and len(sig) >= 2:
            island = True
        if island:
            island_names.append(row.NAME_EN)
            pieces.append((row.NAME_EN, hull.centroid.coords[0], unary_union(sig), hull, max(sum(p.area for p in sig), hull.area)))
        else:
            c = maximum_inscribed_circle(big, tolerance=(maxx - minx) / 5000).coords[0]
            pieces.append((row.NAME_EN, c, big, big, hull.area if OCEAN_AWARE and len(sig) >= 2 and hull.area <= 1.6 * big.area else big.area))
        extra = []
        if row.NAME_EN == "France":
            extra = [("French Guiana", p) for p in polys if p.contains(Point(gx, gy))]
        if row.NAME_EN == "Russia":
            extra = [("Kaliningrad (Russia)", p) for p in polys if p.contains(Point(kx, ky))]
        for nm, p in extra:
            c = maximum_inscribed_circle(p, tolerance=(maxx - minx) / 5000).coords[0]
            pieces.append((nm, c, p, p, p.area))
        for nm, c, geom, home, area in pieces:
            nm = RENAME.get(nm, nm)
            size = float(np.clip(1.6 * np.sqrt(area) * PPU / DPI * 72 / 10, MIN_PT, MAX_PT))
            L = Label()
            L.name, L.area = nm, area
            L.color = darken(fill_of(_, geom.representative_point() if geom.geom_type == "Polygon" else None))
            L.txt = ax.text(c[0], c[1], nm.upper() if size > 14 else nm, fontsize=size, ha="center", va="center", color=L.color, zorder=6, path_effects=halo)
            ext = L.txt.get_window_extent(rend)
            L.w, L.h = ext.width * 1.06, ext.height * 1.15
            L.ax, L.ay = (c[0] - minx) * PPU, (c[1] - miny) * PPU
            L.x, L.y = L.ax, L.ay
            L.geom, L.home = to_px(geom), to_px(home)
            L.own = to_px(row.geometry)  # the whole country (all its islands/exclaves)
            labels.append(L)
    labels.sort(key=lambda l: -l.area)
    N = len(labels)
    print("island-heavy (hull-centred):", island_names)

    # Mode: 'in' = sits on its country; 'out' = would hide its country, so it moves aside with a leader line.
    # Open ocean counts as part of the country, so a label only has to avoid *other* countries' land.
    LAND_PX = to_px(unary_union(list(countries_p.geometry))) if OCEAN_AWARE else None

    def set_size(L, pt):
        L.txt.set_fontsize(pt)
        ext = L.txt.get_window_extent(rend)
        L.w, L.h = ext.width * 1.06, ext.height * 1.15

    def covers(L):
        return rect(L.x, L.y, L.w, L.h).intersection(L.geom).area / L.geom.area if L.geom.area > 0 else 1.0

    for L in labels:
        gb = L.geom.bounds
        pt0 = L.txt.get_fontsize()
        # label hides most of its country: shrink it (down to 60% of its size) before resorting to a leader line
        pt = pt0
        while OCEAN_AWARE and covers(L) >= 0.6 and pt * 0.92 >= max(MIN_PT, 0.6 * pt0):
            pt *= 0.92
            set_size(L, pt)
        L.out = covers(L) >= 0.6 or (L.w > 1.6 * (gb[2] - gb[0]) and (gb[3] - gb[1]) < 6 * L.h)
        if L.out and pt != pt0:
            set_size(L, pt0)

    def local_land(L, pad):
        gx0, gy0, gx1, gy1 = L.own.bounds
        return LAND_PX.intersection(shapely.box(gx0 - pad, gy0 - pad, gx1 + pad, gy1 + pad))

    for L in (labels if OCEAN_AWARE else []):
        pt0 = L.txt.get_fontsize()
        if L.out or pt0 >= 0.8 * MAX_PT:
            continue
        one = L.txt.get_text()
        layouts = [one] + ([two_line(one)] if two_line(one) else [])
        best = (-1.0, one)
        for text in layouts:  # try one-line and two-line; keep whichever lets the label grow most
            L.txt.set_multialignment("center")
            L.txt.set_linespacing(1.0)
            L.txt.set_text(text)
            set_size(L, pt0)
            land = local_land(L, 3 * max(L.w, L.h) * GROW_MAX)
            f = max_growth(L, pt0, land)
            if f > best[0] * 1.001:
                best = (f, text)
        L.txt.set_text(best[1])
        set_size(L, pt0 * best[0])
    OUT_ = np.array([L.out for L in labels])

    # Wrap long names onto two lines when a label sitting on its country is wider than the country.
    for L in labels:
        gb = L.geom.bounds
        if not L.out and " " in L.txt.get_text() and L.w > (1.3 if OCEAN_AWARE else 0.9) * (gb[2] - gb[0]) and (gb[3] - gb[1]) > 2.5 * L.h:
            words = L.txt.get_text().split(" ")
            k = min(range(1, len(words)), key=lambda j: abs(len(" ".join(words[:j])) - len(" ".join(words[j:]))))
            L.txt.set_multialignment("center")
            L.txt.set_linespacing(1.0)
            L.txt.set_text(" ".join(words[:k]) + "\n" + " ".join(words[k:]))
            ext = L.txt.get_window_extent(rend)
            L.w, L.h = ext.width * 1.06, ext.height * 1.15

    for L in (labels if OCEAN_AWARE else []):
        L.land_local = local_land(L, 10 * max(L.w, L.h))

    W_ = np.array([L.w for L in labels])
    H_ = np.array([L.h for L in labels])
    AX = np.array([L.ax for L in labels])
    AY = np.array([L.ay for L in labels])
    X = AX.copy()
    Y = AY.copy()
    OUTLINE_PX = to_px(Polygon(edge), simplify=0.5)
    shapely.prepare(OUTLINE_PX)
    def leader_ends():
        nx, ny = nearest_pt(AX, AY, X, Y, W_, H_)
        return nx, ny

    def cost_batch(i, CX, CY):
        """Cost of placing label i at each candidate centre (arrays of length c)."""
        c = len(CX)
        w, h = W_[i], H_[i]
        R = np.stack([CX - w / 2, CY - h / 2, CX + w / 2, CY + h / 2], -1)  # (c,4)
        mask = np.arange(N) != i
        ox_, oy_ = X[mask], Y[mask]
        ow, oh = W_[mask], H_[mask]
        O = np.stack([ox_ - ow / 2, oy_ - oh / 2, ox_ + ow / 2, oy_ + oh / 2], -1)  # (m,4)
        ovx = np.clip(np.minimum(R[:, None, 2], O[None, :, 2]) - np.maximum(R[:, None, 0], O[None, :, 0]), 0, None)
        ovy = np.clip(np.minimum(R[:, None, 3], O[None, :, 3]) - np.maximum(R[:, None, 1], O[None, :, 1]), 0, None)
        cost = 40 * (ovx * ovy).sum(1) / (h * h)
        # other leaders' dots and lines
        omode = OUT_[mask]
        if omode.any():
            dax, day = AX[mask][omode], AY[mask][omode]
            dr = 0.35 * H_[mask][omode]
            ddx = np.clip(np.abs(CX[:, None] - dax[None]) - w / 2, 0, None)
            ddy = np.clip(np.abs(CY[:, None] - day[None]) - h / 2, 0, None)
            dd = np.hypot(ddx, ddy)
            cost += 15 * (np.clip(dr[None] - dd, 0, None) / dr[None]).sum(1)
            nx_, ny_ = nearest_pt(dax, day, ox_[omode], oy_[omode], ow[omode], oh[omode])
            p0 = np.stack([dax, day], -1)[None]
            p1 = np.stack([nx_, ny_], -1)[None]
            cost += 20 * seg_hits_rects(p0, p1, R[:, None, :]).sum(1)
        if OUT_[i]:
            q0 = np.array([AX[i], AY[i]])[None, None, :]
            nx, ny = nearest_pt(AX[i], AY[i], CX, CY, w, h)
            q1 = np.stack([nx, ny], -1)[:, None, :]
            d = np.hypot(nx - AX[i], ny - AY[i])
            cost += 1.0 * ((d - TARGET * h) / h) ** 2 + 10 * np.clip(0.9 * h - d, 0, None) / h
            # own leader vs other label boxes (unless that box covers our anchor: unavoidable)
            keep = ~((O[:, 0] <= AX[i]) & (AX[i] <= O[:, 2]) & (O[:, 1] <= AY[i]) & (AY[i] <= O[:, 3]))
            cost += 20 * seg_hits_rects(q0, q1, O[None, keep, :]).sum(1)
            # own leader vs other leaders
            if omode.any():
                cost += 6 * seg_cross(q0[:, None, :, :] if False else np.broadcast_to(q0, (c, 1, 2)), q1, np.stack([dax, day], -1)[None], np.stack([nx_, ny_], -1)[None]).sum(1)
        else:
            boxes = shapely.box(R[:, 0], R[:, 1], R[:, 2], R[:, 3])
            if OCEAN_AWARE:
                a_own = shapely.area(shapely.intersection(boxes, labels[i].own))
                other = np.clip(shapely.area(shapely.intersection(boxes, labels[i].land_local)) - a_own, 0, None) / (w * h)
                need = max(min(0.55 * w * h, 0.5 * labels[i].own.area), 1e-6)
                cost += 20 * other + 14 * (1 - np.clip(a_own / need, 0, 1)) + 0.3 * np.hypot(CX - AX[i], CY - AY[i]) / h
            else:
                inside = shapely.area(shapely.intersection(boxes, labels[i].geom)) / (w * h)
                cost += 20 * (1 - inside) + 0.15 * np.hypot(CX - AX[i], CY - AY[i]) / h
        cost += 1000 * ~shapely.contains(OUTLINE_PX, shapely.box(R[:, 0], R[:, 1], R[:, 2], R[:, 3]))  # labels stay inside the globe
        return cost

    def candidates(i, grid=False):
        w, h = W_[i], H_[i]
        cx, cy = [X[i]], [Y[i]]
        for sc in (0.4, 1, 2, 4):
            for a, b in STEPS:
                cx.append(X[i] + a * w * sc)
                cy.append(Y[i] + b * h * sc)
        if OUT_[i]:
            for k in (1.0, 1.4, 2.0, 3.0, 4.5):
                cx += list(AX[i] + np.cos(ANG) * (w / 2 + 1.3 * h) * k)
                cy += list(AY[i] + np.sin(ANG) * 1.8 * h * k)
        else:
            cx.append(AX[i])
            cy.append(AY[i])
            if grid:  # coarse scan of the whole country so a label can jump to where it fits
                gx0, gy0, gx1, gy1 = labels[i].geom.bounds
                nxg, nyg = 14, 9
                us = np.linspace(gx0 - 0.25 * w, gx1 + 0.25 * w, nxg) if OCEAN_AWARE else np.linspace(gx0 + w / 2, max(gx1 - w / 2, gx0 + w / 2), nxg)
                vs = np.linspace(gy0 - 0.5 * h, gy1 + 0.5 * h, nyg) if OCEAN_AWARE else np.linspace(gy0 + h / 2, max(gy1 - h / 2, gy0 + h / 2), nyg)
                for u in us:
                    for v in vs:
                        cx.append(u)
                        cy.append(v)
        return np.array(cx), np.array(cy)

    rng = np.random.default_rng(7)

    def optimize(passes, grid_passes=3):
        for p in range(passes):
            changed = 0
            for i in rng.permutation(N):
                CX, CY = candidates(i, grid=p < grid_passes)
                cst = cost_batch(i, CX, CY)
                cst[0] -= 1e-6
                j = int(np.argmin(cst))
                if j != 0:
                    X[i], Y[i] = CX[j], CY[j]
                    changed += 1
            if changed == 0:
                break

    def label_overlaps():
        bad = []
        for i in range(N):
            m = np.arange(N) != i
            ox = np.minimum(X[i] + W_[i] / 2, X[m] + W_[m] / 2) - np.maximum(X[i] - W_[i] / 2, X[m] - W_[m] / 2)
            oy = np.minimum(Y[i] + H_[i] / 2, Y[m] + H_[m] / 2) - np.maximum(Y[i] - H_[i] / 2, Y[m] - H_[m] / 2)
            if ((ox > 0) & (oy > 0)).any():
                bad.append(i)
        return bad

    # seed 'out' labels on a ring in largest-first order so they start separated, then descend.
    for i in range(N):
        if OUT_[i]:
            CX, CY = candidates(i)
            j = int(np.argmin(cost_batch(i, CX, CY)))
            X[i], Y[i] = CX[j], CY[j]
    optimize(40)
    for attempt in range(12):
        bad = label_overlaps()
        if not bad:
            break
        for i in bad:  # still colliding: shrink the text slightly and retry
            if labels[i].txt.get_fontsize() > 5:
                f = 0.94
                labels[i].txt.set_fontsize(labels[i].txt.get_fontsize() * f)
                W_[i] *= f
                H_[i] *= f
        optimize(15)
    bad = label_overlaps()
    print("passes done; unresolved overlaps:", [labels[i].name for i in bad])

    nx, ny = leader_ends()
    for i, L in enumerate(labels):
        L.txt.set_position((X[i] / PPU + minx, Y[i] / PPU + miny))
        if OUT_[i]:
            x0_, y0_ = AX[i] / PPU + minx, AY[i] / PPU + miny
            x1_, y1_ = nx[i] / PPU + minx, ny[i] / PPU + miny
            ax.plot([x0_, x1_], [y0_, y1_], color=L.color, lw=0.7, zorder=4, solid_capstyle="round")
            ax.plot([x0_], [y0_], marker="o", ms=1.3, color=L.color, zorder=4)
    if PINS:
        _pins = load_pins(PINS)
        draw_pins(ax, _pins, pin_sprite(PIN_IN * DPI))
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        print("pins:", len(_pins))
    fig.savefig(OUT, dpi=DPI, transparent=True)
    print("wrote", OUT, "labels:", N, "leaders:", int(OUT_.sum()))


if __name__ == "__main__":
    main()
