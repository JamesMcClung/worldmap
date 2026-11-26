"""Which way the rivers flow, so that everything downstream of a drawn river is drawn too (a delta's branches, a
river's last stretch below a junction). Natural Earth's river lines run either way, so the flow is inferred: the
mouths are the line ends at the coast, and water runs toward the nearest one along the network."""

import functools
import heapq
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

from worldmap.data import EQUAL_AREA_CRS, load_lakes, load_land, load_rivers
from worldmap.disjoint import DisjointSets

FLOW_JOIN_M = 2_000  # lines meet where an end is this close to another's end or middle (any wider joins the
# sources of neighbouring rivers, so the Danube flowed up the Tisza into the Prut)
SAME_NAME_JOIN_M = 30_000  # ends of one name join this far apart (lakes and reservoirs break a river up: the Mekong
# at Pakse, which left the Mun draining into Thailand)
MOUTH_M = 10_000  # a network's end this close to the coast is a mouth
CANAL = r"canal|csatorna"  # in a name: a canal, left out of the network (the Volga drained down the Don, the Nile
# down the Suez Canal)


@dataclass
class RiverFlow:
    """The rivers cut into pieces at every junction (`pieces`: Natural Earth's columns, lon/lat), each piece flowing
    from node `up` to node `down`, and `order`: the pieces, each after every piece upstream of it."""

    pieces: gpd.GeoDataFrame
    up: np.ndarray
    down: np.ndarray
    order: np.ndarray

    def downstream(self, ranks: np.ndarray) -> np.ndarray:
        """Each piece's rank: the best (smallest) of its own and that of every piece upstream of it."""
        out = np.array(ranks)
        into: dict[int, int] = {}  # node -> the best rank flowing into it
        for k in self.order:
            out[k] = min(out[k], into.get(int(self.up[k]), out[k]))
            into[int(self.down[k])] = min(into.get(int(self.down[k]), out[k]), out[k])
        return out


def split_at_junctions(lines: np.ndarray, join: float) -> tuple[np.ndarray, np.ndarray]:
    """The `lines` (metres) cut where another line's end meets their middle (within `join`), and each piece's line."""
    ends = shapely.points([c for line in lines for c in (line.coords[0], line.coords[-1])])
    cuts: dict[int, set[float]] = {}
    for e, k in shapely.STRtree(lines).query(ends, predicate="dwithin", distance=join).T:
        if k != e // 2:
            at = lines[k].line_locate_point(ends[e])
            if join < at < lines[k].length - join:
                cuts.setdefault(int(k), set()).add(at)
    pieces, of = [], []
    for k, line in enumerate(lines):
        at = [0.0, *sorted(cuts.get(k, ())), line.length]
        pieces += [substring(line, a, b) for a, b in zip(at, at[1:])]
        of += [k] * (len(at) - 1)
    return np.array(pieces), np.array(of, int)


def stretches(lines: BaseGeometry, n: int = 64) -> np.ndarray:
    """`lines` cut into lines of at most `n` segments."""
    out = []
    for line in shapely.get_parts(lines):
        xy = shapely.get_coordinates(line)
        out += [shapely.linestrings(xy[i : i + n + 1]) for i in range(0, len(xy) - 1, n)]
    return np.array(out)


def leaves_of(joined: DisjointSets, n: int) -> np.ndarray:
    """The ends (of `n`) that `joined` joins to no other."""
    node = np.array([joined.find(i) for i in range(n)])
    return np.flatnonzero(np.bincount(node, minlength=n)[node] == 1)


def flow(lines: np.ndarray, names: np.ndarray, ranks: np.ndarray, links: np.ndarray, coast: BaseGeometry, lakes: BaseGeometry, join: float = FLOW_JOIN_M, same_name_join: float = SAME_NAME_JOIN_M, mouth: float = MOUTH_M) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For `lines` (metres; only those `links` join the network): the pieces (split_at_junctions), their line, the
    nodes each flows from and to, and an order of the pieces with each after every piece upstream of it. Ends join
    within `join`, or `same_name_join` for lines of one of the `names` ("" is none). The mouths are the network ends
    within `mouth` of the `coast`; a network without one (it ends in a lake, or a gap cuts it off) drains through
    an end of its most important line (smallest of the `ranks`), the one nearest the coast or the `lakes` (lines).
    Each piece flows toward the nearest mouth along the network."""
    pieces, of = split_at_junctions(lines, join)
    ends = shapely.points([c for p in pieces for c in (p.coords[0], p.coords[-1])])
    linked, name = np.repeat(links[of], 2), np.repeat(names[of], 2)
    joined = DisjointSets(len(ends))
    tree = shapely.STRtree(ends)
    for a, b in tree.query(ends, predicate="dwithin", distance=join).T:
        if linked[a] and linked[b]:
            joined.union(int(a), int(b))
    at_coast = np.zeros(len(ends), bool)  # by short stretches of coast: dwithin of the whole coast took 4 s
    at_coast[np.unique(shapely.STRtree(stretches(coast)).query(ends, predicate="dwithin", distance=mouth)[0])] = True
    for a in leaves_of(joined, len(ends)):  # an inland end bridges to the nearest end of its name (one, or a delta
        # collapses into one node)
        if linked[a] and name[a] and not at_coast[a]:
            near = [int(b) for b in tree.query(ends[a], predicate="dwithin", distance=same_name_join) if b // 2 != a // 2 and linked[b] and name[b] == name[a]]
            if near:
                joined.union(int(a), near[int(np.argmin(shapely.distance(ends[a], ends[near])))])
    node = np.array([joined.find(i) for i in range(len(ends))])
    nodes = node.reshape(-1, 2)
    networks = DisjointSets(len(ends))
    for u, v in nodes:
        networks.union(int(u), int(v))
    leaves = leaves_of(joined, len(ends))
    mouths = {int(node[i]) for i in leaves if at_coast[i]}
    inland: dict[int, list[int]] = {}
    for i in leaves:
        inland.setdefault(networks.find(int(node[i])), []).append(int(i))
    for net in {networks.find(n) for n in mouths}:
        inland.pop(net, None)
    rank = np.repeat(ranks[of], 2)
    for i in inland.values():
        gap = np.minimum(shapely.distance(ends[i], coast), shapely.distance(ends[i], lakes))
        mouths.add(int(node[i[np.lexsort((gap, rank[i]))[0]]]))

    length = shapely.length(pieces)
    nexts: dict[int, list[tuple[int, int]]] = {}
    for k, (u, v) in enumerate(nodes):
        nexts.setdefault(int(u), []).append((int(v), k))
        nexts.setdefault(int(v), []).append((int(u), k))
    dist = {m: 0.0 for m in mouths}
    todo = [(0.0, m) for m in mouths]
    while todo:  # Dijkstra from every mouth at once
        d, n = heapq.heappop(todo)
        if d <= dist[n]:
            for o, k in nexts.get(n, []):
                if d + length[k] < dist.get(o, np.inf):
                    dist[o] = d + length[k]
                    heapq.heappush(todo, (dist[o], o))
    far = np.array([[dist.get(int(n), np.inf) for n in uv] for uv in nodes]).reshape(-1, 2)
    first_up = far[:, 0] >= far[:, 1]
    up, down = np.where(first_up, nodes[:, 0], nodes[:, 1]), np.where(first_up, nodes[:, 1], nodes[:, 0])
    order = np.lexsort((-far.min(axis=1), -far.max(axis=1)))  # farthest from a mouth first (a level piece before the next)
    return pieces, of, up, down, order


@functools.cache
def river_flow() -> RiverFlow:
    """load_rivers' lines cut at their junctions, and which way each piece flows (flow; canals stand alone)."""
    rivers = load_rivers()
    parts, row = shapely.get_parts(rivers.to_crs(EQUAL_AREA_CRS).geometry.values, return_index=True)
    names = rivers.name_en.fillna("")
    canal = (rivers.featurecla == "Canal") | names.str.contains(CANAL, case=False)
    coast, lakes = (shapely.multilinestrings(shapely.get_parts(shapely.boundary(g.to_crs(EQUAL_AREA_CRS).geometry.values))) for g in (load_land(), load_lakes()))
    pieces, of, up, down, order = flow(parts, names.to_numpy()[row], rivers.scalerank.to_numpy()[row], ~canal.to_numpy()[row], coast, lakes)
    table = rivers.drop(columns="geometry").iloc[row[of]].reset_index(drop=True)
    lonlat = gpd.GeoSeries(pieces, crs=EQUAL_AREA_CRS).to_crs(rivers.crs)
    return RiverFlow(gpd.GeoDataFrame(table, geometry=lonlat.values, crs=rivers.crs), up, down, order)
