import numpy as np
from shapely.geometry import LineString

from worldmap.rivers import RiverFlow, flow, split_at_junctions

KM = 1000.0
COAST = LineString([(-500 * KM, 0), (500 * KM, 0)])  # the sea is south of y = 0
NO_LAKES = LineString([(0, 1e9), (1, 1e9)])


def flows(lines: list[LineString], names: list[str], links: list[bool] | None = None, ranks: list[int] | None = None,
          coast: LineString = COAST) -> tuple[np.ndarray, RiverFlow]:
    pieces, of, up, down, order = flow(np.array(lines), np.array(names), np.array(ranks or [1] * len(lines)),
                                       np.array(links or [True] * len(lines)), coast, NO_LAKES)
    return of, RiverFlow(None, up, down, order)  # type: ignore[arg-type]


def shown(lines: list[LineString], names: list[str], drawn: str, **kw: object) -> set[str]:
    """The names of the pieces downstream of the river `drawn` (rank 1, the rest 6)."""
    of, f = flows(lines, names, **kw)  # type: ignore[arg-type]
    ranks = f.downstream(np.where(np.array(names)[of] == drawn, 1, 6))
    return {names[k] for k in of[ranks == 1]}


def test_a_delta_and_the_stretch_below_a_junction_are_downstream() -> None:
    lines = [LineString([(0, 500 * KM), (0, 100 * KM)]),  # the main river, drawn up from its mouth (lines run either way)
             LineString([(0, 100 * KM), (-50 * KM, 1 * KM)]), LineString([(50 * KM, 1 * KM), (0, 100 * KM)]),  # its delta
             LineString([(200 * KM, 300 * KM), (0, 300 * KM)])]  # a tributary meeting its middle
    names = ["Main", "West branch", "East branch", "Tributary"]
    assert shown(lines, names, "Main") == {"Main", "West branch", "East branch"}
    assert shown(lines, names, "Tributary") == {"Main", "West branch", "East branch", "Tributary"}
    assert shown(lines, names, "West branch") == {"West branch"}


def test_sources_close_together_stay_apart() -> None:
    lines = [LineString([(0, 400 * KM), (0, 1 * KM)]), LineString([(10 * KM, 400 * KM), (300 * KM, 1 * KM)])]
    assert shown(lines, ["A", "B"], "A") == {"A"}


def test_a_river_broken_at_a_lake_is_joined_by_name() -> None:
    lines = [LineString([(0, 400 * KM), (0, 220 * KM)]), LineString([(0, 200 * KM), (0, 1 * KM)])]  # 20 km apart
    assert shown(lines, ["A", "A"], "A") == {"A"}
    of, f = flows(lines, ["A", "A"])
    assert f.downstream(np.array([1, 6]))[1] == 1
    of, f = flows(lines, ["A", "B"])  # different names: not joined
    assert f.downstream(np.array([1, 6]))[1] == 6


def test_canals_are_not_part_of_the_network() -> None:
    lines = [LineString([(0, 400 * KM), (0, 1 * KM)]), LineString([(0, 300 * KM), (300 * KM, 1 * KM)])]
    assert shown(lines, ["A", "Canal"], "A", links=[True, False]) == {"A"}
    assert shown(lines, ["A", "Canal"], "A") == {"A", "Canal"}


def test_an_inland_network_drains_through_its_main_river() -> None:
    inland = LineString([(-500 * KM, -1000 * KM), (500 * KM, -1000 * KM)])  # far from both
    lines = [LineString([(0, 400 * KM), (0, 100 * KM)]), LineString([(0, 300 * KM), (-300 * KM, 50 * KM)])]
    # the tributary's source is nearer the coast, but the main river's end wins
    assert shown(lines, ["Main", "Tributary"], "Main", ranks=[1, 5], coast=inland) == {"Main"}
    assert shown(lines, ["Main", "Tributary"], "Tributary", ranks=[1, 5], coast=inland) == {"Main", "Tributary"}


def test_a_line_is_cut_where_another_ends_on_it() -> None:
    lines = np.array([LineString([(0, 0), (100 * KM, 0)]), LineString([(50 * KM, 1 * KM), (50 * KM, 80 * KM)])])
    pieces, of = split_at_junctions(lines, 2 * KM)
    assert [round(p.length / KM) for p in pieces] == [50, 50, 79] and list(of) == [0, 0, 1]


def test_the_nile_delta_and_the_shatt_al_arab_are_drawn_with_their_rivers() -> None:
    from worldmap.rivers import river_flow

    f = river_flow()
    names = f.pieces.name_en.fillna("").to_numpy()

    def below(river: str) -> set[str]:
        return set(names[f.downstream(np.where(names == river, 1, 6)) == 1]) - {river, ""}

    assert below("Nile") == {"Rosetta Branch", "Damietta Branch"}
    assert below("Tigris") == {"Shatt al-Hayy", "Euphrates", "Shatt al-Arab"}  # the Euphrates below their junction
    assert "Chenab" not in below("Indus") and "Prut" not in below("Danube")
