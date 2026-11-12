from worldmap.disjoint import DisjointSets


def test_disjoint_sets() -> None:
    sets = DisjointSets(6)
    sets.union(4, 1)
    sets.union(1, 3)
    sets.union(5, 2)
    assert sets.find(4) == sets.find(1) == sets.find(3) == 3  # b's root stays the root
    assert sets.groups() == [[0], [1, 3, 4], [2, 5]]
