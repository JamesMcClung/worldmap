import numpy as np

from worldmap.labels.priority import label_order, ranks_per_kind, steps


def test_label_order() -> None:
    assert label_order(["countries", "water", "seas"]) == ["countries", "oceans", "seas", "lakes", "straits", "rivers"]
    assert label_order(["rivers", "countries", "rivers"]) == ["rivers", "countries"]


def test_ranks_relative_to_biggest_of_kind() -> None:
    kinds = ["seas"] * 4 + ["lakes"] * 2
    assert list(ranks_per_kind(kinds, [1000, 100.1, 99.9, 1e-9, 5, 0.04])) == [1, 1, 2, 5, 1, 3]


def test_ranks_at_the_steps() -> None:
    assert list(ranks_per_kind(["seas"] * 4, [1000, 100, 1, 0.1])) == [1, 1, 3, 4]


def test_ranks_zero_and_nan() -> None:
    assert list(ranks_per_kind(["seas"] * 3, [10, 0, np.nan])) == [1, 5, 5]


def test_steps_order() -> None:
    kinds = ["countries", "seas", "countries", "seas", "rivers"]
    assert steps(kinds, [1, 100, 1000, 1, 5], ["countries", "rivers", "seas"]) == [(1, [2]), (1, [4]), (1, [1]), (2, [3]), (3, [0])]


def test_steps_biggest_first_and_unasked_kinds_left_out() -> None:
    assert steps(["seas", "seas", "lakes"], [50, 60, 1], ["seas"]) == [(1, [1, 0])]


def test_steps_empty() -> None:
    assert steps([], [], ["countries"]) == []
