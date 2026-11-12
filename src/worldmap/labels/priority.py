"""Priority: labels are placed in steps, most important first. Each kind ranks its labels by full size (area, or a
river's length) against the biggest of that kind in the frame; rank by rank, the kinds follow the --label order."""

from collections.abc import Callable, Sequence

import numpy as np

from worldmap.labels.water import WATER_KINDS

N_RANKS = 5
RANK_STEP = 10.0  # rank k holds the labels down to 1 / RANK_STEP^k of their kind's biggest; the last, the rest

Ranker = Callable[[Sequence[str], Sequence[float]], np.ndarray]


def label_order(label: Sequence[str]) -> list[str]:
    """The --label kinds in order, `water` standing for each water kind in its place, without repeats."""
    water = [k for k in WATER_KINDS if k != "water"]
    return list(dict.fromkeys(k for kind in label for k in (water if kind == "water" else [kind])))


def ranks_per_kind(kinds: Sequence[str], sizes: Sequence[float]) -> np.ndarray:
    """Each label's rank, 1..N_RANKS: how many factors of RANK_STEP it is below the biggest label of its kind.
    A size of 0 (or unknown) is the last rank."""
    size = np.asarray(sizes, float)
    kind_of = np.asarray(kinds, object)
    ranks = np.full(len(size), N_RANKS)
    for kind in set(kinds):
        mine = (kind_of == kind) & (size > 0)  # NaN compares False
        if mine.any():  # rank k: at least biggest / RANK_STEP^k (compared, not logged: exact at the steps)
            floors = size[mine].max() / RANK_STEP ** np.arange(1, N_RANKS)
            ranks[mine] = 1 + (size[mine][:, None] < floors).sum(1)
    return ranks


def steps(kinds: Sequence[str], sizes: Sequence[float], order: Sequence[str], rank: Ranker = ranks_per_kind) -> list[tuple[int, list[int]]]:
    """The placement steps: (rank, label indices, biggest first) for each rank, then each kind of `order` in turn.
    Empty steps, and kinds not in `order`, are left out."""
    ranks = rank(kinds, sizes)
    out = []
    for r in range(1, N_RANKS + 1):
        for kind in order:
            members = [i for i in range(len(kinds)) if kinds[i] == kind and ranks[i] == r]
            if members:
                out.append((r, sorted(members, key=lambda i: -np.nan_to_num(sizes[i]))))
    return out
