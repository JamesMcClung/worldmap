"""Disjoint sets (union-find) over 0..n-1: pieces that meet, joined into groups."""


class DisjointSets:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        """The root of i's set."""
        while self.parent[i] != i:
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        """Join a's set into b's (b's root stays the root)."""
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list[int]]:
        """Each set's members, ascending; sets in order of their smallest member."""
        members: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            members.setdefault(self.find(i), []).append(i)
        return list(members.values())
