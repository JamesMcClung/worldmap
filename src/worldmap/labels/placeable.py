"""What the solvers ask of every label, country (label.Label) or water (water.WaterLabel)."""

from typing import Protocol

from worldmap.labels.geometry import Covered


class Placeable(Covered, Protocol):
    full_size: float  # its whole feature's km2 (a river: km): its priority
    anchor_x: float  # where it belongs, poster px
    anchor_y: float

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> str:  # its --label kind
        ...

    @property
    def min_size(self) -> float:  # pt: it shrinks no further
        ...
