"""What the solvers ask of every label, country (label.Label) or water (water.WaterLabel)."""

from typing import Protocol

import numpy as np

from worldmap.labels.geometry import Covered


class Placeable(Covered, Protocol):
    full_size: float  # its whole feature's km2 (a river: km): its priority
    anchor_x: float  # where it belongs, poster px
    anchor_y: float
    size: float  # pt, as laid out

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> str:  # its --label kind
        ...

    @property
    def min_size(self) -> float:  # pt: it shrinks no further
        ...

    @property
    def parts(self) -> np.ndarray:  # (k, 4) its rects, x0, y0, x1, y1 px relative to its position
        ...

    @property
    def width(self) -> float:  # px: the box round its parts
        ...

    @property
    def height(self) -> float: ...

    @property
    def out(self) -> bool:  # beside its feature with a leader line
        ...

    @property
    def water(self) -> bool:  # a water label: on its feature by its glyphs' centres, never 'out'
        ...
