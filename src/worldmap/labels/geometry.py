"""Vectorised geometry for label placement: rectangles are centre (x, y) plus size (w, h), or [x0, y0, x1, y1] arrays."""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from shapely.geometry.base import BaseGeometry

from worldmap.labels import kernels

COVERAGE_MAX_CELLS = 1_000_000  # a Coverage's cells are 1 px, or bigger where its shape would need more than this


def nearest_point_on_rect(px, py, x, y, w, h) -> tuple[np.ndarray, np.ndarray]:
    """Point of the rectangle centred (x, y), size (w, h), nearest to (px, py). Broadcasts."""
    return np.clip(px, x - w / 2, x + w / 2), np.clip(py, y - h / 2, y + h / 2)


def two_line(text: str) -> str | None:
    """`text` split at the space that best balances the two lines' lengths, or None for a single word."""
    words = text.split(" ")
    if len(words) < 2:
        return None
    k = min(range(1, len(words)), key=lambda j: abs(len(" ".join(words[:j])) - len(" ".join(words[j:]))))
    return " ".join(words[:k]) + "\n" + " ".join(words[k:])


@dataclass
class Coverage:
    """A shape rasterised (by cell centre, cells `step` px from (x0, y0)) as a summed-area table: any box's area
    over it in a few lookups, to within a cell along the box's edges."""
    table: np.ndarray  # (rows + 1, cols + 1): table[j, i] = cells covered in rows < j, columns < i
    x0: float
    y0: float
    step: float

    @classmethod
    def of(cls, geom: BaseGeometry, max_cells: int = COVERAGE_MAX_CELLS) -> "Coverage":
        from affine import Affine
        from rasterio.features import rasterize

        if geom.is_empty:
            return cls(np.zeros((2, 2)), 0.0, 0.0, 1.0)
        gx0, gy0, gx1, gy1 = geom.bounds
        step = max(1.0, math.sqrt((gx1 - gx0) * (gy1 - gy0) / max_cells))
        cols, rows = math.ceil((gx1 - gx0) / step) + 1, math.ceil((gy1 - gy0) / step) + 1
        mask = rasterize([(geom, 1)], out_shape=(rows, cols), transform=Affine(step, 0, gx0, 0, step, gy0), dtype="uint8")
        table = np.zeros((rows + 1, cols + 1))
        table[1:, 1:] = mask.cumsum(0).cumsum(1)
        return cls(table, gx0, gy0, step)

    def below_left(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The shape's area below and left of each (x, y): the table, bilinear between its corners."""
        rows, cols = self.table.shape[0] - 1, self.table.shape[1] - 1
        u, v = np.clip((x - self.x0) / self.step, 0, cols), np.clip((y - self.y0) / self.step, 0, rows)
        i, j = np.minimum(u.astype(int), cols - 1), np.minimum(v.astype(int), rows - 1)
        fu, fv = u - i, v - j
        t = self.table
        return ((1 - fv) * ((1 - fu) * t[j, i] + fu * t[j, i + 1]) + fv * ((1 - fu) * t[j + 1, i] + fu * t[j + 1, i + 1])) * self.step ** 2

    def holds(self, boxes: np.ndarray) -> np.ndarray:
        """Whether each box (..., 4) has all four corners inside the shape (a convex one: wholly inside it)."""
        corners = self.contains(boxes[..., [0, 2, 0, 2]], boxes[..., [1, 1, 3, 3]])
        return corners.all(-1)

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Whether each point's cell is covered (outside the table: no)."""
        rows, cols = self.table.shape[0] - 1, self.table.shape[1] - 1
        u, v = np.floor((x - self.x0) / self.step), np.floor((y - self.y0) / self.step)
        within = (u >= 0) & (u < cols) & (v >= 0) & (v < rows)
        i, j = np.where(within, u, 0).astype(int), np.where(within, v, 0).astype(int)
        t = self.table
        return within & (t[j + 1, i + 1] - t[j, i + 1] - t[j + 1, i] + t[j, i] > 0)

    def area(self, boxes: np.ndarray) -> np.ndarray:
        """The shape's area inside each box (..., 4) [x0, y0, x1, y1]."""
        x0, y0, x1, y1 = (boxes[..., k] for k in range(4))
        return self.below_left(x1, y1) - self.below_left(x0, y1) - self.below_left(x1, y0) + self.below_left(x0, y0)


class Covered(Protocol):
    geom: BaseGeometry
    cover: Coverage | None


def coverage(item: Covered) -> Coverage:
    """A label's geom as a Coverage, made the first time it's needed."""
    if item.cover is None:
        item.cover = Coverage.of(item.geom)
    return item.cover


def on_feature(cover: Coverage, parts: np.ndarray, x: float, y: float, by_area: bool, slices: int) -> tuple[float, np.ndarray]:
    """The share of a label's rects (`parts`, relative to (x, y)) on its feature (`cover`), and which of its pieces
    are on it (in reading order), as the solver judges them (kernels.on_feature): by_area, `slices` slices of the
    box round them; else each rect, by its centre."""
    on = np.zeros(slices if by_area else len(parts), bool)
    share = kernels.on_feature(np.asarray(parts, float), float(x), float(y), by_area, cover.table, cover.x0, cover.y0, cover.step, on)
    return float(share), on
