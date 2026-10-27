"""Vectorised geometry for label placement: rectangles are centre (x, y) plus size (w, h), or [x0, y0, x1, y1] arrays."""

import numpy as np
import shapely
from shapely.geometry import Polygon

PARALLEL_EPS = 1e-9  # segment direction components below this count as parallel to that axis


def rect(x: float, y: float, w: float, h: float) -> Polygon:
    return shapely.box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def nearest_point_on_rect(px, py, x, y, w, h) -> tuple[np.ndarray, np.ndarray]:
    """Point of the rectangle centred (x, y), size (w, h), nearest to (px, py). Broadcasts."""
    return np.clip(px, x - w / 2, x + w / 2), np.clip(py, y - h / 2, y + h / 2)


def segment_hits_rects(p0: np.ndarray, p1: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """p0, p1: (..., 2) broadcastable against rects (..., 4) [x0, y0, x1, y1]; True where the segment p0-p1 touches
    the rect (slab clipping)."""
    d = p1 - p0
    tmin = np.zeros(np.broadcast(d[..., 0], rects[..., 0]).shape)
    tmax = np.ones_like(tmin)
    ok = np.ones(tmin.shape, bool)
    for k in (0, 1):
        dk = d[..., k] * np.ones_like(tmin)
        lo, hi = rects[..., k], rects[..., k + 2]
        p = p0[..., k] * np.ones_like(tmin)
        parallel = np.abs(dk) < PARALLEL_EPS
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - p) / dk, (hi - p) / dk
        t_enter, t_exit = np.minimum(t1, t2), np.maximum(t1, t2)
        ok &= np.where(parallel, (p >= lo) & (p <= hi), True)
        tmin = np.where(parallel, tmin, np.maximum(tmin, t_enter))
        tmax = np.where(parallel, tmax, np.minimum(tmax, t_exit))
    return ok & (tmax >= tmin)


def segments_cross(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """Proper segment-segment intersection (touching doesn't count), broadcasting over leading dims."""

    def cross(o, p, q):
        return (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])

    return (cross(a0, a1, b0) * cross(a0, a1, b1) < 0) & (cross(b0, b1, a0) * cross(b0, b1, a1) < 0)


def two_line(text: str) -> str | None:
    """`text` split at the space that best balances the two lines' lengths, or None for a single word."""
    words = text.split(" ")
    if len(words) < 2:
        return None
    k = min(range(1, len(words)), key=lambda j: abs(len(" ".join(words[:j])) - len(" ".join(words[j:]))))
    return " ".join(words[:k]) + "\n" + " ".join(words[k:])
