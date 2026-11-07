"""Bending text (or any points) along a polyline: shared by the colour-bar titles and the water labels."""

import numpy as np


def arc_lengths(curve: np.ndarray) -> np.ndarray:
    """Cumulative length along an (n, 2) polyline, starting at 0."""
    return np.concatenate([[0], np.cumsum(np.hypot(*np.diff(curve, axis=0).T))])


def along(curve: np.ndarray, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The points at distances `u` along `curve` (cumulative lengths `s`)."""
    return np.stack([np.interp(u, s, curve[:, 0]), np.interp(u, s, curve[:, 1])], axis=-1)


def bend(curve: np.ndarray, u: np.ndarray, v: np.ndarray, h: float) -> np.ndarray:
    """The points `u` along `curve` and `v` along its left normal (the tangent is taken from u - h to u + h);
    shape (len(u), 2). Beyond the ends, the curve's end points are used."""
    s = arc_lengths(curve)
    tangent = along(curve, s, u + h) - along(curve, s, u - h)
    tangent /= np.hypot(*tangent.T)[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)  # to the left of the direction of travel
    return along(curve, s, u) + normal * v[:, None]
