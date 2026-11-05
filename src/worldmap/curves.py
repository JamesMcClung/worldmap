"""Bending text (or any points) along a polyline: shared by the colour-bar titles and the water labels."""

import numpy as np


def arc_lengths(curve: np.ndarray) -> np.ndarray:
    """Cumulative length along an (n, 2) polyline, starting at 0."""
    return np.concatenate([[0], np.cumsum(np.hypot(*np.diff(curve, axis=0).T))])


def bend(curve: np.ndarray, u: np.ndarray, v: np.ndarray, h: float) -> np.ndarray:
    """The points `u` along `curve` and `v` along its left normal (the tangent is taken from u - h to u + h);
    shape (len(u), 2). Beyond the ends, the curve's end points are used."""
    s = arc_lengths(curve)

    def at(d: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(d, s, curve[:, 0]), np.interp(d, s, curve[:, 1])], axis=-1)

    tangent = at(u + h) - at(u - h)
    tangent /= np.hypot(*tangent.T)[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)  # to the left of the direction of travel
    return at(u) + normal * v[:, None]
