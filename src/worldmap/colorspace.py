"""sRGB <-> OKLab / OKLCH conversions (https://bottosson.github.io/posts/oklab/)."""

import numpy as np
from numpy.typing import ArrayLike

RGB = tuple[float, float, float]  # sRGB, each channel in [0, 1]

GAMUT_TOLERANCE = 1e-6  # linear-RGB slack before a colour counts as out of gamut
DARK_FALLBACK: RGB = (0.1, 0.1, 0.1)


def srgb_to_lin(c: ArrayLike) -> np.ndarray:
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c: ArrayLike) -> np.ndarray:
    c = np.asarray(c, float)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def oklab_to_linear_rgb(L: float, a: float, b: float) -> np.ndarray:
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    return np.array([
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ])


def oklch_to_srgb(L: float, C: float, h: float) -> RGB | None:
    """OKLCH (h in degrees) -> sRGB, or None if out of gamut."""
    lin = oklab_to_linear_rgb(L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h)))
    if (lin < -GAMUT_TOLERANCE).any() or (lin > 1 + GAMUT_TOLERANCE).any():
        return None
    r, g, b = np.clip(lin_to_srgb(np.clip(lin, 0, 1)), 0, 1)
    return (r, g, b)


def srgb_to_oklch(rgb: ArrayLike) -> tuple[float, float, float]:
    """sRGB -> OKLCH as (L, C, hue in degrees [0, 360))."""
    lin = srgb_to_lin(rgb)
    l = np.cbrt(0.4122214708 * lin[0] + 0.5363325363 * lin[1] + 0.0514459929 * lin[2])
    m = np.cbrt(0.2119034982 * lin[0] + 0.6806995451 * lin[1] + 0.1073969566 * lin[2])
    s = np.cbrt(0.0883024619 * lin[0] + 0.2817188376 * lin[1] + 0.6299787005 * lin[2])
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    b = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return L, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360


def fit_gamut(L: float, C: float, h: float, steps: int = 40) -> tuple[RGB, float] | None:
    """The most saturated in-gamut colour at lightness L and hue h, trying chroma from C down to 0 in `steps`
    steps. Returns (sRGB, chroma used), or None if even grey is out of gamut."""
    for chroma in np.linspace(C, 0, steps):
        rgb = oklch_to_srgb(L, chroma, h)
        if rgb is not None:
            return rgb, float(chroma)
    return None


def darken(rgb: ArrayLike, L: float = 0.26) -> RGB:
    """Same OKLab hue and (as far as the gamut allows) chroma as `rgb`, at lightness L."""
    _, C, h = srgb_to_oklch(rgb)
    fitted = fit_gamut(L, C, h)
    return fitted[0] if fitted else DARK_FALLBACK
