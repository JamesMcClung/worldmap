import numpy as np
import pytest

from worldmap.colorspace import darken, fit_gamut, oklch_to_srgb, srgb_to_oklch


def test_round_trip() -> None:
    rgb = (0.2, 0.5, 0.8)
    assert oklch_to_srgb(*srgb_to_oklch(rgb)) == pytest.approx(rgb, abs=1e-6)


def test_out_of_gamut_is_none() -> None:
    assert oklch_to_srgb(0.5, 0.5, 30) is None


def test_fit_gamut_reduces_chroma() -> None:
    fitted = fit_gamut(0.5, 0.5, 30)
    assert fitted is not None
    rgb, chroma = fitted
    assert chroma < 0.5
    assert all(0 <= c <= 1 for c in rgb)


def test_darken_keeps_hue() -> None:
    _, _, hue_in = srgb_to_oklch((0.9, 0.6, 0.3))
    L, _, hue_out = srgb_to_oklch(darken((0.9, 0.6, 0.3)))
    assert L == pytest.approx(0.26, abs=1e-3)
    assert abs((hue_out - hue_in + 180) % 360 - 180) < 2
    assert np.isfinite(hue_out)
