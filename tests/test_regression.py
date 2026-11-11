"""Pixel-exact regression tests: each mode's rendered map must match its committed fingerprint."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import TESTS, check_fingerprint, environment, fingerprint, render


def test_render_default(tmp_path: Path) -> None:
    check_fingerprint("default", render(tmp_path))


def test_render_earth(tmp_path: Path) -> None:
    check_fingerprint("earth", render(tmp_path, "--color", "earth"))


def test_render_pins(tmp_path: Path) -> None:
    check_fingerprint("pins", render(tmp_path, "--pins", str(TESTS / "pins.txt")))


def test_render_regions(tmp_path: Path) -> None:
    check_fingerprint("regions", render(tmp_path, "--label", "countries,territories"))


def test_render_disputed(tmp_path: Path) -> None:
    check_fingerprint("disputed", render(tmp_path, "--disputed"))


def test_render_water(tmp_path: Path) -> None:
    check_fingerprint("water", render(tmp_path, "--label", "countries,water"))


def test_render_crop(tmp_path: Path) -> None:
    check_fingerprint("crop", render(tmp_path, "--crop", "France"))


def test_render_crop_water(tmp_path: Path) -> None:
    check_fingerprint("crop_water", render(tmp_path, "--crop", "160:-150,48:72", "--label", "countries,territories,water", "--disputed"))


def test_render_crop_pacific_water(tmp_path: Path) -> None:  # Equal Earth; "water" in the name keys the marine file
    check_fingerprint("crop_pacific_water", render(tmp_path, "--crop", "pacific", "--label", "countries,oceans,seas"))


def test_render_failure_shows_stderr(tmp_path: Path) -> None:
    with pytest.raises(pytest.fail.Exception, match="unrecognized arguments"):
        render(tmp_path, "--no-such-flag")


def test_environment_includes_map_data() -> None:
    env = environment("earth")
    assert {"countries", "states", "lakes", "raster"} <= env.keys()
    assert "raster" not in environment("default")
    assert "marine" in environment("water") and "marine" not in environment("default")
    assert "rivers" in environment("water") and "rivers" not in environment("default")


def tiny_png(tmp_path: Path) -> Path:
    png = tmp_path / "tiny.png"
    Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(png)
    return png


def test_missing_fingerprint_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UPDATE_REFERENCE", raising=False)
    store = tmp_path / "fp.json"
    store.write_text("{}")
    with pytest.raises(pytest.fail.Exception, match="UPDATE_REFERENCE=1"):
        check_fingerprint("nope", tiny_png(tmp_path), store=store)


def test_version_mismatch_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UPDATE_REFERENCE", raising=False)
    png = tiny_png(tmp_path)
    store = tmp_path / "fp.json"
    store.write_text(json.dumps({"tiny": {"sha256": fingerprint(png), "env": {**environment(), "matplotlib": "0.0"}}}))
    with pytest.raises(pytest.skip.Exception, match="matplotlib"):
        check_fingerprint("tiny", png, store=store)


@pytest.mark.xdist_group("topographic_crop")  # 5 GB
def test_render_topographic_crop(tmp_path: Path) -> None:
    check_fingerprint("topographic_crop", render(tmp_path, "--type", "topographic", "--ocean", "--crop", "5:17,40:49"))


@pytest.mark.xdist_group("topographic")  # 3.3 GB each, one at a time
def test_render_topographic_ocean(tmp_path: Path) -> None:
    check_fingerprint("topographic_ocean", render(tmp_path, "--type", "topographic", "--ocean"))


@pytest.mark.xdist_group("topographic")  # 3.3 GB each, one at a time
def test_render_topographic_water(tmp_path: Path) -> None:
    png = render(tmp_path, "--type", "topographic", "--label", "water")
    check_fingerprint("topographic_water", png)
    im = np.asarray(Image.open(png))
    h, w = im.shape[:2]
    assert im[h // 2, 2, 3] == 0 and im[2, w // 10, 3] == 0  # transparent outside the globe: left edge at the equator;
    # top-left corner (no bar there)


def test_environment_includes_topographic_data() -> None:
    assert {"etopo", "glaciers", "ice_shelves", "land", "rasterio", "gdal"} <= environment("topographic_ocean").keys()
    assert "etopo" not in environment("default")
