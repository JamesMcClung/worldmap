"""Pixel-exact regression tests: each mode's rendered map must match its committed fingerprint."""

import json
from pathlib import Path

import pytest
from PIL import Image

from conftest import TESTS, check_fingerprint, environment, fingerprint, render


def test_render_default(tmp_path: Path) -> None:
    check_fingerprint("default", render(tmp_path))


def test_render_earth(tmp_path: Path) -> None:
    check_fingerprint("earth", render(tmp_path, "--earth"))


def test_render_pins(tmp_path: Path) -> None:
    check_fingerprint("pins", render(tmp_path, "--pins", str(TESTS / "pins.txt")))


def test_render_regions(tmp_path: Path) -> None:
    check_fingerprint("regions", render(tmp_path, "--regions"))


def test_render_failure_shows_stderr(tmp_path: Path) -> None:
    with pytest.raises(pytest.fail.Exception, match="unrecognized arguments"):
        render(tmp_path, "--no-such-flag")


def test_environment_includes_map_data() -> None:
    env = environment("earth")
    assert {"countries", "states", "lakes", "raster"} <= env.keys()
    assert "raster" not in environment("default")


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
