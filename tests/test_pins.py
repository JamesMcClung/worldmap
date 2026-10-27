from pathlib import Path

import numpy as np
import pytest

from worldmap.cli import main
from worldmap.pins import PinStyle, load_pins, pin_sprite


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pins.txt"
    path.write_text(text)
    return path


def test_load_pins_parses_comments_and_spaces(tmp_path: Path) -> None:
    assert load_pins(write(tmp_path, "# hi\n10,20\n 30 40 # x\n\n")) == [(10.0, 20.0), (30.0, 40.0)]


def test_load_pins_rejects_garbage(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=":1:"):
        load_pins(write(tmp_path, "abc\n"))


def test_load_pins_rejects_out_of_range(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        load_pins(write(tmp_path, "200,0\n"))


def test_pin_sprite_shape_and_alpha() -> None:
    sprite = pin_sprite(48)
    n = round(2 * PinStyle().pad * 24)
    assert sprite.shape == (n, n, 4)
    assert sprite.min() >= 0 and sprite.max() <= 1
    assert sprite[n // 2, n // 2, 3] == pytest.approx(1)
    assert sprite[0, 0, 3] == 0
    assert np.isfinite(sprite).all()


def test_pick_without_render_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="run make_map first"):
        main(["--pick", "p.txt"])
