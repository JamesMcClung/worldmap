from pathlib import Path

import numpy as np
import pytest

from worldmap.cli import main
from worldmap.pins import PinStyle, check_render_shape, load_pins, pin_sprite, rendered_size_in


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


def test_rendered_size_in(tmp_path: Path) -> None:
    from PIL import Image

    Image.new("RGBA", (1800, 3600)).save(tmp_path / "tall.png", dpi=(200, 200))
    assert rendered_size_in(str(tmp_path / "tall.png")) == pytest.approx(18, abs=1e-3)  # PNG stores dots per metre
    Image.new("RGBA", (4, 4)).save(tmp_path / "bare.png")
    assert rendered_size_in(str(tmp_path / "bare.png")) is None


def test_check_render_shape(tmp_path: Path) -> None:
    from PIL import Image

    from worldmap.frame import make_frame

    for size in [(1800, 923), (1800, 924)]:  # the world at 50 dpi: 923.5 px tall, either way it rounds
        Image.new("RGBA", size).save(tmp_path / "world.png")
        check_render_shape(str(tmp_path / "world.png"), make_frame())
    Image.new("RGBA", (1800, 1800)).save(tmp_path / "square.png")
    with pytest.raises(SystemExit, match="pass the --crop"):
        check_render_shape(str(tmp_path / "square.png"), make_frame())


def test_pick_without_render_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="run make_map first"):
        main(["--pick", "p.txt"])
