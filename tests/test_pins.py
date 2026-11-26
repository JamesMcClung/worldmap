from pathlib import Path

import numpy as np
import pytest

from worldmap.cli import main
from worldmap.pins import PIN_CIRCLES, PinStyle, draw_pin_circles, fit_to_render, load_pins, pin_sprite


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


def test_pin_circles_northernmost_first() -> None:
    import matplotlib.figure

    from worldmap.frame import make_frame

    frame = make_frame()
    ax = matplotlib.figure.Figure().add_subplot()
    circles = draw_pin_circles(ax, frame, [(0, -10), (0, 40)], 0.24)
    assert len(circles) == 2 * len(PIN_CIRCLES)
    centre_y = [c.center[1] for c in circles]
    assert centre_y[0] > centre_y[len(PIN_CIRCLES)]  # the northern pin first, under the southern one
    rim = circles[1]
    assert rim.radius == pytest.approx(0.12 * frame.units_per_in) and rim.center == pytest.approx(frame.to_map.transform(0, 40))


def test_fit_to_render_takes_its_size(tmp_path: Path) -> None:
    from dataclasses import replace

    from PIL import Image

    from worldmap.frame import make_frame

    tall = replace(make_frame(), xmin=0, xmax=1, ymin=0, ymax=2)
    Image.new("RGBA", (1800, 3600)).save(tmp_path / "tall.png", dpi=(200, 200))
    assert fit_to_render(str(tmp_path / "tall.png"), tall).size_in == pytest.approx(18, abs=1e-3)  # PNG stores dots per metre
    Image.new("RGBA", (4, 8)).save(tmp_path / "bare.png")
    assert fit_to_render(str(tmp_path / "bare.png"), tall).size_in is None


def test_fit_to_render_checks_shape(tmp_path: Path) -> None:
    from PIL import Image

    from worldmap.frame import make_frame

    for size in [(1800, 923), (1800, 924)]:  # the world at 50 dpi: 923.5 px tall, either way it rounds
        Image.new("RGBA", size).save(tmp_path / "world.png")
        fit_to_render(str(tmp_path / "world.png"), make_frame())
    Image.new("RGBA", (1800, 1800)).save(tmp_path / "square.png")
    with pytest.raises(SystemExit, match="pass the --crop"):
        fit_to_render(str(tmp_path / "square.png"), make_frame())


def test_pick_without_render_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="run make_map first"):
        main(["--pick", "p.txt"])
