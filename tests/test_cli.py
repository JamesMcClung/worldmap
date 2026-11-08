import subprocess
import sys
from pathlib import Path

import pytest

from worldmap.cli import parse_args
from worldmap.frame import DPI


def test_dpi_flag() -> None:
    assert parse_args([]).dpi == DPI
    assert parse_args(["--dpi", "50"]).dpi == 50
    for bad in ["0", "-1", "x"]:
        with pytest.raises(SystemExit) as e:
            parse_args(["--dpi", bad])
        assert e.value.code == 2


def test_console_script_help() -> None:
    """The installed `make_map` command still works (needs `pip install -e .` after entry-point changes)."""
    out = subprocess.run([Path(sys.executable).parent / "make_map", "-h"], check=True, capture_output=True, text=True).stdout
    assert "--pin-size" in out
    assert "--type" in out and "--ocean" in out


def test_color_flag() -> None:
    assert parse_args([]).color == "colorblind"
    assert parse_args(["--color", "earth"]).color == "earth"
    for bad in [["--color", "pastel"], ["--earth"]]:
        with pytest.raises(SystemExit) as e:
            parse_args(bad)
        assert e.value.code == 2


def test_label_flag() -> None:
    assert parse_args([]).label == {"countries"}
    assert parse_args(["--label", "territories"]).label == {"territories"}
    assert parse_args(["--label", "countries,territories"]).label == {"countries", "territories"}
    assert parse_args(["--label", "none"]).label == set()
    assert parse_args(["--label", "countries,oceans,seas,lakes,water"]).label == {"countries", "oceans", "seas", "lakes", "water"}
    assert parse_args(["--label", "countries,rivers,straits"]).label == {"countries", "rivers", "straits"}
    for bad in [["--label", "continents"], ["--label", "none,countries"], ["--label", ""], ["--regions"]]:
        with pytest.raises(SystemExit) as e:
            parse_args(bad)
        assert e.value.code == 2


def test_type_flag() -> None:
    assert (parse_args([]).type, parse_args([]).ocean) == ("political", False)
    assert (parse_args(["--type", "topographic", "--ocean"]).type, parse_args(["--ocean"]).ocean) == ("topographic", True)
    with pytest.raises(SystemExit) as e:
        parse_args(["--type", "sky"])
    assert e.value.code == 2


def test_pick_uses_crop_frame(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from PIL import Image

    from worldmap import cli

    png = tmp_path / "crop.png"
    Image.new("RGBA", (4, 4)).save(png)
    seen = []
    monkeypatch.setattr(cli, "pick_pins", lambda path, out, frame, *rest: seen.append(frame))
    cli.main(["--pick", str(tmp_path / "p.txt"), "--crop", "5,43:17,49", str(png)])
    assert len(seen) == 1 and not seen[0].globe and "+proj=laea" in seen[0].crs
