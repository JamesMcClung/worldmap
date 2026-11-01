import subprocess
import sys
from pathlib import Path

import pytest

from worldmap.cli import main, parse_args
from worldmap.frame import DPI


def test_ocean_flag_rejected() -> None:
    with pytest.raises(SystemExit) as e:
        main(["--ocean"])
    assert e.value.code == 2


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
    assert "--ocean" not in out


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
    for bad in [["--label", "oceans"], ["--label", "none,countries"], ["--label", ""], ["--regions"]]:
        with pytest.raises(SystemExit) as e:
            parse_args(bad)
        assert e.value.code == 2


def test_disputed_flag() -> None:
    assert parse_args([]).disputed is False
    assert parse_args(["--disputed"]).disputed is True
