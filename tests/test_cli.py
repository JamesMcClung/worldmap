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
