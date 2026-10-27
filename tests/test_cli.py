import subprocess
import sys
from pathlib import Path

import pytest

from worldmap.cli import main


def test_ocean_flag_rejected() -> None:
    with pytest.raises(SystemExit) as e:
        main(["--ocean"])
    assert e.value.code == 2


def test_console_script_help() -> None:
    """The installed `make_map` command still works (needs `pip install -e .` after entry-point changes)."""
    out = subprocess.run([Path(sys.executable).parent / "make_map", "-h"], check=True, capture_output=True, text=True).stdout
    assert "--pin-size" in out
    assert "--ocean" not in out
