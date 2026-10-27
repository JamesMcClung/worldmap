import pytest

from worldmap.make_map import main


def test_ocean_flag_rejected() -> None:
    with pytest.raises(SystemExit) as e:
        main(["--ocean"])
    assert e.value.code == 2
