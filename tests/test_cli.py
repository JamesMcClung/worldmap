import subprocess
import sys
from pathlib import Path

import pytest

from worldmap.cli import draws_rivers, make_map_frame, parse_args
from worldmap.frame import DPI


def test_dpi_flag() -> None:
    assert parse_args([]).dpi == DPI
    assert parse_args(["--dpi", "50"]).dpi == 50
    for bad in ["0", "-1", "x"]:
        with pytest.raises(SystemExit) as e:
            parse_args(["--dpi", bad])
        assert e.value.code == 2


def test_size_parses() -> None:
    assert parse_args([]).size is None
    assert parse_args(["--size", "12"]).size == 12
    for bad in ["0", "-3", "x"]:
        with pytest.raises(SystemExit) as e:
            parse_args(["--size", bad])
        assert e.value.code == 2


def test_size_overrides_crop_default() -> None:
    f = make_map_frame(parse_args(["--crop", "-5:9,42:51", "--size", "3"]))  # below the crop default's floor
    assert max(f.width_in, f.height_in) == pytest.approx(3)


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
    assert parse_args([]).label == ["countries"]
    assert parse_args(["--label", "territories"]).label == ["territories"]
    assert parse_args(["--label", "countries,territories"]).label == ["countries", "territories"]
    assert parse_args(["--label", "none"]).label == []
    assert parse_args(["--label", "countries,oceans,seas,lakes,water"]).label == ["countries", "oceans", "seas", "lakes", "straits", "rivers"]
    assert parse_args(["--label", "seas,water,countries"]).label == ["seas", "oceans", "lakes", "straits", "rivers", "countries"]
    assert parse_args(["--label", "countries,rivers,straits"]).label == ["countries", "rivers", "straits"]
    assert parse_args(["--label", "seas,countries,seas"]).label == ["seas", "countries"]
    for bad in [["--label", "continents"], ["--label", "none,countries"], ["--label", ""], ["--regions"]]:
        with pytest.raises(SystemExit) as e:
            parse_args(bad)
        assert e.value.code == 2


def test_rivers_are_drawn_with_the_flag_or_their_labels() -> None:
    assert draws_rivers(parse_args(["--rivers"])) and draws_rivers(parse_args(["--label", "rivers"]))
    assert not draws_rivers(parse_args([]))


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
    cli.main(["--pick", str(tmp_path / "p.txt"), "--crop", "5:17,43:49", str(png)])
    assert len(seen) == 1 and not seen[0].globe and seen[0].azimuthal


def test_pick_takes_the_size_of_the_render(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from PIL import Image

    from worldmap import cli

    png = tmp_path / "map.png"
    Image.new("RGBA", (1800, 923)).save(png, dpi=(100, 100))  # an 18 in world map
    seen = []
    monkeypatch.setattr(cli, "pick_pins", lambda path, out, frame, *rest: seen.append(frame))
    cli.main(["--pick", str(tmp_path / "p.txt"), str(png)])
    cli.main(["--pick", str(tmp_path / "p.txt"), "--size", "30", str(png)])
    assert seen[0].width_in == pytest.approx(18, abs=1e-3) and seen[1].width_in == 30


def test_pick_on_a_render_of_another_map_exits(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from PIL import Image

    from worldmap import cli

    png = tmp_path / "map.png"
    Image.new("RGBA", (1800, 923)).save(png)  # the world's shape
    monkeypatch.setattr(cli, "pick_pins", lambda *a: pytest.fail("opened the picker"))
    with pytest.raises(SystemExit, match="pass the --crop"):
        cli.main(["--pick", str(tmp_path / "p.txt"), "--crop", "-5:9,42:51", str(png)])


def test_centre_crop_from_the_command_line() -> None:
    from worldmap.cli import make_map_frame, parse_args
    from worldmap.crop import centre_frame

    assert make_map_frame(parse_args(["--crop", "-70,-30/1000km,800km"])).crs == centre_frame((-70, -30, 1000 / 111.19508, 800 / 111.19508)).crs
    with pytest.raises(SystemExit):
        make_map_frame(parse_args(["--crop", "10,50/20"]))


def test_jordan_name_not_in_the_dead_sea() -> None:
    from shapely.geometry import Point

    from worldmap.cli import water_labels
    from worldmap.crop import box_region, crop_frame
    from worldmap.labels.water import WaterLabel
    from worldmap.render import drawable_lakes, merged_lakes

    frame = crop_frame(box_region((35.3, 31.0, 35.7, 31.8)), 0.1)
    lakes = drawable_lakes(frame)
    merged = merged_lakes(lakes)
    _, labels = water_labels({"rivers"}, lakes, merged, frame)
    [jordan] = [lab for lab in labels if isinstance(lab, WaterLabel) and lab.text == "Jordan"]
    assert not frame.to_px(merged, simplify=0).contains(Point(jordan.anchor_x, jordan.anchor_y))
