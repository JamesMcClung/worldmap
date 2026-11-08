from dataclasses import replace

from worldmap.frame import DPI, POSTER_WIDTH_IN, make_frame


def test_world_frame_size_unchanged() -> None:
    f = make_frame()
    assert f.width_in == POSTER_WIDTH_IN and f.height_in == POSTER_WIDTH_IN * f.height / f.width
    assert f.px_per_unit == POSTER_WIDTH_IN * DPI / f.width


def test_tall_frame_long_side_is_poster_width() -> None:
    f = replace(make_frame(), xmin=0, xmax=1, ymin=0, ymax=2)
    assert f.height_in == POSTER_WIDTH_IN and f.width_in == POSTER_WIDTH_IN / 2
    assert f.px_per_unit == POSTER_WIDTH_IN * DPI / 2
