import sys

import pytest
import shapely
from shapely.geometry import LineString, Point, box

from worldmap.cli import make_map_frame, parse_args
from worldmap.frame import make_frame
from worldmap.crop import Candidate, box_region, candidates, choose, crop_frame, describe, matches, parse_box, sphere_centre


def test_parse_box_any_corner_order() -> None:
    assert parse_box("5,43:17,49") == parse_box("17,49:5,43") == (5, 43, 17, 49)


def test_parse_box_crosses_antimeridian_by_narrower_span() -> None:
    assert parse_box("170,50:-160,70") == parse_box("-160,70:170,50") == (170, 50, -160, 70)


def test_parse_box_names_and_errors() -> None:
    assert parse_box("France") is None
    assert parse_box("Saint Kitts and Nevis") is None
    with pytest.raises(ValueError):
        parse_box("5,95:17,49")
    assert parse_box("200,0:10,10") == (10, 0, -160, 10)  # past 180: east from 10, the long way round


def test_box_region_splits_at_antimeridian() -> None:
    assert len(shapely.get_parts(box_region((170, 50, -160, 70)))) == 2
    assert len(shapely.get_parts(box_region((5, 43, 17, 49)))) == 1


def test_sphere_centre_across_antimeridian() -> None:
    lon, lat = sphere_centre(box_region((170, 50, -160, 70)))
    assert abs(abs(lon) - 175) < 0.5 and 55 < lat < 65


def test_crop_frame_france_box() -> None:
    f = crop_frame(box_region((-5, 42, 9, 51)), 0)
    assert not f.globe and "+proj=laea" in f.crs
    assert len(f.clip) == 1 and f.clip[0][0] < -5 and f.clip[0][2] > 9
    assert len(f.outline) == 4


def test_crop_frame_margin() -> None:
    tight, loose = crop_frame(box_region((-5, 42, 9, 51)), 0), crop_frame(box_region((-5, 42, 9, 51)), 0.1)
    assert loose.width == pytest.approx(tight.width * 1.2) and loose.height == pytest.approx(tight.height * 1.2)


def test_clip_boxes_across_antimeridian_and_pole() -> None:
    assert len(crop_frame(box_region((170, 50, -160, 70)), 0).clip) == 2
    f = crop_frame(box_region((-180, 80, 180, 90)), 0)
    assert f.clip == ((-180, f.clip[0][1], 180, 90),)


def test_match_ignores_case_and_accents() -> None:
    cands = [Candidate("Réunion (France)", "territory", Point(55, -21))]
    assert matches("reunion", cands) == cands


def test_matches_exact_first() -> None:
    c = [Candidate("Nigeria", "country", Point(8, 9)), Candidate("Niger", "river", Point(5, 12)), Candidate("Niger", "country", Point(9, 17))]
    assert [(m.text, m.kind) for m in matches("niger", c)] == [("Niger", "country"), ("Niger", "river"), ("Nigeria", "country")]


def test_describe_shows_a_box_that_reproduces_the_crop() -> None:
    from worldmap.crop import CROP_MARGIN

    for c in (Candidate("France", "country", box(-5, 42, 9, 51)), Candidate("Norway", "country", box(5, 58, 31, 71)),
              Candidate("Fiji", "country", shapely.union_all([box(177, -19, 180, -16), box(-180, -17, -179, -16)]))):
        name, spec = describe(c).split(") ")
        assert name == f"{c.text} ({c.kind}"
        b = parse_box(spec)
        assert b is not None
        named, again = crop_frame(c.geom, CROP_MARGIN), crop_frame(box_region(b), 0)
        assert again.width == pytest.approx(named.width, rel=0.01) and again.height == pytest.approx(named.height, rel=0.01)
        lon, lat = again.to_lonlat.transform(0, 0)  # the same centre
        assert named.to_map.transform(lon, lat) == pytest.approx((0, 0), abs=0.01 * named.width)


def test_territories_match_their_parents_full_name() -> None:
    cands = candidates({"countries", "territories"}, False)
    found = [c.text for c in matches("united", cands)]
    assert {"United States", "United Kingdom", "Alaska (US)", "Hawaii (US)", "Bermuda (UK)"} <= set(found)
    assert found.index("United States") < found.index("Alaska (US)")
    assert [c.text for c in matches("united states", cands)][:2] == ["United States", "Alaska (US)"]


def test_checklist_has_no_toggle_all_or_invert_keys() -> None:
    from worldmap.crop import checklist

    keys = {b.keys for b in checklist("pick", ["one", "one"]).application.key_bindings.bindings}
    assert ("a",) not in keys and ("i",) not in keys and (" ",) in keys


def test_candidates_only_from_selected_kinds() -> None:
    texts = {c.text for c in candidates({"countries"}, False)}
    assert "India" in texts and not any("OCEAN" in t or "(" in t for t in texts)


def test_country_candidate_excludes_territories() -> None:
    france = next(c for c in candidates({"countries"}, False) if c.text == "France")
    x0, y0, x1, y1 = france.geom.bounds
    assert x0 > -10 and y0 > 40 and x1 < 10  # no Guiana, Réunion, Mayotte


def test_ambiguous_name_without_tty_exits_with_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    found = matches("niger", [Candidate("Niger", "country", Point(9, 17)), Candidate("Nigeria", "country", Point(8, 9))])
    with pytest.raises(SystemExit, match=r"Niger \(country\) [-\d.,:]+\n.*Nigeria \(country\)"):
        choose("niger", found)


def test_no_match_exits() -> None:
    with pytest.raises(SystemExit, match="no label matches"):
        choose("atlantis", [])


def test_topographic_name_crop_searches_countries() -> None:
    frame = make_map_frame(parse_args(["--type", "topographic", "--crop", "France"]))
    assert not frame.globe


def test_name_crop_needs_labels() -> None:
    with pytest.raises(SystemExit, match="--label"):
        make_map_frame(parse_args(["--crop", "France", "--label", "none"]))


def test_parse_box_whole_longitude_range_and_degenerate_boxes() -> None:
    assert parse_box("-180,70:180,90") == (-180, 70, 180, 90)
    assert not crop_frame(box_region((-180, 70, 180, 90)), 0).globe
    for flat in ("10,40:10,50", "10,40:20,40"):
        with pytest.raises(ValueError, match="no area"):
            parse_box(flat)


def test_candidates_across_the_antimeridian_are_finite() -> None:
    import numpy as np

    cands = candidates({"countries", "oceans", "seas"}, False)
    assert all(np.isfinite(shapely.get_coordinates(c.geom)).all() for c in cands)
    for name in ("Fiji", "Gulf of Anadyr'", "SOUTHERN OCEAN"):
        c = next(c for c in cands if c.text == name)
        assert not crop_frame(c.geom, 0.1).globe


def test_label_scale_grows_with_zoom() -> None:
    from worldmap.crop import LABEL_SCALE_MAX

    assert make_frame().label_scale == 1
    assert crop_frame(box_region((-5, 42, 9, 51)), 0.1).label_scale == LABEL_SCALE_MAX  # France: zoom ~25
    assert 1.3 < crop_frame(box_region((-179, 18, -66, 72)), 0.1).label_scale < 2  # US with Alaska and Hawaii: ~2.5


def test_label_sizes_scale_in_a_crop() -> None:
    import matplotlib.pyplot as plt

    from worldmap.labels.label import MAX_PT, MIN_PT, new_label
    from worldmap.render import new_figure

    f = crop_frame(box_region((-5, 42, 9, 51)), 0.1)
    fig, ax = new_figure(f)
    r = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    huge, tiny = f.canvas, f.canvas.centroid.buffer(1)
    assert new_label(ax, r, f, "Big", (0, 0), huge, huge.area, (0, 0, 0)).text.get_fontsize() == pytest.approx(MAX_PT * f.label_scale)
    assert new_label(ax, r, f, "Small", (0, 0), tiny, tiny.area, (0, 0, 0)).text.get_fontsize() == pytest.approx(MIN_PT * f.label_scale)
    plt.close(fig)


def test_wide_regions_use_equal_earth() -> None:
    from worldmap.crop import CROP_MARGIN

    assert "+proj=laea" in crop_frame(box_region((-179, 18, -66, 72)), CROP_MARGIN).crs  # US with Alaska and Hawaii
    pacific = crop_frame(box_region((120, -60, -70, 60)), 0)
    assert "+proj=eqearth" in pacific.crs and "+lon_0=-155" in pacific.crs
    assert len(pacific.outline) == 4  # the canvas, well inside the globe


def test_pacific_by_name_is_one_candidate_and_crops() -> None:
    from worldmap.crop import CROP_MARGIN
    from worldmap.data import load_countries

    found = [c for c in candidates({"oceans"}, False) if c.text == "PACIFIC OCEAN"]
    assert len(found) == 1
    f = crop_frame(found[0].geom, CROP_MARGIN)
    assert "+proj=eqearth" in f.crs
    raw = load_countries()
    countries = f.visible(f.project(raw))
    fiji = countries[countries.NAME_EN == "Fiji"].geometry.iloc[0]
    split = raw[raw.NAME_EN == "Fiji"].geometry.iloc[0]  # 6 of its islands are cut at +-180
    assert fiji.is_valid and len(shapely.get_parts(fiji)) == len(shapely.get_parts(split)) - 3


def test_whole_world_box_shows_the_whole_width() -> None:
    f = crop_frame(box_region((-180, -85, 180, 85)), 0)
    from pyproj import Transformer

    half = Transformer.from_crs("EPSG:4326", "+proj=eqearth +datum=WGS84", always_xy=True).transform(179.99, 0)[0]
    assert f.width == pytest.approx(2 * half, rel=0.01)


def test_several_names() -> None:
    frame = make_map_frame(parse_args(["--crop", "france, germany"]))
    for lon, lat in [(-0.6, 44.8), (13.4, 52.5)]:  # Bordeaux, Berlin
        assert frame.canvas.contains(Point(*frame.to_map.transform(lon, lat)))
    with pytest.raises(SystemExit, match="atlantis"):
        make_map_frame(parse_args(["--crop", "france,atlantis"]))


def test_parse_box_past_180_goes_the_long_way() -> None:
    assert parse_box("114.8,-81.9:305.7,79.6") == pytest.approx((114.8, -81.9, -54.3, 79.6))  # 191 degrees, across +-180
    assert parse_box("305.7,79.6:114.8,-81.9") == pytest.approx((114.8, -81.9, -54.3, 79.6))
    assert parse_box("260,0:480,10") == (-100, 0, 120, 10)  # 220 degrees, not across
    with pytest.raises(ValueError):
        parse_box("-200,0:300,10")  # more than once round


@pytest.mark.parametrize("name", ["PACIFIC OCEAN", "ATLANTIC OCEAN", "INDIAN OCEAN", "SOUTHERN OCEAN", "ARCTIC OCEAN"])  # round a pole too
def test_describe_reproduces_ocean_crops(name: str) -> None:
    from worldmap.crop import CROP_MARGIN

    c = next(c for c in candidates({"oceans"}, False) if c.text == name)
    b = parse_box(describe(c).rsplit(" ", 1)[1])
    assert b is not None
    named, again = crop_frame(c.geom, CROP_MARGIN), crop_frame(box_region(b), 0)
    assert again.width == pytest.approx(named.width, rel=0.05) and again.height == pytest.approx(named.height, rel=0.05)


def test_frame_extent_of_a_crop_round_the_globe() -> None:
    from worldmap.crop import CROP_MARGIN, frame_extent

    west, south, east, north = frame_extent(crop_frame(box(-180, -70, 180, -55), CROP_MARGIN))
    assert (west, east) == (-180, 180) and south < -70 and north > -55


def test_name_crop_prints_its_box(capsys: pytest.CaptureFixture[str]) -> None:
    frame = make_map_frame(parse_args(["--crop", "france,germany"]))
    out = capsys.readouterr().out
    spec = out.split("--crop ")[1].split()[0]
    b = parse_box(spec)
    assert b is not None and crop_frame(box_region(b), 0).width == pytest.approx(frame.width, rel=0.01)


def test_clip_edges_follow_their_parallels() -> None:
    from worldmap.data import load_countries

    f = crop_frame(box_region((50.6, 70.1, 172.5, 90.0)), 0)  # clipped at 56.7N right round the pole
    c = load_countries()
    russia = f.project(c)[c.NAME_EN == "Russia"].geometry.iloc[0]
    assert russia.is_valid and russia.contains(Point(*f.to_map.transform(75, 66.5)))  # the Gydan peninsula
