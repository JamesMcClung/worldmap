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
    with pytest.raises(ValueError):
        parse_box("200,0:10,10")


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


def test_describe_shows_kind_and_span() -> None:
    colorado = Candidate("Colorado", "river", LineString([(-114.5, 31.8), (-109, 36), (-105.6, 40.5)]))
    assert describe(colorado) == "Colorado (river) 114.5°W 31.8°N to 105.6°W 40.5°N"
    fiji = Candidate("Fiji", "country", shapely.union_all([box(177, -19, 180, -16), box(-180, -17, -179, -16)]))
    assert describe(fiji) == "Fiji (country) 177.0°E 19.0°S to 179.0°W 16.0°S"


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
    with pytest.raises(SystemExit, match=r"Niger \(country\) 9.0°E 17.0°N to 9.0°E 17.0°N\n.*Nigeria \(country\)"):
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
