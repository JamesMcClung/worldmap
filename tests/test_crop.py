import sys

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, box

from worldmap.cli import make_map_frame, parse_args
from worldmap.frame import MapFrame, make_frame
from worldmap.crop import Candidate, box_region, candidates, choose, crop_frame, describe, matches, parse_box, sphere_centre


def test_parse_box_edges() -> None:
    assert parse_box("5:17,43:49") == (5, 43, 17, 49)  # west:east,south:north
    assert parse_box(" 5 : 17 , 43 : 49 ") == (5, 43, 17, 49)


def test_parse_box_open_edges() -> None:
    assert parse_box(":,60:") == (-180, 60, 180, 90)  # everything north of 60N
    assert parse_box(":,:-60") == (-180, -90, 180, -60)
    assert parse_box("-30:60,:") == (-30, -90, 60, 90)
    assert parse_box(":20,0:10") == (-180, 0, 20, 10)


def test_parse_box_west_past_east_crosses_the_antimeridian() -> None:
    assert parse_box("170:-160,50:70") == (170, 50, -160, 70)
    assert parse_box("114.8:-54.3,-81.9:79.6") == (114.8, -81.9, -54.3, 79.6)  # 191 degrees: the Pacific


def test_parse_box_names_and_errors() -> None:
    assert parse_box("France") is None
    assert parse_box("Saint Kitts and Nevis") is None
    for bad in ("5:17,95:49", "200:10,0:10", "5:17,49:43", "10:10,40:50", "5:17,40:40"):  # out of range, north < south, no area
        with pytest.raises(ValueError):
            parse_box(bad)



def test_parse_centre() -> None:
    from worldmap.crop import KM_PER_DEG, parse_centre

    assert parse_centre("10,50/20,10") == (10, 50, 20, 10)  # lon,lat/width,height, bare sizes in degrees
    lon, lat, w, h = parse_centre(" -70 , -30 / 1000km , 500 mi ")  # type: ignore[misc]
    assert (lon, lat) == (-70, -30) and w == pytest.approx(1000 / KM_PER_DEG) and h == pytest.approx(500 * 1.609344 / KM_PER_DEG)
    for name in ("France", "Clutha / Mata-Au", "5:17,43:49"):
        assert parse_centre(name) is None


def test_parse_centre_errors() -> None:
    from worldmap.crop import parse_centre

    for bad in ("200,0/5,5", "0,95/5,5", "10,50/0,5", "10,50/20", "10,50/20,10ft", "10,50/400,10"):
        with pytest.raises(ValueError):
            parse_centre(bad)


def test_centre_crop_is_centred_and_sized_on_the_ground() -> None:
    from pyproj import Geod

    from worldmap.crop import KM_PER_DEG, canvas_centre, centre_frame

    f = centre_frame((2, 46, 20, 10))
    assert "+proj=laea" in f.crs and canvas_centre(f) == pytest.approx((2, 46), abs=1e-6)
    for x, y, half in ((f.xmax, 0, 10), (f.xmin, 0, 10), (0, f.ymax, 5), (0, f.ymin, 5)):  # the edges' middles
        lon, lat = f.to_lonlat.transform(x, y)
        assert Geod(ellps="WGS84").inv(2, 46, lon, lat)[2] / 1000 == pytest.approx(half * KM_PER_DEG, rel=2e-3)


def test_centre_crop_wider_than_azimuthal_is_equal_earth() -> None:
    from worldmap.crop import canvas_centre, centre_frame

    f = centre_frame((-160, 0, 160, 100))
    assert "+proj=eqearth" in f.crs and canvas_centre(f) == pytest.approx((-160, 0), abs=1e-6)


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


def same_map(a: MapFrame, b: MapFrame, tolerance: float) -> bool:
    """Whether b shows what a does: the same size, and its canvas centred on a's."""
    centre = b.to_lonlat.transform((b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2)
    x, y = a.to_map.transform(*centre)
    off = np.hypot(x - (a.xmin + a.xmax) / 2, y - (a.ymin + a.ymax) / 2) / a.width
    return bool(abs(b.width / a.width - 1) < tolerance and abs(b.height / a.height - 1) < tolerance and off < tolerance)


def test_describe_shows_a_box_that_reproduces_the_crop() -> None:
    from worldmap.crop import CROP_MARGIN

    for c in (Candidate("France", "country", box(-5, 42, 9, 51)), Candidate("Norway", "country", box(5, 58, 31, 71)),
              Candidate("Fiji", "country", shapely.union_all([box(177, -19, 180, -16), box(-180, -17, -179, -16)]))):
        name, spec = describe(c).split(") ")
        assert name == f"{c.text} ({c.kind}"
        b = parse_box(spec)
        assert b is not None
        assert same_map(crop_frame(c.geom, CROP_MARGIN), crop_frame(box_region(b), 0), 0.01)


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


@pytest.mark.parametrize("name", ["PACIFIC OCEAN", "ATLANTIC OCEAN", "INDIAN OCEAN", "SOUTHERN OCEAN", "ARCTIC OCEAN"])  # round a pole too
def test_describe_reproduces_ocean_crops(name: str) -> None:
    from worldmap.crop import CROP_MARGIN

    c = next(c for c in candidates({"oceans"}, False) if c.text == name)
    b = parse_box(describe(c).rsplit(" ", 1)[1])
    assert b is not None
    assert same_map(crop_frame(c.geom, CROP_MARGIN), crop_frame(box_region(b), 0), 0.05)


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



@pytest.mark.parametrize("name", ["France", "PACIFIC OCEAN", "SOUTHERN OCEAN"])  # azimuthal, Equal Earth, round a pole
def test_centre_recipe_reproduces_the_crop(name: str) -> None:
    from worldmap.crop import CROP_MARGIN, centre_frame, centre_recipe, centre_spec, parse_centre

    c = next(c for c in candidates({"countries", "oceans"}, False) if c.text == name)
    recipe = centre_recipe(c.geom, CROP_MARGIN)
    assert recipe is not None
    crop = parse_centre(centre_spec(recipe))
    assert crop is not None and same_map(crop_frame(c.geom, CROP_MARGIN), centre_frame(crop), 0.01)


def test_name_crop_prints_its_centre_too(capsys: pytest.CaptureFixture[str]) -> None:
    from worldmap.crop import centre_frame, parse_centre

    frame = make_map_frame(parse_args(["--crop", "france,germany"]))
    crop = parse_centre(capsys.readouterr().out.split(" or --crop ")[1].split()[0])
    assert crop is not None and same_map(frame, centre_frame(crop), 0.01)


def test_clip_edges_follow_their_parallels() -> None:
    from worldmap.data import load_countries

    f = crop_frame(box_region((50.6, 70.1, 172.5, 90.0)), 0)  # clipped at 56.7N right round the pole
    c = load_countries()
    russia = f.project(c)[c.NAME_EN == "Russia"].geometry.iloc[0]
    assert russia.is_valid and russia.contains(Point(*f.to_map.transform(75, 66.5)))  # the Gydan peninsula


@pytest.mark.parametrize("region", [box(-5, 42, 9, 51), box(-180, 60, 180, 90), box(30, 60, 150, 85), box(120, -60, 290 - 360, 60)])
def test_crops_are_centred_on_their_map(region: shapely.Geometry) -> None:
    f = crop_frame(region, 0.1)
    assert abs(f.xmin + f.xmax) / 2 < 1e-3 * f.width
    if "+proj=laea" in f.crs:
        assert abs(f.ymin + f.ymax) / 2 < 1e-3 * f.height


def test_polar_cap_is_centred_on_the_pole() -> None:
    assert float(crop_frame(box(-180, 60, 180, 90), 0).crs.split("+lat_0=")[1].split()[0]) > 89.99


@pytest.mark.parametrize("name, lat", [("ARCTIC OCEAN", 90), ("SOUTHERN OCEAN", -90)])
def test_regions_round_a_pole_get_a_square_map_centred_on_it(name: str, lat: int) -> None:
    from worldmap.crop import CROP_MARGIN

    c = next(c for c in candidates({"oceans"}, False) if c.text == name)
    f = crop_frame(c.geom, CROP_MARGIN)
    assert f"+lat_0={lat}" in f.crs.replace(".000000", "") and f.width == pytest.approx(f.height)
    assert f.xmin == pytest.approx(-f.xmax) and f.ymin == pytest.approx(-f.ymax)
