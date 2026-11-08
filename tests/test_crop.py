import pytest
import shapely

from worldmap.crop import box_region, crop_frame, parse_box, sphere_centre


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
