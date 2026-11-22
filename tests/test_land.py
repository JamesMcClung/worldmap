import pytest

from worldmap.crop import box_region, crop_frame, parse_box
from worldmap.frame import MapFrame, make_frame
from worldmap.labels.curved import LAND_TEXT, Body, CurvedLabel
from worldmap.labels.land import classify, kinds_from_labels, load_land_bodies, make_land_labels, text_of


@pytest.fixture(scope="module")
def frame() -> MapFrame:
    return make_frame()


@pytest.fixture(scope="module")
def world_bodies(frame: MapFrame) -> list[Body]:
    return load_land_bodies({"mountains", "deserts"}, frame)


def test_classify() -> None:
    assert classify("Range/mtn") == "mountains" and classify("Plateau") == "uplands" and classify("Desert") == "deserts"
    assert classify("Plain") == "lowlands" and classify("Gorge") == "lowlands" and classify("Depression") == "lowlands"
    assert classify("Island") is None and classify("Geoarea") is None and classify("Continent") is None


def test_kinds_from_labels() -> None:
    assert kinds_from_labels({"countries", "deserts"}) == {"deserts"}
    assert kinds_from_labels({"seas", "water"}) == set()


def test_text_of() -> None:
    assert text_of(None, "CANADIAN SHIELD") == "Canadian Shield"
    assert text_of(None, "SIERRA MADRE DEL SUR") == "Sierra Madre del Sur"
    assert text_of("Rub' al Khali", "x") == "Rub' al Khali"
    assert text_of(float("nan"), "Alps") == "Alps"
    assert text_of("Ahag", "AHAGGAR MTS.") == "Ahaggar Mountains"  # Natural Earth's English names that are wrong
    assert text_of("Cruach nam Miseag", "Grampian Mts.") == "Grampian Mountains"
    assert text_of("Cordillera Blanca", "CORD. CENTRAL") == "Cordillera Central"
    assert text_of("Appennino Ligure", "APPENNINI") == "Apennines"  # the whole range, not its Ligurian end
    assert text_of("Nullabor Plain", "NULLARBOR PLAIN") == "Nullarbor Plain"
    assert text_of("Stanovoy Range", "STANOVOY UPLAND") == "Stanovoy Upland"  # not the range
    assert text_of("Pamir mountains", "PAMIRS") == "Pamir Mountains"
    assert text_of("Amazon basin", "x") == "Amazon Basin" and text_of("Pontic-Caspian steppe", "x") == "Pontic-Caspian Steppe"


def test_load_land_bodies(world_bodies: list[Body]) -> None:
    texts = [b.text for b in world_bodies]
    assert {b.kind for b in world_bodies} == {"mountains", "deserts"}
    sizes = [b.full_size for b in world_bodies]
    assert sizes == sorted(sizes, reverse=True)
    sahara = next(b for b in world_bodies if b.text == "Sahara")
    assert sahara.full_size > 8e6
    assert {"Himalayas", "Alps", "Andes"} <= set(texts) and "Tibetan Plateau" not in texts


def test_load_land_bodies_full_size_is_whole(world_bodies: list[Body]) -> None:
    west_alps = crop_frame(box_region(parse_box("5:9,44:47")), 0)
    [alps] = [b for b in load_land_bodies({"mountains"}, west_alps) if b.text == "Alps"]
    assert alps.full_size == pytest.approx(next(b for b in world_bodies if b.text == "Alps").full_size)


def test_make_land_labels(frame: MapFrame) -> None:
    andes = [b for b in load_land_bodies({"mountains"}, frame) if b.text == "Andes"]
    [label] = make_land_labels(andes, frame)
    assert isinstance(label, CurvedLabel)
    assert label.style.italic and label.style.text == LAND_TEXT and label.style.tracking_em == 0
    first, last = label.parts[0], label.parts[-1]
    dx = (last[0] + last[2]) / 2 - (first[0] + first[2]) / 2
    dy = (last[1] + last[3]) / 2 - (first[1] + first[3]) / 2
    assert abs(dy) > abs(dx)  # along the range, not across it


def test_same_name_pieces_are_one_feature(world_bodies: list[Body]) -> None:
    """Rows of one name that touch (the Transantarctic Mountains, split at +-180) are one feature, ranked by their
    whole km2; far apart (the Sierra Nevadas of Spain and the US) they are two."""
    from worldmap.data import load_regions

    regions = load_regions()
    split = regions[regions.name_en == "Transantarctic Mountains"].km2.sum()
    [transantarctic] = [b for b in world_bodies if b.text == "Transantarctic Mountains"]
    assert transantarctic.full_size == pytest.approx(split)
    assert len([b for b in world_bodies if b.text == "Sierra Nevada"]) == 2
    assert len([b for b in world_bodies if b.text == "Sahara"]) == 1
