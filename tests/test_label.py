import geopandas as gpd
import matplotlib
import pytest

from worldmap.crop import box_region, crop_frame
from worldmap.data import load_countries, load_disputed
from worldmap.disputed import disputed_label_args, load_disputes
from worldmap.frame import MapFrame, make_frame
from worldmap.labels.label import Label, make_labels
from worldmap.labels.regions import child_regions
from worldmap.render import new_figure


@pytest.fixture(scope="module")
def countries() -> gpd.GeoDataFrame:
    return load_countries()


def labels_in(frame: MapFrame, countries: gpd.GeoDataFrame, territories: bool = False, disputed: bool = False) -> list[Label]:
    """make_labels as cli.main calls it, with every label grey."""
    matplotlib.use("Agg")
    fig, ax = new_figure(frame)
    fig.canvas.draw()
    shown = frame.visible(frame.project(countries))
    children = child_regions(countries, shown, frame) if territories else []
    relabel, extra = disputed_label_args(load_disputes(countries, frame.project(load_disputed())), lambda i, pt: (0.5, 0.5, 0.5)) if disputed else ({}, [])
    labels = make_labels(shown, frame, ax, fig.canvas.get_renderer(), lambda i, pt: (0.5, 0.5, 0.5), children, True, relabel, extra)  # type: ignore[attr-defined]
    import matplotlib.pyplot as plt
    plt.close(fig)
    return labels


def by_name(labels: list[Label]) -> dict[str, Label]:
    return {label.name: label for label in labels}


def test_full_size_is_frame_independent(countries: gpd.GeoDataFrame) -> None:
    world = by_name(labels_in(make_frame(), countries))["France"]
    crop = by_name(labels_in(crop_frame(box_region((-5, 42, 9, 51)), 0), countries))["France"]
    assert world.full_size == pytest.approx(crop.full_size, rel=1e-9)
    assert 5.4e5 <= world.full_size <= 6.6e5
    assert world.kind == "countries"


def test_full_size_of_territories(countries: gpd.GeoDataFrame) -> None:
    labels = by_name(labels_in(make_frame(), countries, territories=True))
    assert labels["Alaska (US)"].kind == "territories"
    assert 1.4e6 <= labels["Alaska (US)"].full_size <= 1.8e6
    assert 6e4 <= labels["Tasmania (Australia)"].full_size <= 7.5e4


def test_full_size_of_breakaways(countries: gpd.GeoDataFrame) -> None:
    labels = by_name(labels_in(make_frame(), countries, disputed=True))
    for name in ("Abkhazia", "Transnistria"):
        assert labels[name].full_size > 0


def test_label_min_size() -> None:
    from test_placement import country_label

    from worldmap.labels.label import SHRINK_MIN_PT

    label = country_label(make_frame(), "Atlantis", -35, 30, 500)
    assert label.min_size == SHRINK_MIN_PT  # the world frame: label_scale 1
    label.label_scale = 2
    assert label.min_size == 2 * SHRINK_MIN_PT
