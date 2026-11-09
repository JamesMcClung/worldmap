import sys
from pathlib import Path

import pytest

from worldmap.data import RIVER_MAX_RANK, is_labeled, load_countries, load_marine, load_rivers, parent_index


def test_dependencies_and_parents() -> None:
    c = load_countries()
    parents = parent_index(c)
    names = {c.NAME_EN[d]: c.NAME_EN[p] for d, p in parents.items()}
    assert len(parents) == 46
    assert names["Greenland"] == "Denmark" and names["Hong Kong"] == "People's Republic of China"
    assert names["Falkland Islands"] == "United Kingdom" and names["Aruba"] == "Netherlands"
    assert not {"Baikonur", "Guantanamo Bay Naval Base", "Palestine", "Siachen Glacier"} & names.keys()
    assert all(is_labeled(c.loc[p]) for p in parents.values())


def test_dependencies_not_labeled() -> None:
    c = load_countries()
    labeled = {r.NAME_EN for _, r in c.iterrows() if is_labeled(r)}
    assert not {"Greenland", "Faroe Islands", "Puerto Rico", "Falkland Islands"} & labeled
    assert {"Palestine", "Kosovo", "Western Sahara", "Israel", "Denmark"} <= labeled
    assert not {"Northern Cyprus", "Somaliland", "Baikonur"} & labeled


def test_load_marine() -> None:
    marine = load_marine()
    assert {"name", "featurecla", "scalerank"} <= set(marine.columns)
    assert (marine.featurecla == "ocean").sum() == 7
    assert "Mediterranean Sea" in set(marine.name)


def test_load_rivers() -> None:
    rivers = load_rivers()
    assert rivers.scalerank.max() <= RIVER_MAX_RANK
    assert "Canal" not in set(rivers.featurecla)
    assert {"Amazonas", "Nile", "Mississippi", "Danube"} <= set(rivers.name)


def fake_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tty: bool, answer: str = "") -> list[str]:
    """Point the cache at tmp_path, stub the network, and return the list the fake download appends its URL to."""
    from worldmap import data

    got: list[str] = []
    monkeypatch.setattr(data, "CACHE", tmp_path)
    monkeypatch.setattr(data, "download_size", lambda url: 466_000_000)
    monkeypatch.setattr(data.urllib.request, "urlretrieve", lambda url, part: (got.append(url), Path(part).write_bytes(b"x")))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    return got


def test_fetch_without_a_terminal_stops_before_downloading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from worldmap.data import ETOPO, fetch

    got = fake_download(monkeypatch, tmp_path, tty=False)
    with pytest.raises(SystemExit, match=rf"(?s)466 MB.*{tmp_path}"):
        fetch(ETOPO)
    assert got == []


def test_fetch_asks_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from worldmap.data import ETOPO, fetch

    got = fake_download(monkeypatch, tmp_path, tty=True, answer="n")
    with pytest.raises(SystemExit):
        fetch(ETOPO)
    assert got == []
    got = fake_download(monkeypatch, tmp_path, tty=True, answer="y")
    assert fetch(ETOPO) == tmp_path / ETOPO and len(got) == 1
